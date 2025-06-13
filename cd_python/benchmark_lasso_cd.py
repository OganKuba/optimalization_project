import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from lasso_ctypes import solve_lasso_cd

from sklearn.linear_model import Lasso
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------
RULES = [
    b"cyclic", b"shuffle", b"random",
    b"block_shuffle",
    b"gs_r", b"gsl_r"
]
SCHEMES = [
    b"prox_lin", b"prox_point",
    b"prox_linear_ext", b"prox_linear_sgd",
    b"prox_linear_svrg", b"bcm"
]

SCHEME_FULL = {
    b"prox_lin"        : "Proximal Linear",
    b"prox_point"      : "Proximal Point",
    b"prox_linear_ext" : "Proximal Linear + Extrapolation",
    b"prox_linear_sgd" : "Stochastic Proximal Linear (SGD)",
    b"prox_linear_svrg": "SVRG (Stochastic Variance‑Reduced)",
    b"bcm"             : "Block Coordinate Minimization",
}
RULE_FULL = {
    b"cyclic"        : "Cyclic",
    b"shuffle"       : "Random Shuffle",
    b"random"        : "Pure Random",
    b"block_shuffle" : "Block Shuffle",
    b"gs_r"          : "Gauss–Southwell‑r",
    b"gsl_r"         : "Gauss–Southwell‑Lipschitz‑r",
}

SYNTH_PARAMS = dict(m=500, n=1000, k=20, noise=0.01,
                    lam_start_factor=0.1,  # start at 10 % λ_max
                    lam_end_factor=0.02,   # end 2 % λ_max
                    eta=0.85)
CALIF_PARAMS = dict(lam_start_factor=0.1,  # 10 % λ_max
                    lam_end_factor=0.005, # 0.5 % λ_max  ← słabsza penalizacja
                    eta=0.8)
SEED = 0


# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------
def pretty(dic, key):
    """Zwróć ładną nazwę; jeśli brak – zwróć sam skrót."""
    return dic.get(key, key.decode() if isinstance(key, bytes) else str(key))

def make_synthetic(p: dict):
    rng = np.random.RandomState(SEED)
    X = rng.randn(p['m'], p['n'])
    # standaryzacja + Fortran‑order (kolumny kolejno w pamięci)
    X = np.asfortranarray(StandardScaler().fit_transform(X))

    beta_true = np.zeros(p['n'])
    beta_true[rng.choice(p['n'], p['k'], replace=False)] = rng.randn(p['k'])
    y = X @ beta_true + p['noise'] * rng.randn(p['m'])
    return X, y, beta_true


def run_c_solver(X, y, *, lam_start, lam_end, eta, rule, scheme):
    """Uruchom C‑solver dla pojedynczej kombinacji rule/scheme."""
    tol = 1e-6
    max_epochs = 1500 if b"svrg" in scheme else 1000
    return solve_lasso_cd(X, y,
                          lam_start=lam_start,
                          lam_end=lam_end,
                          eta=eta,
                          tol=tol,
                          max_epochs=max_epochs,
                          rule=rule,
                          scheme=scheme)


def run_sklearn(X, y, lam_end):
    """Dla porównania: scikit‑learn Lasso (koordynatycyk)."""
    alpha = lam_end / X.shape[0]   # poprawiona skala! λ / m
    t0 = time.perf_counter()
    model = Lasso(alpha=alpha,
                  fit_intercept=False,
                  max_iter=50_000,
                  tol=1e-4,
                  random_state=SEED)
    model.fit(X, y)
    t1 = time.perf_counter()
    return model.coef_, (t1 - t0)


def evaluate(beta, X, y, lam):
    """Zwróć MSE, funkcję celu Lasso i liczbę niezerowych współczynników."""
    mse = mean_squared_error(y, X @ beta)
    resid = y - X @ beta
    loss = 0.5 * np.sum(resid ** 2) + lam * np.sum(np.abs(beta))
    nnz = int(np.count_nonzero(np.abs(beta) > 1e-6))
    return mse, loss, nnz


def bench_dataset(name: str, X, y, lam_start, lam_end, eta, beta_true=None):
    rows = []

    # -- sklearn baseline
    beta_sk, t_sk = run_sklearn(X, y, lam_end)
    mse_sk, loss_sk, nnz_sk = evaluate(beta_sk, X, y, lam_end)
    rows.append(dict(dataset=name, rule_full="sklearn", scheme_full="Lasso",
                     mse=mse_sk, loss=loss_sk, nnz=nnz_sk,
                     epochs=np.nan, time=t_sk))

    # Nagłówek tabeli
    print(f"\n📌 Benchmark: {name}")
    print(f"{'scheme':<38s} {'rule':<30s} {'MSE':>12s} {'nnz':>6s} {'epochs':>7s} {'time (s)':>10s}")
    print(f"{'Lasso (sklearn)':<38s} {'-':<30s} {mse_sk:12.8e} {nnz_sk:6d} {'-':>7s} {t_sk:10.6f}")

    # -- pętla scheme × rule
    for scheme in SCHEMES:
        rules_to_test = [b"shuffle"] if scheme in (b"prox_linear_sgd", b"prox_linear_svrg") else RULES
        for rule in rules_to_test:
            try:
                beta, ep, t = run_c_solver(X, y,
                                           lam_start=lam_start,
                                           lam_end=lam_end,
                                           eta=eta,
                                           rule=rule,
                                           scheme=scheme)
            except Exception as e:
                print(f"{pretty(SCHEME_FULL, scheme):<38s} "
                      f"{pretty(RULE_FULL, rule):<30s} SKIPPED → {e}")
                continue

            mse, loss, nnz = evaluate(beta, X, y, lam_end)

            # wypis na ekran
            print(f"{pretty(SCHEME_FULL, scheme):<38s} "
                  f"{pretty(RULE_FULL, rule):<30s} "
                  f"{mse:12.8e} {nnz:6d} {ep:7d} {t:10.6f}")

            # dopisz do rezultatów
            rows.append(dict(dataset=name,
                             rule_full=pretty(RULE_FULL, rule),
                             scheme_full=pretty(SCHEME_FULL, scheme),
                             mse=mse,
                             loss=loss,
                             nnz=nnz,
                             epochs=ep,
                             time=t))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    # ===== Synthetic ====================================================
    X_syn, y_syn, beta_star = make_synthetic(SYNTH_PARAMS)
    lam_max_syn = np.max(np.abs(X_syn.T @ y_syn))

    lam_start_syn = SYNTH_PARAMS['lam_start_factor'] * lam_max_syn
    lam_end_syn   = SYNTH_PARAMS['lam_end_factor']   * lam_max_syn

    df_syn = bench_dataset("synthetic", X_syn, y_syn,
                           lam_start=lam_start_syn,
                           lam_end=lam_end_syn,
                           eta=SYNTH_PARAMS['eta'],
                           beta_true=beta_star)

    # ===== California Housing ===========================================
    data = fetch_california_housing()
    X_cal = np.asfortranarray(StandardScaler().fit_transform(data.data))
    y_cal = data.target

    lam_max_cal = np.max(np.abs(X_cal.T @ y_cal))
    lam_start_cal = CALIF_PARAMS['lam_start_factor'] * lam_max_cal
    lam_end_cal   = CALIF_PARAMS['lam_end_factor']   * lam_max_cal

    df_cal = bench_dataset("california", X_cal, y_cal,
                           lam_start=lam_start_cal,
                           lam_end=lam_end_cal,
                           eta=CALIF_PARAMS['eta'])

    # ===== Save & quick plots ===========================================
    results = pd.concat([df_syn, df_cal], ignore_index=True)
    results.to_csv("lasso_cd_bench.csv", index=False, float_format="%.12f")
    print("\n✅ Wyniki zapisane do lasso_cd_bench.csv")

    def scatter_plot(df, title):
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.scatter(df['time'], df['mse'])
        ax.set_xlabel("czas [s]")
        ax.set_ylabel("MSE")
        ax.set_title(title)
        for _, row in df.iterrows():
            ax.annotate(f"{row['rule_full']}/{row['scheme_full']}",
                        (row['time'], row['mse']),
                        fontsize=6, alpha=0.6)

        plt.tight_layout()
        return fig

    scatter_plot(df_syn[df_syn['rule_full'] != 'sklearn'],
                 "Synthetic: time vs MSE")
    scatter_plot(df_cal[df_cal['rule_full'] != 'sklearn'],
                 "California Housing: time vs MSE")
    plt.show()


if __name__ == "__main__":
    main()
