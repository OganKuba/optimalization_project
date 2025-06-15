"""
extended_benchmark.py
Copyright 2025

Run coordinate-descent Lasso experiments with:
  • multiple eta values
  • multiple regularisation strengths (lam_end)
  • all schemes / rules from liblasso_cd
and produce a rich set of visualisations.

Usage:
    python extended_benchmark.py
"""

import os, time, itertools, ctypes as ct, numpy as np
from pathlib import Path
from typing  import List, Dict, Tuple
from multiprocessing import Process, Queue


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso

# ──────────────────────────────────────────────────────────────────────────
#  Helpers: data sets
# ──────────────────────────────────────────────────────────────────────────
SEED = 0
_rng  = np.random.default_rng(SEED)

def make_dense(m: int, n: int, k: int = 20, noise: float = 0.01) -> Dict:
    X = _rng.standard_normal((m, n))
    X = StandardScaler().fit_transform(X)
    beta = np.zeros(n); idx = _rng.choice(n, k, replace=False)
    beta[idx] = _rng.standard_normal(k)
    y = X @ beta + noise * _rng.standard_normal(m)
    return {"name": f"dense_{m}x{n}", "X": X, "y": y}

def run_with_timeout(func, args=(), kwargs={}, timeout=120):
    """Run a function with timeout (in seconds) using a separate process."""
    q = Queue()

    def wrapper(q, *args, **kwargs):
        try:
            result = func(*args, **kwargs)
            q.put(result)
        except Exception as e:
            q.put(e)

    p = Process(target=wrapper, args=(q, *args), kwargs=kwargs)
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        raise TimeoutError(f"⏱️ Execution exceeded {timeout} seconds.")
    if not q.empty():
        result = q.get()
        if isinstance(result, Exception):
            raise result
        return result
    else:
        raise TimeoutError("⏱️ No result returned before timeout.")


def make_sparse(m: int, n: int, density: float = 0.01,
                k: int = 20, noise: float = 0.01) -> Dict:
    nnz = int(m * n * density)
    rows = _rng.integers(0, m, nnz)
    cols = _rng.integers(0, n, nnz)
    data = _rng.standard_normal(nnz)
    X = np.zeros((m, n))
    X[rows, cols] = data
    X = StandardScaler().fit_transform(X)
    beta = np.zeros(n); idx = _rng.choice(n, k, replace=False)
    beta[idx] = _rng.standard_normal(k)
    y = X @ beta + noise * _rng.standard_normal(m)
    return {"name": f"sparse{density}_{m}x{n}", "X": X, "y": y}

def california() -> Dict:
    data = fetch_california_housing()
    X = StandardScaler().fit_transform(data.data)
    y = data.target
    return {"name": "california", "X": X, "y": y}

# ──────────────────────────────────────────────────────────────────────────
#  Helpers: C-wrapper, metrics
# ──────────────────────────────────────────────────────────────────────────
LIB = ct.CDLL(str(
    Path("/home/kubog/optimalization_project/cordinate_decent_algorithm"
         "/cmake-build-debug/liblasso_cd.so")))   # adjust if needed

_run = LIB.lasso_cd_run
_run.restype  = ct.c_int
_run.argtypes = [ct.POINTER(ct.c_double), ct.POINTER(ct.c_double),
                 ct.c_int, ct.c_int,
                 ct.POINTER(ct.c_double),
                 ct.c_double, ct.c_double, ct.c_double,
                 ct.c_double, ct.c_int,
                 ct.c_char_p, ct.c_char_p]

def solve_lasso(X: np.ndarray, y: np.ndarray,
                lam_start: float, lam_end: float, eta: float,
                tol: float       = 1e-6,
                max_epochs: int  = 1000,
                rule: bytes      = b"random",
                scheme: bytes    = b"nesterov"
               ) -> Tuple[np.ndarray, int, float]:
    """Thin wrapper around the C library."""
    X_f = np.asfortranarray(X, dtype=np.float64)
    y_c = np.ascontiguousarray(y, dtype=np.float64)
    m, n = X_f.shape
    beta = np.zeros(n, dtype=np.float64)
    t0 = time.perf_counter()
    ep = _run(X_f.ctypes.data_as(ct.POINTER(ct.c_double)),
              y_c .ctypes.data_as(ct.POINTER(ct.c_double)),
              m, n,
              beta.ctypes.data_as(ct.POINTER(ct.c_double)),
              lam_start, lam_end, eta,
              tol, max_epochs,
              rule, scheme)
    t1 = time.perf_counter()
    return beta, ep, t1 - t0

def lam_grid(X: np.ndarray, lam_start_factor: float,
             lam_end_factor: float) -> Tuple[float, float]:
    lam_max = np.max(np.abs(X.T @ X[:, 0]))
    return lam_start_factor * lam_max, lam_end_factor * lam_max

def evaluate(beta: np.ndarray, X: np.ndarray, y: np.ndarray,
             lam: float) -> Tuple[float, float, int]:
    mse   = mean_squared_error(y, X @ beta)
    resid = y - X @ beta
    loss  = 0.5 * np.sum(resid ** 2) + lam * np.sum(np.abs(beta))
    nnz   = int((np.abs(beta) > 1e-6).sum())
    return mse, loss, nnz

# ──────────────────────────────────────────────────────────────────────────
#  Benchmark core
# ──────────────────────────────────────────────────────────────────────────
SCHEMES  = [b"nesterov", b"nesterov_ls",
            b"prox_lin", b"prox_point",
            b"prox_linear_ext", b"prox_linear_sgd",
            b"prox_linear_svrg", b"bcm"]

RULES    = [b"cyclic", b"shuffle", b"random",
            b"block_shuffle", b"gs_r", b"gsl_r"]

ETA_GRID        = [0.3, 0.5, 0.7, 0.85, 0.95]
LAM_END_GRID_FR = [0.1, 0.01, 0.001]   # × lam_max
TOL_DEFAULT     = 1e-6

def run_experiment(dataset: Dict,
                   lam_start_factor: float,
                   eta_values: List[float],
                   lam_end_fracs: List[float]) -> List[Dict]:
    """Run full sweep on a single data set."""
    rows = []
    X, y = dataset["X"], dataset["y"]
    lam_start, _ = lam_grid(X, lam_start_factor, lam_start_factor * 0.2)  # lam_end dummy
    lam_max = lam_start / lam_start_factor

    skip_svrg = dataset["name"] == "dense_1000x5000"

    for lam_frac in lam_end_fracs:
        lam_end = lam_frac * lam_max
        for eta in eta_values:
            for scheme in SCHEMES:
                if skip_svrg and scheme == b"prox_linear_svrg":
                    continue
                if scheme in (b"nesterov", b"nesterov_ls"):
                    rules_to_test = [b"random"]
                elif scheme in (b"prox_linear_sgd", b"prox_linear_svrg"):
                    rules_to_test = [b"shuffle"]
                else:
                    rules_to_test = RULES
                for rule in rules_to_test:
                    max_ep = 300 if scheme == b"prox_linear_svrg" else 1000
                    try:
                        beta, ep, t = run_with_timeout(
                            solve_lasso,
                            args=(X, y),
                            kwargs=dict(
                                lam_start=lam_start,
                                lam_end=lam_end,
                                eta=eta,
                                tol=TOL_DEFAULT,
                                max_epochs=max_ep,
                                rule=rule,
                                scheme=scheme
                            ),
                            timeout=1800  # 30 minutes
                        )
                        mse, loss, nnz = evaluate(beta, X, y, lam_end)
                        rows.append(dict(ds=dataset["name"],
                                         scheme=scheme.decode(),
                                         rule=rule.decode(),
                                         eta=eta,
                                         lam_end_frac=lam_frac,
                                         mse=mse,
                                         loss=loss,
                                         nnz=nnz,
                                         ep=ep,
                                         time=t))
                        print(f"✔ {dataset['name']} | {scheme.decode():<15} {rule.decode():<12} "
                              f"eta={eta:<4} λ/λmax={lam_frac:<5} "
                              f"→ MSE={mse:.3e}, nnz={nnz}, ep={ep}, time={t:.4f}s")
                    except TimeoutError as te:
                        print(f"⏱️ TIMEOUT: {dataset['name']} | {scheme.decode():<15} {rule.decode():<12} "
                              f"eta={eta:<4} λ/λmax={lam_frac:<5} → skipped after 30 min")
                    except Exception as e:
                        print(f"⚠️  Skipped {dataset['name']} "
                              f"{scheme.decode()}-{rule.decode()} "
                              f"(eta={eta}, lam_frac={lam_frac}): {e}")

        # scikit-learn reference once per lam_end
        alpha = lam_end / X.shape[0]
        t0 = time.perf_counter()
        model = Lasso(alpha=alpha, fit_intercept=False,
                      max_iter=50_000, tol=1e-4, random_state=SEED)
        model.fit(X, y)
        t1 = time.perf_counter()
        mse_sk, loss_sk, nnz_sk = evaluate(model.coef_, X, y, lam_end)
        rows.append(dict(ds=dataset["name"],
                         scheme="sklearn",
                         rule="-",
                         eta=np.nan,
                         lam_end_frac=lam_frac,
                         mse=mse_sk,
                         loss=loss_sk,
                         nnz=nnz_sk,
                         ep=np.nan,
                         time=t1 - t0))
        print(f"✔ {dataset['name']} | sklearn         -           "
              f"eta=     λ/λmax={lam_frac:<5} "
              f"→ MSE={mse_sk:.3e}, nnz={nnz_sk}, time={t1 - t0:.4f}s")
    return rows


# ──────────────────────────────────────────────────────────────────────────
#  Visualisations
# ──────────────────────────────────────────────────────────────────────────
def boxplot_mse(df: pd.DataFrame) -> None:
    sns.boxplot(x="scheme", y="mse", data=df, showfliers=False)
    plt.xticks(rotation=45)
    plt.title("MSE distribution per scheme")
    plt.tight_layout()

def barplot_time(df: pd.DataFrame) -> None:
    df_mean = (df.groupby("scheme", as_index=False)
                 .agg(mean_time=("time", "mean")))
    sns.barplot(x="scheme", y="mean_time", data=df_mean, ci=None)
    plt.xticks(rotation=45)
    plt.ylabel("time [s]")
    plt.title("Mean runtime per scheme")
    plt.tight_layout()

def epochs_plot(df: pd.DataFrame) -> None:
    sns.boxplot(x="scheme", y="ep", data=df)
    plt.yscale("log")
    plt.xticks(rotation=45)
    plt.title("Epochs to convergence (log-scale)")
    plt.tight_layout()

def sparsity_heatmap(df: pd.DataFrame) -> None:
    pivot = df.pivot_table(index="scheme", columns="lam_end_frac",
                           values="nnz", aggfunc="median")
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="viridis")
    plt.title("Median nnz vs λ_end fraction")
    plt.ylabel("")
    plt.tight_layout()

def sklearn_relative(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with mse_rel, time_rel columns."""
    ref = (df[df.scheme == "sklearn"][["ds", "lam_end_frac", "mse", "time"]]
           .rename(columns={"mse": "mse_ref", "time": "time_ref"}))
    merged = df.merge(ref, on=["ds", "lam_end_frac"], how="left")
    merged["mse_rel"]  = merged["mse"]  / merged["mse_ref"]
    merged["time_rel"] = merged["time"] / merged["time_ref"]
    return merged

def best_table(df: pd.DataFrame) -> None:
    """Print best schemes per data set for speed and accuracy."""
    best_mse = (df.loc[df.groupby(["ds", "lam_end_frac"])["mse"].idxmin()]
                  [["ds", "lam_end_frac", "scheme", "rule", "mse"]])
    best_time = (df.loc[df.groupby(["ds", "lam_end_frac"])["time"].idxmin()]
                   [["ds", "lam_end_frac", "scheme", "rule", "time"]])
    print("\n=== Best MSE per data set / λ_end ===")
    for _, r in best_mse.iterrows():
        print(f"{r.ds} (λ_end/λ_max={r.lam_end_frac:.3g}) → "
              f"{r.scheme} [{r.rule}], MSE={r.mse:.4e}")
    print("\n=== Shortest time per data set / λ_end ===")
    for _, r in best_time.iterrows():
        print(f"{r.ds} (λ_end/λ_max={r.lam_end_frac:.3g}) → "
              f"{r.scheme} [{r.rule}], time={r.time:.3f}s")

def loss_relative(df: pd.DataFrame) -> pd.DataFrame:
    ref = df[df.scheme == "sklearn"][["ds", "lam_end_frac", "loss"]].rename(columns={"loss": "loss_ref"})
    merged = df.merge(ref, on=["ds", "lam_end_frac"], how="left")
    merged["loss_rel"] = merged["loss"] / merged["loss_ref"]
    return merged

# ──────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────
def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "4")

    DATASETS = [
        make_dense (500, 1000),
        make_dense (1000, 5000),
        make_sparse(500, 1000, 0.01),
        california()
    ]

    all_rows: List[Dict] = []
    for ds in DATASETS:
        print(f"\n▶️  Running grid on {ds['name']} …")
        rows = run_experiment(ds,
                              lam_start_factor=0.1,
                              eta_values=ETA_GRID,
                              lam_end_fracs=LAM_END_GRID_FR)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_csv = "lasso_cd_bench_ext.csv"
    df.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"\n✔️  Saved {out_csv} ({len(df)} rows)")

    df_nosklearn = df[df.scheme != "sklearn"]

    top_mse_rows = (df_nosklearn
                    .sort_values(by=["ds", "lam_end_frac", "mse"])
                    .groupby(["ds", "lam_end_frac"])
                    .head(5)
                    .assign(type="best_mse"))

    top_time_rows = (df_nosklearn
                     .sort_values(by=["ds", "lam_end_frac", "time"])
                     .groupby(["ds", "lam_end_frac"])
                     .head(5)
                     .assign(type="best_time"))

    sklearn_rows = df[df.scheme == "sklearn"].copy()
    sklearn_rows["type"] = "sklearn"

    best_combined = pd.concat([top_mse_rows, top_time_rows, sklearn_rows],
                              ignore_index=True)
    best_combined.to_csv("best_5_per_ds.csv", index=False, float_format="%.6f")
    print(f"\n✔️  Saved best 5 MSE/time per dataset + sklearn to best_5_per_ds.csv")

    best_time_rows = (df[df.scheme != "sklearn"]
                      .sort_values(by=["ds", "lam_end_frac", "time"])
                      .groupby(["ds", "lam_end_frac"])
                      .first()
                      .reset_index()
                      .assign(type="fastest"))

    best_mse_rows = (df[df.scheme != "sklearn"]
                     .sort_values(by=["ds", "lam_end_frac", "mse"])
                     .groupby(["ds", "lam_end_frac"])
                     .first()
                     .reset_index()
                     .assign(type="most_accurate"))

    best_loss_rows = (
        df[df.scheme != "sklearn"]
        .sort_values(by=["ds", "lam_end_frac", "loss"])
        .groupby(["ds", "lam_end_frac"])
        .first()
        .reset_index()
        .assign(type="min_loss")
    )

    best_time_rows.to_csv("best_fastest.csv", index=False, float_format="%.6f")
    best_mse_rows.to_csv("best_mse.csv", index=False, float_format="%.6f")
    best_loss_rows.to_csv("best_loss.csv", index=False, float_format="%.6f")

    print(f"\n✔️  Saved best (fastest & most accurate) methods per dataset to best_single_method_per_ds.csv")

    # ─── Visualisations ────────────────────────────────────────────────
    sns.set_theme(style="whitegrid", font_scale=0.9)
    plt.figure(figsize=(8, 4));  boxplot_mse(df[df.scheme != "sklearn"]);     plt.savefig("boxplot_mse.png")
    plt.figure(figsize=(8, 4));  barplot_time(df[df.scheme != "sklearn"]);    plt.savefig("barplot_time.png")
    plt.figure(figsize=(8, 4));  epochs_plot(df[df.scheme != "sklearn"]);     plt.savefig("epochs.png")
    plt.figure(figsize=(6, 4));  sparsity_heatmap(df[df.scheme != "sklearn"]);plt.savefig("sparsity_heatmap.png")

    plt.figure(figsize=(8, 4));
    sns.boxplot(x="scheme", y="loss", data=df[df.scheme != "sklearn"], showfliers=False)
    plt.xticks(rotation=45);
    plt.title("Objective Loss per Scheme");
    plt.tight_layout()
    plt.savefig("boxplot_loss.png")

    rel = sklearn_relative(df)
    plt.figure(figsize=(8, 4))
    sns.boxplot(x="scheme", y="mse_rel", data=rel[rel.scheme != "sklearn"])
    plt.xticks(rotation=45); plt.axhline(1, ls="--"); plt.title("MSE vs sklearn"); plt.tight_layout()
    plt.savefig("mse_vs_sklearn.png")

    loss_rel_df = loss_relative(df)
    plt.figure(figsize=(8, 4))
    sns.boxplot(x="scheme", y="loss_rel", data=loss_rel_df[loss_rel_df.scheme != "sklearn"])
    plt.xticks(rotation=45)
    plt.axhline(1, ls="--")
    plt.title("Objective Loss vs sklearn");
    plt.tight_layout()
    plt.savefig("loss_vs_sklearn.png")

    plt.figure(figsize=(8, 4))
    sns.boxplot(x="scheme", y="time_rel", data=rel[rel.scheme != "sklearn"])
    plt.xticks(rotation=45); plt.axhline(1, ls="--"); plt.ylabel("time / sklearn"); plt.title("Runtime vs sklearn")
    plt.tight_layout(); plt.savefig("time_vs_sklearn.png")

    # Show interactive windows last
    plt.show()

    # Text summary
    best_table(df[df.scheme != "sklearn"])


if __name__ == "__main__":
    main()
