from runner import solve_lasso
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.linear_model import Lasso


SCHEMES = [
    b"nesterov", b"nesterov_ls",
    b"prox_lin", b"prox_point", b"prox_linear_ext",
    b"prox_linear_sgd", b"prox_linear_svrg", b"bcm"
]

RULES = [
    b"cyclic", b"shuffle", b"random",
    b"block_shuffle", b"gs_r", b"gsl_r"
]

def lam_grid(X, lam_start_factor, lam_end_factor):
    lam_max = np.max(np.abs(X.T @ X[:, 0]))
    return lam_start_factor * lam_max, lam_end_factor * lam_max

def evaluate(beta, X, y, lam):
    mse = mean_squared_error(y, X @ beta)
    resid = y - X @ beta
    loss = 0.5 * np.sum(resid**2) + lam * np.sum(np.abs(beta))
    nnz = int((np.abs(beta) > 1e-6).sum())
    return mse, loss, nnz

def run_sklearn(X, y, lam_end):
    alpha = lam_end / X.shape[0]
    t0 = time.perf_counter()
    model = Lasso(alpha=alpha,
                  fit_intercept=False,
                  max_iter=50000,
                  tol=1e-4,
                  random_state=SEED)
    model.fit(X, y)
    t1 = time.perf_counter()
    return model.coef_, (t1 - t0)

def pretty(d, key):
    return d.get(key, key.decode() if isinstance(key, bytes) else str(key))

SCHEME_FULL = {
    b"nesterov": "Nesterov", b"nesterov_ls": "Nesterov-LS",
    b"prox_lin": "ProxLinear", b"prox_point": "ProxPoint",
    b"prox_linear_ext": "Prox+Extr", b"prox_linear_sgd": "ProxSGD",
    b"prox_linear_svrg": "ProxSVRG", b"bcm": "BCM"
}

RULE_FULL = {
    b"cyclic": "Cyclic", b"shuffle": "Shuffle", b"random": "Random",
    b"block_shuffle": "BlockShuffle", b"gs_r": "GS-r", b"gsl_r": "GSL-r"
}

def run(dataset, *, lam_start_factor, lam_end_factor, eta):
    X, y = dataset["X"], dataset["y"]
    lam_start, lam_end = lam_grid(X, lam_start_factor, lam_end_factor)

    rows = []
    print(f"\n📊 Dataset: {dataset['name']}")
    print(f"{'scheme':<15} {'rule':<16} {'MSE':>12} {'nnz':>6} {'ep':>6} {'time':>9}")
    print("-" * 64)

    skip_svrg = dataset["name"] == "dense_1000x5000"
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
            try:
                max_ep = 300 if scheme == b"prox_linear_svrg" else 1000
                beta, ep, t = solve_lasso(X, y,
                                          lam_start=lam_start,
                                          lam_end=lam_end,
                                          eta=eta,
                                          rule=rule,
                                          scheme=scheme,
                                          max_epochs=max_ep)
                mse, loss, nnz = evaluate(beta, X, y, lam_end)
                print(f"{pretty(SCHEME_FULL, scheme):<15} "
                      f"{pretty(RULE_FULL, rule):<16} "
                      f"{mse:12.4e} {nnz:6} {ep:6} {t:9.4f}")
                rows.append(dict(ds=dataset["name"],
                                 scheme=scheme.decode(),
                                 rule=rule.decode(),
                                 mse=mse,
                                 loss=loss,
                                 nnz=nnz,
                                 ep=ep,
                                 time=t))
            except Exception as e:
                print(f"{pretty(SCHEME_FULL, scheme):<15} "
                      f"{pretty(RULE_FULL, rule):<16} SKIPPED → {e}")

    alpha = lam_end / X.shape[0]
    t0 = time.perf_counter()
    model = Lasso(alpha=alpha, fit_intercept=False, max_iter=50000, tol=1e-4, random_state=0)
    model.fit(X, y)
    t1 = time.perf_counter()

    beta_sk = model.coef_
    t_sk = t1 - t0
    mse_sk, loss_sk, nnz_sk = evaluate(beta_sk, X, y, lam_end)

    print(f"{'Lasso (sklearn)':<15} {'-':<16} "
          f"{mse_sk:12.4e} {nnz_sk:6} {'-':>6} {t_sk:9.4f}")
    rows.append(dict(ds=dataset["name"],
                     scheme="sklearn",
                     rule="-",
                     mse=mse_sk,
                     loss=loss_sk,
                     nnz=nnz_sk,
                     ep=np.nan,
                     time=t_sk))

    return rows
