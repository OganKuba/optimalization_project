import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from lasso_ctypes import solve_lasso_cd

# Uwzględniaj też blockową wersję:
rules = [
    b"cyclic", b"shuffle", b"random",
    b"block_shuffle",
    b"gs_r",
    b"gsl_r"
]
schemes = [
    b"prox_lin", b"prox_point",
    b"prox_linear_ext", b"prox_linear_sgd",
    b"prox_linear_svrg",
    b"bcm"
]



# ----- Synthetic -----
m, n, k = 500, 1000, 20
np.random.seed(0)
X_syn = np.random.randn(m, n)
beta_star = np.zeros(n)
beta_star[np.random.choice(n, k, False)] = np.random.randn(k)
y_syn = X_syn @ beta_star + 0.01 * np.random.randn(m)

print("=== Synthetic ===")
for r in rules:
    for s in schemes:
        try:
            β, ep, t = solve_lasso_cd(X_syn, y_syn,
                                      lam_start=1.0, lam_end=0.02, eta=0.85,
                                      rule=r, scheme=s)
        except Exception as e:
            print(f"SKIP rule={r.decode()} scheme={s.decode()} → {e}")
            continue

        mse = np.mean((β - beta_star) ** 2)
        nnz = np.sum(np.abs(β) > 1e-6)
        rel_err = np.linalg.norm(β - beta_star) / np.linalg.norm(beta_star)
        resid = y_syn - X_syn @ β
        lasso_loss = 0.5 * np.sum(resid**2) + 0.02 * np.sum(np.abs(β))
        support_true = set(np.flatnonzero(beta_star))
        support_est  = set(np.flatnonzero(np.abs(β) > 1e-6))
        tp = len(support_true & support_est)
        prec = tp / len(support_est) if support_est else 0.0
        rec  = tp / len(support_true)

        print(f"rule={r.decode():13s} scheme={s.decode():10s} "
              f"MSE={mse:.2e}  loss={lasso_loss:.2f}  "
              f"nnz={nnz:3d}  rel_err={rel_err:.2e}  "
              f"prec={prec:.2f}  rec={rec:.2f}  "
              f"epochs={ep:4d}  time={t:.3f}s")

# ----- Real Data -----
data = fetch_california_housing()
X = StandardScaler().fit_transform(data.data)
y = data.target

print("\n=== California housing ===")
for r in rules:
    for s in schemes:
        try:
            β, ep, t = solve_lasso_cd(X, y,
                                      lam_start=0.05, lam_end=0.001, eta=0.9,
                                      rule=r, scheme=s)
        except Exception as e:
            print(f"SKIP rule={r.decode()} scheme={s.decode()} → {e}")
            continue

        mse = mean_squared_error(y, X @ β)
        nnz = np.sum(np.abs(β) > 1e-6)
        resid = y - X @ β
        lasso_loss = 0.5 * np.sum(resid**2) + 0.01 * np.sum(np.abs(β))

        print(f"rule={r.decode():13s} scheme={s.decode():10s} "
              f"MSE={mse:.4f}  loss={lasso_loss:.3f}  "
              f"nnz={nnz:2d}  epochs={ep:4d}  time={t:.3f}s")
