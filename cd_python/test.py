import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from lasso_ctypes import solve_lasso_cd

m, n, k = 100, 200, 10
np.random.seed(0)

X = np.asfortranarray(StandardScaler().fit_transform(np.random.randn(m, n)))
beta_true = np.zeros(n)
beta_true[np.random.choice(n, k, replace=False)] = np.random.randn(k)
y = X @ beta_true + 0.01 * np.random.randn(m)
y = (y - y.mean()) / y.std()

lam_max = np.max(np.abs(X.T @ y))
lam = 0.1 * lam_max

beta, epochs, t = solve_lasso_cd(  # <-- poprawione!
    X, y,
    lam_start=lam, lam_end=lam,
    tol=1e-6, max_epochs=2000,
    rule=b"shuffle",
    scheme=b"prox_linear_svrg"
)

print("MSE        :", mean_squared_error(y, X @ beta))
print("nnz        :", (np.abs(beta) > 1e-6).sum())
print("epochs     :", epochs)
print(f"time (s)   : {t:.4f}")
