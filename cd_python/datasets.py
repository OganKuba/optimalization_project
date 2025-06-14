import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

SEED = 0
_rng  = np.random.default_rng(SEED)

# --------- Synthetic ---------- #
def make_dense(m, n, k=20, noise=0.01):
    X = _rng.standard_normal((m, n))
    X = StandardScaler().fit_transform(X)          # zero-mean, unit-var
    beta = np.zeros(n); idx = _rng.choice(n, k, False)
    beta[idx] = _rng.standard_normal(k)
    y = X @ beta + noise * _rng.standard_normal(m)
    return {"name": f"dense_{m}x{n}", "X": X, "y": y}

def make_sparse(m, n, density=0.01, k=20, noise=0.01):
    nnz = int(m*n*density)
    rows = _rng.integers(0, m, nnz)
    cols = _rng.integers(0, n, nnz)
    data = _rng.standard_normal(nnz)
    X = np.zeros((m, n))
    X[rows, cols] = data
    X = StandardScaler().fit_transform(X)          # skala kolumn
    beta = np.zeros(n); idx = _rng.choice(n, k, False)
    beta[idx] = _rng.standard_normal(k)
    y = X @ beta + noise * _rng.standard_normal(m)
    return {"name": f"sparse{density}_{m}x{n}", "X": X, "y": y}

# --------- Real ---------- #
def california():
    data = fetch_california_housing()
    X = StandardScaler().fit_transform(data.data)
    y = data.target
    return {"name": "california", "X": X, "y": y}
