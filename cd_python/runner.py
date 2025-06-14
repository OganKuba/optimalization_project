import time, ctypes as ct, numpy as np
from pathlib import Path

# ---- dynamic lib (dostosuj ścieżkę!) ----
LIB = ct.CDLL(str(Path("/home/kubog/optimalization_project/cordinate_decent_algorithm"
                       "/cmake-build-debug/liblasso_cd.so")))

_run = LIB.lasso_cd_run
_run.restype  = ct.c_int
_run.argtypes = [ct.POINTER(ct.c_double), ct.POINTER(ct.c_double),
                 ct.c_int, ct.c_int,                     # m,n
                 ct.POINTER(ct.c_double),                # beta_out
                 ct.c_double, ct.c_double, ct.c_double,  # λ_start,end,η
                 ct.c_double, ct.c_int,                  # tol,max_epochs
                 ct.c_char_p, ct.c_char_p]               # rule,scheme

def solve_lasso(X, y, *, lam_start, lam_end, eta,
                tol=1e-6, max_epochs=1000,
                rule=b"random", scheme=b"nesterov"):
    Xf   = np.asfortranarray(X, dtype=np.float64)
    y    = np.ascontiguousarray(y, dtype=np.float64)
    m,n  = Xf.shape
    beta = np.zeros(n, dtype=np.float64)

    t0 = time.perf_counter()
    ep = _run(Xf.ctypes.data_as(ct.POINTER(ct.c_double)),
              y .ctypes.data_as(ct.POINTER(ct.c_double)),
              m, n,
              beta.ctypes.data_as(ct.POINTER(ct.c_double)),
              lam_start, lam_end, eta,
              tol, max_epochs,
              rule, scheme)
    t1 = time.perf_counter()
    return beta, ep, t1-t0
