# lasso_ctypes.py
import ctypes as ct
import numpy as np
from pathlib import Path
import time

# automatyczna ścieżka: <this file>/../build/liblasso_cd.so
lib_path = Path("/home/kubog/optimalization_project/cordinate_decent_algorithm/cmake-build-debug/liblasso_cd.so")
lib = ct.CDLL(str(lib_path))

_run = lib.lasso_cd_run
_run.restype  = ct.c_int
_run.argtypes = [ct.POINTER(ct.c_double),  # X
                 ct.POINTER(ct.c_double),  # y
                 ct.c_int, ct.c_int,       # m, n
                 ct.POINTER(ct.c_double),  # beta_out
                 ct.c_double, ct.c_double, ct.c_double,  # λ_start,end,η
                 ct.c_double, ct.c_int,    # tol, max_epochs
                 ct.c_char_p, ct.c_char_p] # rule , scheme


def solve_lasso_cd(X, y,
                   lam_start=1.0, lam_end=0.01, eta=0.8,
                   tol=1e-6, max_epochs=1000,
                   rule=b"shuffle",
                   scheme=b"prox_lin"):
    """Wrapper ‑ zwraca (beta, epochs, time)"""
    Xf = np.asfortranarray(X, dtype=np.float64)   # kolumnowo
    y  = np.ascontiguousarray(y, dtype=np.float64)
    m, n = Xf.shape
    beta = np.zeros(n, dtype=np.float64)

    t0 = time.perf_counter()
    epochs = _run(Xf.ctypes.data_as(ct.POINTER(ct.c_double)),
                  y.ctypes.data_as(ct.POINTER(ct.c_double)),
                  m, n,
                  beta.ctypes.data_as(ct.POINTER(ct.c_double)),
                  lam_start, lam_end, eta,
                  tol, max_epochs,
                  rule, scheme)
    t1 = time.perf_counter()
    return beta, epochs, t1 - t0
