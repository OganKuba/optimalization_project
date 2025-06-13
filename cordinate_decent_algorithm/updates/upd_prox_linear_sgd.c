#include "../core/cd_engine.h"
#include "../utils.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BATCH_SZ 256  // Size of the mini-batch for stochastic updates

// Initialization: seed RNG (required for random sampling)
static int sgd_init(CDState *st) {
    (void)st;
    srand((unsigned)time(NULL));
    return 0;
}

/*
 * Computes dot product over a subset of rows (mini-batch)
 * xcol: column of X (feature vector)
 * vec:  residual vector
 * idx:  list of row indices
 * k:    number of samples in the mini-batch
 */
static double dot_subset(const double *xcol, const double *vec, const int *idx, int k) {
    double sum = 0.0;
    for (int p = 0; p < k; ++p)
        sum += xcol[idx[p]] * vec[idx[p]];
    return sum;
}

/*
 * Stochastic prox-linear coordinate update (with mini-batch SGD)
 */
static void prox_linear_sgd_update(CDState *st, int j)
{
    const int m = st->m;
    const double *Xj = st->X + (size_t)j * m;

    /* --- 1. losowy mini‑batch ----------------------------- */
    int idx[BATCH_SZ];
    for (int t = 0; t < BATCH_SZ; ++t)
        idx[t] = rand() % m;

    /* --- 2. nie‑zaniżony gradient ------------------------- */
    double grad_j = -((double)m / BATCH_SZ) *
                    dot_subset(Xj, st->resid, idx, BATCH_SZ);

    /* --- 3. krok prox‑linear ------------------------------ */
    double Lj = st->norm2[j];
    double z  = st->beta[j] - grad_j / Lj;
    double new_beta = shrink(z, st->lam / Lj);
    double delta    = new_beta - st->beta[j];
    if (delta == 0.0) return;
    st->beta[j] = new_beta;

    /* --- 4. **pełna** aktualizacja residuum --------------- */
    /*    (koszt O(m) – i tak tańszy niż 20 k epok!)          */
    for (int i = 0; i < m; ++i)
        st->resid[i] -= delta * Xj[i];

    /* --- 5. co epokę odświeżamy residuum globalnie -------- */
    /*       (zapewnia spójność w obliczu błędów zaokrągleń)  */
    static long step = 0;
    if (++step % st->n == 0) {          /* raz na epokę      */
        memcpy(st->resid, st->y, m * sizeof(double));
        for (int jj = 0; jj < st->n; ++jj)
            if (st->beta[jj] != 0.0)
                st->axpy(-st->beta[jj],
                         st->X + (size_t)jj * m,
                         st->resid, m);
    }

    if (st->grad) st->grad[j] = grad_j;
}


// Exported update scheme: stochastic prox-linear
const CDUpdateScheme SCHEME_PROX_LINEAR_SGD = {
    .init     = sgd_init,
    .update_j = prox_linear_sgd_update
};
