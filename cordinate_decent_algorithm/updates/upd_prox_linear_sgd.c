#include "../core/cd_engine.h"
#include "../utils.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BATCH_SZ 256  // Mini-batch size for SGD updates

/* ---------- Initialization ---------- */

// Seed the random number generator
static int sgd_init(CDState* st)
{
    (void)st;
    srand((unsigned)time(NULL));
    return 0;
}

/* ---------- Utility: Dot product over a mini-batch ---------- */

static double dot_subset(const double* xcol, const double* vec, const int* idx, int k)
{
    double sum = 0.0;
    for (int p = 0; p < k; ++p)
        sum += xcol[idx[p]] * vec[idx[p]];
    return sum;
}

/* ---------- Main SGD Update ---------- */

/*
 * Stochastic coordinate-wise prox-linear update:
 * - Samples a mini-batch of residuals
 * - Approximates gradient
 * - Applies soft-thresholding (Lasso)
 */
static void prox_linear_sgd_update(CDState* st, int j)
{
    const int m = st->m;
    const double* Xj = st->X + (size_t)j * m;

    // 1. Sample mini-batch indices
    int idx[BATCH_SZ];
    for (int t = 0; t < BATCH_SZ; ++t)
        idx[t] = rand() % m;

    // 2. Estimate gradient using sampled rows
    double grad_j = -((double)m / BATCH_SZ) *
        dot_subset(Xj, st->resid, idx, BATCH_SZ);

    // 3. Proximal update
    double Lj = st->norm2[j];
    double z = st->beta[j] - grad_j / Lj;
    double new_beta = shrink(z, st->lam / Lj);
    double delta = new_beta - st->beta[j];

    if (delta == 0.0)
        return;

    st->beta[j] = new_beta;

    // 4. Full residual update (to maintain consistency)
    for (int i = 0; i < m; ++i)
        st->resid[i] -= delta * Xj[i];

    // 5. Recompute residual every full epoch (to correct drift)
    static long step = 0;
    if (++step % st->n == 0)
    {
        memcpy(st->resid, st->y, m * sizeof(double));
        for (int jj = 0; jj < st->n; ++jj)
        {
            if (st->beta[jj] != 0.0)
                st->axpy(-st->beta[jj],
                         st->X + (size_t)jj * m,
                         st->resid, m);
        }
    }

    // Optional: cache gradient
    if (st->grad)
        st->grad[j] = grad_j;
}

/* ---------- Exported Update Scheme ---------- */

const CDUpdateScheme SCHEME_PROX_LINEAR_SGD = {
    .init = sgd_init,
    .update_j = prox_linear_sgd_update
};
