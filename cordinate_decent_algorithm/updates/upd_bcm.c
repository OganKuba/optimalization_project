#include "../core/cd_engine.h"
#include "../utils.h"
#include <stddef.h>

/*
 * Block Coordinate Minimization (BCM):
 * - Updates a block of coordinates at once
 * - Uses proximal gradient steps (e.g., for Lasso)
 * - Efficiently maintains residuals
 */

// j0: starting coordinate index (will be aligned to block)
static void bcm_update(CDState* st, int j0)
{
    int bs = st->block_size;
    int m = st->m;

    // Align j0 to block boundary
    j0 -= (j0 % bs);

    double* delta = st->block_tmp; // temp buffer for β updates

    // -------- 1. Compute gradient for each coordinate in block --------
    for (int k = 0; k < bs; ++k)
    {
        const double* Xk = st->X + (size_t)(j0 + k) * m;
        delta[k] = -dot(Xk, st->resid, m); // g_k = -X_kᵀ r
    }

    // -------- 2. Apply proximal update (soft-thresholding) ------------
    for (int k = 0; k < bs; ++k)
    {
        double Lj = st->norm2[j0 + k];
        double newb = shrink(st->beta[j0 + k] - delta[k] / Lj, st->lam / Lj);
        delta[k] = newb - st->beta[j0 + k]; // change in β
        st->beta[j0 + k] = newb;
    }

    // -------- 3. Update residual vector: r ← r - X_block * Δ ---------
    for (int k = 0; k < bs; ++k)
    {
        if (delta[k] != 0.0)
        {
            const double* Xk = st->X + (size_t)(j0 + k) * m;
            axpy(-delta[k], Xk, st->resid, m); // r -= delta * Xk
        }
    }
}

/* Exported update scheme: block coordinate update with no init step */
const CDUpdateScheme SCHEME_BCM = {
    .init = NULL,
    .update_j = bcm_update
};
