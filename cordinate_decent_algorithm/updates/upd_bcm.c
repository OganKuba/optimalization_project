#include "../core/cd_engine.h"
#include "../utils.h"
#include <stddef.h>

/*
 * Block coordinate update:
 * - Computes gradients for a block of coordinates
 * - Applies proximal (Lasso-style) updates
 * - Efficiently updates residuals
 */
static void bcm_update(CDState *st, int j0) /* j0 = index of first coordinate in block */
{
    int bs = st->block_size; // block size
    j0 -= (j0 % bs);
    int m = st->m; // number of data samples (rows)
    double *delta = st->block_tmp; // temporary array to store updates

    /* -------- Compute gradient for each column in the block -------- */
    for (int k = 0; k < bs; ++k) {
        const double *Xk = st->X + (size_t) (j0 + k) * m; // pointer to column j0 + k
        delta[k] = -dot(Xk, st->resid, m); // gradient g_k = -X_k^T * r
    }

    /* -------- Perform proximal update (soft thresholding) -------- */
    for (int k = 0; k < bs; ++k) {
        double Lj = st->norm2[j0 + k]; // squared norm of column j
        double newb = shrink(st->beta[j0 + k] - delta[k] / Lj, st->lam / Lj); // prox step
        delta[k] = newb - st->beta[j0 + k]; // store change in beta
        st->beta[j0 + k] = newb; // update beta
    }

    /* -------- Update residual vector (r := r - X_j * delta_j) -------- */
    for (int k = 0; k < bs; ++k) {
        if (delta[k] != 0.0) {
            axpy(-delta[k], // scale
                 st->X + (size_t) (j0 + k) * m, // column vector X_j
                 st->resid, m); // r := r - delta * X_j
        }
    }
}

/* Exported update scheme: no initialization needed */
const CDUpdateScheme SCHEME_BCM = {
    .init = NULL,
    .update_j = bcm_update
};
