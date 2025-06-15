#include <stddef.h>
#include "../utils.h"
#include "../core/cd_engine.h"
#include "updates.h"

/*
 * Proximal Linear Update (for Lasso):
 *
 * Performs coordinate-wise update using:
 *   β_j ← shrink( β_j - grad_j / L_j, λ / L_j )
 *
 * where:
 *   grad_j = -X_jᵀ r         (r = y - Xβ)
 *   L_j    = ||X_j||²        (precomputed column norm²)
 *   shrink = soft-thresholding operator
 */
static void prox_linear_update(CDState* st, int j)
{
    int m = st->m;
    const double* Xj = st->X + (size_t)j * m;

    // Compute gradient: grad_j = -X_jᵀ r
    double grad_j = -dot(Xj, st->resid, m);
    double L_j = st->norm2[j];

    // Proximal update (soft-thresholding)
    double new_beta = shrink(st->beta[j] - grad_j / L_j, st->lam / L_j);
    double delta = new_beta - st->beta[j];

    if (delta == 0.0)
        return;

    // Update β and residual
    st->beta[j] = new_beta;
    axpy(-delta, Xj, st->resid, m);

    // Optionally update stored gradient
    if (st->grad)
        st->grad[j] = grad_j;
}

/* Exported coordinate-wise proximal update scheme */
const CDUpdateScheme SCHEME_PROX_LINEAR = {
    .init = NULL,
    .update_j = prox_linear_update
};
