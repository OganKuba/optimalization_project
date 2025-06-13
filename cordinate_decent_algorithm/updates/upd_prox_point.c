#include <stddef.h>
#include "../utils.h"
#include "../core/cd_engine.h"
#include "updates.h"

/*
 * Prox-point coordinate update:
 * Uses a fixed step size (alpha = 1 / L_j) to perform a soft-thresholded update.
 * This is a standard surrogate-based approach for minimizing:
 *
 *     f(beta) + lambda * ||beta||_1
 */
static void prox_point_update(CDState *st, int j)
{
    int m = st->m;
    const double *Xj = st->X + (size_t)j * m;

    // Compute gradient: grad_j = -X_j^T * residual
    double grad_j = -dot(Xj, st->resid, m);
    double L_j = st->norm2[j];

    // Fixed step size (alpha), often set to 1 / L_j
    double alpha = 1.0 / L_j;

    // Compute proximal center: beta_j - alpha * grad_j
    double prox_center = st->beta[j] - alpha * grad_j;

    // Apply soft-thresholding with scaled lambda
    double new_beta = shrink(prox_center, st->lam * alpha);

    // Compute update
    double delta = new_beta - st->beta[j];
    if (delta == 0.0)
        return;

    // Apply update to beta_j
    st->beta[j] = new_beta;

    // Update residual: r = r - delta * X_j
    axpy(-delta, Xj, st->resid, m);

    // Optionally store gradient for use by selection rules
    if (st->grad)
        st->grad[j] = grad_j;
}

// Exported update scheme using the prox-point method
const CDUpdateScheme SCHEME_PROX_POINT = {
    .init     = NULL,
    .update_j = prox_point_update
};
