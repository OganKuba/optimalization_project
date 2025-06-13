#include <stddef.h>
#include "../utils.h"
#include "../core/cd_engine.h"
#include "updates.h"

/*
 * Proximal linear update (coordinate-wise for Lasso):
 *
 *  new_beta_j = shrink( beta_j - (1 / L_j) * grad_j , lambda / L_j )
 *
 * where:
 *  - grad_j = -X_j^T * r          (since r = y - X * beta)
 *  - L_j = ||X_j||^2              (squared norm of feature j)
 *  - shrink() is the soft-thresholding operator
 */
static void prox_linear_update(CDState *st, int j)
{
    int m = st->m;  // number of data points
    const double *Xj = st->X + (size_t)j * m;  // pointer to column j of X

    // Compute gradient: grad_j = -X_j^T * residual
    double grad_j = -dot(Xj, st->resid, m);       // partial derivative w.r.t. beta_j
    double L_j = st->norm2[j];                    // precomputed squared norm of X_j

    // Apply proximal (Lasso) update via soft-thresholding
    double new_beta = shrink(st->beta[j] - grad_j / L_j, st->lam / L_j);
    double delta = new_beta - st->beta[j];

    // If no change, exit early
    if (delta == 0.0)
        return;

    // Update beta
    st->beta[j] = new_beta;

    // Update residual vector: r = r - delta * X_j
    axpy(-delta, Xj, st->resid, m);

    // (Optional) store gradient if used by selection rule (e.g., Gauss-Southwell-s)
    if (st->grad)
        st->grad[j] = grad_j;
}

/*
 * Exported update scheme for coordinate-wise prox-linear (e.g., for Lasso)
 */
const CDUpdateScheme SCHEME_PROX_LINEAR = {
    .init      = NULL,                // No initialization needed
    .update_j  = prox_linear_update   // Main update function
};
