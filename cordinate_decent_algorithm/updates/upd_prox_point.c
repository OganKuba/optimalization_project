#include <stddef.h>
#include "../utils.h"
#include "../core/cd_engine.h"
#include "updates.h"

/*
 * Prox-Point Coordinate Update:
 *
 * Applies a fixed-step proximal gradient step to minimize:
 *     f(β) + λ‖β‖₁
 *
 * Update rule:
 *     β_j ← shrink(β_j - α·∇_j, α·λ),  where α = 1 / L_j
 */
static void prox_point_update(CDState* st, int j)
{
    int m = st->m;
    const double* Xj = st->X + (size_t)j * m;

    // Gradient: ∇_j = -X_jᵀ r
    double grad_j = -dot(Xj, st->resid, m);
    double L_j = st->norm2[j];
    double alpha = 1.0 / L_j;

    // Proximal step
    double prox_center = st->beta[j] - alpha * grad_j;
    double new_beta = shrink(prox_center, st->lam * alpha);

    double delta = new_beta - st->beta[j];
    if (delta == 0.0)
        return;

    // Update β and residual
    st->beta[j] = new_beta;
    axpy(-delta, Xj, st->resid, m);

    // Optionally store gradient
    if (st->grad)
        st->grad[j] = grad_j;
}

/* Exported update scheme using Prox-Point method */
const CDUpdateScheme SCHEME_PROX_POINT = {
    .init = NULL,
    .update_j = prox_point_update
};
