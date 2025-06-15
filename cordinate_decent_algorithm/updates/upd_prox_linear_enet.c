#include "../core/cd_engine.h"
#include "../utils.h"

// Default L2-to-L1 regularization ratio (can be overridden at compile time)
#ifndef ENET_L2_RATIO
#define ENET_L2_RATIO 0.1
#endif

/* ---------- Initialization ---------- */

// No-op init for Elastic Net scheme
static int enet_init(CDState* st)
{
    (void)st;
    return 0;
}

/* ---------- Proximal Operator ---------- */

// Elastic Net proximal operator:
// First apply soft-thresholding (L1), then scale by 1 / (1 + λ₂)
static inline double prox_enet(double theta, double lambda1, double lambda2)
{
    double u = shrink(theta, lambda1);
    return u / (1.0 + lambda2);
}

/* ---------- Update Function ---------- */

// Coordinate-wise update for Elastic Net
static void prox_linear_enet_update(CDState* st, int j)
{
    int m = st->m;
    const double* Xj = st->X + (size_t)j * m;

    // Compute gradient: ∇_j = -X_jᵀ r
    double grad_j = -dot(Xj, st->resid, m);
    double Lj = st->norm2[j];

    // Compute L1 and L2 regularization scaled by Lj
    double lambda1 = st->lam / Lj;
    double lambda2 = (ENET_L2_RATIO * st->lam) / Lj;

    // Intermediate value for proximal update
    double theta = st->beta[j] - grad_j / Lj;

    // Elastic Net proximal step
    double new_beta = prox_enet(theta, lambda1, lambda2);
    double delta = new_beta - st->beta[j];

    if (delta == 0.0)
        return;

    // Update β and residual
    st->beta[j] = new_beta;
    axpy(-delta, Xj, st->resid, m);

    // Optionally update gradient cache
    if (st->grad)
        st->grad[j] = grad_j;
}

/* ---------- Exported Scheme ---------- */

const CDUpdateScheme SCHEME_PROX_LINEAR_ENET = {
    .init = enet_init,
    .update_j = prox_linear_enet_update
};
