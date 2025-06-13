#include "../core/cd_engine.h"
#include "../utils.h"

// Default ratio of L2 regularization to L1 (can be overridden at compile time)
#ifndef ENET_L2_RATIO
#define ENET_L2_RATIO 0.1
#endif

// Initialization function (does nothing in this case)
static int enet_init(CDState *st) {
    (void)st;
    return 0;
}

/*
 * Proximal operator for Elastic Net:
 * Applies soft-thresholding (L1) followed by scaling (L2).
 */
static inline double prox_enet(double theta, double lambda1, double lambda2)
{
    double u = shrink(theta, lambda1);    // Apply soft-thresholding
    return u / (1.0 + lambda2);           // Apply L2 shrinkage
}

/*
 * Coordinate-wise update for Elastic Net:
 * 1. Compute partial gradient for coordinate j
 * 2. Perform Elastic Net proximal update
 * 3. Update model coefficient and residual vector
 */
static void prox_linear_enet_update(CDState *st, int j)
{
    int m = st->m;  // number of training samples
    const double *Xj = st->X + (size_t)j * m;  // column j of design matrix

    // Compute gradient: g_j = -X_j^T * residual
    double grad_j = -dot(Xj, st->resid, m);
    double Lj = st->norm2[j];  // squared norm of column j

    // Regularization parameters
    double lambda1 = st->lam / Lj;                      // L1 term
    double lambda2 = (ENET_L2_RATIO * st->lam) / Lj;    // L2 term

    // Compute intermediate value theta for proximal update
    double theta = st->beta[j] - grad_j / Lj;

    // Apply Elastic Net proximal operator
    double new_beta = prox_enet(theta, lambda1, lambda2);

    // Compute the change in beta
    double delta = new_beta - st->beta[j];
    if (delta == 0.0)
        return;  // No update needed

    // Update beta
    st->beta[j] = new_beta;

    // Update residual: r = r - delta * X_j
    axpy(-delta, Xj, st->resid, m);

    // Optionally store the gradient (used by some selection rules)
    if (st->grad)
        st->grad[j] = grad_j;
}

/*
 * Elastic Net update scheme structure
 * - Used by the coordinate descent engine
 */
const CDUpdateScheme SCHEME_PROX_LINEAR_ENET = {
    .init     = enet_init,
    .update_j = prox_linear_enet_update
};
