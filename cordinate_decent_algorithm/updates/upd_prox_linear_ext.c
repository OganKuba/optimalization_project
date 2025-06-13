#include "../core/cd_engine.h"
#include "../utils.h"
#include <stddef.h>

#define OMEGA 0.5  // Fixed extrapolation weight (can be generalized later)

static int ext_init(CDState *st) {
    (void)st;
    return 0;
}

static void prox_linear_ext_update(CDState *st, int j)
{
    int m = st->m;
    const double *Xj = st->X + (size_t)j * m;

    // Extrapolated point: x_hat = beta + omega * (beta - beta_prev)
    double beta_prev = st->beta_prev[j];
    double beta_curr = st->beta[j];
    double x_hat = beta_curr + OMEGA * (beta_curr - beta_prev);

    // Gradient approximation at current residual
    double grad_j = -dot(Xj, st->resid, m);
    double L_j = st->norm2[j];

    // Prox-linear step centered at extrapolated point
    double prox_center = x_hat - grad_j / L_j;
    double new_beta = shrink(prox_center, st->lam / L_j);

    double delta = new_beta - beta_curr;
    if (delta == 0.0) {
        // No change; update history and return
        st->beta_prev[j] = beta_curr;
        return;
    }

    // Apply update
    st->beta[j] = new_beta;

    // Update residual: r = r - delta * X_j
    axpy(-delta, Xj, st->resid, m);

    // Optional: store gradient if used elsewhere
    if (st->grad)
        st->grad[j] = grad_j;

    // Update history for next extrapolation
    st->beta_prev[j] = beta_curr;
}

const CDUpdateScheme SCHEME_PROX_LINEAR_EXT = {
    .init     = ext_init,
    .update_j = prox_linear_ext_update
};
