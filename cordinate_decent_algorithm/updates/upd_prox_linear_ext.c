#include "../core/cd_engine.h"
#include "../utils.h"
#include <stddef.h>

#define OMEGA 0.5  // Fixed extrapolation weight (can be made adaptive)

/* ---------- Initialization ---------- */

static int ext_init(CDState* st)
{
    (void)st;
    return 0;
}

/* ---------- Update Function ---------- */

/*
 * Extrapolated Prox-Linear Update:
 * Applies a momentum-style update by extrapolating β before the proximal step.
 *
 * x̂_j = β_j + ω (β_j - β_prev_j)
 * β_j ← shrink(x̂_j - grad / L_j, λ / L_j)
 */
static void prox_linear_ext_update(CDState* st, int j)
{
    int m = st->m;
    const double* Xj = st->X + (size_t)j * m;

    // Extrapolated point: x̂_j
    double beta_prev = st->beta_prev[j];
    double beta_curr = st->beta[j];
    double x_hat = beta_curr + OMEGA * (beta_curr - beta_prev);

    // Gradient: ∇_j = -X_jᵀ r
    double grad_j = -dot(Xj, st->resid, m);
    double L_j = st->norm2[j];

    // Proximal update at extrapolated point
    double prox_input = x_hat - grad_j / L_j;
    double new_beta = shrink(prox_input, st->lam / L_j);
    double delta = new_beta - beta_curr;

    if (delta == 0.0)
    {
        // No change; just update history
        st->beta_prev[j] = beta_curr;
        return;
    }

    // Update β and residual
    st->beta[j] = new_beta;
    axpy(-delta, Xj, st->resid, m);

    // Optional: update stored gradient
    if (st->grad)
        st->grad[j] = grad_j;

    // Store current beta as previous for next round
    st->beta_prev[j] = beta_curr;
}

/* ---------- Exported Update Scheme ---------- */

const CDUpdateScheme SCHEME_PROX_LINEAR_EXT = {
    .init = ext_init,
    .update_j = prox_linear_ext_update
};
