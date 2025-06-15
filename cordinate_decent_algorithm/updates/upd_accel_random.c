#include <math.h>
#include <string.h>
#include <cblas.h>

#include "../core/cd_engine.h"
#include "utils.h"

/* ---------- Compute γ_{k+1} based on γ_k, σ, and n ---------- */
static inline double gamma_next(double g_prev, double sigma, int n) {
    double a = 1.0;
    double b = -1.0 / n - g_prev * g_prev * sigma / n;
    double c = -g_prev * g_prev;
    double disc = b * b - 4.0 * a * c;
    return (-b + sqrt(disc)) / (2.0 * a);
}

/* ---------- Initialization ---------- */
static int nesterov_fast_init(CDState *st) {
    st->gamma_prev = 0.0;
    st->vr_counter = 0;
    memcpy(st->v_buf, st->beta, (size_t)st->n * sizeof(double));
    return 0;
}

/* ---------- Coordinate Update Step ---------- */
static void nesterov_fast_update_j(CDState *st, int j) {
    int m = st->m, n = st->n;
    const double *Xj = st->X + (size_t)j * m;

    // Momentum parameters
    double gamma = gamma_next(st->gamma_prev, st->sigma, n);
    double alpha = (n - gamma * st->sigma) / (gamma * ((double)n * n - st->sigma));
    double beta  = 1.0 - gamma * st->sigma / n;

    // Gradient and Lipschitz constant
    double Lj   = st->norm2[j] + st->sigma;
    double grad = -cblas_ddot(m, Xj, 1, st->resid, 1) + st->sigma * st->beta[j];

    // Intermediate point y_j
    double yj     = alpha * st->v_buf[j] + (1.0 - alpha) * st->beta[j];
    double x_new  = shrink(yj - grad / Lj, st->lam / Lj);
    double dx     = x_new - st->beta[j];

    // Update beta and residual
    if (dx) {
        st->beta[j] = x_new;
        cblas_daxpy(m, -dx, Xj, 1, st->resid, 1);
    }

    // Adaptive restart condition
    if ((x_new - yj) * dx > 0.0) {
        gamma        = 0.0;
        alpha        = 1.0 / n;
        beta         = 1.0;
        st->v_buf[j] = st->beta[j];
    }

    // Update velocity buffer
    st->v_buf[j] = beta * st->v_buf[j]
                 + (1.0 - beta) * yj
                 - (gamma / Lj) * grad;

    st->gamma_prev = gamma;
    st->vr_counter++;
}

/* ---------- Exported Update Scheme ---------- */
const CDUpdateScheme SCHEME_NESTEROV = {
    .init     = nesterov_fast_init,
    .update_j = nesterov_fast_update_j
};
