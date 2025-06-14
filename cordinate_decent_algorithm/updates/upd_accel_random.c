/* --------------------------------------------------------------- *
 *  Minimal-Lazy Nesterov (σ=0, bez rv_buf)                        *
 * --------------------------------------------------------------- */
#include <math.h>
#include <cblas.h>
#include <stdio.h>

#include "../core/cd_engine.h"
#include "utils.h"

/* ---------- parametry ---------- */
#define LOG_EVERY 40          /* epok */

/* ---------- γ_{k+1} ------------ */
static inline double gamma_next(double g_prev, int n)
{
    double disc = 1.0 + 4.0 * n * n * g_prev * g_prev;
    return 0.5 * (1.0 + sqrt(disc)) / n;
}

/* ---------- init --------------- */
static int nesterov_init(CDState *st)
{
    st->gamma_prev = 0.0;
    st->vr_counter = 0;
    /* v_buf = beta (start) */
    memcpy(st->v_buf, st->beta, (size_t)st->n * sizeof(double));
    st->sigma = 0.0;           /* koniecznie σ = 0 */
    return 0;
}

/* ---------- pojedynczy krok ---- */
static void nesterov_update_j(CDState *st, int j)
{
    const int m = st->m, n = st->n;
    const double *Xj = st->X + (size_t)j*m;
    double Lj = st->norm2[j];

    double gamma = gamma_next(st->gamma_prev, n);
    double alpha = 1.0 / (gamma * n);          /* β = 1 */

    /* grad = -⟨Xj, resid⟩ */
    double grad = -cblas_ddot(m, Xj, 1, st->resid, 1);

    /* prox-step */
    double yj = alpha * st->v_buf[j] + (1.0 - alpha) * st->beta[j];
    double x_new = shrink(yj - grad / Lj, st->lam / Lj);

    double dx = x_new - st->beta[j];
    st->beta[j] = x_new;
    if (dx)
        cblas_daxpy(m, -dx, Xj, 1, st->resid, 1);

    /* momentum */
    double dv = -(gamma / Lj) * grad;
    st->v_buf[j] += dv;

    st->gamma_prev = gamma;
    st->vr_counter++;

    /* lekkie logowanie */
    if (LOG_EVERY && j == 0 && st->vr_counter % (LOG_EVERY*n) == 0)
        printf("[ep %ld] γ=%.3e α=%.3e ‖β‖=%.3e\n",
               st->vr_counter/n, gamma, alpha,
               sqrt(st->dot(st->beta, st->beta, n)));
}

/* ---------- rejestracja -------- */
const CDUpdateScheme SCHEME_NESTEROV = {
    .init     = nesterov_init,
    .update_j = nesterov_update_j
};
