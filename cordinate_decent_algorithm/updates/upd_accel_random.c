/* ------------------------------------------------------------------ *
 *  Lazy-Nesterov Accelerated Randomized Coordinate Descent           *
 *  O(|X_j|) per step (Lee–Sidford trick, Section 3.5)                *
 * ------------------------------------------------------------------ */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "../core/cd_engine.h"
#include "utils.h"

/* ---------- pomocnicze ------------------------------------------------ */
static inline double gamma_next(double g_prev, double sigma, int n)
{
    if (sigma == 0.0) {
        double disc = 1.0 + 4.0 * n * n * g_prev * g_prev;
        return 0.5 * (1.0 / n) * (1.0 + sqrt(disc));
    } else {
        double A = sigma * g_prev * g_prev + n;
        double B = -(sigma * g_prev * g_prev + 1.0);
        double C = g_prev * g_prev;
        double disc = B * B - 4.0 * A * C;
        return (-B + sqrt(disc)) / (2.0 * A);
    }
}

/* ---------- init ------------------------------------------------------ */
static int nesterov_init(CDState *st)
{
    st->gamma_prev = 0.0;
    st->vr_counter = 0;
    return 0;
}

/* ---------- pojedynczy krok ------------------------------------------ */
static void nesterov_update_j(CDState *st, int j)
{
    int    n   = st->n,   m = st->m;
    double Lj  = st->norm2[j];
    const double *Xj = st->X + (size_t)j * m;

    double gamma = gamma_next(st->gamma_prev, st->sigma, n);
    double alpha = (n - gamma * st->sigma) / (gamma * (n * n - st->sigma));
    double beta  = 1.0 - gamma * st->sigma / n;

    /* --- ∇_j f(y_k): pojedynczy fused dot() --------------------------- */
    double g_r = 0.0, g_rv = 0.0;
    for (int i = 0; i < m; ++i) {
        double x = Xj[i];
        g_r  += x * st->resid[i];
        g_rv += x * st->rv_buf[i];
    }

    double grad = -( alpha * (st->rv_a * g_rv + st->rv_b * g_r)
                   + (1.0 - alpha) * g_r );

    if (st->vr_counter < 1000 && j == 0) {
        double grad_ref = 0.0;
        for (int i = 0; i < m; ++i) {
            double Axj = 0.0;
            for (int jj = 0; jj < st->n; ++jj)
                Axj += st->X[i + jj * m] * st->beta[jj];  // pełne Ax

            double ri = st->y[i] - Axj;  // resid = y - Ax
            grad_ref += Xj[i] * (-ri);   // ∇_j f = -⟨Xj, r⟩
        }

        printf("grad lazy = %.4e   ref = %.4e   ratio = %.4e\n",
               grad, grad_ref, grad / grad_ref);
    }

    if (j == 0 && st->vr_counter % st->n == 0) {
        printf("[epoch %ld] γ = %.4e, α = %.4e, β = %.4e, ‖β‖ = %.4e\n",
               st->vr_counter / st->n, gamma, alpha, beta,
               sqrt(st->dot(st->beta, st->beta, st->n)));
        fflush(stdout);
    }

    /* --- proximal krok ------------------------------------------------ */
    double yj = alpha * (st->v_a * st->beta[j] + st->v_b * st->v_buf[j])
              + (1.0 - alpha) * st->beta[j];

    double x_tilde = yj - grad / Lj;
    double denom   = 1.0 + st->sigma / Lj;
    double x_new   = shrink(x_tilde / denom, st->lam / (Lj + st->sigma));
    double dx = x_new - st->beta[j];
    if (dx != 0.0) st->axpy(-dx, Xj, st->resid, m);
    st->beta[j] = x_new;

    /* --- aktualizacja lazy-skalarów ---------------------------------- */
    double c1 = beta + (1.0 - beta) * alpha;
    double c2 = (1.0 - beta) * (1.0 - alpha);
    st->rv_a *= c1;
    st->rv_b  = c1 * st->rv_b + c2;
    st->v_a  *= beta;
    st->v_b   = beta * st->v_b + (1.0 - beta);

    /* --- rzadka aktualizacja ∆v -------------------------------------- */
    double dv = -(gamma / Lj) * grad;
    if (dv != 0.0) {
        st->v_buf[j] += dv;
        st->axpy(-dv, Xj, st->rv_buf, m);
    }

    /* --- restart co epokę (n kroków) --------------------------------- */
    if (++st->vr_counter % n == 0) {
        for (int i = 0; i < m; ++i)
            st->rv_buf[i] = st->rv_a * st->rv_buf[i] + st->rv_b * st->resid[i];

        st->rv_a = 1.0;
        st->rv_b = 0.0;
        st->v_a  = 0.0;
        st->v_b  = 1.0;

        // 👉 NIE resetujemy gamma_prev (pęd ma pozostać)
        st->gamma_prev = gamma;
    } else {
        st->gamma_prev = gamma;
    }
}

/* ---------- definicja schematu --------------------------------------- */
const CDUpdateScheme SCHEME_NESTEROV = {
    .init     = nesterov_init,
    .update_j = nesterov_update_j
};
