#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <cblas.h>

#include "../core/cd_engine.h"
#include "../utils.h"

/*
 * Nesterov-LS: Low-storage accelerated coordinate descent
 * Based on Wright's formulation (Algorithm 4), uses 2x2 matrix updates
 * with adaptive restart and limited memory (no full vectors needed).
 */

// Internal state structure for low-storage scheme
typedef struct
{
    // 2×2 matrix B_k stored column-wise: [B11 B12; B21 B22]
    double B11, B12, B21, B22;

    // Sparse copies of auxiliary vectors: v̂ and ŷ (only updated entry j)
    double* v_hat;
    double* y_hat;
} LSBuf;

/* Compute γ_{k+1} from γ_k, σ, and n */
static inline double gamma_next(double g_prev, double sigma, int n)
{
    double a = 1.0;
    double b = -1.0 / n - g_prev * g_prev * sigma / n;
    double c = -g_prev * g_prev;
    double disc = b * b - 4.0 * a * c;
    return (-b + sqrt(disc)) / (2.0 * a);
}

/* Initialize scheme state */
static int nesterov_ls_init(CDState* st)
{
    LSBuf* ls = calloc(1, sizeof(LSBuf));
    ls->B11 = ls->B22 = 1.0; // B₀ = I
    ls->B12 = ls->B21 = 0.0;

    ls->v_hat = calloc(st->n, sizeof(double));
    ls->y_hat = calloc(st->n, sizeof(double));

    // v̂₀ = ŷ₀ = β₀
    memcpy(ls->v_hat, st->beta, st->n * sizeof(double));
    memcpy(ls->y_hat, st->beta, st->n * sizeof(double));

    st->scheme_data = ls;
    st->gamma_prev = 0.0;
    st->vr_counter = 0;
    return 0;
}

/* Return coordinate j from linear combination of v̂ and ŷ */
static inline double ls_get_yj(const LSBuf* ls, int j)
{
    return ls->v_hat[j] * ls->B12 + ls->y_hat[j] * ls->B22;
}

/* Perform one coordinate update */
static void nesterov_ls_update_j(CDState* st, int j)
{
    LSBuf* ls = (LSBuf*)st->scheme_data;
    const int m = st->m;
    const int n = st->n;
    const double* Xj = st->X + (size_t)j * m;

    // Step parameters (γ, α, β)
    double gamma = gamma_next(st->gamma_prev, st->sigma, n);
    double alpha = (n - gamma * st->sigma) / (gamma * ((double)n * n - st->sigma));
    double beta = 1.0 - gamma * st->sigma / n;

    // Current ỹ_j
    double yj = ls_get_yj(ls, j);

    // Gradient and prox step
    double Lj = st->norm2[j] + st->sigma;
    double grad = -cblas_ddot(m, Xj, 1, st->resid, 1) + st->sigma * yj;
    double x_new = shrink(yj - grad / Lj, st->lam / Lj);
    double dx = x_new - st->beta[j];

    // Update β and residual if changed
    if (dx)
    {
        st->beta[j] = x_new;

        if (fabs(st->beta[j]) > 1e6)
            st->beta[j] = copysign(1e6, st->beta[j]); // safeguard

        cblas_daxpy(m, -dx, Xj, 1, st->resid, 1);
    }

    // Define R_k matrix
    double R11 = beta;
    double R12 = alpha * beta;
    double R21 = 1.0 - beta;
    double R22 = 1.0 - alpha * beta;

    // S_k components (only for row j)
    double coef1 = gamma / Lj;
    double coef2 = 1.0 - alpha + alpha * gamma;
    double Sj1 = coef1 * grad;
    double Sj2 = coef2 * grad;

    // B_{k+1} = B_k · R_k
    double B11 = ls->B11 * R11 + ls->B12 * R21;
    double B12 = ls->B11 * R12 + ls->B12 * R22;
    double B21 = ls->B21 * R11 + ls->B22 * R21;
    double B22 = ls->B21 * R12 + ls->B22 * R22;

    // Check invertibility of B_{k+1}
    double det = B11 * B22 - B12 * B21;
    if (fabs(det) < 1e-12 || !isfinite(det))
    {
        // Restart if matrix is near-singular or NaN/Inf
        ls->B11 = ls->B22 = 1.0;
        ls->B12 = ls->B21 = 0.0;
        st->gamma_prev = 0.0;
        return;
    }

    // Inverse of B_{k+1}
    double Inv11 = B22 / det, Inv12 = -B12 / det;
    double Inv21 = -B21 / det, Inv22 = B11 / det;

    // Update sparse vectors v̂_j and ŷ_j
    double delta_v = Sj1 * Inv11 + Sj2 * Inv21;
    double delta_y = Sj1 * Inv12 + Sj2 * Inv22;
    ls->v_hat[j] -= delta_v;
    ls->y_hat[j] -= delta_y;

    // Optional adaptive restart
    if ((x_new - yj) * dx > 0.0)
    {
        ls->B11 = ls->B22 = 1.0;
        ls->B12 = ls->B21 = 0.0;
        ls->v_hat[j] = ls->y_hat[j] = st->beta[j];
        gamma = 0.0;
    }

    st->gamma_prev = gamma;
    st->vr_counter++;
}

/* Export update scheme */
const CDUpdateScheme SCHEME_NESTEROV_LS = {
    .init = nesterov_ls_init,
    .update_j = nesterov_ls_update_j
};
