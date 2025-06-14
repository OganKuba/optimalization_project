#include <math.h>      // sqrt, fabs
#include <string.h>    // memcpy
#include <stdlib.h>    // calloc, malloc, free
#include <cblas.h>     // cblas_ddot, cblas_daxpy

#include "../core/cd_engine.h"   // CDState, CDUpdateScheme itd.
#include "../utils.h"            // shrink, axpy, dot, itp.

typedef struct {
    /* 2×2 macierz Bk (kolumnowo):  [B11 B12; B21 B22] */
    double B11, B12, B21, B22;
    /* rozrzedzone kopie wektorów  v̂  i  ŷ */
    double *v_hat;
    double *y_hat;
} LSBuf;

static inline double gamma_next(double g_prev, double sigma, int n) {
    double a = 1.0;
    double b = -1.0 / n - g_prev * g_prev * sigma / n;
    double c = -g_prev * g_prev;
    double disc = b * b - 4.0 * a * c;
    return (-b + sqrt(disc)) / (2.0 * a);
}

static int nesterov_ls_init(CDState *st)
{
    LSBuf *ls = calloc(1, sizeof(LSBuf));
    ls->B11 = ls->B22 = 1.0;  /* B0 = I */
    ls->B12 = ls->B21 = 0.0;
    ls->v_hat = calloc(st->n, sizeof(double));
    ls->y_hat = calloc(st->n, sizeof(double));

    /*  v̂0 = β0 , ŷ0 = β0  (bo ṽ0 = ỹ0 = β0) */
    memcpy(ls->v_hat, st->beta, st->n * sizeof(double));
    memcpy(ls->y_hat, st->beta, st->n * sizeof(double));

    st->scheme_data = ls;          /* nowe pole w CDState lub reuse rule_data */
    st->gamma_prev  = 0.0;
    st->vr_counter  = 0;
    return 0;
}

static inline double ls_get_yj(const LSBuf *ls, int j)
{
    /* ỹ_j = v̂_j * B12  +  ŷ_j * B22 */
    return ls->v_hat[j] * ls->B12 + ls->y_hat[j] * ls->B22;
}

static void nesterov_ls_update_j(CDState *st, int j)
{
    LSBuf  *ls = (LSBuf*)st->scheme_data;
    const int m = st->m;
    const double *Xj = st->X + (size_t)j * m;

    /* --------- 1. Parametry kroku (γ, α, β) ---------------- */
    const int n = st->n;
    double gamma = gamma_next(st->gamma_prev, st->sigma, n);
    double alpha = (n - gamma * st->sigma) /
                   (gamma * ((double)n * n - st->sigma));
    double beta  = 1.0 - gamma * st->sigma / n;

    /* --------- 2. Odczyt ỹ_j z  v̂, ŷ  -------------------- */
    double yj   = ls_get_yj(ls, j);

    /* --------- 3. Gradient i krok prox -------------------- */
    double Lj   = st->norm2[j] + st->sigma;
    double grad = -cblas_ddot(m, Xj, 1, st->resid, 1) + st->sigma * yj;
    double x_new = shrink(yj - grad / Lj, st->lam / Lj);
    double dx    = x_new - st->beta[j];

    /* --------- 4. Aktualizacja β i residua ---------------- */
    if (dx) {
        st->beta[j] = x_new;
        cblas_daxpy(m, -dx, Xj, 1, st->resid, 1);
    }

    /* --------- 5. Wyznaczenie elementów R_k --------------- */
    double R11 =  beta;
    double R12 =  alpha * beta;
    double R21 =  1.0 - beta;
    double R22 =  1.0 - alpha * beta;

    /* --------- 6. Wyznaczenie S_k  (tylko rząd j) ---------- */
    double coef1 =  gamma / Lj;                 /* dla v-składnika */
    double coef2 = (1.0 - alpha + alpha*gamma); /* dla y-składnika */
    double Sj1 = coef1 * grad;
    double Sj2 = coef2 * grad;

    /* --------- 7. Zaktualizuj B_{k+1} = B_k R_k ----------- */
    double B11 = ls->B11*R11 + ls->B12*R21;
    double B12 = ls->B11*R12 + ls->B12*R22;
    double B21 = ls->B21*R11 + ls->B22*R21;
    double B22 = ls->B21*R12 + ls->B22*R22;
    ls->B11 = B11;  ls->B12 = B12;
    ls->B21 = B21;  ls->B22 = B22;

    /* --------- 8. (B_{k+1})^{-1}  z dwóch liczb ----------- */
    double det = B11*B22 - B12*B21;
    double Inv11 =  B22 / det,   Inv12 = -B12 / det;
    double Inv21 = -B21 / det,   Inv22 =  B11 / det;

    /* --------- 9. Zaktualizuj  v̂_j  i  ŷ_j  -------------- */
    double delta_v = Sj1*Inv11 + Sj2*Inv21;
    double delta_y = Sj1*Inv12 + Sj2*Inv22;
    ls->v_hat[j] -= delta_v;
    ls->y_hat[j] -= delta_y;

    /* --------- 10. Adaptive restart (opcjonalnie) -------- */
    if ((x_new - yj) * dx > 0.0) {
        ls->B11 = ls->B22 = 1.0;
        ls->B12 = ls->B21 = 0.0;
        ls->v_hat[j] = ls->y_hat[j] = st->beta[j];
        gamma = 0.0;
    }

    st->gamma_prev = gamma;
    st->vr_counter++;
}

const CDUpdateScheme SCHEME_NESTEROV_LS = {
    .init     = nesterov_ls_init,
    .update_j = nesterov_ls_update_j
};


