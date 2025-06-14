/* ------------------------------------------------------------------ *
 *  Lazy-Nesterov (σ = 0) – szybka, losowa aktualizacja współrzędnej  *
 *  Wersja zoptymalizowana:                                           *
 *      – BLAS (OpenBLAS / MKL)                                       *
 *      – alpha = 1/(γ n), beta = 1                                   *
 *      – flush rv_buf co FLUSH_EVERY epok                            *
 * ------------------------------------------------------------------ */
#include <math.h>
#include <string.h>
#include <cblas.h>

#include "../core/cd_engine.h"     /* CDState, axpy/dot prototypy */
#include "utils.h"

/* ----------- Ustawienia praktyczne ------------ */
#define LOG_EVERY       50         /* drukuj co 50 epok (0 – wyłącz) */
#define FLUSH_EVERY     2          /* zagęszczenie rv co 2 epoki     */

/* ---------- pomocnicza: γ_{k+1} (σ = 0) -------- */
static inline double gamma_next(double g_prev, int n)
{
    /* γ² − γ/n − g_prev² = 0  ⇒  γ = (1 + √(1 + 4 n² g²)) / (2n) */
    double disc = 1.0 + 4.0 * n * n * g_prev * g_prev;
    return 0.5 * (1.0 + sqrt(disc)) / n;
}

/* ---------- init -------------------------------- */
static int nesterov_init(CDState *st)
{
    st->gamma_prev = 0.0;       /* γ_{-1} */
    st->vr_counter = 0;
    /* σ zostawiamy = 0.0 – nie zmieniamy w state */
    return 0;
}

/* ---------- pojedyncza aktualizacja j ---------- */
static void nesterov_update_j(CDState *st, int j)
{
    const int    n = st->n;
    const int    m = st->m;
    const double *Xj = st->X + (size_t)j * m;   /* kolumna j */
    double Lj = st->norm2[j];

    /* ---- współczynniki pędu ---- */
    double gamma = gamma_next(st->gamma_prev, n);
    double alpha = 1.0 / (gamma * n);           /* β = 1        */

    /* ---- gradient współrzędny ---- */
    double g_r  = cblas_ddot(m, Xj, 1, st->resid, 1);
    double g_rv = cblas_ddot(m, Xj, 1, st->rv_buf, 1);
    double grad = -(alpha * g_rv + (1.0 - alpha) * g_r);

    /* opcjonalne, lekkie logowanie */
    if (LOG_EVERY && j == 0 &&
        st->vr_counter % (LOG_EVERY * n) == 0)
    {
        double normB = sqrt(st->dot(st->beta, st->beta, n));
        printf("[ep %ld] γ=%.3e α=%.3e ‖β‖=%.3e\n",
               st->vr_counter / n, gamma, alpha, normB);
        fflush(stdout);
    }

    /* ---- prox-krok Lasso (λ₂ = 0) ---- */
    double yj = alpha * st->v_buf[j] + (1.0 - alpha) * st->beta[j];
    double x_new = shrink(yj - grad / Lj, st->lam / Lj);

    double dx = x_new - st->beta[j];
    st->beta[j] = x_new;
    if (dx)
        cblas_daxpy(m, -dx, Xj, 1, st->resid, 1);        /* resid ← resid - dx·Xj */

    /* ---- momentum ---- */
    double dv = -(gamma / Lj) * grad;
    st->v_buf[j] += dv;
    if (dv)
        cblas_daxpy(m, -dv, Xj, 1, st->rv_buf, 1);       /* rv_buf ← rv_buf - dv·Xj */

    /* ---- flush co FLUSH_EVERY epok ---- */
    if (++st->vr_counter % (FLUSH_EVERY * n) == 0)
        memcpy(st->rv_buf, st->resid, (size_t)m * sizeof(double));

    st->gamma_prev = gamma;
}

/* ---------- rejestracja schematu --------------- */
const CDUpdateScheme SCHEME_NESTEROV = {
    .init     = nesterov_init,
    .update_j = nesterov_update_j
};

/* -----------------  **NOTATKI**  -----------------
 * • Macierz X powinna być przechowywana w layout-cie
 *   column-major (Fortran) – wtedy kolumna Xj jest
 *   kolejnym blokiem pamięci i BLAS działa bez kopiowania.
 *
 * • Jeśli chcesz sample'ować współrzędne proporcjonalnie
 *   do L_j (importance sampling): stwórz własny `rule`
 *   w folderze rules/ i zamiast czystego "pure random"
 *   zwracaj indeks j ~ L_j / Σ L_k.
 *
 * • Przy bardzo rzadkich macierzach rozważ powrót do
 *   ręcznych pętli + #pragma omp simd (BLAS nie zawsze
 *   wykorzystuje sparsity).
 * -------------------------------------------------- */
