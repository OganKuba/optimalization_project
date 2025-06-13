/*
 *  prox_linear_svrg.c  –  schemat SVRG dla LASSO w silniku CD
 *  (wersja z mini‑batchami, ochroną NaN/Inf i clippingiem gradientu)
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../core/cd_engine.h"
#include "../utils.h"

/* ---------------- helpers ------------------------------------------- */
static inline double X_ij(const double *X, int m, int i, int j)
{ return X[(size_t)j * m + i]; }

static inline int sane(double x)          /* 1 = OK, 0 = NaN/Inf */
{ return !(isnan(x) || isinf(x)); }

/* ---------------- initial snapshot ---------------------------------- */
static int svrg_init(CDState *st)
{
    const int m = st->m, n = st->n;

    memcpy(st->resid_snap, st->y, m * sizeof(double));          /* r̃ = y         */

    for (int j = 0; j < n; ++j) {                               /* g̃ = −Xᵀ r̃    */
        double g = -st->dot(st->X + (size_t)j * m, st->resid_snap, m);
        st->grad_snap[j] = sane(g) ? g : 0.0;
    }
    st->vr_counter = 0;
    return 0;
}

/* ---------------- snapshot refresh ---------------------------------- */
static void maybe_refresh_snapshot(CDState *st)
{
    const int m = st->m;
    const int n = st->n;
    const long every = m / 4;           /* ¼ m zamiast m        */
    if (st->vr_counter % every) return;

    memcpy(st->beta_snap , st->beta , n * sizeof(double));
    memcpy(st->resid_snap, st->y    , m * sizeof(double));

    /* r̃ = y − X β̃ */
    for (int j = 0; j < n; ++j) {
        double bj = st->beta_snap[j];
        if (!sane(bj) || bj == 0.0) continue;
        st->axpy(-bj, st->X + (size_t)j * m, st->resid_snap, m);
    }

    /* g̃ = −Xᵀ r̃ */
    for (int j = 0; j < n; ++j) {
        double g = -st->dot(st->X + (size_t)j * m, st->resid_snap, m);
        st->grad_snap[j] = sane(g) ? g : 0.0;
    }

    st->vr_counter = 0;
}


/* ---------------- SVRG coordinate update --------------------------- */
#define BATCH      100          /* mini‑batch size                      */
#define GRAD_CLIP  1e6         /* zabezpieczenie przed eksplozją       */
#define DELTA_CLIP  0.25

static void prox_linear_svrg_update(CDState *st, int j)
{
    const int m  = st->m;
    const double *Xj = st->X + (size_t)j * m;

    /* --- mini‑batch gradient ---------------------------------------- */
    double g_curr = 0.0, g_snap = 0.0;
    int    used   = 0;

    for (int b = 0; b < BATCH; ++b) {
        int i = rand() % m;
        double r  = st->resid[i];
        double rs = st->resid_snap[i];
        if (!sane(r) || !sane(rs)) continue;

        double xij = X_ij(st->X, m, i, j);
        g_curr += -xij * r;
        g_snap += -xij * rs;
        ++used;
    }
    if (used < BATCH / 2) { ++st->vr_counter; maybe_refresh_snapshot(st); return; }

    double grad_j = ((double)m / used) * (g_curr - g_snap) + st->grad_snap[j];
    if (!sane(grad_j) || fabs(grad_j) > GRAD_CLIP) {
// #ifdef CD_DEBUG
//         printf("❗ grad_j=%e (j=%d) — skip\n", grad_j, j);
// #endif
        ++st->vr_counter; maybe_refresh_snapshot(st); return;
    }

    double L_j = st->norm2[j];
    if (L_j < 1e-12 || !sane(L_j)) { ++st->vr_counter; maybe_refresh_snapshot(st); return; }

    /* --- prox‑linear step ------------------------------------------ */
    double z        = st->beta[j] - grad_j / L_j;
    double new_beta = shrink(z, st->lam / L_j);
    if (!sane(new_beta)) { ++st->vr_counter; maybe_refresh_snapshot(st); return; }

    double delta = new_beta - st->beta[j];
    if (fabs(delta) < 1e-12) { ++st->vr_counter; maybe_refresh_snapshot(st); return; }
    if (fabs(delta) > DELTA_CLIP) delta = copysign(DELTA_CLIP, delta);

    st->beta[j] = st->beta[j] + delta;
    if (fabs(st->beta[j]) > 50.0)                 /* twardy limit */
        st->beta[j] = copysign(50.0, st->beta[j]);

    st->axpy(-delta, Xj, st->resid, m);

// #ifdef CD_DEBUG
//     if (st->vr_counter < 20)
//         printf("[DBG] j=%d  |grad|=%.3e  δ=%.3e  β=%.3e\n",
//                j, fabs(grad_j), delta, st->beta[j]);
// #endif

    ++st->vr_counter;
    maybe_refresh_snapshot(st);
}

/* ---------------- scheme descriptor ---------------------------------- */
const CDUpdateScheme SCHEME_PROX_LINEAR_SVRG = {
    .init     = svrg_init,
    .update_j = prox_linear_svrg_update
};
