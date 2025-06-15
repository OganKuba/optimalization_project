/*
 *  prox_linear_svrg.c – SVRG scheme for Lasso in the CD engine
 *  Uses mini-batches, protects against NaN/Inf, and applies gradient clipping
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../core/cd_engine.h"
#include "../utils.h"

/* ---------- Helpers ---------- */

// Access element (i,j) of column-major matrix
static inline double X_ij(const double* X, int m, int i, int j)
{
    return X[(size_t)j * m + i];
}

// Check if value is finite and real
static inline int sane(double x)
{
    return !(isnan(x) || isinf(x));
}

/* ---------- Initialization: Take Initial Snapshot ---------- */

static int svrg_init(CDState* st)
{
    const int m = st->m, n = st->n;

    memcpy(st->resid_snap, st->y, m * sizeof(double)); // r̃ = y

    // g̃ = -Xᵀ r̃
    for (int j = 0; j < n; ++j)
    {
        double g = -st->dot(st->X + (size_t)j * m, st->resid_snap, m);
        st->grad_snap[j] = sane(g) ? g : 0.0;
    }

    st->vr_counter = 0;
    return 0;
}

/* ---------- Refresh Snapshot Periodically ---------- */

static void maybe_refresh_snapshot(CDState* st)
{
    const int m = st->m;
    const int n = st->n;
    const long every = m / 4; // Refresh every ¼ epoch

    if (st->vr_counter % every) return;

    memcpy(st->beta_snap, st->beta, n * sizeof(double));
    memcpy(st->resid_snap, st->y, m * sizeof(double));

    // r̃ = y − X β̃
    for (int j = 0; j < n; ++j)
    {
        double bj = st->beta_snap[j];
        if (!sane(bj) || bj == 0.0) continue;
        st->axpy(-bj, st->X + (size_t)j * m, st->resid_snap, m);
    }

    // g̃ = −Xᵀ r̃
    for (int j = 0; j < n; ++j)
    {
        double g = -st->dot(st->X + (size_t)j * m, st->resid_snap, m);
        st->grad_snap[j] = sane(g) ? g : 0.0;
    }

    st->vr_counter = 0;
}

/* ---------- SVRG Coordinate Update ---------- */

#define BATCH       100     // Mini-batch size
#define GRAD_CLIP   1e6     // Gradient clip threshold
#define DELTA_CLIP  0.25    // Step size clip

static void prox_linear_svrg_update(CDState* st, int j)
{
    const int m = st->m;
    const double* Xj = st->X + (size_t)j * m;

    // --- 1. Mini-batch gradient ---
    double g_curr = 0.0, g_snap = 0.0;
    int used = 0;

    for (int b = 0; b < BATCH; ++b)
    {
        int i = rand() % m;
        double r = st->resid[i];
        double rs = st->resid_snap[i];
        if (!sane(r) || !sane(rs)) continue;

        double xij = X_ij(st->X, m, i, j);
        g_curr += -xij * r;
        g_snap += -xij * rs;
        ++used;
    }

    if (used < BATCH / 2)
    {
        ++st->vr_counter;
        maybe_refresh_snapshot(st);
        return;
    }

    double grad_j = ((double)m / used) * (g_curr - g_snap) + st->grad_snap[j];

    if (!sane(grad_j) || fabs(grad_j) > GRAD_CLIP)
    {
        ++st->vr_counter;
        maybe_refresh_snapshot(st);
        return;
    }

    // --- 2. Proximal Update ---
    double L_j = st->norm2[j];
    if (L_j < 1e-12 || !sane(L_j))
    {
        ++st->vr_counter;
        maybe_refresh_snapshot(st);
        return;
    }

    double z = st->beta[j] - grad_j / L_j;
    double new_beta = shrink(z, st->lam / L_j);
    if (!sane(new_beta))
    {
        ++st->vr_counter;
        maybe_refresh_snapshot(st);
        return;
    }

    double delta = new_beta - st->beta[j];
    if (fabs(delta) < 1e-12)
    {
        ++st->vr_counter;
        maybe_refresh_snapshot(st);
        return;
    }

    // Clip step size to prevent instability
    if (fabs(delta) > DELTA_CLIP)
        delta = copysign(DELTA_CLIP, delta);

    // --- 3. Update β and residual ---
    st->beta[j] += delta;

    // Hard clamp on β (overflow protection)
    if (fabs(st->beta[j]) > 50.0)
        st->beta[j] = copysign(50.0, st->beta[j]);

    st->axpy(-delta, Xj, st->resid, m);

    ++st->vr_counter;
    maybe_refresh_snapshot(st);

    // Uncomment for debugging
    /*
    if (st->vr_counter < 20)
        printf("[DBG] j=%d |grad|=%.3e δ=%.3e β=%.3e\n",
               j, fabs(grad_j), delta, st->beta[j]);
    */
}

/* ---------- Exported Update Scheme ---------- */

const CDUpdateScheme SCHEME_PROX_LINEAR_SVRG = {
    .init = svrg_init,
    .update_j = prox_linear_svrg_update
};
