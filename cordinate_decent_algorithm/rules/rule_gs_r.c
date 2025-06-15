#include "../core/cd_engine.h"
#include <stdlib.h>
#include <math.h>
#include "utils.h"

/* ------------------------------------------------------------------
 * Gauss–Southwell‑r (GS‑r) and GSL‑r coordinate selection rules
 *
 * GS‑r:  selects coordinate with largest |δ_j|
 * GSL‑r: selects coordinate with largest |δ_j| / sqrt(L_j)
 *        (L_j = column norm squared of feature j)
 * ------------------------------------------------------------------ */

// Internal structure holding rule-specific state
typedef struct
{
    int* order; // Sorted coordinate indices
    int use_lipschitz; // 0 = GS‑r, 1 = GSL‑r
} GSRData;

/* ---------------- Initialization ---------------- */

// GS‑r: unscaled scores
static int gsr_init_gs(CDState* st)
{
    if (!st->grad)
        st->grad = (double*)malloc(st->n * sizeof(double));

    GSRData* d = (GSRData*)malloc(sizeof(GSRData));
    d->order = (int*)malloc(st->n * sizeof(int));
    d->use_lipschitz = 0;

    st->rule_data = d;
    return 0;
}

// GSL‑r: scaled scores (by sqrt(L_j))
static int gsr_init_gsl(CDState* st)
{
    if (!st->grad)
        st->grad = (double*)malloc(st->n * sizeof(double));

    GSRData* d = (GSRData*)malloc(sizeof(GSRData));
    d->order = (int*)malloc(st->n * sizeof(int));
    d->use_lipschitz = 1;

    st->rule_data = d;
    return 0;
}

/* ---------------- Epoch Setup ---------------- */

// Lasso-style update step: δ_j = prox result - β_j
static inline double step_len(const CDState* st, int j)
{
    double gj = st->grad[j];
    double Lj = st->norm2[j];
    double newb = shrink(st->beta[j] - gj / Lj, st->lam / Lj);
    return newb - st->beta[j];
}

// Helper for sorting coordinates by score
typedef struct
{
    int idx;
    double score;
} ScoreIndex;

// Sort descending by score
static int compare_scores_desc(const void* a, const void* b)
{
    double diff = ((ScoreIndex*)b)->score - ((ScoreIndex*)a)->score;
    return (diff > 0) - (diff < 0);
}

// Compute scores and sort coordinates
static void gsr_begin(CDState* st, int epoch)
{
    (void)epoch;
    int n = st->n, m = st->m;
    const double *X = st->X, *resid = st->resid;

    // Compute full gradient: ∇f = -Xᵀ r
    for (int j = 0; j < n; ++j)
    {
        const double* Xj = X + (size_t)j * m;
        st->grad[j] = -dot(Xj, resid, m);
    }

    GSRData* d = (GSRData*)st->rule_data;
    ScoreIndex* scores = (ScoreIndex*)malloc(n * sizeof(ScoreIndex));

    // Compute scores
    for (int j = 0; j < n; ++j)
    {
        double delta = fabs(step_len(st, j));
        double score = d->use_lipschitz ? delta / sqrt(st->norm2[j]) : delta;
        scores[j].idx = j;
        scores[j].score = score;
    }

    // Sort by score
    qsort(scores, n, sizeof(ScoreIndex), compare_scores_desc);

    // Store order
    for (int i = 0; i < n; ++i)
        d->order[i] = scores[i].idx;

    free(scores);
}

/* ---------------- Rule Behavior ---------------- */

// Return next coordinate index
static int gsr_next(CDState* st, int idx)
{
    GSRData* d = (GSRData*)st->rule_data;
    return d->order[idx];
}

// Free memory used by the rule
static void gsr_cleanup(CDState* st)
{
    GSRData* d = (GSRData*)st->rule_data;
    if (!d) return;

    free(d->order);
    free(d);

    st->rule_data = NULL;
    st->rule_cleanup = NULL;
}

/* ---------------- Exported Rule Definitions ---------------- */

const CDIndexRule RULE_GS_R = {
    .init = gsr_init_gs,
    .begin_epoch = gsr_begin,
    .next_j = gsr_next,
    .end_epoch = NULL,
    .cleanup = gsr_cleanup
};

const CDIndexRule RULE_GSL_R = {
    .init = gsr_init_gsl,
    .begin_epoch = gsr_begin,
    .next_j = gsr_next,
    .end_epoch = NULL,
    .cleanup = gsr_cleanup
};
