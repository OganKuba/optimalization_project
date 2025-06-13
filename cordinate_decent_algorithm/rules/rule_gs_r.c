#include "../core/cd_engine.h"
#include <stdlib.h>
#include <math.h>
#include "utils.h"

/* ------------------------------------------------------------------ *
 *  Gauss–Southwell‑r (max step length) & GSL‑r (scaled by 1/sqrt(Lj)) *
 *  Two versions of greedy coordinate selection rule:
 *    - GS‑r: select coordinate with largest |δ_j|
 *    - GSL‑r: select coordinate with largest |δ_j| / sqrt(L_j)
 * ------------------------------------------------------------------ */

// Structure to hold internal data for the rule
typedef struct {
    int   *order;          // permutation of coordinate indices, sorted descending by score
    int    use_lipschitz;  // 0 = GS‑r, 1 = GSL‑r
} GSRData;

/* Initialize for GS‑r variant (unscaled) */
static int gsr_init_gs(CDState *st)
{
    if (!st->grad)
        st->grad = (double*)malloc(st->n * sizeof(double));  // Allocate gradient if not present
    GSRData *d = (GSRData*)malloc(sizeof(GSRData));
    d->order = (int*)malloc(st->n * sizeof(int));
    d->use_lipschitz = 0;
    st->rule_data = d;
    return 0;
}

/* Initialize for GSL‑r variant (scaled by Lipschitz constant) */
static int gsr_init_gsl(CDState *st)
{
    if (!st->grad)
        st->grad = (double*)malloc(st->n * sizeof(double));
    GSRData *d = (GSRData*)malloc(sizeof(GSRData));
    d->order = (int*)malloc(st->n * sizeof(int));
    d->use_lipschitz = 1;
    st->rule_data = d;
    return 0;
}

/* Compute coordinate-wise step δ_j in Lasso (prox-linear update) */
static inline double step_len(const CDState *st, int j)
{
    double gj = st->grad[j];
    double Lj = st->norm2[j];
    double newb = shrink(st->beta[j] - gj / Lj, st->lam / Lj);
    return newb - st->beta[j];  // δ_j
}

/* Helper struct for sorting scores with indices */
typedef struct {
    int idx;
    double score;
} ScoreIndex;

/* Comparator function for qsort: sort by descending score */
static int compare_scores_desc(const void *a, const void *b)
{
    double diff = ((ScoreIndex*)b)->score - ((ScoreIndex*)a)->score;
    return (diff > 0) - (diff < 0);  // equivalent to sign(b - a)
}

/* Begin of epoch: compute full gradient, score for each coordinate, and sort */
static void gsr_begin(CDState *st, int epoch)
{
    (void)epoch;
    int n = st->n, m = st->m;
    const double *X = st->X;
    const double *resid = st->resid;

    // Compute full gradient ∇f = -X^T * resid
    for (int j = 0; j < n; ++j) {
        const double *Xj = X + (size_t)j * m;
        st->grad[j] = -dot(Xj, resid, m);
    }

    GSRData *d = (GSRData*)st->rule_data;

    // Compute score_j = |δ_j| or |δ_j| / sqrt(L_j), depending on variant
    ScoreIndex *scores = (ScoreIndex*)malloc(n * sizeof(ScoreIndex));
    for (int j = 0; j < n; ++j) {
        double abs_delta = fabs(step_len(st, j));
        double score = d->use_lipschitz ? abs_delta / sqrt(st->norm2[j])
                                        : abs_delta;
        scores[j].idx = j;
        scores[j].score = score;
    }

    // Sort coordinates by descending score
    qsort(scores, n, sizeof(ScoreIndex), compare_scores_desc);

    // Store sorted order in d->order[]
    for (int i = 0; i < n; ++i)
        d->order[i] = scores[i].idx;

    free(scores);
}

/* Select next coordinate index from precomputed order */
static int gsr_next(CDState *st, int idx)
{
    GSRData *d = (GSRData*)st->rule_data;
    return d->order[idx];
}

/* Free dynamically allocated memory for the rule */
static void gsr_cleanup(CDState *st)
{
    GSRData *d = (GSRData*)st->rule_data;
    st->rule_data   = NULL;
    st->rule_cleanup = NULL;
    free(d->order);
    free(d);
}

/* --------------------- Two rule instances: GS‑r and GSL‑r --------------------- */
const CDIndexRule RULE_GS_R = {
    .init        = gsr_init_gs,
    .begin_epoch = gsr_begin,
    .next_j      = gsr_next,
    .end_epoch   = NULL,
    .cleanup     = gsr_cleanup
};

const CDIndexRule RULE_GSL_R = {
    .init        = gsr_init_gsl,
    .begin_epoch = gsr_begin,
    .next_j      = gsr_next,
    .end_epoch   = NULL,
    .cleanup     = gsr_cleanup
};
