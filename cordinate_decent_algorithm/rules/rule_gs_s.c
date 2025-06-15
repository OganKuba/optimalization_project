#include <stdlib.h>
#include <math.h>
#include "../core/cd_engine.h"

/*
 * Gauss–Southwell‑s (GS‑s) Rule:
 * Selects coordinates with largest |∇_j f(β)| in descending order.
 */

// Internal state for GS‑s rule
typedef struct
{
    int* order; // Sorted coordinate indices by |gradient|
} GSSData;

/* ---------- Initialization ---------- */

// Allocate memory for coordinate order
static int gss_init(CDState* st)
{
    GSSData* d = (GSSData*)malloc(sizeof(GSSData));
    d->order = (int*)malloc(st->n * sizeof(int));
    st->rule_data = d;
    return 0;
}

/* ---------- Sorting Helpers ---------- */

// Struct for pairing gradient values with indices
typedef struct
{
    int index;
    double abs_grad;
} GradItem;

// Sort by descending |gradient|
static int cmp_desc(const void* a, const void* b)
{
    double diff = ((GradItem*)b)->abs_grad - ((GradItem*)a)->abs_grad;
    return (diff > 0) - (diff < 0);
}

/* ---------- Epoch Logic ---------- */

// Compute order of coordinates by |gradient|
static void gss_begin(CDState* st, int epoch)
{
    (void)epoch;
    GSSData* d = (GSSData*)st->rule_data;

    GradItem* items = (GradItem*)malloc(st->n * sizeof(GradItem));
    for (int j = 0; j < st->n; ++j)
    {
        items[j].index = j;
        items[j].abs_grad = fabs(st->grad[j]);
    }

    qsort(items, st->n, sizeof(GradItem), cmp_desc);

    for (int j = 0; j < st->n; ++j)
        d->order[j] = items[j].index;

    free(items);
}

/* ---------- Coordinate Selection ---------- */

// Return coordinate index by precomputed order
static int gss_next(CDState* st, int idx)
{
    GSSData* d = (GSSData*)st->rule_data;
    return d->order[idx];
}

/* ---------- Cleanup ---------- */

// Free allocated memory
static void gss_cleanup(CDState* st)
{
    GSSData* d = (GSSData*)st->rule_data;
    free(d->order);
    free(d);
    st->rule_data = NULL;
    st->rule_cleanup = NULL;
}

/* ---------- Exported Rule Definition ---------- */

const CDIndexRule RULE_GS_S = {
    .init = gss_init,
    .begin_epoch = gss_begin,
    .next_j = gss_next,
    .end_epoch = NULL,
    .cleanup = gss_cleanup
};
