#include <stdlib.h>
#include <math.h>
#include "../core/cd_engine.h"

// Structure to store index ordering for greedy coordinate selection
typedef struct {
    int *order;
} GSSData;

// Initialization: allocate memory for the ordering array
static int gss_init(CDState *st) {
    GSSData *d = (GSSData *) malloc(sizeof(GSSData));
    d->order = (int *) malloc(st->n * sizeof(int));
    st->rule_data = d;
    return 0;
}

// Struct used for sorting gradient values with qsort
typedef struct {
    int index; // Coordinate index
    double abs_grad; // Absolute gradient value
} GradItem;

// Comparison function for qsort: descending order by |gradient|
static int cmp_desc(const void *a, const void *b) {
    double diff = ((GradItem *) b)->abs_grad - ((GradItem *) a)->abs_grad;
    return (diff > 0) - (diff < 0); // Returns -1, 0, or 1
}

// Called once per epoch: rank coordinates by |gradient| in descending order
static void gss_begin(CDState *st, int epoch) {
    (void) epoch;
    GSSData *d = (GSSData *) st->rule_data;

    // Allocate temporary array of gradient values and indices
    GradItem *items = (GradItem *) malloc(st->n * sizeof(GradItem));
    for (int j = 0; j < st->n; ++j) {
        items[j].index = j;
        items[j].abs_grad = fabs(st->grad[j]);
    }

    // Sort using quicksort (O(n log n)) by descending |gradient|
    qsort(items, st->n, sizeof(GradItem), cmp_desc);

    // Extract sorted indices into the order array
    for (int j = 0; j < st->n; ++j) {
        d->order[j] = items[j].index;
    }

    free(items); // Free temporary sort buffer
}

// Return the coordinate index for the current step based on precomputed order
static int gss_next(CDState *st, int idx) {
    GSSData *d = (GSSData *) st->rule_data;
    return d->order[idx];
}

// Cleanup: free allocated memory for ordering data
static void gss_cleanup(CDState *st) {
    GSSData *d = (GSSData *) st->rule_data;
    st->rule_data   = NULL;
    st->rule_cleanup = NULL;
    free(d->order);
    free(d);
}

// Final definition of the greedy sorted coordinate selection rule
const CDIndexRule RULE_GS_S = {
    .init = gss_init, // Called once at start
    .begin_epoch = gss_begin, // Called once per epoch to rank coordinates
    .next_j = gss_next, // Called each iteration to select next index
    .end_epoch = NULL, // Not used
    .cleanup = gss_cleanup // Free memory when done
};
