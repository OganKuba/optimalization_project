#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../core/cd_engine.h"

/*
 * Shuffle Rule:
 * At each epoch, randomly permutes coordinate indices using Fisher–Yates.
 * Ensures uniform, unbiased coordinate updates.
 */

// Internal rule state
typedef struct
{
    int* perm; // Random permutation of coordinate indices
} ShufData;

/* ---------- Initialization ---------- */

// Allocate memory and seed RNG
static int shuf_init(CDState* st)
{
    ShufData* d = (ShufData*)malloc(sizeof(ShufData));
    d->perm = (int*)malloc(st->n * sizeof(int));
    st->rule_data = d;

    srand((unsigned)time(NULL));
    return 0;
}

/* ---------- Epoch Setup ---------- */

// Generate a new random permutation
static void shuf_begin(CDState* st, int epoch)
{
    (void)epoch;
    ShufData* d = (ShufData*)st->rule_data;

    for (int j = 0; j < st->n; ++j)
        d->perm[j] = j;

    for (int j = st->n - 1; j > 0; --j)
    {
        int k = rand() % (j + 1);
        int tmp = d->perm[j];
        d->perm[j] = d->perm[k];
        d->perm[k] = tmp;
    }
}

/* ---------- Coordinate Selection ---------- */

// Return j-th index from the shuffled order
static int shuf_next(CDState* st, int idx)
{
    ShufData* d = (ShufData*)st->rule_data;
    return d->perm[idx];
}

/* ---------- Cleanup ---------- */

// Free allocated memory
static void shuf_cleanup(CDState* st)
{
    ShufData* d = (ShufData*)st->rule_data;
    free(d->perm);
    free(d);
    st->rule_data = NULL;
    st->rule_cleanup = NULL;
}

// No-op end-of-epoch function
static void shuf_end(CDState* st)
{
    (void)st;
}

/* ---------- Rule Definition ---------- */

const CDIndexRule RULE_SHUFFLE = {
    .init = shuf_init,
    .begin_epoch = shuf_begin,
    .next_j = shuf_next,
    .end_epoch = shuf_end,
    .cleanup = shuf_cleanup
};
