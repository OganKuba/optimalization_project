/* rules/rule_block_shuffle.c */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../core/cd_engine.h"

/*
 * Block Shuffle Rule:
 * Randomly shuffles block order at each epoch.
 * Used for block coordinate descent.
 */

// Rule-specific state: stores block permutation
typedef struct {
    int *perm;  // Permutation of block indices
} BPerm;

// Forward declaration
static void blk_cleanup(CDState *st);

/*
 * Initialize the rule:
 * - Allocate block permutation array
 * - Set cleanup function
 * - Seed RNG
 */
static int blk_init(CDState *st) {
    int B = st->n_blocks;

    BPerm *d = malloc(sizeof(BPerm));
    if (!d) return -1;

    d->perm = malloc(B * sizeof(int));
    if (!d->perm) {
        free(d);
        return -1;
    }

    st->rule_data = d;
    st->rule_cleanup = blk_cleanup;

    srand((unsigned)time(NULL));
    return 0;
}

/*
 * Begin each epoch:
 * - Initialize identity permutation [0, ..., B-1]
 * - Shuffle using Fisher-Yates
 */
static void blk_begin(CDState *st, int epoch) {
    (void)epoch;  // Unused
    int B = st->n_blocks;
    BPerm *d = st->rule_data;

    for (int i = 0; i < B; ++i)
        d->perm[i] = i;

    // Shuffle
    for (int i = B - 1; i > 0; --i) {
        int k = rand() % (i + 1);
        int tmp = d->perm[i];
        d->perm[i] = d->perm[k];
        d->perm[k] = tmp;
    }
}

/*
 * Return next coordinate index:
 * - idx is linear index [0, n)
 * - Map it to a coordinate in a shuffled block
 */
static int blk_next(CDState *st, int idx) {
    BPerm *d = st->rule_data;
    int b = idx / st->block_size;        // Block number
    int offset = idx % st->block_size;   // Position within block
    return d->perm[b] * st->block_size + offset;
}

/*
 * Free rule-specific memory
 */
static void blk_cleanup(CDState *st) {
    BPerm *d = (BPerm *)st->rule_data;
    if (!d) return;

    free(d->perm);
    free(d);

    st->rule_data = NULL;
    st->rule_cleanup = NULL;
}

/*
 * Exported rule definition
 */
const CDIndexRule RULE_BLOCK_SHUFFLE = {
    .init        = blk_init,
    .begin_epoch = blk_begin,
    .next_j      = blk_next,
    .end_epoch   = NULL,
    .cleanup     = blk_cleanup
};
