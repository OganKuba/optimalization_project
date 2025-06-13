/* rules/rule_block_shuffle.c */

#include <stdio.h>

#include "../core/cd_engine.h"
#include <stdlib.h>
#include <time.h>

/*
 * Block Shuffle Rule:
 * In each epoch, randomly shuffle the order of coordinate blocks.
 * Used in coordinate descent when updates are done block-wise.
 */

// Structure to hold block permutation
typedef struct {
    int *perm;  // permutation of block indices
} BPerm;

static void blk_cleanup(CDState *st);

/*
 * Initialize rule:
 * - Allocate memory for block permutation array
 * - Seed the random number generator
 */


static int blk_init(CDState *st)
{
    int B = st->n_blocks;
    // fprintf(stderr, "[DEBUG] blk_init: n_blocks = %d\n", B);

    BPerm *d = malloc(sizeof(BPerm));
    d->perm  = malloc(B * sizeof(int));
    st->rule_data = d;
    st->rule_cleanup = blk_cleanup;

    srand((unsigned)time(NULL));
    return 0;
}

/*
 * Begin epoch:
 * - Create identity permutation [0, 1, ..., B-1]
 * - Shuffle using Fisher-Yates algorithm (uniform random permutation)
 */
static void blk_begin(CDState *st, int epoch)
{
    (void)epoch;  // unused
    int B = st->n_blocks;
    BPerm *d = st->rule_data;

    // Initialize permutation to [0, 1, ..., B-1]
    for (int i = 0; i < B; ++i)
        d->perm[i] = i;

    // Shuffle permutation using Fisher-Yates
    for (int i = B - 1; i > 0; --i) {
        int k = rand() % (i + 1);
        int t = d->perm[i];
        d->perm[i] = d->perm[k];
        d->perm[k] = t;
    }
}

/*
 * Get index of the next coordinate block:
 * - Returns the index of the first coordinate in the shuffled block
 */
static int blk_next(CDState *st, int idx)
{
    BPerm *d = st->rule_data;
    int b        = idx / st->block_size;        /* number of block      */
    int offset   = idx % st->block_size;        /* positon in block  */
    return d->perm[b] * st->block_size + offset;
}

static void blk_cleanup(CDState *st)
{
    BPerm *d = (BPerm*)st->rule_data;
    if (!d) return;
    // fprintf(stderr, "[DEBUG] blk_cleanup: perm=%p, struct=%p\n", (void*)d->perm, (void*)d);
    free(d->perm);
    free(d);
    st->rule_data    = NULL;
    st->rule_cleanup = NULL;
    // fprintf(stderr,"[DEBUG] blk_cleanup: perm free\n");
}

/*
 * Free memory used by the rule
 *

/*
 * Exported rule definition:
 * - Uses block shuffling
 * - No end-of-epoch hook needed
 */
const CDIndexRule RULE_BLOCK_SHUFFLE = {
    .init        = blk_init,
    .begin_epoch = blk_begin,
    .next_j      = blk_next,
    .end_epoch   = NULL,
    .cleanup     = blk_cleanup
};
