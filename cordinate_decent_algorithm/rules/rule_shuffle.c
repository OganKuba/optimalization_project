#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../core/cd_engine.h"

// Structure to hold permutation data for shuffle-based coordinate selection
typedef struct {
    int *perm; // Array storing a random permutation of coordinate indices
} ShufData;

/* Initialize the shuffle rule: allocate the permutation array once */
static int shuf_init(CDState *st) {
    // Allocate memory for the ShufData structure
    ShufData *d = (ShufData *) malloc(sizeof(ShufData));

    // Allocate an integer array of length n to store permutations
    d->perm = (int *) malloc(st->n * sizeof(int));

    // Store the pointer in the generic rule_data field
    st->rule_data = d;

    // Seed the random number generator
    srand((unsigned) time(NULL));

    return 0; // Return 0 to indicate success
}

/* Fisher–Yates shuffle: run at the beginning of each epoch */
static void shuf_begin(CDState *st, int epoch) {
    (void) epoch; // Unused parameter

    // Retrieve the shuffle data from the generic pointer
    ShufData *d = (ShufData *) st->rule_data;

    // Initialize perm array with identity permutation: 0, 1, 2, ..., n-1
    for (int j = 0; j < st->n; ++j)
        d->perm[j] = j;

    // Shuffle the array using Fisher–Yates algorithm
    for (int j = st->n - 1; j > 0; --j) {
        int k = rand() % (j + 1); // Pick a random index between 0 and j
        int tmp = d->perm[j]; // Swap perm[j] with perm[k]
        d->perm[j] = d->perm[k];
        d->perm[k] = tmp;
    }
}

/* Return the j-th coordinate index from the current permutation */
static int shuf_next(CDState *st, int idx) {
    ShufData *d = (ShufData *) st->rule_data;
    return d->perm[idx]; // Return the idx-th element of the shuffled order
}

/* Free the resources allocated by this shuffle rule */
static void shuf_cleanup(CDState *st) {
    ShufData *d = (ShufData *) st->rule_data;
    st->rule_data   = NULL;
    st->rule_cleanup = NULL;
    free(d->perm); // Free the permutation array
    free(d); // Free the structure itself
}

/* End-of-epoch hook: not needed for this rule */
static void shuf_end(CDState *st) { (void) st; }

/* Define the full coordinate selection rule for randomized shuffle */
const CDIndexRule RULE_SHUFFLE = {
    .init = shuf_init, // Called once before optimization starts
    .begin_epoch = shuf_begin, // Called at the start of each epoch (shuffles indices)
    .next_j = shuf_next, // Called to get the next coordinate index to update
    .end_epoch = shuf_end, // Called at the end of each epoch (no-op here)
    .cleanup = shuf_cleanup // Called to free allocated memory at the end
};
