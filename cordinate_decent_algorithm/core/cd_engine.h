#ifndef CD_ENGINE_H
#define CD_ENGINE_H

#include "state.h"  // Definition of CDState

// Coordinate selection rule interface
typedef struct
{
    int (*init)(CDState* st); // Optional: setup rule data
    void (*begin_epoch)(CDState* st, int k); // Called at start of each epoch
    int (*next_j)(CDState* st, int idx); // Select next coordinate j
    void (*end_epoch)(CDState* st); // Called at end of epoch
    void (*cleanup)(CDState* st); // Free rule-related memory
} CDIndexRule;

// Update scheme interface
typedef struct
{
    int (*init)(CDState* st); // Optional: setup scheme
    void (*update_j)(CDState* st, int j); // Update coordinate j
} CDUpdateScheme;

// Run coordinate descent
int cd_run(CDState* st,
           const CDIndexRule* rule,
           const CDUpdateScheme* scheme,
           int max_epochs,
           double tol);

// Initialize CDState with input data
CDState cd_create_state(const double* X,
                        const double* y,
                        int m, int n,
                        double lam);

// Free all memory used by CDState
void cd_free_state(CDState* st);

// Set block size (for block coordinate descent)
void cd_set_block_size(CDState* st, int block_size);

#endif
