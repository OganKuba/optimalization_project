#ifndef CD_ENGINE_H
#define CD_ENGINE_H

#include "state.h"  // Include the definition of CDState structure

// Structure defining the interface for a coordinate selection rule (Index Rule)
typedef struct {
    int (*init)(CDState *st); // Initialize rule (e.g., allocate memory, setup data)
    void (*begin_epoch)(CDState *st, int k); // Called at the beginning of each epoch (iteration over all coordinates)
    int (*next_j)(CDState *st, int idx); // Selects the next coordinate index j to update
    void (*end_epoch)(CDState *st); // Called at the end of each epoch (cleanup or tracking)
    void (*cleanup)(CDState *st); // Frees any memory or data used by the rule (NEW)
} CDIndexRule;

// Structure defining the update scheme used to modify coordinates
typedef struct {
    int (*init)(CDState *st); // Optional: initialize the update scheme
    void (*update_j)(CDState *st, int j); // Required: performs the update on coordinate j
} CDUpdateScheme;

// Main function that runs the coordinate descent algorithm
int cd_run(CDState *st, // Pointer to the optimization state
           const CDIndexRule *rule, // Pointer to coordinate selection rule
           const CDUpdateScheme *scheme, // Pointer to coordinate update scheme
           int max_epochs, // Maximum number of full passes (epochs)
           double tol); // Convergence tolerance (e.g., for stopping criterion)

// Creates and initializes a CDState object with the input data
CDState cd_create_state(const double *X, // Feature matrix X (flattened, size m*n)
                        const double *y, // Response vector y (size m)
                        int m, int n, // m = number of samples, n = number of features
                        double lam); // Regularization parameter lambda

// Frees memory and resources used by the CDState object
void cd_free_state(CDState *st);

// Sets the block size for block coordinate descent (if supported)
void cd_set_block_size(CDState *st, int block_size);

#endif