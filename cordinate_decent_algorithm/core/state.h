#ifndef STATE_H
#define STATE_H

#include <stddef.h>

// Structure representing the state for a coordinate descent algorithm
typedef struct CDState {
    // Input data
    const double *X; // Design matrix (features), size m x n (flattened)
    const double *y; // Response vector, size m
    int m, n; // m = number of samples, n = number of features
    double lam; // Regularization parameter (lambda)

    // Model state
    double *beta; // Current coefficient vector (size n)
    double *resid; // Residuals: y - X * beta (size m)
    const double *norm2; // Precomputed L2 norm squared of each column of X (size n)

    // Function pointers for vector operations
    void (*axpy)(double a, const double *, double *, int); // Performs y += a * x
    double (*dot)(const double *, const double *, int); // Computes dot product: x · y

    // Rule-related data and cleanup
    void *rule_data; // Optional data used by coordinate selection rule
    void (*rule_cleanup)(struct CDState *); // Cleanup function for rule_data (NEW)

    // Gradient storage
    double *grad; // Gradient vector of the loss function (size n)

    // Block-wise coordinate descent (optional)
    int block_size; // Size of each coordinate block
    int n_blocks; // Total number of blocks
    double *block_tmp; // Temporary buffer for block computations

    // Previous iteration data
    double *beta_prev; // Previous beta vector (used for convergence checks or VR)

    /* ---- For Variance Reduction (VR) methods ---- */
    double *beta_snap; // Snapshot of beta: β̃ (used in VR techniques)
    double *resid_snap; // Snapshot residuals: r̃ = y - X * β̃
    double *grad_snap; // Snapshot gradient: ∇f(β̃) (size n)
    long vr_counter; // Counter for VR steps (used to trigger snapshots periodically)
} CDState;

#endif
