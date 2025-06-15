#ifndef STATE_H
#define STATE_H

#include <stddef.h>

// Coordinate Descent state structure
typedef struct CDState
{
    // Input data
    const double* X; // Feature matrix (flattened), size m × n
    const double* y; // Response vector, size m
    int m, n; // m = samples, n = features
    double lam; // Regularization parameter

    // Model variables
    double* beta; // Current coefficients, size n
    double* resid; // Residuals: y - Xβ, size m
    const double* norm2; // Precomputed column norm² of X, size n

    // Basic vector ops
    void (*axpy)(double a, const double*, double*, int); // y += a·x
    double (*dot)(const double*, const double*, int); // x·y

    // Rule-specific data
    void* rule_data; // Custom rule state
    void (*rule_cleanup)(struct CDState*); // Cleanup callback

    // Gradient (if used)
    double* grad; // Gradient vector, size n

    // Block coordinate descent
    int block_size; // Block size
    int n_blocks; // Number of blocks
    double* block_tmp; // Temp buffer for block updates

    // Previous iterate
    double* beta_prev; // For convergence check / momentum

    // Variance Reduction (VR)
    double* beta_snap; // Snapshot β̃
    double* resid_snap; // Snapshot residuals
    double* grad_snap; // Snapshot gradient
    long vr_counter; // Counter for snapshot triggering

    // Acceleration (e.g., FISTA-like)
    double* y_aux; // yₖ
    double* v_aux; // vₖ
    double* resid_y; // Residual for yₖ
    double* resid_v; // Residual for vₖ
    double gamma_prev; // γₖ₋₁
    double sigma; // Strong convexity modulus

    // Linear combinations: used for acceleration
    double rv_a, rv_b; // resid_v = rv_a * rv_buf + rv_b * resid
    double v_a, v_b; // v = v_a * beta + v_b * v_buf
    double* rv_buf; // Residual buffer (default: y)
    double* v_buf; // Velocity buffer (default: beta)

    void* scheme_data; // Optional scheme-specific data
} CDState;

#endif
