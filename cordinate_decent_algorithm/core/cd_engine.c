#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "cd_engine.h"
#include "../utils.h"

// Main coordinate descent routine
int cd_run(CDState* st,
           const CDIndexRule* rule,
           const CDUpdateScheme* scheme,
           int max_epochs,
           double tol)
{
    // Clean up old rule-specific data
    if (st->rule_data)
    {
        if (st->rule_cleanup)
            st->rule_cleanup(st);
        else
            free(st->rule_data);
        st->rule_data = NULL;
        st->rule_cleanup = NULL;
    }

    // Initialize the coordinate selection rule
    if (rule->init)
    {
        rule->init(st);
        if (rule->cleanup && st->rule_data)
            st->rule_cleanup = rule->cleanup;
    }

    // Initialize update scheme if needed
    if (scheme->init)
        scheme->init(st);

    // Allocate memory for previous iterate
    double* beta_prev = (double*)malloc(st->n * sizeof(double));
    if (!beta_prev) return -1;

    int epoch = 0;
    double lam_prev = st->lam;

    const int PATIENCE = 10; // Early stopping patience
    int no_improve_epochs = 0;

    // Coordinate descent iterations
    for (epoch = 0; epoch < max_epochs; ++epoch)
    {
        memcpy(st->beta_prev, st->beta, st->n * sizeof(double));
        memcpy(beta_prev, st->beta, st->n * sizeof(double));

        // Reset momentum buffer if lambda has changed
        if (st->lam != lam_prev)
        {
            st->gamma_prev = 0.0;
            memcpy(st->v_buf, st->beta, st->n * sizeof(double));
            lam_prev = st->lam;
        }

        if (rule->begin_epoch)
            rule->begin_epoch(st, epoch);

        // Update each coordinate
        for (int idx = 0; idx < st->n; ++idx)
        {
            int j0 = rule->next_j(st, idx);
            scheme->update_j(st, j0);
        }

        if (rule->end_epoch)
            rule->end_epoch(st);

        // Check for convergence: max |β_k - β_{k-1}|
        double maxdiff = 0.0;
        for (int j = 0; j < st->n; ++j)
        {
            double d = fabs(st->beta[j] - beta_prev[j]);
            if (d > maxdiff) maxdiff = d;
        }

        if (maxdiff < tol)
        {
            no_improve_epochs++;
        }
        else
        {
            no_improve_epochs = 0;
        }

        if (no_improve_epochs >= PATIENCE)
            break;
    }

    free(beta_prev);
    return epoch;
}

// Initializes a coordinate descent state structure
CDState cd_create_state(const double* X, const double* y,
                        int m, int n, double lam)
{
    CDState st;

    st.X = X;
    st.y = y;
    st.m = m;
    st.n = n;
    st.lam = lam;

    st.block_size = 1;
    st.n_blocks = n;
    st.block_tmp = NULL;

    // Allocate buffers for iterates and residuals
    st.beta_snap = (double*)calloc(n, sizeof(double));
    st.resid_snap = (double*)malloc(m * sizeof(double));
    st.grad_snap = (double*)calloc(n, sizeof(double));
    st.beta = (double*)calloc(n, sizeof(double));
    st.beta_prev = (double*)calloc(n, sizeof(double));
    st.resid = (double*)malloc(m * sizeof(double));
    st.y_aux = (double*)calloc(n, sizeof(double));
    st.v_aux = (double*)calloc(n, sizeof(double));
    st.resid_y = (double*)malloc(m * sizeof(double));
    st.resid_v = (double*)calloc(m, sizeof(double));
    st.rv_buf = (double*)malloc(m * sizeof(double));
    st.v_buf = (double*)malloc(n * sizeof(double));

    memcpy(st.resid_snap, y, m * sizeof(double));
    memcpy(st.resid, y, m * sizeof(double));
    memcpy(st.resid_v, y, m * sizeof(double));
    memcpy(st.rv_buf, y, m * sizeof(double));

    memcpy(st.y_aux, st.beta, n * sizeof(double));
    memcpy(st.v_aux, st.beta, n * sizeof(double));
    memcpy(st.v_buf, st.beta, n * sizeof(double));
    memcpy(st.resid_y, y, m * sizeof(double));

    // Momentum/variance reduction parameters
    st.rv_a = 1.0;
    st.rv_b = 0.0;
    st.v_a = 0.0;
    st.v_b = 1.0;
    st.gamma_prev = 0.0;
    st.sigma = 1e-4;

    st.vr_counter = 0;

    // Precompute squared column norms of X
    st.norm2 = precompute_col_norm2(X, m, n);

    // Function pointers for linear algebra ops
    st.axpy = axpy;
    st.dot = dot;

    st.rule_data = NULL;
    st.rule_cleanup = NULL;
    st.grad = NULL;

    return st;
}

// Frees memory allocated in CDState
void cd_free_state(CDState* st)
{
    free((void*)st->norm2);
    free(st->beta);
    free(st->resid);
    free(st->block_tmp);
    free(st->beta_prev);

    free(st->beta_snap);
    free(st->resid_snap);
    free(st->grad_snap);

    free(st->y_aux);
    free(st->v_aux);
    free(st->resid_y);
    free(st->resid_v);

    free(st->rv_buf);
    free(st->v_buf);

    if (st->rule_data)
    {
        if (st->rule_cleanup)
            st->rule_cleanup(st);
        else
            free(st->rule_data);
        st->rule_data = NULL;
        st->rule_cleanup = NULL;
    }

    if (st->grad)
        free(st->grad);
}

// Set block size for block-coordinate descent
void cd_set_block_size(CDState* st, int bs)
{
    if (bs < 1 || st->n % bs != 0)
    {
        exit(1); // Invalid block size
    }

    st->block_size = bs;
    st->n_blocks = st->n / bs;

    if (st->block_tmp)
        free(st->block_tmp);

    st->block_tmp = (bs > 1) ? malloc(bs * sizeof(double)) : NULL;

    // Reinitialize rule if it depends on block size
    if (st->rule_data)
    {
        if (st->rule_cleanup)
            st->rule_cleanup(st);
        else
            free(st->rule_data);
        st->rule_data = NULL;
        st->rule_cleanup = NULL;
    }
}
