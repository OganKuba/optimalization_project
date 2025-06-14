#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/cd_engine.h"
#include "rules/rules.h"
#include "updates/updates.h"

#ifdef _WIN32
#  define DLL_EXPORT __declspec(dllexport)
#else
#  define DLL_EXPORT
#endif

/*
 * Main interface for running Lasso via coordinate descent.
 *
 * Parameters:
 *  - X, y: input data matrix and target vector
 *  - m, n: number of rows and columns in X
 *  - beta_out: output array for model coefficients
 *  - lam_start, lam_end: regularization path (lambda max to min)
 *  - eta: geometric decay factor for lambda
 *  - tol: convergence tolerance
 *  - max_epochs: max epochs per lambda
 *  - rule_name: name of coordinate selection rule
 *  - sch_name: name of update scheme
 *
 * Returns: total number of epochs run
 */
DLL_EXPORT int lasso_cd_run(const double *X, const double *y,
                            int m, int n,
                            double *beta_out,
                            double lam_start, double lam_end, double eta,
                            double tol, int max_epochs,
                            const char *rule_name,
                            const char *sch_name)
{
    // Default rule: cyclic
    const CDIndexRule *rule = &RULE_CYCLIC;

    // Choose coordinate selection rule
    if (strcmp(rule_name, "gs_r") == 0)
        rule = &RULE_GS_R;
    else if (strcmp(rule_name, "gsl_r") == 0)
        rule = &RULE_GSL_R;
    else if (strcmp(rule_name, "shuffle") == 0)
        rule = &RULE_SHUFFLE;
    else if (strcmp(rule_name, "random") == 0)
        rule = &RULE_RANDOM;
    else if (strcmp(rule_name, "block_shuffle") == 0)
        rule = &RULE_BLOCK_SHUFFLE;

    // Default update scheme: standard prox-linear
    const CDUpdateScheme *sch = &SCHEME_PROX_LINEAR;

    // Choose update scheme
    if (strcmp(sch_name, "prox_linear_enet") == 0)
        sch = &SCHEME_PROX_LINEAR_ENET;
    else if (strcmp(sch_name, "prox_linear_svrg") == 0)
        sch = &SCHEME_PROX_LINEAR_SVRG;
    else if (strcmp(sch_name, "prox_linear_sgd") == 0)
        sch = &SCHEME_PROX_LINEAR_SGD;
    else if (strcmp(sch_name, "prox_linear_ext") == 0)
        sch = &SCHEME_PROX_LINEAR_EXT;
    else if (strcmp(sch_name, "prox_point") == 0)
        sch = &SCHEME_PROX_POINT;
    else if (strcmp(sch_name, "bcm") == 0)
        sch = &SCHEME_BCM;
    else if (strcmp(sch_name, "nesterov") == 0)
        sch = &SCHEME_NESTEROV;
    else if (strcmp(sch_name, "nesterov_ls") == 0)
        sch = &SCHEME_NESTEROV_LS;

    // Create and initialize solver state
    CDState st = cd_create_state(X, y, m, n, lam_start);
    st.rule_cleanup = rule->cleanup;

    // Choose block size for BCM or block rules
    int block_size = (n % 10 == 0 ? 10 :
                      n % 5  == 0 ? 5  :
                      n % 4  == 0 ? 4  :
                      n % 3  == 0 ? 3  :
                      n % 2  == 0 ? 2  : 1);

    // Force block indexing if using BCM
    if (strcmp(sch_name, "bcm") == 0) {
        rule = &RULE_BLOCK_SHUFFLE;  // BCM requires block selection
        st.rule_cleanup = rule->cleanup;
        cd_set_block_size(&st, block_size);
    }
    // Or if block_shuffle is explicitly requested
    else if (strcmp(rule_name, "block_shuffle") == 0) {
        cd_set_block_size(&st, block_size);
    }
    // Otherwise use single-coordinate updates
    else {
        cd_set_block_size(&st, 1);
    }

    // Coordinate descent path over multiple lambda values
    double lam = lam_start;
    int total_ep = 0;

    while (lam >= lam_end) {
        st.lam = lam;
        total_ep += cd_run(&st, rule, sch, max_epochs, tol);
        lam *= eta;  // decay lambda geometrically
    }

    // Copy final beta to output
    for (int j = 0; j < n; ++j)
        beta_out[j] = st.beta[j];

    cd_free_state(&st);
    return total_ep;
}
