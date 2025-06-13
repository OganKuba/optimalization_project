#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "cd_lasso.h"
#include "utils.h"
#include "cd_path.h"

/* ---------- helper : gaussian RNG (Box‑Muller) ---------- */
static double gauss_rand()
{
    double u = (rand() + 1.0) / (RAND_MAX + 2.0);
    double v = (rand() + 1.0) / (RAND_MAX + 2.0);
    return sqrt(-2.0 * log(u)) * cos(2.0 * M_PI * v);
}

/* ---------- synthetic data generator -------------------- */
void make_synthetic(double *X, double *y,
                    double *beta_star,
                    int m, int n, int sparsity, double noise_std)
{
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            X[j * m + i] = gauss_rand();

    for (int j = 0; j < n; ++j) beta_star[j] = 0.0;
    for (int k = 0; k < sparsity; ++k) {
        int idx = rand() % n;
        beta_star[idx] = gauss_rand();
    }

    for (int i = 0; i < m; ++i) {
        double s = 0.0;
        for (int j = 0; j < n; ++j)
            s += X[j * m + i] * beta_star[j];
        y[i] = s + noise_std * gauss_rand();
    }
}

/* ---------- count non‑zeros ----------------------------- */
int count_nnz(const double *beta, int n, double tol)
{
    int nnz = 0;
    for (int j = 0; j < n; ++j)
        if (fabs(beta[j]) > tol) ++nnz;
    return nnz;
}

/* ======================================================== */
int main(void)
{
    srand((unsigned)time(NULL));

    /* problem size + ground‑truth sparsity */
    int    m = 500;        /* samples   */
    int    n = 1000;       /* features  */
    int    k = 20;         /* nnz in β* */
    double noise_std = 0.01;

    /* allocate arrays */
    double *X          = (double *)malloc((size_t)m * n * sizeof(double));
    double *y          = (double *)malloc((size_t)m     * sizeof(double));
    double *beta_star  = (double *)malloc((size_t)n     * sizeof(double));

    make_synthetic(X, y, beta_star, m, n, k, noise_std);

    /* ----- LASSO path parameters ----- */
    double lam_start  = 1.0;
    double lam_end    = 0.01;
    double eta        = 0.8;    /* lambda *= eta */
    double tol        = 1e-6;
    int    max_epochs = 1000;

    /* run CD‑path (shuffled rule) */
    CDResult sol = cd_path(X, y, m, n,
                           lam_start, lam_end, eta,
                           tol, max_epochs,
                           "shuffle");

    /* ------- evaluation -------- */
    /* mean‑squared‑error to ground truth */
    double mse = 0.0;
    for (int j = 0; j < n; ++j) {
        double d = sol.beta[j] - beta_star[j];
        mse += d * d;
    }
    mse /= n;

    printf("\n=== Synthetic test ===\n");
    printf("True sparsity  : %d/%d\n", k, n);
    printf("Found sparsity : %d/%d (|β|>1e-4)\n",
           count_nnz(sol.beta, n, 1e-4), n);
    printf("MSE(β_hat, β*) : %.6g\n", mse);
    printf("Final loss     : %.6g\n", sol.loss);
    printf("Total epochs   : %d\n", sol.epochs);

    /* clean‑up */
    free(X); free(y); free(beta_star); free(sol.beta);
    return 0;
}
