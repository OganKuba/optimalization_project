#include <stdlib.h>
#include <math.h>
#include "utils.h"

double dot(const double *a, const double *b, int len)
{
    double s = 0.0;
    for (int i = 0; i < len; ++i) s += a[i] * b[i];
    return s;
}

void axpy(double alpha, const double *x, double *y, int len)
{
    for (int i = 0; i < len; ++i) y[i] += alpha * x[i];
}

double col_norm2(const double *X, int m, int col)
{
    const double *col_ptr = X + (size_t)col * m;
    double s = 0.0;
    for (int i = 0; i < m; ++i) s += col_ptr[i] * col_ptr[i];
    return s;
}

double *precompute_col_norm2(const double *X, int m, int n){
    double *norm2 = (double *)malloc((size_t)n * sizeof(double));
    if (!norm2) return NULL;
    for (int j = 0; j < n; ++j)
        norm2[j] = col_norm2(X, m, j);
    return norm2;
}

int converged(const double *beta_old,
                     const double *beta_new, int n,
                     double loss_old, double loss_new,
                     double tol)
{
    if (fabs(loss_new - loss_old) < tol) return 1;

    double max_diff = 0.0;
    for (int j = 0; j < n; ++j) {
        double d = fabs(beta_new[j] - beta_old[j]);
        if (d > max_diff) max_diff = d;
    }
    return (max_diff < tol);
}

double max_abs(const double *v, int n)
{
    double m = 0.0;
    for (int i = 0; i < n; ++i) {
        double a = fabs(v[i]);
        if (a > m) m = a;
    }
    return m;
}

double shrink(double z, double tau)
{
    if (z >  tau) return z - tau;
    if (z < -tau) return z + tau;
    return 0.0;
}


