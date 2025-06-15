//
// Created by kubog on 07.05.2025.
//

#ifndef UTILS_H
#define UTILS_H

double dot(const double* a, const double* b, int len);
void axpy(double alpha, const double* x, double* y, int len);
double col_norm2(const double* X, int m, int col);

double shrink(double z, double tau);

double* precompute_col_norm2(const double* X, int m, int n);

int converged(const double* beta_old,
              const double* beta_new, int n,
              double loss_old, double loss_new,
              double tol);

double max_abs(const double* v, int n);


#endif //UTILS_H
