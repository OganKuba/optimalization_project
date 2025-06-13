#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include "io.h"

static void die(const char *msg, char *fname){
    // fprintf(stderr, "%s: %s\n", msg, fname);
    exit(EXIT_FAILURE);
}

double *load_vector(const char* file, int m){
    FILE *fp = fopen(file, "r");
    if (!fp) die("cannot open vector file", file);

    double *v = (double *)malloc((size_t)m * sizeof(double));
    if (!v) die("malloc failed (vector)", file);

    for (int i = 0; i < m; ++i) {
        if (fscanf(fp, " %lf", &v[i]) != 1)
            die("unexpected EOF or format error (vector)", file);
    }
    fclose(fp);
    return v;
}

double *load_matrix(const char *file, int m, int n){
    FILE *fp = fopen(file, "r");
    if (!fp) die("cannot open matrix file", file);

    double *X = (double *)malloc((size_t)m * n * sizeof(double));
    if (!X) die("malloc failed (matrix)", file);

    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            double val;
            if (fscanf(fp, " %lf", &val) != 1)
                die("unexpected EOF or format error (matrix)", file);
            X[j * m + i] = val;
        }
    fclose(fp);
    return X;
}

void free_all(int n_ptrs, ...)
{
    va_list ap;
    va_start(ap, n_ptrs);
    for (int k = 0; k < n_ptrs; ++k) {
        void *p = va_arg(ap, void *);
        free(p);
    }
    va_end(ap);
}
