//
// Created by kubog on 07.05.2025.
//

#ifndef IO_H
#define IO_H

double *load_vector(const char *file, int m);
double *load_matrix(const char *file, int m, int n);  // column-major
void    free_all(int n_ptrs, ...);

#endif //IO_H
