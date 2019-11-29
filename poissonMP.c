#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <unistd.h>
#include <sys/time.h>
// #include <mpi.h>

// Rectangle
#define A1 -1.0
#define A2 2.0
#define B1 -2.0
#define B2 2.0

#define A_DIFF 3.0
#define B_DIFF 4.0

// Grid
#define M1 80
#define N1 80

// Epsilon
#define EPS 0.001

// For comparison
double xi(int i, double stepX) {
    return A1 + i * stepX;
}

double yj(int j, double stepY) {
    return B1 + j * stepY;
}
// --------------

// Given functions
double u1(double x, double y) {
    return exp(1 - (x + y) * (x + y));
}

double deriv_u1(double x, double y) {
    return -2 * (x + y) * exp(1 - (x + y) * (x + y));
}

double k2(double x) {
    return 4 + x;
}

double q3(double x, double y) {
    return (x + y) * (x + y);
}

double F(double x, double y) {
    return u1(x, y) * ((x + y) * (x + y + 2) - (4 + x) * (-4 + 8 * (x + y) * (x + y)));
}

// Boundary conditions
double gammaA1(double y) {
    return exp(y * (2 - y));
}

double gammaA2(double y) {
    return exp(-4*y - y*y - 3);
}

double gammaB1(double x) {
    return exp(4*x - x*x - 3);
}

double gammaB2(double x) {
    return exp(-4*x - x*x - 3);
}

// Step X
double h1(int m) {
    return A_DIFF / m;
}

// Step Y
double h2(int n) {
    return B_DIFF / n;
}

// Dot product
double dotProduct(double* u, double* v, int dim, double stepX, double stepY) {
    double result = 0;
    // #pragma omp parallel
    #pragma omp parallel for schedule (static) reduction(+:result)
    for (int i = 0; i < dim; i++) {
        //printf(" Thread %d: %d\n", omp_get_thread_num(), i);
        result += u[i] * v[i];
    }
    return result * stepX * stepY;
}

double** getMatrixA(int m, int n, int dim, double stepX, double stepY, double* xs, double* ys) {
    double stepCoeffX = 1 / (stepX * stepX);
    double stepCoeffY = 1 / (stepY * stepY);
    double **A;
    A = (double **)malloc(dim * sizeof(double *));
    for (int i = 0; i < dim; i++) {
        *(A + i) = (double *)malloc(dim * sizeof(double));
        for (int j = 0; j < dim; j++) {
            A[i][j] = 0;
        }
    }

    A[0][0] =
            stepCoeffX * (k2(xs[1] + 0.5 * stepX) + k2(xs[1] - 0.5 * stepX)) +
            stepCoeffY * (2 * k2(xs[1])) + q3(xs[1], ys[1]);

    A[0][1] = -stepCoeffY * k2(xs[1]);
    A[0][n - 1] = -stepCoeffX * k2(xs[1] + 0.5 * stepX);

    #pragma omp parallel for schedule (static)
    for (int j = 2; j < n; j++) {
        A[j - 1][j - 2] = -stepCoeffY * k2(xs[1]);

        A[j - 1][j - 1] =
            stepCoeffX * (k2(xs[1] + 0.5 * stepX) + k2(xs[1] - 0.5 * stepX)) +
            stepCoeffY * (2 * k2(xs[1])) + q3(xs[1], ys[j]);

        A[j - 1][j] = -stepCoeffY * k2(xs[1]);
        A[j - 1][j + n - 2] = -stepCoeffX * k2(xs[1] + 0.5 * stepX);
    }

    #pragma omp parallel for schedule (static)
    for (int i = 2; i < m - 1; i++) {
        for (int j = 1; j < n; j++) {
            A[(n - 1)*(i - 1) + (j - 1)][(n - 1)*(i - 1) + (j - 1) - (n - 1)] = -stepCoeffX * k2(xs[i] - 0.5 * stepX);
            A[(n - 1)*(i - 1) + (j - 1)][(n - 1)*(i - 1) + (j - 1) - 1] = -stepCoeffY * k2(xs[i]);

            A[(n - 1)*(i - 1) + (j - 1)][(n - 1)*(i - 1) + (j - 1)] =
                stepCoeffX * (k2(xs[i] + 0.5 * stepX) + k2(xs[i] - 0.5 * stepX)) +
                stepCoeffY * (2 * k2(xs[i])) + q3(xs[i], ys[j]);

            A[(n - 1)*(i - 1) + (j - 1)][(n - 1)*(i - 1) + (j - 1) + 1] = -stepCoeffY * k2(xs[i]);
            A[(n - 1)*(i - 1) + (j - 1)][(n - 1)*(i - 1) + (j - 1) + (n - 1)] = -stepCoeffX * k2(xs[i] + 0.5 * stepX);
        }
    }
    #pragma omp parallel for schedule (static)
    for (int j = 1; j < n - 1; j++) {
        A[(n - 1)*(m - 2) + (j - 1)][(n - 1)*(m - 2) + (j - 1) - (n - 1)] = -stepCoeffX * k2(xs[m - 1] - 0.5 * stepX);
        A[(n - 1)*(m - 2) + (j - 1)][(n - 1)*(m - 2) + (j - 1) - 1] = -stepCoeffY * k2(xs[m - 1]);

        A[(n - 1)*(m - 2) + (j - 1)][(n - 1)*(m - 2) + (j - 1)] =
            stepCoeffX * (k2(xs[m - 1] + 0.5 * stepX) + k2(xs[m - 1] - 0.5 * stepX)) +
            stepCoeffY * (2 * k2(xs[m - 1])) + q3(xs[m - 1], ys[j]);

        A[(n - 1)*(m - 2) + (j - 1)][(n - 1)*(m - 2) + (j - 1) + 1] = -stepCoeffY * k2(xs[m - 1]);
    }

    A[dim - 1][dim - n] = -stepCoeffX * k2(xs[m - 1] - 0.5 * stepX);
    A[dim - 1][dim - 2] = -stepCoeffY * k2(xs[m - 1]);
    A[dim - 1][dim - 1] =
            stepCoeffX * (k2(xs[m - 1] + 0.5 * stepX) + k2(xs[m - 1] - 0.5 * stepX)) +
            stepCoeffY * (2 * k2(xs[m - 1])) + q3(xs[m - 1], ys[n - 1]);

    return A;
}

double* getVectorB(int m, int n, int dim, double stepX, double stepY, double* xs, double* ys) {
    double stepCoeffX = 1 / (stepX * stepX);
    double stepCoeffY = 1 / (stepY * stepY);
    double* B;
    B = (double *)malloc(dim * sizeof(double));

    #pragma omp parallel for schedule (static)
    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            B[(n - 1)*(i - 1) + (j - 1)] = F(xs[i], ys[j]);
        }
    }
    #pragma omp parallel for schedule (static)
    for (int j = 1; j < n; j++) {
        B[j - 1] += stepCoeffX * k2(xs[1] - 0.5 * stepX) * gammaA1(ys[j]);
    }
    #pragma omp parallel for schedule (static)
    for (int j = 1; j < n; j++) {
        B[(n - 1)*(m - 2) + (j - 1)] += stepCoeffX * k2(xs[m - 1] + 0.5 * stepX) * gammaA2(ys[j]);
    }
    B[0] += stepCoeffY * k2(xs[1]) * gammaB1(xs[1]);
    B[dim - 1] += stepCoeffY * k2(xs[m - 1]) * gammaB2(xs[m - 1]);

    return B;
}

double* iterativeMethod(int m, int n, double stepX, double stepY) {
    int dim = (m - 1) * (n - 1);
    double stepCoeffX = 1 / (stepX * stepX);
    double stepCoeffY = 1 / (stepY * stepY);
    
    // Grid nodes
    double* xs;
    xs = (double *)malloc(m * sizeof(double));
    for (int i = 0; i < m; i++) {
        xs[i] = A1 + i * stepX;
    }

    double* ys;
    ys = (double *)malloc(n * sizeof(double));
    for (int j = 0; j < n; j++) {
        ys[j] = B1 + j * stepY;
    }

    // Initializing with 2
    double* w;
    w = (double *)malloc(dim * sizeof(double));
    for (int i = 0; i < dim; i++) {
        w[i] = 2;
    }

    double* wDiff;
    wDiff = (double *)malloc(dim * sizeof(double));

    double **A = getMatrixA(m, n, dim, stepX, stepY, xs, ys);

    // printf("Matrix A\n");
    // for (int i = 0; i < dim; i++) {
    //     for (int j = 0; j < dim; j++) {
    //         printf("%f ", A[i][j]);
    //     }
    //     printf("\n");
    // }

    double* B = getVectorB(m, n, dim, stepX, stepY, xs, ys);

    // printf("Vector B\n");
    // for (int i = 0; i < dim; i++) {
    //     printf("%f\n", B[i]);
    // }

    double* r;
    r = (double *)malloc(dim * sizeof(double));

    double* Ar;
    Ar = (double *)malloc(dim * sizeof(double));

    // int k = 0;
    do {
        // Find r
        #pragma omp parallel for schedule (static)
        for (int i = 0; i < dim; i++) {
            r[i] = 0;
            for (int j = 0; j < dim; j++) {
                if (fabs(A[i][j]) > 10e-7) {
                    r[i] += A[i][j] * w[j];
                }
            }
            r[i] -= B[i];
        }

        // Find tau
        #pragma omp parallel for schedule (static)
        for (int i = 0; i < dim; i++) {
            Ar[i] = 0;
            for (int j = 0; j < dim; j++) {
                if (fabs(A[i][j]) > 10e-7) {
                    Ar[i] += A[i][j] * r[j];
                }
            }
        }
        double tau = dotProduct(Ar, r, dim, stepX, stepY) / dotProduct(Ar, Ar, dim, stepX, stepY);

        #pragma omp parallel for schedule (static)
        for (int i = 0; i < dim; i++) {
            double temp = w[i];
            w[i] = w[i] - tau * r[i];
            wDiff[i] = w[i] - temp;
        }
        //k++;
    } while (sqrt(dotProduct(wDiff, wDiff, dim, stepX, stepY)) > EPS);
    // } while ((sqrt(dotProduct(wDiff, wDiff, dim, stepX, stepY)) > EPS) && (k < dim));
    //} while (k < dim);
    //printf("K: %d\n", k);
    free(A);
    free(B);
    free(wDiff);
    free(Ar);
    free(xs);
    free(ys);
    return w;
}

int main() {

    struct timeval start, end;

    gettimeofday(&start, NULL);

    FILE *f = fopen("error80.txt", "w");
    if (f == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }

    printf("Starting the program\n");

    double stepX1 = h1(M1);
    double stepY1 = h2(N1);

    double* res = iterativeMethod(M1, N1, stepX1, stepY1);

    double error = 0;
    #pragma omp parallel for schedule (static) reduction(+:error)
    for (int i = 1; i < M1; i++) {
        for (int j = 1; j < N1; j++) {
            error += (u1(xi(i, stepX1), yj(j, stepY1)) - res[(N1 - 1)*(i - 1) + (j - 1)]) * (u1(xi(i, stepX1), yj(j, stepY1)) - res[(N1 - 1)*(i - 1) + (j - 1)]);
        }
    }
    gettimeofday(&end, NULL);
    double time_taken = end.tv_sec + end.tv_usec / 1e6 -
                        start.tv_sec - start.tv_usec / 1e6; // in seconds
    fprintf(f, "Error: %f\n", error);
    fprintf(f, "Execution time: %f\n", time_taken);

    free(res);
    fclose(f);
    return 0;
}
