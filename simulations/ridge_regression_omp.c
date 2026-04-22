/*
 * ridge_regression_omp.c  –  v2_streaming
 *
 * OpenMP-parallelised ridge regression.
 *
 * Changes vs v1_batch:
 *   - Results are printed immediately as each run completes (no large
 *     cost[] array stored in memory).
 *   - Mean and variance are accumulated online via Welford's algorithm,
 *     so only O(1) state is kept across runs.
 *   - Output order is non-deterministic (threads finish in any order);
 *     this is fine since the samples are i.i.d.
 *
 * Cost = (RSS + p * lambda * ||w||^2) / (2*p)
 * where w = (X'X + lambda*p*I)^{-1} X'y
 *
 * Build (macOS with Accelerate framework + OpenMP via Homebrew libomp):
 *   gcc -O2 -DACCELERATE_NEW_LAPACK -Xpreprocessor -fopenmp \
 *       -I$(brew --prefix libomp)/include -L$(brew --prefix libomp)/lib \
 *       -o ridge_regression_omp ridge_regression_omp.c \
 *       -framework Accelerate -lomp -lm
 *
 * Build (Linux with OpenBLAS):
 *   gcc -O2 -fopenmp -o ridge_regression_omp ridge_regression_omp.c \
 *       -lblas -llapack -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <omp.h>

/* LAPACK/BLAS declarations */
#ifdef __APPLE__
#  include <Accelerate/Accelerate.h>
#  define USE_CBLAS
#else
#  include <cblas.h>
extern void dsysv_(char*, int*, int*, double*, int*, int*, double*, int*,
                   double*, int*, int*);
#  define USE_CBLAS
#endif

/* ------------------------------------------------------------------ */
/* Box-Muller transform: thread-safe normal random variable            */
/* ------------------------------------------------------------------ */
static double randn_bm(unsigned int *seed, int *have_spare, double *spare)
{
    if (*have_spare) {
        *have_spare = 0;
        return *spare;
    }

    double u, v, s;
    do {
        u = (double)rand_r(seed) / RAND_MAX * 2.0 - 1.0;
        v = (double)rand_r(seed) / RAND_MAX * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);

    double mul = sqrt(-2.0 * log(s) / s);
    *spare      = v * mul;
    *have_spare = 1;
    return u * mul;
}

/* ------------------------------------------------------------------ */
/* ridge_run: one run of ridge regression, returns the cost value.    */
/* ------------------------------------------------------------------ */
static double ridge_run(int p, int N, double lambda,
                        unsigned int *seed, int *have_spare, double *spare)
{
    int i, j;

    double *X = (double*)malloc((size_t)N * p * sizeof(double));
    if (!X) { perror("malloc X"); exit(1); }
    for (i = 0; i < N * p; i++)
        X[i] = randn_bm(seed, have_spare, spare);

    double inv_sqrt_p = 1.0 / sqrt((double)p);
    double *beta = (double*)malloc((size_t)p * sizeof(double));
    if (!beta) { perror("malloc beta"); exit(1); }
    for (i = 0; i < p; i++)
        beta[i] = inv_sqrt_p;

    double *noise = (double*)malloc((size_t)N * sizeof(double));
    if (!noise) { perror("malloc noise"); exit(1); }
    for (i = 0; i < N; i++)
        noise[i] = randn_bm(seed, have_spare, spare);

    double *y = (double*)malloc((size_t)N * sizeof(double));
    if (!y) { perror("malloc y"); exit(1); }
    cblas_dgemv(CblasColMajor, CblasNoTrans, N, p, 1.0, X, N, beta, 1, 0.0, y, 1);
    for (i = 0; i < N; i++)
        y[i] += noise[i];

    double *A = (double*)malloc((size_t)p * p * sizeof(double));
    if (!A) { perror("malloc A"); exit(1); }
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, p, p, N, 1.0, X, N, X, N, 0.0, A, p);

    double reg = lambda * (double)p;
    for (j = 0; j < p; j++)
        A[j * p + j] += reg;

    double *rhs = (double*)malloc((size_t)p * sizeof(double));
    if (!rhs) { perror("malloc rhs"); exit(1); }
    cblas_dgemv(CblasColMajor, CblasTrans, N, p, 1.0, X, N, y, 1, 0.0, rhs, 1);

    {
        char   uplo  = 'U';
        int    n     = p, nrhs = 1, lda = p, ldb = p, info;
        int   *ipiv  = (int*)malloc((size_t)p * sizeof(int));
        if (!ipiv) { perror("malloc ipiv"); exit(1); }

        double work_query;
        int    lwork = -1;
        dsysv_(&uplo, &n, &nrhs, A, &lda, ipiv, rhs, &ldb,
               &work_query, &lwork, &info);

        lwork        = (int)work_query;
        double *work = (double*)malloc((size_t)lwork * sizeof(double));
        if (!work) { perror("malloc work"); exit(1); }

        dsysv_(&uplo, &n, &nrhs, A, &lda, ipiv, rhs, &ldb,
               work, &lwork, &info);

        free(work);
        free(ipiv);

        if (info != 0) {
            fprintf(stderr, "dsysv_ failed with info = %d\n", info);
            exit(1);
        }
    }
    double *w = rhs;

    double *resid = (double*)malloc((size_t)N * sizeof(double));
    if (!resid) { perror("malloc resid"); exit(1); }
    memcpy(resid, y, (size_t)N * sizeof(double));
    cblas_dgemv(CblasColMajor, CblasNoTrans, N, p, -1.0, X, N, w, 1, 1.0, resid, 1);

    double rss = 0.0;
    for (i = 0; i < N; i++)
        rss += resid[i] * resid[i];

    double w_sq = 0.0;
    for (i = 0; i < p; i++)
        w_sq += w[i] * w[i];

    double cost = (rss + (double)p * lambda * w_sq) / (2.0 * (double)p);

    free(X);
    free(beta);
    free(noise);
    free(y);
    free(A);
    free(rhs);
    free(resid);

    return cost;
}

/* ------------------------------------------------------------------ */
/* main                                                                 */
/* ------------------------------------------------------------------ */
int main(void)
{
    srand((unsigned)time(NULL));

    int    p      = 70;
    int    N      = 100;
    double lambda = 0.5;
    int    runs   = 1e8;

    unsigned int base_seed = (unsigned int)time(NULL);

    /*
     * Welford's online algorithm for mean and variance.
     * Shared state protected by a critical section (updated once per run,
     * negligible overhead vs. the BLAS/LAPACK work inside ridge_run).
     */
    long   count = 0;
    double mean  = 0.0;
    double M2    = 0.0;   /* accumulated sum of squared deviations */

    #pragma omp parallel
    {
        unsigned int seed       = base_seed ^ (unsigned int)(omp_get_thread_num() * 1000003 + 1);
        int          have_spare = 0;
        double       spare      = 0.0;

        #pragma omp for schedule(dynamic)
        for (int run = 0; run < runs; run++) {
            double c = ridge_run(p, N, lambda, &seed, &have_spare, &spare);

            #pragma omp critical
            {
                /* Print immediately — partial output is preserved on interrupt */
                printf("%f\n", c);

                /* Welford update */
                count++;
                double delta = c - mean;
                mean += delta / (double)count;
                M2   += delta * (c - mean);
            }
        }
    }

    double variance = (count > 1) ? M2 / (double)count : 0.0;
    fprintf(stderr, "runs=%ld  mean=%.6f  std=%.6f\n", count, mean, sqrt(variance));

    return 0;
}
