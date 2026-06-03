/*
 * lasso_regression_omp.c
 *
 * OpenMP sample generator for the Lasso large-deviation experiment.
 *
 * This file intentionally follows the normalization used in ldlr_v3.m:
 *
 *     A_ij = N(0,1) / sqrt(p)
 *     y = A beta + epsilon,       epsilon_i ~ N(0,1)
 *     E(w) = 1/2 ||y - A w||^2 + (lambda/2) ||w||_1
 *     output = E(w*) / p
 *
 * The default beta is constant, beta_j = beta0, matching the old ridge C
 * code after its X/sqrt(p) normalization is made explicit.
 *
 * Build without OpenMP:
 *     cc -O2 -Wall -Wextra -o lasso_regression_omp lasso_regression_omp.c -lm
 *
 * Build with OpenMP:
 *     gcc -O2 -Wall -Wextra -fopenmp -o lasso_regression_omp \
 *         lasso_regression_omp.c -lm
 *
 * Example:
 *     ./lasso_regression_omp --p 70 --N 100 --lambda 0.5 --runs 100000 \
 *         > lasso_output1e5_p70_N100_1.txt
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#  include <omp.h>
#else
static int omp_get_thread_num(void) { return 0; }
static int omp_get_max_threads(void) { return 1; }
#endif

typedef struct {
    int p;
    int N;
    long long runs;
    double lambda;
    double beta0;
    unsigned int seed;
    int max_iter;
    int power_iters;
    double tol;
    int quiet;
} Config;

static void *xmalloc(size_t nbytes)
{
    void *ptr = malloc(nbytes);
    if (!ptr) {
        perror("malloc");
        exit(1);
    }
    return ptr;
}

static void usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s [options]\n"
        "\n"
        "Options:\n"
        "  --p INT              Number of features (default: 70)\n"
        "  --N INT              Number of samples per regression (default: 100)\n"
        "  --lambda FLOAT       Lasso lambda; penalty is lambda/2 * ||w||_1 (default: 0.5)\n"
        "  --runs INT           Number of Monte Carlo samples (default: 100000)\n"
        "  --beta0 FLOAT        Constant beta value beta_j = beta0 (default: 1)\n"
        "  --seed INT           Base RNG seed (default: current time)\n"
        "  --max-iter INT       FISTA iterations per sample (default: 2000)\n"
        "  --tol FLOAT          FISTA relative objective tolerance (default: 1e-7)\n"
        "  --power-iters INT    Power iterations for Lipschitz estimate (default: 30)\n"
        "  --quiet              Suppress configuration banner on stderr\n"
        "  --help               Show this help\n",
        prog);
}

static long long parse_ll(const char *s, const char *name)
{
    char *end = NULL;
    long long v = strtoll(s, &end, 10);
    if (!s[0] || (end && *end)) {
        fprintf(stderr, "Invalid integer for %s: %s\n", name, s);
        exit(2);
    }
    return v;
}

static int parse_int(const char *s, const char *name)
{
    long long v = parse_ll(s, name);
    if (v < -2147483647LL || v > 2147483647LL) {
        fprintf(stderr, "Integer out of range for %s: %s\n", name, s);
        exit(2);
    }
    return (int)v;
}

static double parse_double(const char *s, const char *name)
{
    char *end = NULL;
    double v = strtod(s, &end);
    if (!s[0] || (end && *end) || !isfinite(v)) {
        fprintf(stderr, "Invalid floating-point value for %s: %s\n", name, s);
        exit(2);
    }
    return v;
}

static Config default_config(void)
{
    Config cfg;
    cfg.p = 70;
    cfg.N = 100;
    cfg.runs = 100000LL;
    cfg.lambda = 0.5;
    cfg.beta0 = 1.0;
    cfg.seed = (unsigned int)time(NULL);
    cfg.max_iter = 2000;
    cfg.power_iters = 30;
    cfg.tol = 1e-7;
    cfg.quiet = 0;
    return cfg;
}

static Config parse_args(int argc, char **argv)
{
    Config cfg = default_config();

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (strcmp(arg, "--help") == 0) {
            usage(argv[0]);
            exit(0);
        } else if (strcmp(arg, "--quiet") == 0) {
            cfg.quiet = 1;
        } else {
            if (i + 1 >= argc) {
                fprintf(stderr, "Missing value after %s\n", arg);
                usage(argv[0]);
                exit(2);
            }
            const char *val = argv[++i];
            if (strcmp(arg, "--p") == 0) {
                cfg.p = parse_int(val, arg);
            } else if (strcmp(arg, "--N") == 0) {
                cfg.N = parse_int(val, arg);
            } else if (strcmp(arg, "--lambda") == 0) {
                cfg.lambda = parse_double(val, arg);
            } else if (strcmp(arg, "--runs") == 0) {
                cfg.runs = parse_ll(val, arg);
            } else if (strcmp(arg, "--beta0") == 0) {
                cfg.beta0 = parse_double(val, arg);
            } else if (strcmp(arg, "--seed") == 0) {
                cfg.seed = (unsigned int)parse_ll(val, arg);
            } else if (strcmp(arg, "--max-iter") == 0) {
                cfg.max_iter = parse_int(val, arg);
            } else if (strcmp(arg, "--tol") == 0) {
                cfg.tol = parse_double(val, arg);
            } else if (strcmp(arg, "--power-iters") == 0) {
                cfg.power_iters = parse_int(val, arg);
            } else {
                fprintf(stderr, "Unknown option: %s\n", arg);
                usage(argv[0]);
                exit(2);
            }
        }
    }

    if (cfg.p <= 0 || cfg.N <= 0 || cfg.runs <= 0) {
        fprintf(stderr, "--p, --N, and --runs must be positive.\n");
        exit(2);
    }
    if (cfg.lambda < 0.0 || cfg.max_iter <= 0 || cfg.power_iters <= 0 || cfg.tol <= 0.0) {
        fprintf(stderr, "--lambda must be nonnegative; solver parameters must be positive.\n");
        exit(2);
    }

    return cfg;
}

static double randn_bm(unsigned int *seed, int *have_spare, double *spare)
{
    if (*have_spare) {
        *have_spare = 0;
        return *spare;
    }

    double u, v, s;
    do {
        u = ((double)rand_r(seed) + 1.0) / ((double)RAND_MAX + 2.0) * 2.0 - 1.0;
        v = ((double)rand_r(seed) + 1.0) / ((double)RAND_MAX + 2.0) * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);

    double mul = sqrt(-2.0 * log(s) / s);
    *spare = v * mul;
    *have_spare = 1;
    return u * mul;
}

static double soft_threshold_scalar(double x, double theta)
{
    if (x > theta) {
        return x - theta;
    }
    if (x < -theta) {
        return x + theta;
    }
    return 0.0;
}

static double lasso_objective(int p, const double *G, const double *b,
                              double yy, double alpha, const double *w)
{
    double quad = 0.0;
    double dot = 0.0;
    double l1 = 0.0;

    for (int j = 0; j < p; j++) {
        double gwj = 0.0;
        const double *Gj = G + (size_t)j * p;
        for (int k = 0; k < p; k++) {
            gwj += Gj[k] * w[k];
        }
        quad += w[j] * gwj;
        dot += b[j] * w[j];
        l1 += fabs(w[j]);
    }

    return 0.5 * (quad - 2.0 * dot + yy) + alpha * l1;
}

static double largest_eigenvalue_power(int p, const double *G, int power_iters,
                                       double *tmp)
{
    double *v = (double*)xmalloc((size_t)p * sizeof(double));
    double inv = 1.0 / sqrt((double)p);
    for (int j = 0; j < p; j++) {
        v[j] = inv;
    }

    for (int it = 0; it < power_iters; it++) {
        for (int j = 0; j < p; j++) {
            double s = 0.0;
            const double *Gj = G + (size_t)j * p;
            for (int k = 0; k < p; k++) {
                s += Gj[k] * v[k];
            }
            tmp[j] = s;
        }

        double nrm = 0.0;
        for (int j = 0; j < p; j++) {
            nrm += tmp[j] * tmp[j];
        }
        nrm = sqrt(nrm);
        if (nrm <= 0.0 || !isfinite(nrm)) {
            free(v);
            return 1e-15;
        }
        for (int j = 0; j < p; j++) {
            v[j] = tmp[j] / nrm;
        }
    }

    for (int j = 0; j < p; j++) {
        double s = 0.0;
        const double *Gj = G + (size_t)j * p;
        for (int k = 0; k < p; k++) {
            s += Gj[k] * v[k];
        }
        tmp[j] = s;
    }

    double lambda_max = 0.0;
    for (int j = 0; j < p; j++) {
        lambda_max += v[j] * tmp[j];
    }

    free(v);
    if (lambda_max <= 0.0 || !isfinite(lambda_max)) {
        return 1e-15;
    }
    return lambda_max;
}

static void solve_lasso_fista(int p, const double *G, const double *b,
                              double yy, double alpha, int max_iter,
                              double tol, int power_iters, double *w)
{
    double *v = (double*)xmalloc((size_t)p * sizeof(double));
    double *w_new = (double*)xmalloc((size_t)p * sizeof(double));
    double *grad = (double*)xmalloc((size_t)p * sizeof(double));
    double *tmp = (double*)xmalloc((size_t)p * sizeof(double));

    for (int j = 0; j < p; j++) {
        w[j] = 0.0;
        v[j] = 0.0;
    }

    double L = largest_eigenvalue_power(p, G, power_iters, tmp);
    double step = 1.0 / fmax(L, 1e-15);
    double tk = 1.0;
    double obj_prev = INFINITY;

    for (int it = 1; it <= max_iter; it++) {
        for (int j = 0; j < p; j++) {
            double gv = -b[j];
            const double *Gj = G + (size_t)j * p;
            for (int k = 0; k < p; k++) {
                gv += Gj[k] * v[k];
            }
            grad[j] = gv;
        }

        for (int j = 0; j < p; j++) {
            w_new[j] = soft_threshold_scalar(v[j] - step * grad[j],
                                             alpha * step);
        }

        double tk_new = 0.5 * (1.0 + sqrt(1.0 + 4.0 * tk * tk));
        double coeff = (tk - 1.0) / tk_new;
        for (int j = 0; j < p; j++) {
            v[j] = w_new[j] + coeff * (w_new[j] - w[j]);
        }

        if (it == 1 || it % 10 == 0) {
            double obj = lasso_objective(p, G, b, yy, alpha, w_new);
            double rel = fabs(obj_prev - obj) / fmax(1.0, fabs(obj));
            if (rel < tol) {
                memcpy(w, w_new, (size_t)p * sizeof(double));
                break;
            }
            obj_prev = obj;
        }

        memcpy(w, w_new, (size_t)p * sizeof(double));
        tk = tk_new;
    }

    free(v);
    free(w_new);
    free(grad);
    free(tmp);
}

static double lasso_run(const Config *cfg, unsigned int *seed,
                        int *have_spare, double *spare)
{
    int p = cfg->p;
    int N = cfg->N;
    double inv_sqrt_p = 1.0 / sqrt((double)p);
    double alpha = 0.5 * cfg->lambda;

    double *A = (double*)xmalloc((size_t)N * p * sizeof(double));
    double *y = (double*)xmalloc((size_t)N * sizeof(double));
    double *G = (double*)xmalloc((size_t)p * p * sizeof(double));
    double *b = (double*)xmalloc((size_t)p * sizeof(double));
    double *w = (double*)xmalloc((size_t)p * sizeof(double));

    for (int j = 0; j < p; j++) {
        for (int i = 0; i < N; i++) {
            A[(size_t)j * N + i] = randn_bm(seed, have_spare, spare) * inv_sqrt_p;
        }
    }

    for (int i = 0; i < N; i++) {
        double signal = 0.0;
        for (int j = 0; j < p; j++) {
            signal += A[(size_t)j * N + i] * cfg->beta0;
        }
        y[i] = signal + randn_bm(seed, have_spare, spare);
    }

    double yy = 0.0;
    for (int i = 0; i < N; i++) {
        yy += y[i] * y[i];
    }

    for (int j = 0; j < p; j++) {
        for (int k = 0; k < p; k++) {
            double s = 0.0;
            const double *Aj = A + (size_t)j * N;
            const double *Ak = A + (size_t)k * N;
            for (int i = 0; i < N; i++) {
                s += Aj[i] * Ak[i];
            }
            G[(size_t)j * p + k] = s;
        }
    }

    for (int j = 0; j < p; j++) {
        double s = 0.0;
        const double *Aj = A + (size_t)j * N;
        for (int i = 0; i < N; i++) {
            s += Aj[i] * y[i];
        }
        b[j] = s;
    }

    solve_lasso_fista(p, G, b, yy, alpha, cfg->max_iter, cfg->tol,
                      cfg->power_iters, w);
    double E = lasso_objective(p, G, b, yy, alpha, w);
    double xi = E / (double)p;

    free(A);
    free(y);
    free(G);
    free(b);
    free(w);

    return xi;
}

int main(int argc, char **argv)
{
    Config cfg = parse_args(argc, argv);

    if (!cfg.quiet) {
        fprintf(stderr,
            "lasso_regression_omp: p=%d N=%d r=%.8g lambda=%.8g "
            "beta0=%.8g runs=%lld max_iter=%d tol=%.3g threads=%d seed=%u\n",
            cfg.p, cfg.N, (double)cfg.p / (double)cfg.N, cfg.lambda,
            cfg.beta0, cfg.runs, cfg.max_iter, cfg.tol,
            omp_get_max_threads(), cfg.seed);
    }

    long long count = 0;
    double mean = 0.0;
    double M2 = 0.0;

    #pragma omp parallel
    {
        unsigned int seed = cfg.seed
            ^ (unsigned int)(omp_get_thread_num() * 1000003u + 0x9e3779b9u);
        int have_spare = 0;
        double spare = 0.0;

        #pragma omp for schedule(dynamic)
        for (long long run = 0; run < cfg.runs; run++) {
            double xi = lasso_run(&cfg, &seed, &have_spare, &spare);

            #pragma omp critical
            {
                printf("%.17g\n", xi);

                count++;
                double delta = xi - mean;
                mean += delta / (double)count;
                M2 += delta * (xi - mean);
            }
        }
    }

    double variance = (count > 1) ? M2 / (double)count : 0.0;
    fprintf(stderr, "runs=%lld  mean=%.12g  std=%.12g\n",
            count, mean, sqrt(variance));

    return 0;
}
