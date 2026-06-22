/*
 * ols_sampler.c
 *
 * OpenMP-parallelised Monte Carlo sampler for the OLS large-deviation study.
 * Extracted from ldlr_sampler.c (the OLS branch of one_trial in the MATLAB
 * ldlr_v3_seedsafe.m).  The theory / rate-function analysis is done separately.
 *
 * Model (P < N):
 *     A = randn(N,P)/sqrt(P),   beta ~ dist,   eps ~ N(0,1)^N,   y = A*beta + eps
 *     w = argmin ||y - A w||^2          (ordinary least squares)
 *     E = 0.5 ||y - A w||^2
 *     xi = E / P
 *
 * KEY FACT (why this file is so short).  The OLS residual is r = (I - H) y with
 * H = A (A'A)^{-1} A' the hat matrix.  Since (I - H) A = 0, the signal term is
 * annihilated: r = (I - H) eps, so
 *     RSS = eps' (I - H) eps   ~   chi^2_{N-P}
 * independently of beta and of the realisation of A.  Consequences:
 *   (1) the "fast" path draws g ~ N(0,1)^{N-P} and sets E = 0.5 g'g -- no design
 *       matrix, no solve, exact in distribution;
 *   (2) the scaled-cumulant generating function and rate function are closed
 *       form, with  a = (1-r)/(2r),  Phi(s) = a*log(1+s),  xi(s) = a/(1+s),
 *       so  E[xi] = a  and  Var[xi] ~ a/P.
 * The "exact" path (--exact) does the genuine least-squares solve via dgels and
 * is provided only to verify (1) and (2) empirically; it must agree with the
 * fast path for any choice of beta.
 *
 * Differences vs the original ridge_regression_omp.c style:
 *   - xoshiro256** PRNG (SplitMix64 seeding) instead of rand_r; rand_r's short
 *     period / weak high bits corrupt exactly the deep tails the rate function
 *     lives in.
 *   - Per-run reseeding by run index: sample r is reproducible regardless of
 *     thread count or scheduling (mirrors cfg.seed + 1000003*t in the MATLAB).
 *   - Per-thread output buffering + a pairwise Welford reduction (no per-sample
 *     critical section).
 *   - Raw little-endian float64 output by default plus a .meta sidecar; --text
 *     reproduces "one xi per line".
 *
 * Build (Linux, OpenBLAS):
 *   gcc -O3 -march=native -fopenmp -o ols_sampler ols_sampler.c -lopenblas -lm
 * Build (reference BLAS/LAPACK):
 *   gcc -O3 -fopenmp -o ols_sampler ols_sampler.c -llapack -lblas -lm
 * Build (macOS, Accelerate + libomp):
 *   clang -O3 -DACCELERATE_NEW_LAPACK -Xpreprocessor -fopenmp \
 *       -I$(brew --prefix libomp)/include -L$(brew --prefix libomp)/lib \
 *       -o ols_sampler ols_sampler.c -framework Accelerate -lomp -lm
 *
 * Examples:
 *   ./ols_sampler --N 100 --P 70 --runs 1000000 --out ols.bin
 *   ./ols_sampler --N 100 --P 70 --runs 100000 --exact --beta student --beta-df 5
 *   ./ols_sampler --N 100 --P 70 --runs 200000 --text --out -
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>
#include <omp.h>

#ifdef __APPLE__
#  include <Accelerate/Accelerate.h>
#else
#  include <cblas.h>
extern void dgels_(char*, int*, int*, int*, double*, int*, double*, int*,
                   double*, int*, int*);
#endif

extern void openblas_set_num_threads(int) __attribute__((weak));

/* ====================================================================== */
/* PRNG: SplitMix64 seeding + xoshiro256** + Box-Muller normals           */
/* ====================================================================== */
typedef struct { uint64_t s[4]; int have_spare; double spare; } rng_t;

static inline uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static inline uint64_t rotl(uint64_t x, int k){ return (x<<k)|(x>>(64-k)); }

static void rng_seed(rng_t *r, uint64_t seed) {
    uint64_t sm = seed;
    for (int i=0;i<4;i++) r->s[i]=splitmix64(&sm);
    r->have_spare=0; r->spare=0.0;
    for (int i=0;i<8;i++){
        uint64_t t=r->s[1]<<17;
        r->s[2]^=r->s[0]; r->s[3]^=r->s[1];
        r->s[1]^=r->s[2]; r->s[0]^=r->s[3];
        r->s[2]^=t; r->s[3]=rotl(r->s[3],45);
    }
}
static inline uint64_t rng_u64(rng_t *r){
    uint64_t result=rotl(r->s[1]*5,7)*9, t=r->s[1]<<17;
    r->s[2]^=r->s[0]; r->s[3]^=r->s[1];
    r->s[1]^=r->s[2]; r->s[0]^=r->s[3];
    r->s[2]^=t; r->s[3]=rotl(r->s[3],45);
    return result;
}
static inline double rng_unif(rng_t *r){
    uint64_t x=rng_u64(r)>>11;                 /* 53 bits */
    return (x+0.5)*(1.0/9007199254740992.0);
}
static double rng_normal(rng_t *r){
    if (r->have_spare){ r->have_spare=0; return r->spare; }
    double u,v,s;
    do { u=2.0*rng_unif(r)-1.0; v=2.0*rng_unif(r)-1.0; s=u*u+v*v; }
    while (s>=1.0||s==0.0);
    double mul=sqrt(-2.0*log(s)/s);
    r->spare=v*mul; r->have_spare=1;
    return u*mul;
}
/* Gamma(shape,scale) Marsaglia-Tsang; only used for student beta. */
static double rng_gamma(rng_t *r, double shape, double scale){
    if (shape<1.0){ double g=rng_gamma(r,shape+1.0,scale); return g*pow(rng_unif(r),1.0/shape); }
    double d=shape-1.0/3.0, c=1.0/sqrt(9.0*d);
    for(;;){
        double z=rng_normal(r), v=1.0+c*z;
        if (v<=0.0) continue;
        v=v*v*v;
        double u=rng_unif(r);
        if (log(u) < 0.5*z*z + d - d*v + d*log(v)) return scale*d*v;
    }
}
static double rng_student_t(rng_t *r, double df){
    double z=rng_normal(r), g=rng_gamma(r,df/2.0,2.0);   /* chi2(df) */
    return z/sqrt(g/df);
}

/* ====================================================================== */
/* Configuration                                                          */
/* ====================================================================== */
typedef struct {
    long      N, P;
    long long runs;
    uint64_t  seed;
    int       threads;
    char      beta_kind[24];
    double    beta_mean, beta_std, beta_value, beta_rho, beta_df;
    int       exact;          /* 0 => fast chi-square path */
    char      out[1024];
    int       text_output;
} config;

static void set_defaults(config *c){
    c->N=100; c->P=70; c->runs=1000000; c->seed=12345ULL; c->threads=0;
    strcpy(c->beta_kind,"normal");
    c->beta_mean=0.0; c->beta_std=1.0; c->beta_value=1.0; c->beta_rho=0.2; c->beta_df=5.0;
    c->exact=0; strcpy(c->out,"ols.bin"); c->text_output=0;
}
static void usage(const char *p){
    fprintf(stderr,
"Usage: %s [options]   (ordinary least squares, requires P<N)\n"
"  --N N --P P                       problem size\n"
"  --runs R                          number of Monte Carlo trials\n"
"  --seed S                          base seed (reproducible per run)\n"
"  --threads T                       OpenMP threads (0 = default)\n"
"  --exact                           genuine least-squares solve (dgels)\n"
"                                    instead of the chi-square fast path\n"
"  --beta KIND                       normal|constant|rademacher|\n"
"                                    sparse_rademacher|student (exact path only;\n"
"                                    OLS RSS is beta-independent)\n"
"  --beta-mean M --beta-std S --beta-value V --beta-rho R --beta-df D\n"
"  --out FILE                        output path ('-' = stdout, text)\n"
"  --text                            text output (one xi per line)\n", p);
}
static int need_arg(int i,int argc,const char *f){
    if(i+1>=argc){ fprintf(stderr,"missing value for %s\n",f); exit(2);} return i+1;
}
static void parse_args(int argc,char**argv,config*c){
    for(int i=1;i<argc;i++){
        const char*a=argv[i];
        if      (!strcmp(a,"--N"))          { i=need_arg(i,argc,a); c->N=atol(argv[i]); }
        else if (!strcmp(a,"--P"))          { i=need_arg(i,argc,a); c->P=atol(argv[i]); }
        else if (!strcmp(a,"--runs"))       { i=need_arg(i,argc,a); c->runs=atoll(argv[i]); }
        else if (!strcmp(a,"--seed"))       { i=need_arg(i,argc,a); c->seed=strtoull(argv[i],NULL,10); }
        else if (!strcmp(a,"--threads"))    { i=need_arg(i,argc,a); c->threads=atoi(argv[i]); }
        else if (!strcmp(a,"--exact"))      { c->exact=1; }
        else if (!strcmp(a,"--beta"))       { i=need_arg(i,argc,a); strncpy(c->beta_kind,argv[i],sizeof c->beta_kind-1); }
        else if (!strcmp(a,"--beta-mean"))  { i=need_arg(i,argc,a); c->beta_mean=atof(argv[i]); }
        else if (!strcmp(a,"--beta-std"))   { i=need_arg(i,argc,a); c->beta_std=atof(argv[i]); }
        else if (!strcmp(a,"--beta-value")) { i=need_arg(i,argc,a); c->beta_value=atof(argv[i]); }
        else if (!strcmp(a,"--beta-rho"))   { i=need_arg(i,argc,a); c->beta_rho=atof(argv[i]); }
        else if (!strcmp(a,"--beta-df"))    { i=need_arg(i,argc,a); c->beta_df=atof(argv[i]); }
        else if (!strcmp(a,"--out"))        { i=need_arg(i,argc,a); strncpy(c->out,argv[i],sizeof c->out-1); }
        else if (!strcmp(a,"--text"))       { c->text_output=1; }
        else if (!strcmp(a,"--help")||!strcmp(a,"-h")) { usage(argv[0]); exit(0); }
        else { fprintf(stderr,"unknown option: %s\n",a); usage(argv[0]); exit(2); }
    }
}
static void validate(config*c){
    for(char*p=c->beta_kind;*p;++p)*p=(char)tolower((unsigned char)*p);
    if(c->N<=0||c->P<=0||c->runs<=0){ fprintf(stderr,"N,P,runs must be positive\n"); exit(2);}
    if(c->P>=c->N){ fprintf(stderr,"OLS benchmark requires P<N\n"); exit(2);}
    int ok=!strcmp(c->beta_kind,"normal")||!strcmp(c->beta_kind,"constant")||
           !strcmp(c->beta_kind,"rademacher")||!strcmp(c->beta_kind,"sparse_rademacher")||
           !strcmp(c->beta_kind,"student");
    if(!ok){ fprintf(stderr,"unsupported beta kind\n"); exit(2);}
    if(!strcmp(c->beta_kind,"student")&&c->beta_df<=2.0){
        fprintf(stderr,"student beta requires df>2\n"); exit(2);}
}

/* beta sampling (matches sample_beta in the MATLAB) -- exact path only */
static void sample_beta(rng_t *r,const config*c,double*beta,long P){
    if(!strcmp(c->beta_kind,"normal")){
        for(long i=0;i<P;i++) beta[i]=c->beta_mean+c->beta_std*rng_normal(r);
    } else if(!strcmp(c->beta_kind,"constant")){
        for(long i=0;i<P;i++) beta[i]=c->beta_value;
    } else if(!strcmp(c->beta_kind,"rademacher")){
        double amp=c->beta_std;
        for(long i=0;i<P;i++) beta[i]=(rng_unif(r)<0.5)?-amp:amp;
    } else if(!strcmp(c->beta_kind,"sparse_rademacher")){
        double rho=c->beta_rho, amp=c->beta_std/sqrt(rho>1e-300?rho:1e-300);
        for(long i=0;i<P;i++){
            if(rng_unif(r)<rho) beta[i]=(rng_unif(r)<0.5)?-amp:amp; else beta[i]=0.0;
        }
    } else { /* student */
        double scl=c->beta_std/sqrt(c->beta_df/(c->beta_df-2.0));
        for(long i=0;i<P;i++) beta[i]=scl*rng_student_t(r,c->beta_df);
    }
}

/* ====================================================================== */
/* Trials -> xi = E/P                                                     */
/* ====================================================================== */
/* fast path: RSS ~ chi^2_{N-P}, distribution-free in beta and A */
static inline double trial_fast(rng_t *r, const config *c){
    long P=c->P, m=c->N-c->P;
    double rss=0.0;
    for(long i=0;i<m;i++){ double g=rng_normal(r); rss+=g*g; }
    return 0.5*rss/(double)P;
}

/* exact path: form A,beta,y and do the least-squares solve.
 * dgels overwrites y with [solution(0..P-1); residuals(P..N-1)],
 * so RSS = sum_{i=P}^{N-1} y[i]^2. */
typedef struct { double *A,*beta,*y,*work; int lwork; } ws_t;

static double trial_exact(rng_t *r, const config *c, ws_t *ws){
    long N=c->N, P=c->P;
    double inv=1.0/sqrt((double)P);
    for(long k=0;k<N*P;k++) ws->A[k]=inv*rng_normal(r);
    sample_beta(r,c,ws->beta,P);
    for(long i=0;i<N;i++) ws->y[i]=rng_normal(r);          /* eps */
    cblas_dgemv(CblasColMajor,CblasNoTrans,(int)N,(int)P,
                1.0,ws->A,(int)N,ws->beta,1,1.0,ws->y,1);  /* y += A beta */
    char tr='N'; int m=(int)N,n=(int)P,nrhs=1,lda=(int)N,ldb=(int)N,info;
    dgels_(&tr,&m,&n,&nrhs,ws->A,&lda,ws->y,&ldb,ws->work,&ws->lwork,&info);
    if(info!=0){ fprintf(stderr,"dgels info=%d\n",info); exit(1); }
    double rss=0.0; for(long i=P;i<N;i++) rss+=ws->y[i]*ws->y[i];
    return 0.5*rss/(double)P;
}

/* ====================================================================== */
/* Welford                                                                */
/* ====================================================================== */
typedef struct { long long n; double mean, M2; } welford;
static inline void welford_push(welford*a,double x){
    a->n++; double d=x-a->mean; a->mean+=d/(double)a->n; a->M2+=d*(x-a->mean);
}
static void welford_merge(welford*a,const welford*b){
    if(b->n==0)return; if(a->n==0){*a=*b;return;}
    long long n=a->n+b->n; double d=b->mean-a->mean;
    a->mean=a->mean+d*(double)b->n/(double)n;
    a->M2=a->M2+b->M2+d*d*(double)a->n*(double)b->n/(double)n;
    a->n=n;
}
static inline uint64_t seed_for_run(uint64_t base,long long run){
    uint64_t x=base+0x9E3779B97F4A7C15ULL*(uint64_t)(run+1);
    return splitmix64(&x);
}

/* ====================================================================== */
/* main                                                                   */
/* ====================================================================== */
int main(int argc,char**argv){
    config cfg; set_defaults(&cfg);
    parse_args(argc,argv,&cfg); validate(&cfg);
    if(openblas_set_num_threads) openblas_set_num_threads(1);
    if(cfg.threads>0) omp_set_num_threads(cfg.threads);

    /* query dgels workspace once (exact path only) */
    int g_lwork=1;
    if(cfg.exact){
#ifndef __APPLE__
        int info,lwork=-1; double q;
        char tr='N'; int m=(int)cfg.N,n=(int)cfg.P,nrhs=1,lda=(int)cfg.N,ldb=(int)cfg.N;
        dgels_(&tr,&m,&n,&nrhs,NULL,&lda,NULL,&ldb,&q,&lwork,&info);
        g_lwork=(int)q; if(g_lwork<1) g_lwork=1;
#else
        g_lwork=64*(int)cfg.P;
#endif
    }

    FILE *fout=NULL; int to_stdout=(!strcmp(cfg.out,"-"));
    if(to_stdout){ cfg.text_output=1; fout=stdout; }
    else { fout=fopen(cfg.out,cfg.text_output?"w":"wb"); if(!fout){perror("fopen out");return 1;} }

    double r = (double)cfg.P/cfg.N;
    double a_theory = (1.0-r)/(2.0*r);            /* E[xi]; Var[xi] ~ a/P */
    fprintf(stderr,
        "OLS sampler: N=%ld P=%ld r=%.6g runs=%lld path=%s\n"
        "  beta=%s (exact path only; RSS is beta-independent)\n"
        "  seed=%llu threads=%d out=%s format=%s\n"
        "  closed form: E[xi]=(1-r)/(2r)=%.12g  Var[xi]~a/P=%.12g\n",
        cfg.N,cfg.P,r,cfg.runs, cfg.exact?"exact-LS":"chi-square",
        cfg.beta_kind,(unsigned long long)cfg.seed,cfg.threads,cfg.out,
        cfg.text_output?"text":"binary", a_theory, a_theory/(double)cfg.P);

    welford global={0,0.0,0.0};
    double t0=omp_get_wtime();
    const size_t BUFCAP=1u<<16;

    #pragma omp parallel
    {
        ws_t ws={0}; ws.lwork=g_lwork;
        if(cfg.exact){
            ws.A   =malloc((size_t)cfg.N*cfg.P*sizeof(double));
            ws.beta=malloc((size_t)cfg.P*sizeof(double));
            ws.y   =malloc((size_t)cfg.N*sizeof(double));
            ws.work=malloc((size_t)g_lwork*sizeof(double));
        }
        rng_t rng; welford loc={0,0.0,0.0};
        double *buf=malloc(BUFCAP*sizeof(double)); size_t bn=0;

        #pragma omp for schedule(dynamic, 4096)
        for(long long run=0;run<cfg.runs;run++){
            rng_seed(&rng, seed_for_run(cfg.seed,run));
            double xi = cfg.exact ? trial_exact(&rng,&cfg,&ws) : trial_fast(&rng,&cfg);
            welford_push(&loc,xi);
            buf[bn++]=xi;
            if(bn==BUFCAP){
                #pragma omp critical (io)
                { if(cfg.text_output) for(size_t k=0;k<bn;k++) fprintf(fout,"%.10g\n",buf[k]);
                  else fwrite(buf,sizeof(double),bn,fout); }
                bn=0;
            }
        }
        if(bn){
            #pragma omp critical (io)
            { if(cfg.text_output) for(size_t k=0;k<bn;k++) fprintf(fout,"%.10g\n",buf[k]);
              else fwrite(buf,sizeof(double),bn,fout); }
        }
        #pragma omp critical (reduce)
        { welford_merge(&global,&loc); }

        free(buf);
        if(cfg.exact){ free(ws.A); free(ws.beta); free(ws.y); free(ws.work); }
    }

    double secs=omp_get_wtime()-t0;
    if(!to_stdout) fclose(fout);

    double var_pop    = (global.n>1)? global.M2/(double)global.n        : 0.0;
    double var_sample = (global.n>1)? global.M2/(double)(global.n-1)    : 0.0;
    fprintf(stderr,
        "done: n=%lld mean_xi=%.12g var_xi(sample)=%.12g time=%.2fs (%.3g trials/s)\n"
        "  vs theory: E[xi]=%.12g  Var[xi]~a/P=%.12g\n",
        global.n, global.mean, var_sample, secs, global.n/(secs>0?secs:1),
        a_theory, a_theory/(double)cfg.P);

    if(!to_stdout){
        char meta[1100]; snprintf(meta,sizeof meta,"%s.meta",cfg.out);
        FILE*fm=fopen(meta,"w");
        if(fm){
            fprintf(fm,
                "model=ols\nN=%ld\nP=%ld\nr=%.12g\nlambda=0\nruns=%lld\nn=%lld\n"
                "beta_kind=%s\nbeta_mean=%.12g\nbeta_std=%.12g\nbeta_value=%.12g\n"
                "beta_rho=%.12g\nbeta_df=%.12g\nseed=%llu\nexact=%d\n"
                "format=%s\ndtype=float64\nmean_xi=%.15g\nvar_xi_sample=%.15g\n"
                "theory_mean_xi=%.15g\ntheory_var_xi=%.15g\n",
                cfg.N,cfg.P,r,cfg.runs,global.n,
                cfg.beta_kind,cfg.beta_mean,cfg.beta_std,cfg.beta_value,cfg.beta_rho,cfg.beta_df,
                (unsigned long long)cfg.seed,cfg.exact,
                cfg.text_output?"text":"binary_le",global.mean,var_sample,
                a_theory,a_theory/(double)cfg.P);
            fclose(fm);
            fprintf(stderr,"wrote %s and %s\n",cfg.out,meta);
        }
    }
    return 0;
}