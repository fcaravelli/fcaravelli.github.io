function results = ldlr_v3_seedsafe(cfg)
%LDLR_SINGLE_SIMPLE  Single-file MATLAB code for large deviations in random linear regression.
%
% CLEAR FUNCTION CALLS
%   results = ldlr_single_simple();
%   results = ldlr_single_simple(cfg);
%
% MINIMAL EXAMPLE
%   cfg = struct();
%   cfg.model = 'ridge';          % 'ols', 'ridge', or 'lasso'
%   cfg.N = 600;
%   cfg.P = 480;
%   cfg.lambda = 0.5;
%   cfg.nTrials = 2000;
%   cfg.beta.kind = 'normal';     % 'normal','constant','rademacher','sparse_rademacher','student'
%   cfg.beta.mean = 0;
%   cfg.beta.std = 1;
%   results = ldlr_single_simple(cfg);
%
% OBJECTIVE AND DATA MODEL
%   y = X beta / sqrt(P) + epsilon,  epsilon_i ~ N(0,1), X_ij ~ N(0,1)
%   H_lambda(w) = 1/2 || y - X w / sqrt(P) ||_2^2 + sum_j g_lambda(w_j)
%
% PENALTIES
%   OLS:   g_lambda(w) = 0, implemented for P<N.
%   Ridge: g_lambda(w) = lambda*w^2/2.
%   Lasso: g_lambda(w) = lambda*|w|/2.
%
% OUTPUTS
%   results.cfg       configuration actually used
%   results.sim       samples of xi=E/P
%   results.theory    Phi(s), xi(s), psi(xi) from finite-s saddle equations
%   results.analysis  logarithmic histogram, empirical rate, Gaussian approximation
%
% FILE OUTPUTS
%   A .mat file and TWO .png figures are written in cfg.outDir:
%     *_Phi.png       theoretical scaled cumulant generating function Phi(s)
%     *_rate.png      rate function + empirical dots + Gaussian approximation
%
% NOTES
%   The theoretical Ridge/Lasso calculation uses the annealed-beta finite-s
%   saddle equations of the supplementary material. OLS uses the exact
%   chi-square result for r=P/N<1.

if nargin < 1 || isempty(cfg)
    cfg = default_config();
end
cfg = merge_defaults(cfg, default_config());
cfg = validate_config(cfg);
ensure_dir(cfg.outDir);

fprintf('LDLR single-file run started: %s\n', datestr(now));
fprintf('case=%s, N=%d, P=%d, r=%.8g, lambda=%.8g, trials=%d\n', ...
    cfg.model, cfg.N, cfg.P, cfg.P/cfg.N, cfg.lambda, cfg.nTrials);

rng(safe_seed(cfg.seed), 'twister');
sim = simulate_ldlr(cfg);
theory = compute_theory_rate(cfg);
analysis = analyze_empirical_rate(sim.xi, cfg, theory);

stamp = datestr(now, 'yyyymmdd_HHMMSS');
base = sprintf('ldlr_%s_N%d_P%d_lam%.4g_%s', cfg.model, cfg.N, cfg.P, cfg.lambda, stamp);
basePath = fullfile(cfg.outDir, base);
plot_ldlr_outputs(cfg, theory, analysis, basePath);

results = struct();
results.cfg = cfg;
results.config = cfg;                  % easy alias
results.sim = sim;
results.theory = theory;
results.analysis = analysis;
% Easy-to-use output fields for plotting or further analysis
results.data = struct();
results.data.xiSamples = sim.xi;
results.data.energySamples = sim.E;
results.data.s = theory.s;
results.data.Phi = theory.Phi;
results.data.xiTheory = theory.xiSorted;
results.data.psiTheory = theory.psiSorted;
results.data.xiHist = analysis.hist.centers;
results.data.logHistDensity = []; % intentionally not plotted
results.data.xiRateEmpirical = analysis.rate.xi;
results.data.psiEmpirical = analysis.rate.psi;
results.data.xiGaussian = analysis.gauss.xi;
results.data.psiGaussian = analysis.gauss.psi;
results.data.outputFolder = cfg.outDir;

outMat = [basePath '.mat'];
try
    save(outMat, 'results', '-v7.3');
catch
    save(outMat, 'results');
end
fprintf('Saved results: %s\n', outMat);
fprintf('LDLR single-file run finished: %s\n', datestr(now));
end

% ========================================================================
% DEFAULTS AND CONFIGURATION
% ========================================================================
function cfg = default_config()
cfg.model       = 'ridge';
cfg.N           = 600;
cfg.P           = 480;
cfg.lambda      = 0.5;
cfg.nTrials     = 2000;
cfg.seed        = 12345;
cfg.outDir      = fullfile(pwd, 'output_ldlr_single');

cfg.beta.kind   = 'normal';
cfg.beta.mean   = 0;
cfg.beta.std    = 1;
cfg.beta.value  = 1;
cfg.beta.rho    = 0.2;
cfg.beta.df     = 5;

cfg.useParallel = false;
cfg.nWorkers    = 0;

cfg.sim.ols.fastChiSquare = true;
cfg.sim.lasso.maxIter = 2000;
cfg.sim.lasso.tol     = 1e-7;
cfg.sim.lasso.verbose = false;
cfg.sim.lasso.powerIters = 30;

cfg.theory.sGrid = unique([linspace(-0.85,-0.05,65), linspace(-0.045,0.045,41), linspace(0.05,8,140)]);
cfg.theory.nHermiteZ    = 80;
cfg.theory.nHermiteBeta = 80;
cfg.theory.solverTol    = 1e-9;
cfg.theory.solverMaxIt  = 2000;
cfg.theory.verbose      = false;

cfg.hist.nBins = 80;
cfg.hist.minCount = 2;

% Plot controls.  The Gaussian curve is drawn across the full theory range,
% not just the narrow empirical-sample range.  Empirical points are thinned
% for readability.
cfg.plot.empiricalMaxDots = 35;
cfg.plot.empiricalMarkerSize = 7;
end

function cfg = merge_defaults(cfg, def)
fn = fieldnames(def);
for i = 1:numel(fn)
    f = fn{i};
    if ~isfield(cfg, f) || isempty(cfg.(f))
        cfg.(f) = def.(f);
    elseif isstruct(def.(f)) && isstruct(cfg.(f))
        cfg.(f) = merge_defaults(cfg.(f), def.(f));
    end
end
end

function cfg = validate_config(cfg)
cfg.model = lower(cfg.model);
if ~ismember(cfg.model, {'ols','ridge','lasso'})
    error('cfg.model must be ''ols'', ''ridge'', or ''lasso''.');
end
if cfg.N <= 0 || cfg.P <= 0 || cfg.nTrials <= 0
    error('cfg.N, cfg.P and cfg.nTrials must be positive.');
end
cfg.N = round(cfg.N); cfg.P = round(cfg.P); cfg.nTrials = round(cfg.nTrials);
if strcmp(cfg.model,'ols') && cfg.P >= cfg.N
    error('OLS exact benchmark is implemented only for P<N.');
end
if cfg.lambda < 0
    error('cfg.lambda must be nonnegative.');
end
if strcmp(cfg.model,'ols')
    cfg.lambda = 0;
end
if ~isfield(cfg,'beta') || ~isfield(cfg.beta,'kind')
    error('Please specify cfg.beta.kind.');
end
cfg.beta.kind = lower(cfg.beta.kind);
if ~ismember(cfg.beta.kind, {'normal','constant','rademacher','sparse_rademacher','student'})
    error('Unsupported cfg.beta.kind.');
end
if strcmp(cfg.beta.kind,'student') && cfg.beta.df <= 2
    error('Student beta distribution requires cfg.beta.df > 2 for finite variance.');
end
end

function ensure_dir(d)
if ~exist(d, 'dir')
    mkdir(d);
end
end

% ========================================================================
% MONTE CARLO SIMULATION
% ========================================================================
function sim = simulate_ldlr(cfg)
if cfg.useParallel
    maybe_start_pool(cfg.nWorkers);
end
nT = cfg.nTrials;
xi = nan(nT,1);
E  = nan(nT,1);

fprintf('Simulating %d trials...\n', nT);
t0 = tic;
if cfg.useParallel
    parfor t = 1:nT
        [E(t), xi(t)] = one_trial(cfg, safe_seed(cfg.seed + 1000003*t)); %#ok<PFBNS>
    end
else
    reportEvery = max(1, round(nT/20));
    for t = 1:nT
        [E(t), xi(t)] = one_trial(cfg, safe_seed(cfg.seed + 1000003*t));
        if mod(t, reportEvery) == 0 || t == nT
            fprintf('  %d/%d trials done (%.1fs)\n', t, nT, toc(t0));
        end
    end
end
sim.E = E;
sim.xi = xi;
sim.meanXi = mean_nan(xi);
sim.varXi = var_nan(xi);
sim.wallTime = toc(t0);
fprintf('Simulation done: mean xi=%.12g, var xi=%.12g, time=%.1fs\n', ...
    sim.meanXi, sim.varXi, sim.wallTime);
end


function s = safe_seed(seed)
% Convert any numeric seed into a valid MATLAB rng seed in [0, 2^32-1).
% This prevents failures in long simulations where cfg.seed + const*t
% exceeds MATLAB's allowed seed range.
M = 2^32 - 1;
s = mod(floor(double(seed)), M);
if ~isfinite(s) || s < 0
    s = 1;
end
s = double(s);
end

function [E, xi] = one_trial(cfg, seed)
rng(safe_seed(seed), 'twister');
N = cfg.N; P = cfg.P; lambda = cfg.lambda;
dist = beta_dist(cfg.beta, seed);
A = randn(N,P) / sqrt(P);
beta = dist.sample(P);
epsv = randn(N,1);
y = A*beta + epsv;

switch cfg.model
    case 'ols'
        if cfg.sim.ols.fastChiSquare
            g = randn(N-P,1);
            E = 0.5 * (g' * g);
        else
            w = A \ y;
            res = y - A*w;
            E = 0.5 * (res' * res);
        end
    case 'ridge'
        G = A' * A;
        b = A' * y;
        M = G + lambda * speye(P);
        [R,flag] = chol(M, 'lower');
        if flag == 0
            w = R' \ (R \ b);
        else
            w = M \ b;
        end
        res = y - A*w;
        E = 0.5*(res'*res) + 0.5*lambda*(w'*w);
    case 'lasso'
        opts = cfg.sim.lasso;
        [w, ~] = solve_lasso_fista(A, y, lambda/2, opts);
        res = y - A*w;
        E = 0.5*(res'*res) + 0.5*lambda*sum(abs(w));
end
xi = E / P;
end

function maybe_start_pool(nWorkers)
p = gcp('nocreate');
if isempty(p)
    if nWorkers > 0
        parpool(nWorkers);
    else
        parpool;
    end
end
end

function [w, out] = solve_lasso_fista(A, y, alpha, opts)
maxIter = getfield_default(opts,'maxIter',2000);
tol = getfield_default(opts,'tol',1e-7);
verbose = getfield_default(opts,'verbose',false);
powerIters = getfield_default(opts,'powerIters',30);
P = size(A,2);
L = power_lipschitz(A, powerIters);
step = 1 / max(L, eps);

w = zeros(P,1);
v = w;
tk = 1;
objPrev = inf;
for it = 1:maxIter
    r = A*v - y;
    grad = A' * r;
    wNew = soft_threshold(v - step*grad, alpha*step);
    tkNew = 0.5*(1 + sqrt(1 + 4*tk^2));
    v = wNew + ((tk - 1)/tkNew) * (wNew - w);
    if mod(it,10)==0 || it==1
        rr = A*wNew - y;
        obj = 0.5*(rr'*rr) + alpha*sum(abs(wNew));
        rel = abs(objPrev - obj) / max(1, abs(obj));
        if verbose && mod(it,100)==0
            fprintf('  FISTA it=%d obj=%.12g rel=%.3g\n', it, obj, rel);
        end
        if rel < tol
            w = wNew;
            objPrev = obj;
            break;
        end
        objPrev = obj;
    end
    w = wNew;
    tk = tkNew;
end
out.iter = it;
out.obj = objPrev;
out.L = L;
end

function L = power_lipschitz(A, nIt)
P = size(A,2);
v = randn(P,1);
v = v / norm(v);
for k = 1:nIt
    v = A'*(A*v);
    nv = norm(v);
    if nv == 0
        L = eps;
        return;
    end
    v = v / nv;
end
Av = A*v;
L = Av' * Av;
end

% ========================================================================
% THEORY: PHI(S), LEGENDRE RATE, SADDLE SOLVER
% ========================================================================
function theory = compute_theory_rate(cfg)
r = cfg.P / cfg.N;
sGrid = cfg.theory.sGrid(:)';
model = cfg.model;
theory = struct();
theory.r = r;
theory.sGridInput = sGrid;

switch model
    case 'ols'
        if r >= 1
            error('OLS theory implemented for r<1.');
        end
        sGrid = sGrid(sGrid > -0.999);
        a = (1-r)/(2*r);
        Phi = a * log1p(sGrid);
        xi = a ./ (1 + sGrid);
        psi = Phi - sGrid .* xi;
        [~,i0] = min(abs(sGrid));
        psi = psi - psi(i0);
        theory.s = sGrid(:);
        theory.Phi = Phi(:);
        theory.xi = xi(:);
        theory.psi = psi(:);
        theory.meanXi = a;
        theory.vPerP = a;
        theory.status = ones(numel(sGrid),1);
        theory.theta = [];
    otherwise
        quad = make_theory_quadrature(cfg);
        Phi = nan(numel(sGrid),1);
        theta = nan(numel(sGrid),4);
        status = zeros(numel(sGrid),1);
        Sval = nan(numel(sGrid),1);
        th0 = initial_theta(cfg, quad);
        for k = 1:numel(sGrid)
            s = sGrid(k);
            if abs(s) < 1e-9
                s = 1e-9;
            end
            [th, S, ok] = solve_saddle_general(s, cfg, quad, th0);
            theta(k,:) = th(:)';
            Sval(k) = S;
            Phi(k) = -S;
            status(k) = ok;
            if ok
                th0 = th;
            end
            if cfg.theory.verbose
                fprintf('  theory %s: s=% .5g Phi=% .10g ok=%d\n', model, s, Phi(k), ok);
            end
        end
        good = isfinite(Phi) & status>0;
        sGrid = sGrid(good); Phi = Phi(good); theta = theta(good,:); Sval = Sval(good); status = status(good);
        [sGrid,ord] = sort(sGrid(:));
        Phi = Phi(ord); theta = theta(ord,:); Sval = Sval(ord); status = status(ord);
        xi = gradient(Phi, sGrid);
        psi = Phi - sGrid .* xi;
        [~,i0] = min(abs(sGrid));
        psi = psi - psi(i0);
        [meanXi, vPerP] = local_cumulants(sGrid, Phi);
        theory.s = sGrid;
        theory.Phi = Phi;
        theory.xi = xi;
        theory.psi = psi;
        theory.theta = theta;
        theory.S = Sval;
        theory.status = status;
        theory.meanXi = meanXi;
        theory.vPerP = vPerP;
end

[theory.xiSorted, ordXi] = sort(theory.xi(:));
theory.psiSorted = theory.psi(ordXi);
theory.sSortedByXi = theory.s(ordXi);
end

function quad = make_theory_quadrature(cfg)
[zx, zw] = gauss_hermite(cfg.theory.nHermiteZ);
zNodes = sqrt(2)*zx;
zWeights = zw/sqrt(pi);
dist = beta_dist(cfg.beta, cfg.seed);
[bNodes,bWeights] = dist.quad(cfg.theory.nHermiteBeta);
[Z,B] = ndgrid(zNodes(:), bNodes(:));
[Wz,Wb] = ndgrid(zWeights(:), bWeights(:));
quad.z = Z(:);
quad.beta = B(:);
quad.w = Wz(:).*Wb(:);
quad.w = quad.w / sum(quad.w);
quad.betaSecondMoment = dist.secondMoment;
end

function th0 = initial_theta(cfg, quad)
r = cfg.P/cfg.N;
lam = cfg.lambda;
if strcmp(cfg.model,'ridge') && lam > 0
    disc = sqrt(4*lam*r^2 + ((lam-1)*r + 1)^2);
    chi = (disc - lam*r + r - 1) / (2*lam*r);
    chihat = (sqrt((lam+1)^2*r^2 + 2*(lam-1)*r + 1) - r*(lam+1) + 1)/(4*r);
    denom = (2*chihat + lam)^2;
    m2 = quad.betaSecondMoment;
    A = 1/denom;
    B = 1/(r*(1+chi)^2);
    q0 = (lam^2*m2*A + A*B) / max(1 - A*B, 1e-12);
    qhat0 = -(q0+1)/(2*r*(1+chi)^2);
    tau = sqrt(max(-2*qhat0, 1e-12));
else
    if r < 1
        chi = r/(1-r);
        chihat = (1-r)/(2*r);
        q0 = max(chi,1e-3);
        tau = sqrt(max((1-r)/r, 1e-12));
    else
        chi = 1;
        chihat = 1/(2*r*(1+chi));
        q0 = 1;
        tau = 1;
    end
end
th0 = log([max(q0,1e-10), max(chi,1e-10), max(chihat,1e-10), max(tau,1e-10)]);
end

function [theta, S, ok] = solve_saddle_general(s, cfg, quad, theta0)
obj = @(th) saddle_residual(th, s, cfg, quad);
ok = false;
theta = theta0;
usedFsolve = false;
try
    fopts = optimoptions('fsolve','Display','off', ...
        'FunctionTolerance',cfg.theory.solverTol, ...
        'StepTolerance',cfg.theory.solverTol, ...
        'MaxIterations',cfg.theory.solverMaxIt, ...
        'MaxFunctionEvaluations',20000);
    [theta,~,exitflag] = fsolve(obj, theta0, fopts);
    ok = exitflag > 0;
    usedFsolve = true;
catch
    opts = optimset('Display','off', 'TolX', cfg.theory.solverTol, ...
        'TolFun', cfg.theory.solverTol, 'MaxIter', cfg.theory.solverMaxIt, ...
        'MaxFunEvals', 20000);
    [theta, fval, exitflag] = fminsearch(@(th) sum(obj(th).^2), theta0, opts);
    ok = exitflag > 0 && fval < 1e-8;
end

res = obj(theta);
if norm(res) > 1e-5
    bestTheta = theta;
    bestNorm = norm(res);
    opts = optimset('Display','off', 'TolX', cfg.theory.solverTol, ...
        'TolFun', cfg.theory.solverTol, 'MaxIter', cfg.theory.solverMaxIt, ...
        'MaxFunEvals', 20000);
    for scale = [0.1, 0.3, 1.0]
        thTry0 = theta0 + scale*randn(size(theta0));
        if usedFsolve
            try
                [thTry,~,~] = fsolve(obj, thTry0, fopts);
                rn = norm(obj(thTry));
            catch
                [thTry, fval] = fminsearch(@(th) sum(obj(th).^2), thTry0, opts);
                rn = sqrt(fval);
            end
        else
            [thTry, fval] = fminsearch(@(th) sum(obj(th).^2), thTry0, opts);
            rn = sqrt(fval);
        end
        if rn < bestNorm
            bestNorm = rn;
            bestTheta = thTry;
        end
    end
    theta = bestTheta;
    ok = bestNorm < 1e-5;
end

[S, valid] = action_general(theta, s, cfg, quad);
ok = ok && valid && isfinite(S) && isreal(S);
end

function F = saddle_residual(theta, s, cfg, quad)
[q0, chi, chihat, tau, qhat0] = unpack_theta(theta);
r = cfg.P/cfg.N;
x = 1 + chi + s*(q0 + 1);
if x <= 0 || chi <= 0 || chihat <= 0 || tau <= 0 || ~isfinite(x)
    F = 1e6*ones(4,1);
    return;
end
[tm_u2, tm_zu] = tilted_moments(theta, s, cfg, quad);
F = zeros(4,1);
F(1) = chihat - 1/(2*r*(1+chi));
F(2) = qhat0 - (1/s)*(1/(2*r*x) - chihat);
F(3) = q0 - tm_u2;
F(4) = chi + s*q0 - tm_zu/tau;
F = F ./ [max(1,abs(chihat)); max(1,abs(qhat0)); max(1,abs(q0)); max(1,abs(chi+s*q0))];
end

function [tm_u2, tm_zu] = tilted_moments(theta, s, cfg, quad)
[~, ~, chihat, tau] = unpack_theta(theta);
[u, m] = scalar_u_m(cfg.model, cfg.lambda, chihat, tau, quad.beta, quad.z);
logw = log(max(quad.w, realmin)) - s*m;
mx = max(logw);
wtilt = exp(logw - mx);
wtilt = wtilt / sum(wtilt);
tm_u2 = sum(wtilt .* (u.^2));
tm_zu = sum(wtilt .* (quad.z .* u));
end

function [S, valid] = action_general(theta, s, cfg, quad)
[q0, chi, chihat, tau, qhat0] = unpack_theta(theta);
r = cfg.P/cfg.N;
x = 1 + chi + s*(q0+1);
valid = x>0 && chi>0 && chihat>0 && tau>0;
if ~valid
    S = NaN;
    return;
end
[~, m] = scalar_u_m(cfg.model, cfg.lambda, chihat, tau, quad.beta, quad.z);
logAvg = logsumexp(log(max(quad.w, realmin)) - s*m);
S = s*(qhat0*chi + q0*chihat) + s^2*q0*qhat0 ...
    - (1/(2*r))*log(x/(1+chi)) + logAvg;
valid = isfinite(S) && isreal(S);
end

function [u, m, wstar] = scalar_u_m(model, lambda, chihat, tau, beta, z)
% tau = sqrt(-2*qhat0), qhat0<0.
% Potential: V = chihat*u^2 - tau*z*u + g_lambda(beta-u), u=beta-w.
switch model
    case 'ridge'
        u = (lambda*beta + tau*z) ./ (2*chihat + lambda);
        wstar = beta - u;
        m = chihat*u.^2 - tau*z.*u + 0.5*lambda*wstar.^2;
    case 'lasso'
        a = beta - (tau/(2*chihat))*z;
        theta = lambda/(4*chihat);
        wstar = soft_threshold(a, theta);
        u = beta - wstar;
        m = chihat*u.^2 - tau*z.*u + 0.5*lambda*abs(wstar);
    otherwise
        error('Numerical saddle is used only for ridge/lasso.');
end
end

function [q0, chi, chihat, tau, qhat0] = unpack_theta(theta)
v = exp(theta(:));
q0 = v(1);
chi = v(2);
chihat = v(3);
tau = v(4);
qhat0 = -0.5*tau^2;
end

function [meanXi, vPerP] = local_cumulants(s, Phi)
if numel(s) < 3
    meanXi = NaN;
    vPerP = NaN;
    return;
end
[~,idx] = sort(abs(s));
idx = idx(1:min(11,numel(idx)));
ss = s(idx);
pp = Phi(idx);
[ss,ord] = sort(ss);
pp = pp(ord);
if numel(ss) >= 5
    deg = min(4,numel(ss)-1);
    p = polyfit(ss, pp, deg);
    dp = polyder(p);
    ddp = polyder(dp);
    meanXi = polyval(dp, 0);
    vPerP = -polyval(ddp, 0);
else
    d1 = gradient(Phi,s);
    d2 = gradient(d1,s);
    [~,i0] = min(abs(s));
    meanXi = d1(i0);
    vPerP = -d2(i0);
end
if ~isfinite(vPerP) || vPerP <= 0
    d1 = gradient(Phi,s);
    d2 = gradient(d1,s);
    [~,i0] = min(abs(s));
    meanXi = d1(i0);
    vPerP = max(-d2(i0), eps);
end
end

% ========================================================================
% EMPIRICAL RATE AND PLOTS
% ========================================================================
function analysis = analyze_empirical_rate(xi, cfg, theory)
xi = xi(isfinite(xi));
P = cfg.P;
nb = cfg.hist.nBins;
[counts, edges] = histcounts(xi, nb, 'Normalization', 'count');
centers = 0.5*(edges(1:end-1)+edges(2:end));
widths = diff(edges);
pdf = counts(:) ./ (numel(xi) * widths(:));
centers = centers(:);
counts = counts(:);
mask = counts >= cfg.hist.minCount & pdf > 0;
psiEmp = nan(size(centers));
psiEmp(mask) = -log(pdf(mask)) / P;
if any(mask)
    psiEmp = psiEmp - min(psiEmp(mask));
end
analysis.hist.centers = centers;
analysis.hist.edges = edges;
analysis.hist.counts = counts;
analysis.hist.pdf = pdf;
analysis.hist.logPdf = log(max(pdf, realmin));
analysis.rate.xi = centers(mask);
analysis.rate.psi = psiEmp(mask);
analysis.rate.counts = counts(mask);

mu = theory.meanXi;
v = theory.vPerP;
if ~isfinite(v) || v <= 0
    v = P * var_nan(xi);
end
% Draw the Gaussian approximation over the full theoretical xi-range,
% so it extends visibly to the left and right of the empirical cloud.
finiteTheoryXi = theory.xiSorted(isfinite(theory.xiSorted) & isfinite(theory.psiSorted));
if ~isempty(finiteTheoryXi)
    xLeft = min([finiteTheoryXi(:); centers(:)]);
    xRight = max([finiteTheoryXi(:); centers(:)]);
else
    span = max(centers) - min(centers);
    if ~isfinite(span) || span <= 0
        span = max(1, abs(mu));
    end
    xLeft = min(centers) - 2*span;
    xRight = max(centers) + 2*span;
end
xLeft = max(0, xLeft);
if xRight <= xLeft
    xRight = xLeft + max(1, abs(mu));
end
xg = linspace(xLeft, xRight, 700)';
psiG = (xg - mu).^2 / (2*v);
analysis.gauss.xi = xg;
analysis.gauss.psi = psiG;
analysis.gauss.meanXi = mu;
analysis.gauss.vPerP = v;
analysis.gauss.logPdfFiniteN = -0.5*log(2*pi*v/P) - P*psiG;
end

function plot_ldlr_outputs(cfg, theory, analysis, basePath)
ensure_dir(fileparts(basePath));

% ------------------------------------------------------------------------
% PLOT 1: theoretical scaled cumulant generating function Phi(s)
% ------------------------------------------------------------------------
fig1 = figure('Visible','on', 'Name', 'LDLR theoretical Phi(s)');
plot(theory.s, theory.Phi, '-', 'LineWidth', 2);
xlabel('s');
ylabel('\Phi(s)');
grid on;
title(sprintf('%s: theoretical scaled cumulant generating function', upper(cfg.model)));
set(fig1, 'Color', 'w');
saveas(fig1, [basePath '_Phi.png']);

% ------------------------------------------------------------------------
% PLOT 2: theoretical rate, empirical rate, Gaussian approximation.
% The log-density histogram is deliberately NOT plotted, to keep the figure clean.
% ------------------------------------------------------------------------
fig2 = figure('Visible','on', 'Name', 'LDLR rate function NO RED HISTOGRAM');
set(fig2, 'Color', 'w');
clf(fig2);
set(fig2, 'Color', 'w');
% Absolutely no log-density / histogram curve is plotted here.
validRate = ~isnan(analysis.rate.psi) & isfinite(analysis.rate.psi);
hold on;
if any(validRate)
    idx = find(validRate);
    maxDots = max(1, cfg.plot.empiricalMaxDots);
    if numel(idx) > maxDots
        idx = idx(round(linspace(1, numel(idx), maxDots)));
    end
    plot(analysis.rate.xi(idx), analysis.rate.psi(idx), 'o', ...
        'MarkerSize', cfg.plot.empiricalMarkerSize, 'LineWidth', 1.3, ...
        'MarkerEdgeColor', [0.20 0.00 0.70], ...
        'MarkerFaceColor', [0.70 0.55 1.00], ...
        'DisplayName', 'empirical rate');
end
plot(theory.xiSorted, theory.psiSorted, '-', 'LineWidth', 2.3, ...
    'Color', [0.00 0.00 0.00], ...
    'DisplayName', 'theory rate');
plot(analysis.gauss.xi, analysis.gauss.psi, '--', 'LineWidth', 2.3, ...
    'Color', [0.47 0.67 0.19], ...
    'DisplayName', 'Gaussian approximation');

% Focus x-axis on the empirical simulation region, with a small margin.
if any(validRate)
    xEmp = analysis.rate.xi(validRate);
    xMin = min(xEmp);
    xMax = max(xEmp);
    xSpan = xMax - xMin;

    if xSpan <= 0 || ~isfinite(xSpan)
        xSpan = max(0.05*abs(mean(xEmp)), 1e-3);
    end

    pad = 0.20*xSpan;   % increase to 0.5 for wider, decrease to 0.2 for tighter
    xlim([max(0, xMin - pad), xMax + pad]);
end



ylabel('\psi(\xi)');

finitePsi = [analysis.rate.psi(validRate); theory.psiSorted(isfinite(theory.psiSorted)); analysis.gauss.psi(isfinite(analysis.gauss.psi))];
if ~isempty(finitePsi)
    yMax = quantile_local(finitePsi, 0.98);
    if isfinite(yMax) && yMax > 0
        ylim([0, 1.15*yMax]);
    end
end

xlabel('\xi = E/P');
title(sprintf('%s: rate function and Gaussian approximation, N=%d, P=%d, \lambda=%.4g', ...
    upper(cfg.model), cfg.N, cfg.P, cfg.lambda));
grid on;
legend('Location','best');
saveas(fig2, [basePath '_rate.png']);

fprintf('\nTwo plots are open and were saved here:\n');
fprintf('  %s\n', [basePath '_Phi.png']);
fprintf('  %s\n\n', [basePath '_rate.png']);
end

function q = quantile_local(x, p)
x = sort(x(isfinite(x)));
if isempty(x)
    q = NaN;
    return;
end
p = min(max(p,0),1);
idx = 1 + (numel(x)-1)*p;
lo = floor(idx); hi = ceil(idx);
if lo == hi
    q = x(lo);
else
    q = x(lo) + (idx-lo)*(x(hi)-x(lo));
end
end

% ========================================================================
% BETA DISTRIBUTIONS AND QUADRATURE
% ========================================================================
function dist = beta_dist(betaCfg, seed)
if nargin < 2
    seed = 1;
end
kind = lower(betaCfg.kind);
dist.kind = kind;
dist.sample = @(n) sample_beta(betaCfg, n);
dist.quad = @(n) quad_beta(betaCfg, n, seed);
dist.secondMoment = estimate_second_moment(betaCfg);
end

function b = sample_beta(c, n)
switch lower(c.kind)
    case 'normal'
        mu = getfield_default(c,'mean',0);
        sig = getfield_default(c,'std',1);
        b = mu + sig*randn(n,1);
    case 'constant'
        b = getfield_default(c,'value',1) * ones(n,1);
    case 'rademacher'
        amp = getfield_default(c,'std',1);
        b = amp * sign(rand(n,1)-0.5);
        b(b==0) = amp;
    case 'sparse_rademacher'
        rho = getfield_default(c,'rho',0.2);
        amp = getfield_default(c,'std',1) / sqrt(max(rho,eps));
        active = rand(n,1) < rho;
        b = zeros(n,1);
        b(active) = amp * sign(rand(sum(active),1)-0.5);
        b(b==0 & active) = amp;
    case 'student'
        df = getfield_default(c,'df',5);
        sig = getfield_default(c,'std',1);
        b = sig * student_t_rand(df,n,1) / sqrt(df/(df-2));
end
end

function [nodes, weights] = quad_beta(c, n, seed)
switch lower(c.kind)
    case 'normal'
        [x,w] = gauss_hermite(n);
        mu = getfield_default(c,'mean',0);
        sig = getfield_default(c,'std',1);
        nodes = mu + sqrt(2)*sig*x;
        weights = w / sqrt(pi);
    case 'constant'
        nodes = getfield_default(c,'value',1);
        weights = 1;
    case 'rademacher'
        amp = getfield_default(c,'std',1);
        nodes = [-amp; amp];
        weights = [0.5; 0.5];
    case 'sparse_rademacher'
        rho = getfield_default(c,'rho',0.2);
        amp = getfield_default(c,'std',1) / sqrt(max(rho,eps));
        nodes = [-amp; 0; amp];
        weights = [rho/2; 1-rho; rho/2];
    case 'student'
        % Deterministic empirical quadrature. This avoids requiring Statistics Toolbox.
        old = rng;
        rng(safe_seed(seed+991),'twister');
        nodes = sample_beta(c, n);
        weights = ones(n,1)/n;
        rng(old);
end
nodes = nodes(:);
weights = weights(:) / sum(weights);
end

function m2 = estimate_second_moment(c)
switch lower(c.kind)
    case 'normal'
        m2 = getfield_default(c,'std',1)^2 + getfield_default(c,'mean',0)^2;
    case 'constant'
        m2 = getfield_default(c,'value',1)^2;
    case {'rademacher','sparse_rademacher','student'}
        m2 = getfield_default(c,'std',1)^2;
    otherwise
        m2 = NaN;
end
end

function [x,w] = gauss_hermite(n)
% Nodes and weights for integral exp(-x^2) f(x) dx.
i = (1:n-1)';
a = sqrt(i/2);
CM = diag(a,1) + diag(a,-1);
[V,D] = eig(CM);
[x,idx] = sort(diag(D));
V = V(:,idx);
w = sqrt(pi) * (V(1,:)').^2;
end

function y = soft_threshold(x, theta)
y = sign(x) .* max(abs(x) - theta, 0);
end

function y = logsumexp(a)
mx = max(a);
y = mx + log(sum(exp(a-mx)));
end

function v = getfield_default(s, f, d)
if isfield(s,f) && ~isempty(s.(f))
    v = s.(f);
else
    v = d;
end
end

function m = mean_nan(x)
x = x(isfinite(x));
if isempty(x)
    m = NaN;
else
    m = mean(x);
end
end

function v = var_nan(x)
x = x(isfinite(x));
if numel(x) <= 1
    v = NaN;
else
    v = var(x,0);
end
end

function x = student_t_rand(df, m, n)
% Student-t random numbers without relying on trnd/statistics toolbox.
% t = Z / sqrt(ChiSquare_df/df).
if nargin < 3
    n = 1;
end
z = randn(m,n);
g = gamma_rand(df/2, 2, m, n); % chi-square(df)
x = z ./ sqrt(g/df);
end

function x = gamma_rand(shape, scale, m, n)
% Marsaglia-Tsang gamma RNG for shape>0. Returns Gamma(shape,scale).
% This avoids using gamrnd from Statistics Toolbox.
if nargin < 4
    n = 1;
end
x = zeros(m,n);
for idx = 1:numel(x)
    a = shape;
    if a < 1
        % Boost to a+1, then transform.
        x(idx) = gamma_rand(a+1, scale, 1, 1) * rand()^(1/a);
    else
        d = a - 1/3;
        c = 1/sqrt(9*d);
        accepted = false;
        while ~accepted
            z = randn();
            v = (1 + c*z)^3;
            if v > 0
                u = rand();
                if log(u) < 0.5*z^2 + d - d*v + d*log(v)
                    x(idx) = scale*d*v;
                    accepted = true;
                end
            end
        end
    end
end
end
