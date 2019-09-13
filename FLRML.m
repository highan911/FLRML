function [L,output] = FLRML(X, trip, d, opts)
    addpath(genpath('./tools'))
    
    MAX_RANK_SVD = 3000;
    if isfield(opts, 'max_rank_svd') && opts.max_rank_svd > 0
        MAX_RANK_SVD = opts.max_rank_svd;
    end

    

    if(iscell(X))
        U = X{1};
        Sigma = X{2};
        V = X{3};
        r = length(Sigma);
        n = size(V,1);
        D = size(U,1);
        time_svd = 0;
    else    
    %% SVD

        if(opts.record>=0)
            fprintf('SVD\n');
        end

        t0 = tic;

        n = size(X,2);
        D = size(X,1);

        colLength = 1 ./ sqrt(sum(X.^2,1));
        X = X * diag(sparse(colLength));

        [U,Sigma,V] = betapca(X, min([D, n, MAX_RANK_SVD]));
        Sigma = diag(Sigma);
        r=numel(find(Sigma>1e-5));
        U = U(:,1:r);
        V = V(:,1:r);
        Sigma = Sigma(1:r);

        time_svd = toc(t0);
    end
    

    n_trip = size(trip,1); 
    
    v = ones(n_trip,1);

    T_array = sparse(trip(:,2),trip(:,1),v,n,n);
    T_array = sum(T_array,1)';
    T_array = 1./(T_array+1);
    
    
    v = [ v ; - v ];
    i = [ trip(:,2);trip(:,3)];
    j = [ trip(:,1);trip(:,1)];

    C = sparse(i,j,v,n,n);
    
    [L,output] = FLRML_core_stiefel(U, Sigma, V, C, T_array, d, opts);
    
    output.time_svd=time_svd;
end





function [L,output] = FLRML_core_stiefel(U, Sigma, V, C, T_array, d, opts)

addpath(genpath('./tools/beta_pca'))
addpath(genpath('./tools/FOptM-share'))

QUIET = false;
if isfield(opts, 'record') && opts.record==0
    QUIET = true;
end

if ~isfield(opts, 'margin_ratio') || opts.margin_ratio <= 0
    opts.margin_ratio = 1;
end


if(~QUIET)
    fprintf('constant initializing\n');
end

t0 = tic;

r = size(U,2);
VTCT = V' * C * diag(sparse(T_array));


assert(r >= d, 'r >= d')


P = [orth(rand(d,d)); zeros(r-d, d)];
s = rand(d ,1).^2;


time_constinit = toc(t0);


%% Main Iteration

[P, s, output]= OptStiefel_WithUpdateK(P, s, V, VTCT, opts);

    
%% optput

output.time_constinit=time_constinit;

B = diag(sqrt(s)) * P';

L = B * diag(1./Sigma) * U';

end



%% Updating K

function [K, Const, MARGIN, nls] = init_K(P, s, V, VTCT, opts)
    %{
        COMMENTS:

        S_half := sqrt(S)
        
        Y := S_half P' V' = S_half Y0
        Z := - Y' Y C T = - Y0' S Y0 C T = - Y0' S P' VTCT
        diag(Z) := - Z0 * s;
    %}

    UPDATE_ITERS = 10;

    Y0 = (V * P)';
    Z0 = ( Y0 .* (P' * VTCT) )';
    
    n = size(Y0,2);
    
    if(opts.record >= 1)
        fprintf('variable initializing');
    end
    
    for nls=1:UPDATE_ITERS
        
        if(opts.record >= 1)
            fprintf('.');
        end
        
        n_sample = min(n, 100);

        some_Y_ids = randperm(n,n_sample);
        some_Y = Y0(:,some_Y_ids);
        norm_some_Y  = s' * some_Y.^2 ;
        norm_average = sum(norm_some_Y) / n_sample;

        MARGIN = opts.margin_ratio * norm_average;
        
        z = - Z0 * s;
        z = z + MARGIN;

        lambda = (z>0);

        K = -VTCT*diag(sparse(lambda))*V;

        k = -diag(P' * K * P);

        s_new = smooth(k);
        
        Const = sum(lambda) * MARGIN;
        
        test1 = lambda' * z;
        test2 = P' .* (P'*K);
        test2 = sum(test2'*s_new) + Const;
        
        if(abs(test1-test2) / max(abs(test1), abs(test2)) < 1e-2)
            break
        end
        
        s = s_new;
        
        
    end

end

function [K, Const, F] = update_K(P, s, V, VTCT, MARGIN)
    Y0 = (V * P)';
    Z0 = ( Y0 .* (P' * VTCT) )';
    
    z0 = - Z0 * s;
    z = z0 + MARGIN;
    
    lambda = (z>0);
    
    K = -VTCT*diag(sparse(lambda))*V;
    
    Const = sum(lambda) * MARGIN;
    
    F = lambda' * z;
end



%% smoothed_fun_OCP
function sig = sigmoid(x)
    sig = 1 ./ (1 + exp (-x));
end

function smo = smooth(x)
    smo = - log(sigmoid (-x) );
    
    mask = (smo == Inf);
    smo(mask) = x(mask);
end

function [F, G, s] = fun_OCP(P, K, Const)

    k = -diag(P' * K * P);
    
    sig = sigmoid(k);
    s = smooth(k);
        
    F = -1/2 * s' * k + Const;
    
    q = -1/2 * ( s + k .* sig );
    
    G = -(K + K') * P * diag(q);

end






%% Edited from OptStiefelGBB.m
function [X, s, out]= OptStiefel_WithUpdateK(X, s, V_const, VTCT, opts)
%-------------------------------------------------------------------------
% curvilinear search algorithm for optimization on Stiefel manifold
%
%   min F(X), S.t., X'*X = I_k, where X \in R^{n,k}
%
%   H = [G, X]*[X -G]'
%   U = 0.5*tau*[G, X];    V = [X -G]
%   X(tau) = X - 2*U * inv( I + V'*U ) * V'*X
%
%   -------------------------------------
%   U = -[G,X];  V = [X -G];  VU = V'*U;
%   X(tau) = X - tau*U * inv( I + 0.5*tau*VU ) * V'*X
%
%
% Input:
%           X --- n by k matrix such that X'*X = I
%
%        opts --- option structure with fields:
%                 record = 0, no print out
%                 mxitr       max number of iterations
%                 xtol        stop control for ||X_k - X_{k-1}||
%                 gtol        stop control for the projected gradient
%                 ftol        stop control for |F_k - F_{k-1}|/(1+|F_{k-1}|)
%                             usually, max{xtol, gtol} > ftol
%   
% Output:
%           X --- solution
%         Out --- output information
%
% -------------------------------------
% For example, consider the eigenvalue problem F(X) = -0.5*Tr(X'*A*X);
%
% function demo
% 
% function [F, G] = fun(X,  A)
%   G = -(A*X);
%   F = 0.5*sum(dot(G,X,1));
% end
% 
% n = 1000; k = 6;
% A = randn(n); A = A'*A;
% opts.record = 0; %
% opts.mxitr  = 1000;
% opts.xtol = 1e-5;
% opts.gtol = 1e-5;
% opts.ftol = 1e-8;
% 
% X0 = randn(n,k);    X0 = orth(X0);
% tic; [X, out]= OptStiefelGBB(X0, @fun, opts, A); tsolve = toc;
% out.fval = -2*out.fval; % convert the function value to the sum of eigenvalues
% fprintf('\nOptM: obj: %7.6e, itr: %d, nfe: %d, cpu: %f, norm(XT*X-I): %3.2e \n', ...
%             out.fval, out.itr, out.nfe, tsolve, norm(X'*X - eye(k), 'fro') );
% 
% end
% -------------------------------------
%
% Reference: 
%  Z. Wen and W. Yin
%  A feasible method for optimization with orthogonality constraints
%
%-------------------------------------------------------------------------

%% Size information

if isempty(X)
    error('input X is an empty matrix');
else
    [n, k] = size(X);
end

if isfield(opts, 'xtol')
    if opts.xtol < 0 || opts.xtol > 1
        opts.xtol = 1e-6;
    end
else
    opts.xtol = 1e-6;
end

if isfield(opts, 'gtol')
    if opts.gtol < 0 || opts.gtol > 1
        opts.gtol = 1e-6;
    end
else
    opts.gtol = 1e-6;
end

if isfield(opts, 'ftol')
    if opts.ftol < 0 || opts.ftol > 1
        opts.ftol = 1e-3;
    end
else
    opts.ftol = 1e-3;
end

% parameters for control the linear approximation in line search
if isfield(opts, 'rho')
   if opts.rho < 0 || opts.rho > 1
        opts.rho = 1e-4;
   end
else
    opts.rho = 1e-4;
end

% factor for decreasing the step size in the backtracking line search
if isfield(opts, 'eta')
   if opts.eta < 0 || opts.eta > 1
        opts.eta = 0.1;
   end
else
    opts.eta = 0.2;
end

% parameters for updating C by HongChao, Zhang
if isfield(opts, 'gamma')
   if opts.gamma < 0 || opts.gamma > 1
        opts.gamma = 0.85;
   end
else
    opts.gamma = 0.85;
end

if isfield(opts, 'tau')
   if opts.tau < 0 || opts.tau > 1e3
        opts.tau = 1e-3;
   end
else
    opts.tau = 1e-3;
end

if isfield(opts, 'taumax')
   if opts.taumax < 0 || opts.taumax > 1e40
        opts.taumax = 1e20;
   end
else
    opts.taumax = 1e20;
end

if isfield(opts, 'taumin')
   if opts.taumin < 0 || opts.taumin > 1e20
        opts.taumin = 1e-20;
   end
else
    opts.taumin = 1e-20;
end

if isfield(opts, 'tautol')
    if opts.tautol < 0 || opts.tautol > 1
        opts.tautol = opts.taumin;
    end
else
    opts.tautol = 1e-10;
end

% parameters for the  nonmontone line search by Raydan
if ~isfield(opts, 'STPEPS')
    opts.STPEPS = 1e-10;
end

if isfield(opts, 'nt')
    if opts.nt < 0 || opts.nt > 100
        opts.nt = 5;
    end
else
    opts.nt = 5;
end

if isfield(opts, 'projG')
    switch opts.projG
        case {1,2}; otherwise; opts.projG = 1;
    end
else
    opts.projG = 1;
end

if isfield(opts, 'iscomplex')
    switch opts.iscomplex
        case {0, 1}; otherwise; opts.iscomplex = 0;
    end
else
    opts.iscomplex = 0;
end

if isfield(opts, 'mxitr')
    if opts.mxitr < 0 || opts.mxitr > 2^20
        opts.mxitr = 30;
    end
else
    opts.mxitr = 30;
end

if ~isfield(opts, 'record')
    opts.record = 0;
end


%-------------------------------------------------------------------------------
% copy parameters
xtol = opts.xtol;
gtol = opts.gtol;
ftol = opts.ftol;
tautol = opts.tautol;
rho  = opts.rho;
STPEPS = opts.STPEPS;
eta   = opts.eta;
gamma = opts.gamma;
iscomplex = opts.iscomplex;
record = opts.record;

nt = opts.nt;   crit = ones(nt, 3);

invH = true; if k < n/2; invH = false;  eye2k = eye(2*k); end

%% Initial function value and gradient
% prepare for iterations

t0 = tic;

[K, Const, MARGIN, itr_init] = init_K(X, s, V_const, VTCT, opts);


memory_rec = memory;
max_memory = memory_rec.MemUsedMATLAB;

[F, G, s] = fun_OCP(X, K, Const);  

out.nfe = 1;  
GX = G'*X;

if invH
    GXT = G*X';  H = 0.5*(GXT - GXT');  RX = H*X;
else
    if opts.projG == 1
        U =  [G, X];    V = [X, -G];       VU = V'*U;
    elseif opts.projG == 2
        GB = G - 0.5*X*(X'*G);
        U =  [GB, X];    V = [X, -GB];       VU = V'*U;
    end
    %U =  [G, X];    VU = [GX', X'*X; -(G'*G), -GX];   
    %VX = VU(:,k+1:end); %VX = V'*X;
    VX = V'*X;
end
dtX = G - X*GX;     nrmG  = norm(dtX, 'fro');
    
Q = 1; Cval = F;  tau = opts.tau;

fvals=[];
fvals = [fvals;F];

%% Print iteration header if debug == 1
if (opts.record >= 1)
    fid = 1;
    fprintf(fid, '\n ----------- Gradient Method with Line search ----------- \n');
    fprintf(fid, '%4s %8s %8s %10s %10s %10s %4s \n', 'Iter', 'tau', 'F(X)', 'nrmG', 'XDiff', 'FDiff', 'nls');
    fprintf(fid, '%4d  %3.2e  %4.3e  %3.2e  %3.2e  %3.2e  %2d\n', 0, 0, F, 0, 0, 0, 0);
end

time_varinit = toc(t0);
t0 = tic;


%% main iteration
for itr = 1 : opts.mxitr
    XP = X;     FP = F;   GP = G;   dtXP = dtX;
     % scale step size

    nls = 1; deriv = rho*nrmG^2; %deriv
    while 1
        % calculate G, F,
        if invH
            [X, ~] = linsolve(eye(n) + tau*H, XP - tau*RX);
        else
            [aa, ~] = linsolve(eye2k + (0.5*tau)*VU, VX);
            X = XP - U*(tau*aa);
        end
        %if norm(X'*X - eye(k),'fro') > 1e-6; error('X^T*X~=I'); end
        if ~isreal(X) && ~iscomplex ; error('X is complex'); end
        
        [F,  G, s] = fun_OCP(X, K, Const);
        out.nfe = out.nfe + 1;
        
        if F <= Cval - tau*deriv || nls >= 25 || tau<=opts.tautol
            break;
        end
        tau = eta*tau;          nls = nls+1;
    end  
    
    GX = G'*X;
    if invH
        GXT = G*X';  H = 0.5*(GXT - GXT');  RX = H*X;
    else
        if opts.projG == 1
            U =  [G, X];    V = [X, -G];       VU = V'*U;
        elseif opts.projG == 2
            GB = G - 0.5*X*(X'*G);
            U =  [GB, X];    V = [X, -GB];     VU = V'*U; 
        end
        %U =  [G, X];    VU = [GX', X'*X; -(G'*G), -GX];
        %VX = VU(:,k+1:end); % VX = V'*X;
        VX = V'*X;
    end
    dtX = G - X*GX;    nrmG  = norm(dtX, 'fro');
    
    S = X - XP;         XDiff = norm(S,'fro')/sqrt(n);
    tau = opts.tau; 
    
    if iscomplex
        %Y = dtX - dtXP;     SY = (sum(sum(real(conj(S).*Y))));
        Y = dtX - dtXP;     SY = abs(sum(sum(conj(S).*Y)));
        if mod(itr,2)==0; tau = sum(sum(conj(S).*S))/SY; 
        else tau = SY/sum(sum(conj(Y).*Y)); end    
    else
        %Y = G - GP;     SY = abs(sum(sum(S.*Y)));
        Y = dtX - dtXP;     SY = abs(sum(sum(S.*Y)));
        if mod(itr,2)==0; tau = sum(sum(S.*S))/SY;
        else tau  = SY/sum(sum(Y.*Y)); end
        

    end
    tau = max(min(tau, opts.taumax), opts.taumin);
    
    
    [K, Const, F] = update_K(X, s, V_const, VTCT, MARGIN);
    fvals = [fvals;F];
    
    memory_rec = memory;
    max_memory = max(max_memory, memory_rec.MemUsedMATLAB);
    
    FDiff = abs(FP-F)/(abs(FP)+1);
    
    if (record >= 1)
        fprintf('%4d  %3.2e  %4.3e  %3.2e  %3.2e  %3.2e  %2d\n', ...
            itr, tau, F, nrmG, XDiff, FDiff, nls);
    end
    
    if (XDiff < xtol && tau <= tautol ) || FDiff < ftol
        if itr <= 2
            ftol = 0.1*ftol;
            xtol = 0.1*xtol;
            gtol = 0.1*gtol;
        else
            out.msg = 'converge';
            break;
        end
    end
    
    Qp = Q; Q = gamma*Qp + 1; Cval = (gamma*Qp*Cval + F)/Q;
    
    
end


time_iter = toc(t0);

if itr >= opts.mxitr
    out.msg = 'exceed max iteration';
end

out.feasi = norm(X'*X-eye(k),'fro');
if  out.feasi > 1e-13
    out.nfe = out.nfe + 1;
    out.feasi = norm(X'*X-eye(k),'fro');
end

out.nrmG = nrmG;
out.fval = F;
out.fvals = fvals;
out.itr = itr;
out.itr_init=itr_init;
out.time_varinit = time_varinit;
out.time_iter = time_iter;
out.max_memory = max_memory;

end





