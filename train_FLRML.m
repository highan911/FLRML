addpath(genpath('./tools'))


opts = struct();

opts.max_rank_svd = 3000;
opts.mxitr = 20;
opts.margin_ratio = 1;


opts.record = 1;
    
d = 100;

t0 = tic;

%{
    COMMENTS:
    
    Use
        [L,output] = FLRML(X, trip, d, opts);
    for original data, and use 
        [L,output] = FLRML({U, Sigma, V}, trip, d, opts);
    for data after SVD preprocessing.
%}


%[L,output] = FLRML({U, Sigma, V}, trip, d, opts);
[L,output] = FLRML(X, trip, d, opts);


t1 = toc(t0);