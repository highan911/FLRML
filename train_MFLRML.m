addpath(genpath('./tools'))

d = 100;




opts = struct();

opts.record = 0;

opts.batch_trips = 80;
opts.batch_iters = 20;


opts.margin_ratio = 1;


t0 = tic;

[L,output] = MFLRML(X, trip, d, opts);

t1 = toc(t0);

