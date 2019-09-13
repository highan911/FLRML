dataname = 'TDT2';

addpath(genpath('./tools'))
load([dataname,'.mat']);

MAX_RANK_SVD = 3000;
EPS = 1e-5;


colLength = 1 ./ sqrt(sum(X.^2,1));
X = X * diag(sparse(colLength));

colLength = 1 ./ sqrt(sum(X_test.^2,1));
X_test = X_test * diag(sparse(colLength));

D = size(X, 1);
n = size(X, 2);



fprintf('svd\n');

t0 = tic;
[U,Sigma,V] = betapca(X, min([D, n, MAX_RANK_SVD]));

Sigma = diag(Sigma);

r=numel(find(Sigma>EPS));
U = U(:,1:r);
V = V(:,1:r);
Sigma = Sigma(1:r);

preTime = toc(t0);

fprintf('saving\n');

save([dataname,'_svd.mat'], 'Sigma', 'U', 'V', 'r', 'preTime', '-v7.3')