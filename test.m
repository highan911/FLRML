addpath(genpath('./tools'))


BlockSize = 250;
KNN_NUM = 5;

N_Cat = 40;

Y = full(L*X);
Y_test = full(L*X_test);
colLength = sqrt(sum(Y.^2,1));
Y1 = Y * diag(sparse(1 ./ colLength));
colLength = sqrt(sum(Y_test.^2,1));
Y1_test = Y_test * diag(sparse(1 ./ colLength));


fprintf('KNN\n');
[Precision_KNN, labels_predict] = KNN(Y1, Y1_test, labels_train, labels_test, KNN_NUM, N_Cat, BlockSize);
