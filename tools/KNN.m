function [Precision_KNN, labels_predict] = KNN(Vecs_train, Vecs_test, labels_train, labels_test, KNN_NUM, N_Cat, BlockSize)

    
	N_train = size(Vecs_train, 2);
    N_test = size(Vecs_test, 2);
    
    N_block = ceil(N_test/BlockSize);
    
    labels_predict = [];
    
    for j = 1 : N_block
        
        fprintf('.');
        
        istart = (j-1)*BlockSize + 1;
        iend = min(j*BlockSize, N_test);
        
        N_test_block = iend - istart + 1;
        
        Vecs_test_block = Vecs_test(:, istart:iend);
        
        
        Dist = DistMutual(Vecs_train, Vecs_test_block);
        [topKVals,Id]=sort(Dist,'ascend');


        Id = Id(1:KNN_NUM, :);
        topKVals = topKVals(1:KNN_NUM, :);


        topKCats = zeros(KNN_NUM, N_test_block);

        for i=1:N_test_block
            topKCats(:,i) = labels_train(Id(:, i)); 
        end

        MaxCat = sparse(N_Cat, N_test_block);
        for i=1:KNN_NUM
            MaxCat = MaxCat + sparse(topKCats(i,:), 1:N_test_block, topKVals(i,:), N_Cat, N_test_block);
        end
        
        [~, labels_predict_block] = max(MaxCat);
        labels_predict = [labels_predict;labels_predict_block'];
        
    end
    Precision_KNN = Precision(labels_test, labels_predict);
end

function Dist = DistMutual(Vecs_train, Vecs_test)
    N_train = size(Vecs_train, 2);
    N_test = size(Vecs_test, 2);

    xTx = sum(Vecs_train.^2, 1);
    yTy = sum(Vecs_test.^2, 1);
    xTy = Vecs_train'*Vecs_test;
    Dist = xTx' * ones(1,N_test) + ones(N_train, 1) * yTy - 2 * xTy;
end

function P = Precision(labels, labels_predict)
    if(size(labels,1)~= size(labels_predict,1))
        labels = labels';
    end
    Delta = labels - labels_predict;
    P = length(find(Delta == 0)) / length(labels);
end