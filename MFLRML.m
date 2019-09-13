function [L,output] = MFLRML(X, trip, d, opts)
    addpath(genpath('./tools'))
    
    n = size(X,2);
    D = size(X,1);
    
    BATCH_TRIPS = 100;
    if isfield(opts, 'batch_trips')
        if(opts.batch_trips>0)
            BATCH_TRIPS = opts.batch_trips;
        end
    end
    BATCH_TRIPS = min(BATCH_TRIPS, n);
    BATCH_TRIPS = max(BATCH_TRIPS, ceil(d/3) + 10);
    
    BATCH_ITERS = 30;
    if isfield(opts, 'batch_iters')
        if(opts.batch_iters>0)
            BATCH_ITERS = opts.batch_iters;
        end
    end



    opts.record = 0;
    

    n_trip = size(trip,1); 
    
    v = ones(n_trip,1);

    v = [ v ; - v ];
    i = [ trip(:,2);trip(:,3)];
    j = [ trip(:,1);trip(:,1)];

    C = sparse(i,j,v,n,n);


    fprintf('start');

    L=0;
    output = {};

    for J = 1:BATCH_ITERS
        fprintf('.');

        if(mod(J,100)==0)
            fprintf('\n');
        end
        
        trip_ids = randperm(n_trip);
        trip_ids = trip_ids(1:BATCH_TRIPS);
        trips_J =  trip(trip_ids, :);
        XJ_ids = [trips_J(:,1);trips_J(:,2);trips_J(:,3)];
        XJ_ids = unique(XJ_ids);

        nJ = numel(XJ_ids);
        
        if(nJ < d+10)
            new_XJ_ids = randperm(n)';
            new_XJ_ids = new_XJ_ids(1:d+10-nJ);
            XJ_ids = [XJ_ids;new_XJ_ids];
            XJ_ids = unique(XJ_ids);
            nJ = numel(XJ_ids);
        end
        
        XJ = X(:, XJ_ids);
        CJ = C(XJ_ids,XJ_ids);
        

        T_array_J = abs(CJ);
        T_array_J = sum(T_array_J,1)';
        numel_CJ=sum(T_array_J);
        T_array_J = T_array_J/2;
        T_array_J = 1./(T_array_J+1);
        
        if(J==1)
            %initialize
            L = rand(d,D);

            YJ = L * XJ;
            norm_average = mean(sum(YJ.^2,1));
            MARGIN = opts.margin_ratio * norm_average;
        end
            
        
        YJ = L * XJ;
        CJTJ =  CJ * diag(sparse(T_array_J));

        zJ = sum(YJ.* (YJ*CJTJ),1)' + MARGIN;

        LambdaJ = (zJ>0); 

        fval = LambdaJ' * zJ;
        outputJ.fval = fval;

        %SVD
        [U,Sigma,V] = betapca(XJ, min([D, nJ]));
        Sigma = diag(Sigma);
        r=numel(find(Sigma>1e-5));
        U = U(:,1:r);
        V = V(:,1:r);
        Sigma = Sigma(1:r);
        %ENDSVD

        B = L * U * diag(Sigma);

        [P,S,Q] = betapca(B', min([d,r]));
        S = S.^2;
        
        K = - V' * CJTJ * diag(sparse(LambdaJ)) * V;

        GP =  (K + K') * P * S;

        GP_TProj =  GP - P * GP' * P;


        P_new = P - GP_TProj;
        [UU,~,VV] = betapca(P_new, min([d,r]));
        P_new = UU * VV';

        k = -diag(P' * K * P);
        s_new = smooth(k);

        B_new = Q * diag(real(sqrt(s_new))) * P_new';
    
        L_new = B_new * diag(1./Sigma) * U';
        rate = 1 / sqrt(J);
        L = (1-rate) * L + rate * L_new;

        
        outputJ.XJ_ids=XJ_ids;
        outputJ.numel_CJ=numel_CJ;
        output{J} = outputJ;

    end
    
    
    function sig = sigmoid(x)
        sig = 1 ./ (1 + exp (-x));
    end

    function smo = smooth(x)
        smo = - log(sigmoid (-x) );
        mask = (smo == Inf);
        smo(mask) = x(mask);
    end
    
end

