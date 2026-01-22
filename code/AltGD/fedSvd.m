function [U,S]  = fedSvd(Xzeros,r,Tsvd,numWrkrs)
    [n,q] = size(Xzeros);
    M = randn(n,r);
    X_ = reshape(Xzeros,[n,q/numWrkrs,numWrkrs]); 
    X_ = parallel.pool.Constant(X_);
    for t = 1 : Tsvd
        parfor j = 1 : numWrkrs
            %Xj = (X_(:,:,j));
            %tmp = (X_(:,:,j))'*M; %M_(:,:,j) should be an n x r matrix
            %M_(:,:,j) = (X_(:,:,j))*tmp;
            tmp =  X_.Value(:,:,j)'*M;
            M_(:,:,j) = X_.Value(:,:,j)*tmp;
        end
        M = sum(M_,3);
        if t == Tsvd
            S = diag(sqrt(vecnorm(M)));
            U = orth(M);
        else
            M = orth(M);
        end
    end
end