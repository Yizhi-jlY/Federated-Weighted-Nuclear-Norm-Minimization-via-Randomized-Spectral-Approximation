function [U,S,V]  = fedSvd_UV(Xzeros,r,Tsvd,numWrkrs)
    [n,q] = size(Xzeros);
    M = randn(n,r);
    X_ = reshape(Xzeros,[n,q/numWrkrs,numWrkrs]); 
    X_ = parallel.pool.Constant(X_);
    for t = 1 : Tsvd
        parfor j = 1 : numWrkrs
            tmp =  X_.Value(:,:,j)'*M;
            M_(:,:,j) = X_.Value(:,:,j)*tmp;
        end
        M = sum(M_,3);
        if t == Tsvd
            S = diag(sqrt(vecnorm(M)));
            U = orth(M);
            parfor j = 1 : numWrkrs
                V_(:,:,j) = U'*X_.Value(:,:,j); 
            end
            V = reshape(V_,[r,q]);
        else
            M = orth(M);
        end
    end
    V = diag(1./diag(S))*V;
    V = V';
end