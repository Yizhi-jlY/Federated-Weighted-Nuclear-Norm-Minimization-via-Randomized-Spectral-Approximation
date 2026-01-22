function [spectral_err, frob_err, sv_err, recon_err] = compute_errors(data, U_fed, S_fed, V_fed, params)
    % Compute various error metrics
    % Used to evaluate the accuracy of the SVD decomposition
    
    % Standard SVD as a baseline
    [U_true, S_true, V_true] = svd(data.A_true, 'econ');
    U_true = U_true(:, 1:params.k);
    S_true = S_true(1:params.k, 1:params.k);
    V_true = V_true(:, 1:params.k);
    
    % Reconstruct matrices
    A_fed = U_fed * S_fed * V_fed';
    A_true_k = U_true * S_true * V_true';
    
    % Spectral norm error
    spectral_err = norm(A_fed - A_true_k, 2) / norm(A_true_k, 2);
    
    % Frobenius norm error
    frob_err = norm(A_fed - A_true_k, 'fro') / norm(A_true_k, 'fro');
    
    % Singular value error
    sv_fed = diag(S_fed);
    sv_true = diag(S_true);
    sv_err = norm(sv_fed - sv_true, 2) / norm(sv_true, 2);
    
    % Reconstruction error (relative to the original matrix)
    recon_err = norm(A_fed - data.A_true, 'fro') / norm(data.A_true, 'fro');
end