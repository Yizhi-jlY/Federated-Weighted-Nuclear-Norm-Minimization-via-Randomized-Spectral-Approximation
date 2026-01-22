function privacy_info = analyze_privacy_protection(A_clients, Omega_ill, R_Y_clients, B, params)
    % Analyze the privacy protection effect

    % Calculate the true covariance matrix
    A_global = [];
    for i = 1:length(A_clients)
        A_global = [A_global; A_clients{i}];
    end
    true_cov = A_global' * A_global;

    % Simulate the server's attempt to reconstruct the covariance matrix
    S_Y = zeros(size(Omega_ill, 2));
    for i = 1:length(R_Y_clients)
        S_Y = S_Y + R_Y_clients{i}' * R_Y_clients{i};
    end

    % Attempt to reconstruct the covariance (this is an ill-posed problem)
    % Use the least-squares solution as a reconstruction attempt
    try
        reconstructed_cov1 = pinv(Omega_ill') * S_Y * pinv(Omega_ill);
    catch
        reconstructed_cov1 = zeros(size(true_cov));
    end
    
    % Since the server also has the variable B, try to recover the covariance matrix using B
    reconstructed_cov2 = B' * B;

    % Calculate the errors separately
    error1 = norm(true_cov - reconstructed_cov1, 'fro') / norm(true_cov, 'fro');
    error2 = norm(true_cov - reconstructed_cov2, 'fro') / norm(true_cov, 'fro');

    % Select the reconstruction result with the minimum error
    if error1 < error2
        privacy_info.reconstructed_covariance = reconstructed_cov1;
        recon_error = error1;
    else
        privacy_info.reconstructed_covariance = reconstructed_cov2;
        recon_error = error2;
    end

    privacy_info.condition_number = cond(Omega_ill);

    % Calculate the information leakage metric (based on reconstruction error)
    privacy_info.information_leakage = 1 / (1 + recon_error);
end