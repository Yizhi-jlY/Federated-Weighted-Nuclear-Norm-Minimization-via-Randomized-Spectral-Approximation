function results = experiment_rho_effect(data, params)
    % Experiment 1: Effect of the ill-conditioned projector parameter rho
    % Evaluate the effect of different rho values on algorithm performance
    
    rho_values = linspace(0, 1, 10);
    num_rho = length(rho_values);
    
    results.rho_values = rho_values;
    results.spectral_errors = zeros(num_rho, params.monte_carlo_runs);
    results.frobenius_errors = zeros(num_rho, params.monte_carlo_runs);
    results.singular_value_errors = zeros(num_rho, params.monte_carlo_runs);
    results.reconstruction_errors = zeros(num_rho, params.monte_carlo_runs);
    
    for i = 1:num_rho
        fprintf('Testing rho = %.2f\n', rho_values(i));
        
        params_temp = params;
        params_temp.rho = rho_values(i);
        
        for run = 1:params.monte_carlo_runs
            % Run Federated SVD
            [U_fed, S_fed, V_fed, ~] = federated_randomized_svd_parallel(data.A_clients, params_temp);
            
            % Calculate errors
            [spectral_err, frob_err, sv_err, recon_err] = compute_errors(data, U_fed, S_fed, V_fed, params);
            
            results.spectral_errors(i, run) = spectral_err;
            results.frobenius_errors(i, run) = frob_err;
            results.singular_value_errors(i, run) = sv_err;
            results.reconstruction_errors(i, run) = recon_err;
        end
    end
    
    % Calculate statistics
    results.spectral_mean = mean(results.spectral_errors, 2);
    results.spectral_std = std(results.spectral_errors, 0, 2);
    results.frobenius_mean = mean(results.frobenius_errors, 2);
    results.frobenius_std = std(results.frobenius_errors, 0, 2);
    results.sv_mean = mean(results.singular_value_errors, 2);
    results.sv_std = std(results.singular_value_errors, 0, 2);
    results.recon_mean = mean(results.reconstruction_errors, 2);
    results.recon_std = std(results.reconstruction_errors, 0, 2);
end