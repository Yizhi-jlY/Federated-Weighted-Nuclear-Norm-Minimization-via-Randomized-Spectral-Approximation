function results = experiment_p_over_effect(data, params)
    % Experiment 2: Effect of oversampling parameter p_over
    % Evaluate the effect of different p_over values on algorithm performance
    
    p_over_values = [5, 10, 15, 20, 25, 30];
    num_p_over = length(p_over_values);
    
    results.p_over_values = p_over_values;
    results.spectral_errors = zeros(num_p_over, params.monte_carlo_runs);
    results.frobenius_errors = zeros(num_p_over, params.monte_carlo_runs);
    results.singular_value_errors = zeros(num_p_over, params.monte_carlo_runs);
    results.reconstruction_errors = zeros(num_p_over, params.monte_carlo_runs);
    
    for i = 1:num_p_over
        fprintf('Testing p_over = %d\n', p_over_values(i));
        
        params_temp = params;
        params_temp.p_over = p_over_values(i);
        
        for run = 1:params.monte_carlo_runs
            [U_fed, S_fed, V_fed, ~] = federated_randomized_svd_parallel(data.A_clients, params_temp);
            
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