function results = experiment_dp_tradeoff(data, params, dp_cfg)
% experiment_dp_tradeoff - Evaluate privacy-utility trade-off under (epsilon, delta)-DP
% Inputs:
%   data     : struct with A_clients and ground truth fields (same as other experiments)
%   params   : base parameter struct for federated_randomized_svd_parallel
%   dp_cfg   : struct with fields
%                .eps_values : array of epsilon values to sweep
%                .delta      : DP delta (default 1e-5)
%                .clip_C     : clipping threshold C (default 1.0)
%                .seed       : base random seed for DP noise (default 555)
%
% Outputs:
%   results  : struct containing per-epsilon errors and summary statistics

    if nargin < 3, dp_cfg = struct(); end
    if ~isfield(dp_cfg, 'eps_values'), dp_cfg.eps_values = [0.2, 0.5, 1, 2, 5]; end
    if ~isfield(dp_cfg, 'delta'),      dp_cfg.delta = 1e-5; end
    if ~isfield(dp_cfg, 'clip_C'),     dp_cfg.clip_C = 1.0; end
    if ~isfield(dp_cfg, 'seed'),       dp_cfg.seed = 555; end

    eps_values = dp_cfg.eps_values;
    num_eps = numel(eps_values);

    results.eps_values = eps_values;
    results.spectral_errors = zeros(num_eps, params.monte_carlo_runs);
    results.frobenius_errors = zeros(num_eps, params.monte_carlo_runs);
    results.singular_value_errors = zeros(num_eps, params.monte_carlo_runs);
    results.reconstruction_errors = zeros(num_eps, params.monte_carlo_runs);
    results.dp_sigma = zeros(num_eps, 1);

    for idx = 1:num_eps
        eps_val = eps_values(idx);
        fprintf('Testing epsilon = %.3f\n', eps_val);

        params_temp = params;
        params_temp.dp_enable = true;
        params_temp.dp_clip_C = dp_cfg.clip_C;
        params_temp.dp_epsilon = eps_val;
        params_temp.dp_delta = dp_cfg.delta;
        params_temp.dp_seed = dp_cfg.seed + idx * 100; % vary seed per epsilon to avoid reuse

        results.dp_sigma(idx) = params_temp.dp_clip_C * sqrt(2 * log(1.25 / params_temp.dp_delta)) / params_temp.dp_epsilon;

        for run = 1:params.monte_carlo_runs
            [U_fed, S_fed, V_fed, ~] = federated_randomized_svd_parallel(data.A_clients, params_temp);
            [spectral_err, frob_err, sv_err, recon_err] = compute_errors(data, U_fed, S_fed, V_fed, params);

            results.spectral_errors(idx, run) = spectral_err;
            results.frobenius_errors(idx, run) = frob_err;
            results.singular_value_errors(idx, run) = sv_err;
            results.reconstruction_errors(idx, run) = recon_err;
        end
    end

    % Aggregate statistics
    results.spectral_mean = mean(results.spectral_errors, 2);
    results.spectral_std = std(results.spectral_errors, 0, 2);
    results.frobenius_mean = mean(results.frobenius_errors, 2);
    results.frobenius_std = std(results.frobenius_errors, 0, 2);
    results.sv_mean = mean(results.singular_value_errors, 2);
    results.sv_std = std(results.singular_value_errors, 0, 2);
    results.recon_mean = mean(results.reconstruction_errors, 2);
    results.recon_std = std(results.reconstruction_errors, 0, 2);
end
