function results = experiment_rank_adaptation_attack()
% experiment_rank_adaptation_attack - Stress test when true rank is unknown/slowly decaying
% Compares our federated randomized SVD (correctly adapting to rank) against
% AltGDMin configured with an underestimated rank.

    rng(2025);

    % Problem setup
    results.num_clients = 4;
    results.m_total = 400;
    results.n = 200;
    results.k_true = 12;
    results.k_wrong = 6; % Under-estimated rank fed to AltGDMin (Abbasi-style)

    % Slowly decaying spectrum to mimic ambiguous rank
    sv_true = (1 ./ (1:results.k_true)).^0.3;
    S_true = diag(sv_true);
    results.sv_true = sv_true;

    % Generate orthonormal factors
    [U_true, ~] = qr(randn(results.m_total, results.k_true), 0);
    [V_true, ~] = qr(randn(results.n, results.k_true), 0);
    A_true = U_true * S_true * V_true';

    % Partition rows to clients for federated SVD
    rows_per_client = results.m_total / results.num_clients;
    A_clients = cell(results.num_clients, 1);
    for i = 1:results.num_clients
        idx_start = (i - 1) * rows_per_client + 1;
        idx_end = i * rows_per_client;
        A_clients{i} = A_true(idx_start:idx_end, :);
    end

    % --- Our method: federated randomized SVD (no DP here) ---
    params_fed = struct('k', results.k_true, 'p_over', 8, 'rho', 0.1, 'q', 1, ...
                        'enable_client_correction', false, 'upload_corrected_sv', false);
    [U_fed, S_fed, V_fed, ~] = federated_randomized_svd_parallel(A_clients, params_fed);
    A_fed = U_fed * S_fed * V_fed';
    results.error_fed = norm(A_fed - A_true, 'fro') / norm(A_true, 'fro');

    % --- Abbasi baseline: AltGDMin with underestimated rank ---
    mask_full = true(results.m_total, results.n);
    alt_params = struct('m', results.m_total, 'n', results.n, 'p', results.num_clients, ...
                        'r', results.k_wrong, 'maxiter', 20, 'tol', 1e-6, ...
                        'eta_c', 0.8, 'Tsvd', 10, 'p_obs', 1, 'L_true', A_true);
    alt_out = altGDMin_T(A_true, mask_full, alt_params);
    results.error_alt = relative_error(alt_out.A_hat, A_true);

    % Aggregate for easy plotting
    results.errors = [results.error_fed; results.error_alt];
end
