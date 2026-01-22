clc
clear
close all

% Privacy-Preserving Federated SVD  Algorithm Implementation and Experiment
% Main function: Run all experiments

fprintf('=== Federated Randomized SVD Algorithm Experiment ===\n');

% Set random seed for reproducibility
rng(42);

% 1. Parameter settings
    
params.num_clients = 8;           % Number of clients
params.m_total = 1000;            % Total number of rows
params.n = 500;                   % Number of columns
params.k = 20;                    % Target rank
params.p_over = 10;               % Oversampling parameter
params.rho = 0.1;                 % Decay factor for the ill-conditioned projector
params.q = 0;                     % Number of power iterations
params.noise_level = 0;         % Noise level
params.monte_carlo_runs = 10;     % Number of Monte Carlo runs
params.enable_client_correction = false;  % Whether to enable client-side correction
params.upload_corrected_sv = true;       % Whether to upload corrected singular values

% 2. Generate test data
  % Generate test data
    % Generate simulated data based on given parameters
    
    fprintf('Generating test data...\n');
    
    % Generate a low-rank matrix
    U_true = randn(params.m_total, params.k);
    [U_true, ~] = qr(U_true, 0);
    
    % Generate singular values (decreasing)
    sv_true = logspace(0, -2, params.k);
    S_true = diag(sv_true);
    
    V_true = randn(params.n, params.k);
    [V_true, ~] = qr(V_true, 0);
    
    % Construct the ground truth matrix
    A_true = U_true * S_true * V_true';
    
    % Add noise
    A_true = A_true + params.noise_level * randn(params.m_total, params.n);
    
    % Distribute data to clients
    m_per_client = floor(params.m_total / params.num_clients);
    data.A_clients = cell(params.num_clients, 1);
    data.m_clients = zeros(params.num_clients, 1);
    
    start_idx = 1;
    for i = 1:params.num_clients
        if i == params.num_clients
            % The last client gets all remaining rows
            end_idx = params.m_total;
        else
            end_idx = start_idx + m_per_client - 1;
        end
        
        data.A_clients{i} = A_true(start_idx:end_idx, :);
        data.m_clients(i) = size(data.A_clients{i}, 1);
        start_idx = end_idx + 1;
    end
    
    % Store ground truth values for comparison
    data.A_true = A_true;
    data.U_true = U_true;
    data.S_true = S_true;
    data.V_true = V_true;
    data.sv_true = sv_true;
    
    fprintf('Data generation complete: %d clients, total dimensions %dx%d\n', ...
        params.num_clients, params.m_total, params.n);



% 3. Run all experiments

    % Run all experiments
    % Perform a series of experiments to evaluate the performance of the Federated Randomized SVD algorithm

    % Define font size options for all plots
    fontSizeOptions = struct(...
        'title', 16, ...
        'labels', 24, ...
        'ticks', 16, ...
        'legend', 18 ...
    );

    % --- Setup directory for saving figures ---
    save_dir = '.'; % Define the folder name for saved images


    % --- Experiment 1: Effect of Ill-Conditioned Projector Parameter rho ---
    fprintf('\n--- Experiment 1: Effect of Ill-Conditioned Projector Parameter rho ---\n');
    exp1_results = experiment_rho_effect(data, params);

    h1 = figure; % Assign a handle to the figure
    set(gcf, 'Position', [100, 100, 800, 600]); % Uniform figure size for publication
    errorbar(exp1_results.rho_values, exp1_results.spectral_mean, exp1_results.spectral_std, ...
        'Color', [0, 0.447, 0.741], 'LineStyle', '-', 'Marker', 'o', 'LineWidth', 3, 'DisplayName', 'Spectral Norm Error');
    hold on;
    errorbar(exp1_results.rho_values, exp1_results.frobenius_mean, exp1_results.frobenius_std, ...
        'Color', [0.85, 0.325, 0.098], 'LineStyle', '-', 'Marker', 's', 'LineWidth', 3, 'DisplayName', 'Frobenius Norm Error');
    % errorbar(exp1_results.rho_values, exp1_results.sv_mean, exp1_results.sv_std, ...
    %     'Color', [0.466, 0.674, 0.188], 'LineStyle', '-', 'Marker', '^', 'LineWidth', 3, 'DisplayName', 'Singular Value Error');
    % errorbar(exp1_results.rho_values, exp1_results.recon_mean, exp1_results.recon_std, ...
    %     'Color', [0.494, 0.184, 0.556], 'LineStyle', '-', 'Marker', 'd', 'LineWidth', 3, 'DisplayName', 'Reconstruction Error');
    hold off;
    
    % Apply font size settings
    set(gca, 'FontSize', fontSizeOptions.ticks);
    xlabel('$\rho$ Values', 'Interpreter', 'latex', 'FontSize', fontSizeOptions.labels);
    ylabel('Relative Error', 'FontSize', fontSizeOptions.labels);
    % title('Effect of Ill-Conditioned Projector Parameter $\rho$', 'FontSize', fontSizeOptions.title);
    lgd = legend('Location', 'best');
    lgd.FontSize = fontSizeOptions.legend;
    grid on;
    box on;
    
    % --- Save the figure ---
    print(h1, fullfile(save_dir, 'effect_of_rho.eps'), '-depsc'); % Save as an EPS vector graphic
    print(h1, fullfile(save_dir, 'effect_of_rho_high_res.png'), '-dpng', '-r300'); % Save as a 300 DPI PNG


    % --- Singular Value Comparison Plot Setup ---
    % Set random seed for reproducibility (used in random projections for Federated SVD)
    rng(42);
    % Define test parameters
    num_clients = 5; % Number of clients, representing nodes participating in federated learning
    m = 200; % Number of rows in each client's data matrix
    n = 500; % Number of columns in each client's data matrix
    k = 15; % Target decomposition rank, i.e., number of retained singular values
    p_over = 1; % Oversampling parameter to improve accuracy of randomized SVD
    q = 2; % Number of power iterations to enhance subspace capture
    % Define the different rho values to test
    rho_values = [0.001, 0.1, 0.2, 0.5];
    % Initialize parameter structure
    params.k = k;
    params.p_over = p_over;
    params.q = q;
    params.enable_client_correction = 1;
    params.upload_corrected_sv = 0;
    % Generate simple deterministic client data
    A_clients = cell(num_clients, 1);
    A_full = [];
    for i = 1:num_clients
        % Generate a simple low-rank matrix for each client
        S = diag([10, 8, 6, 4, 2, zeros(1, m-5)]);
        U = rand(m);
        V = rand(n, m);
        A_clients{i} = U * S * V';
        A_full = [A_full; A_clients{i}];
    end
    % Run standard SVD for comparison
    [U_true, S_true, V_true] = svd(A_full, 'econ');
    U_true = U_true(:, 1:k);
    S_true = S_true(1:k, 1:k);
    V_true = V_true(:, 1:k);
    singular_values_true = diag(S_true);
    % Create a new figure window for plotting singular value comparison
    h2 = figure; % Assign a handle to the figure
    set(gcf, 'Position', [100, 100, 800, 600]); % Uniform figure size for publication
    % Plot singular values from standard SVD
    semilogy(1:k, singular_values_true, ...
        'Color', 'k', 'LineStyle', '-', 'Marker', 'o', 'LineWidth', 3, 'DisplayName', 'Standard SVD');
    hold on;
    % Define plotting colors and markers (colorblind-friendly palette)
    colors = {[0.85, 0.325, 0.098], [0.929, 0.694, 0.125], [0.494, 0.184, 0.556], [0.466, 0.674, 0.188]};
    markers = {'+', '*', 'x', 's'};
    % Loop over different rho values for testing
    for i = 1:length(rho_values)
        current_rho = rho_values(i);
        params.rho = current_rho;
        % Run Federated Randomized SVD algorithm
        [~, S_k, ~, ~] = federated_randomized_svd(A_clients, params);
        singular_values_fed = diag(S_k);
        % Plot singular values from Federated SVD using different colors and markers
        semilogy(1:k, singular_values_fed, ...
            'Color', colors{i}, ...
            'Marker', markers{i}, ...
            'LineStyle', '-', ...
            'LineWidth', 3, ...
            'DisplayName', ['Federated SVD, $\rho$ = ' num2str(current_rho)]);
    end
    hold off;

    % Apply font size settings
    set(gca, 'FontSize', fontSizeOptions.ticks);
    xlabel('Index', 'FontSize', fontSizeOptions.labels);
    ylabel('Singular Values (Log Scale)', 'FontSize', fontSizeOptions.labels);
    % title('Singular Value Comparison for Different $\rho$ Values', 'FontSize', fontSizeOptions.title);
    lgd = legend('Location', 'best', 'Interpreter', 'latex');
    lgd.FontSize = fontSizeOptions.legend;
    grid on;
    box on;

    % --- Save the figure ---
    print(h2, fullfile(save_dir, 'singular_value_comparison.eps'), '-depsc'); % Save as an EPS vector graphic
    print(h2, fullfile(save_dir, 'singular_value_comparison_high_res.png'), '-dpng', '-r300'); % Save as a 300 DPI PNG


    % --- Experiment 2: Effect of Oversampling Parameter p_over (MODIFIED SECTION) ---
    exp2_results = experiment_p_over_effect(data, params);
    h3 = figure; % Assign a handle to the figure
    set(gcf, 'Position', [100, 100, 800, 600]); % Uniform figure size for publication
    % Use the same style as Experiment 1 for consistency
    errorbar(exp2_results.p_over_values, exp2_results.spectral_mean, exp2_results.spectral_std, ...
        'Color', [0, 0.447, 0.741], 'LineStyle', '-', 'Marker', 'o', 'LineWidth', 3, 'DisplayName', 'Spectral Norm Error');
    hold on;
    errorbar(exp2_results.p_over_values, exp2_results.frobenius_mean, exp2_results.frobenius_std, ...
        'Color', [0.85, 0.325, 0.098], 'LineStyle', '-', 'Marker', 's', 'LineWidth', 3, 'DisplayName', 'Frobenius Norm Error');
    errorbar(exp2_results.p_over_values, exp2_results.sv_mean, exp2_results.sv_std, ...
        'Color', [0.466, 0.674, 0.188], 'LineStyle', '-', 'Marker', '^', 'LineWidth', 3, 'DisplayName', 'Singular Value Error');
    errorbar(exp2_results.p_over_values, exp2_results.recon_mean, exp2_results.recon_std, ...
        'Color', [0.494, 0.184, 0.556], 'LineStyle', '-', 'Marker', 'd', 'LineWidth', 3, 'DisplayName', 'Reconstruction Error');
    hold off;
    
    % Apply font size settings
    set(gca, 'FontSize', fontSizeOptions.ticks);
    % Use LaTeX interpreter for mathematical symbols
    xlabel('Oversampling Parameter $p_{\mathrm{over}}$', 'Interpreter', 'latex', 'FontSize', fontSizeOptions.labels);
    ylabel('Relative Error', 'FontSize', fontSizeOptions.labels);

    set(gca, 'YScale', 'log'); % Set the y-axis to a logarithmic scale
    % title('Effect of Oversampling Parameter $p_{\mathrm{over}}$', 'FontSize', fontSizeOptions.title); % Commented out for academic papers
lgd = legend('Location', 'northeast');
lgd.FontSize = fontSizeOptions.legend;
    grid on;
    box on;

    % --- Save the figure ---
    print(h3, fullfile(save_dir, 'effect_of_oversampling.eps'), '-depsc'); % Save as an EPS vector graphic
    print(h3, fullfile(save_dir, 'effect_of_oversampling_high_res.png'), '-dpng', '-r300'); % Save as a 300 DPI PNG


    % --- Experiment 3: Privacy-Utility Trade-off (epsilon sweep) ---
    fprintf('\n--- Experiment 3: Privacy-Utility Trade-off (epsilon sweep) ---\n');
    dp_cfg = struct('eps_values', [0.2, 0.5, 1, 2, 5], 'delta', 1e-5, 'clip_C', 1.0, 'seed', 777);
    exp3_results = experiment_dp_tradeoff(data, params, dp_cfg);

    h4 = figure;
    set(gcf, 'Position', [120, 120, 800, 600]);
    errorbar(exp3_results.eps_values, exp3_results.frobenius_mean, exp3_results.frobenius_std, ...
        'Color', [0.00, 0.45, 0.74], 'LineStyle', '-', 'Marker', 'o', 'LineWidth', 3, 'DisplayName', 'Frobenius Error');
    hold on;
    errorbar(exp3_results.eps_values, exp3_results.recon_mean, exp3_results.recon_std, ...
        'Color', [0.85, 0.33, 0.10], 'LineStyle', '-', 'Marker', 's', 'LineWidth', 3, 'DisplayName', 'Reconstruction Error');
    yline(exp3_results.recon_mean(end), '--', 'Color', [0.30, 0.30, 0.30], 'LineWidth', 1.5, 'DisplayName', 'Noiseless baseline (approx)');
    hold off;

    set(gca, 'FontSize', fontSizeOptions.ticks);
    xlabel('$\epsilon$ (privacy budget, larger = weaker privacy)', 'Interpreter', 'latex', 'FontSize', fontSizeOptions.labels);
    ylabel('Relative Error', 'FontSize', fontSizeOptions.labels);
    lgd = legend('Location', 'northeast');
    lgd.FontSize = fontSizeOptions.legend;
    grid on; box on;

    print(h4, fullfile(save_dir, 'privacy_utility_tradeoff.eps'), '-depsc');
    print(h4, fullfile(save_dir, 'privacy_utility_tradeoff_high_res.png'), '-dpng', '-r300');


    % --- Experiment 4: Rank Adaptation Attack (AltGDMin mis-specified rank) ---
    fprintf('\n--- Experiment 4: Rank Adaptation Attack (AltGDMin underestimates rank) ---\n');
    exp4_results = experiment_rank_adaptation_attack();

    h5 = figure;
    set(gcf, 'Position', [140, 140, 680, 480]);
    bar(categorical({'FedWNNM (ours)', 'AltGDMin (r mis-set)'}), exp4_results.errors);
    ylabel('Relative Reconstruction Error', 'FontSize', fontSizeOptions.labels);
    set(gca, 'FontSize', fontSizeOptions.ticks);
    title(sprintf('True rank = %d, AltGDMin rank = %d', exp4_results.k_true, exp4_results.k_wrong), 'FontSize', fontSizeOptions.title);
    grid on; box on;

    print(h5, fullfile(save_dir, 'rank_adaptation_attack.eps'), '-depsc');
    print(h5, fullfile(save_dir, 'rank_adaptation_attack_high_res.png'), '-dpng', '-r300');


    % --- Experiment 5: Visual reconstruction attack illustration ---
    fprintf('\n--- Experiment 5: Visual reconstruction attack (original vs. no-DP vs. DP) ---\n');
    reconstruction_attack_demo();

