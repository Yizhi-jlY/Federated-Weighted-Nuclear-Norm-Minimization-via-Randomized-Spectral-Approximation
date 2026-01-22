%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                   Main Function: Robustness and Parameter Sensitivity Analysis
%                                  (Monte Carlo Simulation Version)
%
%   Description:
%       This script performs robustness and parameter sensitivity analyses.
%       It includes several sub-experiments that can be enabled or disabled.
%       To ensure statistical reliability, a Monte Carlo (MC) simulation
%       mechanism is used, repeating each experiment multiple times on
%       different random datasets and averaging the results.
%
%       The included analyses are:
%       - Sensitivity analysis of the hyperparameter C for FedWNNM.
%       - Performance impact analysis of the oversampling parameter p_over for Federated SVD.
%       - Performance impact analysis of the parameter rho for Federated SVD.
%
%   Supported Modes:
%       1. 'run_and_analyze': Run new experiments and analyze the results.
%       2. 'analyze_only':    Load and analyze results from a specified folder
%                             without running new experiments.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 1. Environment Initialization
clc;
clear;
close all;
fprintf('============================================================\n');
fprintf('         Experiment 3: Robustness and Sensitivity Analysis (MC)       \n');
fprintf('============================================================\n\n');

% Set random seed for reproducibility
rng(2025, 'twister');

%% 2. Path Management
codeFolderPath0 = fileparts(fileparts(mfilename('fullpath')));
codeFolderPath = fullfile(codeFolderPath0, 'code');
fprintf('Code root directory located: %s\n', codeFolderPath);
utils_path = fullfile(codeFolderPath, 'utils');
addpath(genpath(utils_path));
fprintf('[Path Management] Added utility functions path: %s\n\n', utils_path);

%% 3. Experiment Parameter Configuration
% -------------------- Run Mode Control --------------------
ANALYSIS_MODE = 'analyze_only';
% For 'analyze_only' mode, specify the target results directory.
% Example: TARGET_RESULTS_DIR = 'results/Experiment3/20250902_170212';
TARGET_RESULTS_DIR = 'results/Experiment3/20250902_170212';

% -------------------- Sub-experiment Run Flags --------------------
% --- Federated Algorithms ---
run_flags.FedWNNM_MC     = 1;
run_flags.altGDMin_T     = 1; % Federated AltGD
run_flags.altGDMinCntrl_T = 1;
run_flags.altMinCntrl_T   = 1;

% --- Centralized Baseline Algorithms ---
run_flags.WNNM_MC        = 1;
run_flags.AltGD           = 1;
run_flags.altMinPrvt_T   = 1;

% -------------------- Data Generation Parameters --------------------
params.m = 100;        % Number of matrix rows
params.n = 100;        % Number of matrix columns
params.r = 4;          % True rank

% -------------------- General Algorithm Parameters --------------------
params.maxiter = 350;  % Maximum number of iterations
params.tol = 1e-5;     % Convergence tolerance
params.mc = 4;         % Number of Monte Carlo simulations

% -------------------- Federated and Randomized SVD Parameters --------------------
params.p = 4;          % Number of federated clients
params.r_est = params.r + 10; % Estimated rank for Randomized SVD
params.p_over = 10;    % Oversampling parameter (default)
params.rho = 0.5;      % Split ratio for federated SVD (default)

% -------------------- Parameter Ranges for Sensitivity Analysis --------------------
% Experiment: FedWNNM weight C range
exp_ranges.C = [0.1, 0.5, 1, 2, 5, 10];
% Experiment: Oversampling parameter p_over range
exp_ranges.p_over = [5, 10, 15, 20, 25];
% Experiment: Split ratio rho range
exp_ranges.rho = [0.1, 0.3, 0.5, 0.7, 0.9, 1];

fprintf('Experiment parameters configured.\n\n');

%% 4. Experiment Execution (only in 'run_and_analyze' mode)
if strcmpi(ANALYSIS_MODE, 'run_and_analyze')

    % --- Create main results directory ---
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    results_dir = fullfile('results', 'Experiment3', timestamp);
    if ~exist(results_dir, 'dir'), mkdir(results_dir); end
    fprintf('All new results will be saved in: %s\n\n', results_dir);

    % --- 4.1 Sub-experiment: Sensitivity of FedWNNM to C ---
    if run_flags.FedWNNM_MC
        exp_name = 'sensitivity_fedwnnm_C';
        exp_dir = fullfile(results_dir, exp_name);
        if ~exist(exp_dir, 'dir'), mkdir(exp_dir); end
        fprintf('--- Starting sub-experiment: [%s] ---\n', exp_name);

        p_obs_fixed = 0.3;
        algo_name = 'FedWNNM_MC';

        for C_val = exp_ranges.C
            fprintf('  Testing FedWNNM with C = %.2f\n', C_val);

            for mc_iter = 1:params.mc
                fprintf('    MC iteration: %d/%d\n', mc_iter, params.mc);
                [data, mask, L_true, U_true, S_true] = generate_synthetic_data(params, p_obs_fixed);
                [algo_path, base_parameters] = setup_algorithm_env(codeFolderPath, algo_name, params, L_true, U_true, S_true);

                parameters = base_parameters;
                parameters.C = C_val; % Set the current C value

                result = run_specific_algorithm(algo_name, data, mask, parameters);
                result_filename = fullfile(exp_dir, sprintf('result_%s_C_%.2f_mc_%d.mat', algo_name, C_val, mc_iter));
                save(result_filename, 'result');
                if ~isempty(algo_path), rmpath(genpath(algo_path)); end
            end
        end
        fprintf('--- Sub-experiment [%s] completed ---\n\n', exp_name);
    end

    % --- 4.2 Sub-experiment: Sensitivity of FedSVD to p_over (using FedWNNM) ---
    if run_flags.FedWNNM_MC
        exp_name = 'sensitivity_frsvd_pover';
        exp_dir = fullfile(results_dir, exp_name);
        if ~exist(exp_dir, 'dir'), mkdir(exp_dir); end
        fprintf('--- Starting sub-experiment: [%s] ---\n', exp_name);

        p_obs_fixed = 0.3;
        algo_name = 'FedWNNM_MC'; % Use FedWNNM_MC as the test platform

        for pover_val = exp_ranges.p_over
            fprintf('  Testing FedSVD with p_over = %d\n', pover_val);

            for mc_iter = 1:params.mc
                fprintf('    MC iteration: %d/%d\n', mc_iter, params.mc);
                [data, mask, L_true, U_true, S_true] = generate_synthetic_data(params, p_obs_fixed);
                [algo_path, base_parameters] = setup_algorithm_env(codeFolderPath, algo_name, params, L_true, U_true, S_true);

                parameters = base_parameters;
                parameters.p_over = pover_val; % Set the current p_over value

                result = run_specific_algorithm(algo_name, data, mask, parameters);
                result_filename = fullfile(exp_dir, sprintf('result_%s_pover_%d_mc_%d.mat', algo_name, pover_val, mc_iter));
                save(result_filename, 'result');
                if ~isempty(algo_path), rmpath(genpath(algo_path)); end
            end
        end
        fprintf('--- Sub-experiment [%s] completed ---\n\n', exp_name);
    end

    % --- 4.3 Sub-experiment: Sensitivity of FedSVD to rho (using FedWNNM) ---
    if run_flags.FedWNNM_MC
        exp_name = 'sensitivity_frsvd_rho';
        exp_dir = fullfile(results_dir, exp_name);
        if ~exist(exp_dir, 'dir'), mkdir(exp_dir); end
        fprintf('--- Starting sub-experiment: [%s] ---\n', exp_name);

        p_obs_fixed = 0.3;
        algo_name = 'FedWNNM_MC'; % Use FedWNNM_MC as the test platform

        for rho_val = exp_ranges.rho
            fprintf('  Testing FedSVD with rho = %.2f\n', rho_val);

            for mc_iter = 1:params.mc
                fprintf('    MC iteration: %d/%d\n', mc_iter, params.mc);
                [data, mask, L_true, U_true, S_true] = generate_synthetic_data(params, p_obs_fixed);
                [algo_path, base_parameters] = setup_algorithm_env(codeFolderPath, algo_name, params, L_true, U_true, S_true);

                parameters = base_parameters;
                parameters.rho = rho_val; % Set the current rho value

                result = run_specific_algorithm(algo_name, data, mask, parameters);
                result_filename = fullfile(exp_dir, sprintf('result_%s_rho_%.2f_mc_%d.mat', algo_name, rho_val, mc_iter));
                save(result_filename, 'result');
                if ~isempty(algo_path), rmpath(genpath(algo_path)); end
            end
        end
        fprintf('--- Sub-experiment [%s] completed ---\n\n', exp_name);
    end
end % End of 'run_and_analyze' mode

%% 5. Results Analysis and Plotting
fprintf('============================================================\n');
fprintf('                 Loading, Aggregating, and Analyzing Results                \n');
fprintf('============================================================\n\n');

% Determine the directory for analysis
if strcmpi(ANALYSIS_MODE, 'run_and_analyze')
    results_dir_for_analysis = results_dir;
    fprintf('Mode: [Run and Analyze]. Analyzing results from the current run:\n%s\n\n', results_dir_for_analysis);
else
    if isempty(TARGET_RESULTS_DIR) || ~exist(TARGET_RESULTS_DIR, 'dir')
        error('Error: In "analyze_only" mode, please set a valid TARGET_RESULTS_DIR.');
    end
    results_dir_for_analysis = TARGET_RESULTS_DIR;
    fprintf('Mode: [Analyze Only]. Loading results from specified directory:\n%s\n\n', results_dir_for_analysis);
end

% --- 5.1 Analyze and plot sensitivity of FedWNNM to C ---
if run_flags.FedWNNM_MC
    exp_name = 'sensitivity_fedwnnm_C';
    exp_dir = fullfile(results_dir_for_analysis, exp_name);
    if exist(exp_dir, 'dir')
        fprintf('--- Analyzing: [%s] ---\n', exp_name);
        C_vals_agg = []; psnr_vals_agg = [];

        for C_val = exp_ranges.C
            file_pattern = fullfile(exp_dir, sprintf('result_FedWNNM_MC_C_%.2f_mc_*.mat', C_val));
            result_files = dir(file_pattern);

            if isempty(result_files), continue; end

            temp_psnr = [];
            for k = 1:length(result_files)
                loaded_data = load(fullfile(exp_dir, result_files(k).name));
                temp_psnr(end+1) = loaded_data.result.relative_error;
            end

            C_vals_agg(end+1) = C_val;
            psnr_vals_agg(end+1) = mean(temp_psnr);
        end

        [C_vals_sorted, sort_idx] = sort(C_vals_agg);
        psnr_vals_sorted = psnr_vals_agg(sort_idx);
        plot_sensitivity_curve(C_vals_sorted, psnr_vals_sorted, 'C', 'FedWNNM', results_dir_for_analysis, 'semilogx');
        fprintf('FedWNNM-C sensitivity curve generated.\n\n');
    else
        fprintf('Results directory for sub-experiment [%s] not found. Skipping analysis.\n', exp_name);
    end
end

% --- 5.2 Analyze and plot sensitivity of FedSVD to p_over ---
if run_flags.FedWNNM_MC
    exp_name = 'sensitivity_frsvd_pover';
    exp_dir = fullfile(results_dir_for_analysis, exp_name);
    if exist(exp_dir, 'dir')
        fprintf('--- Analyzing: [%s] ---\n', exp_name);
        pover_vals_agg = []; psnr_vals_agg = []; time_vals_agg = [];

        for pover_val = exp_ranges.p_over
            file_pattern = fullfile(exp_dir, sprintf('result_FedWNNM_MC_pover_%d_mc_*.mat', pover_val));
            result_files = dir(file_pattern);

            if isempty(result_files), continue; end

            temp_psnr = []; temp_time = [];
            for k = 1:length(result_files)
                loaded_data = load(fullfile(exp_dir, result_files(k).name));
                temp_psnr(end+1) = loaded_data.result.relative_error;
                temp_time(end+1) = loaded_data.result.total_time;
            end

            pover_vals_agg(end+1) = pover_val;
            psnr_vals_agg(end+1) = mean(temp_psnr);
            time_vals_agg(end+1) = mean(temp_time);
        end

        [pover_vals_sorted, sort_idx] = sort(pover_vals_agg);
        psnr_vals_sorted = psnr_vals_agg(sort_idx);
        time_vals_sorted = time_vals_agg(sort_idx);
        plot_sensitivity_pover(pover_vals_sorted, psnr_vals_sorted, time_vals_sorted, results_dir_for_analysis);
        fprintf('FedSVD-p_over sensitivity curve generated.\n\n');
    else
        fprintf('Results directory for sub-experiment [%s] not found. Skipping analysis.\n', exp_name);
    end
end

% --- 5.3 Analyze and plot sensitivity of FedSVD to rho ---
if run_flags.FedWNNM_MC
    exp_name = 'sensitivity_frsvd_rho';
    exp_dir = fullfile(results_dir_for_analysis, exp_name);
    if exist(exp_dir, 'dir')
        fprintf('--- Analyzing: [%s] ---\n', exp_name);
        rho_vals_agg = []; psnr_vals_agg = [];

        for rho_val = exp_ranges.rho
            file_pattern = fullfile(exp_dir, sprintf('result_FedWNNM_MC_rho_%.2f_mc_*.mat', rho_val));
            result_files = dir(file_pattern);

            if isempty(result_files), continue; end

            temp_psnr = [];
            for k = 1:length(result_files)
                loaded_data = load(fullfile(exp_dir, result_files(k).name));
                temp_psnr(end+1) = loaded_data.result.relative_error;
            end

            rho_vals_agg(end+1) = rho_val;
            psnr_vals_agg(end+1) = mean(temp_psnr);
        end

        [rho_vals_sorted, sort_idx] = sort(rho_vals_agg);
        psnr_vals_sorted = psnr_vals_agg(sort_idx);
        plot_sensitivity_curve(rho_vals_sorted, psnr_vals_sorted, '\rho', 'FedWNNM', results_dir_for_analysis, 'plot');
        fprintf('FedWNNM-rho sensitivity curve generated.\n\n');
    else
        fprintf('Results directory for sub-experiment [%s] not found. Skipping analysis.\n', exp_name);
    end
end

fprintf('\nAll analysis tasks are complete!\n');


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                          Helper Functions (Inline)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [data, mask, L_true, U_true, S_true] = generate_synthetic_data(params, p_obs)
    % Generate synthetic low-rank matrix data.
    % NOTE: The random seed is not reset here to ensure diversity across MC iterations.
    U_true_gen = randn(params.m, params.r);
    V_true_gen = randn(params.n, params.r);
    L_true = U_true_gen * V_true_gen';
    
    % Normalize to [0, 1]
    L_min = min(L_true(:));
    L_max = max(L_true(:));
    L_true = (L_true - L_min) / (L_max - L_min);
    
    [U_true_orth, S_true_diag, ~] = svd(L_true, 'econ');
    U_true = U_true_orth(:, 1:params.r);
    S_true_diag = diag(S_true_diag);
    S_true = S_true_diag(1:params.r);
    
    % Create observation mask
    num_observed = round(params.m * params.n * p_obs);
    omega = randperm(params.m * params.n, num_observed);
    mask = zeros(params.m, params.n);
    mask(omega) = 1;
    data = L_true .* mask;
end

function [algo_path, parameters] = setup_algorithm_env(code_root, algo_name, base_params, L, U, S)
    % Set up algorithm environment: dynamic paths and parameters.
    folder_map = containers.Map(...
        {'AltGD', 'altGDMinCntrl_T', 'altGDMin_T', 'altMinCntrl_T', 'altMinPrvt_T', ...
         'FedWNNM_MC', 'WNNM_MC'}, ...
        {'AltGD', 'AltGD', 'AltGD', 'AltGD', 'AltGD', ...
         'Fed_WNNM', 'WNNM'});

    if isKey(folder_map, algo_name)
        folder_name = folder_map(algo_name);
        algo_path = fullfile(code_root, folder_name);
        addpath(genpath(algo_path));
    else
        warning('Folder mapping for %s not found.', algo_name);
        algo_path = '';
    end

    % Set common parameters
    parameters = struct(...
        'm', base_params.m, 'n', base_params.n, 'maxiter', base_params.maxiter, ...
        'tol', base_params.tol, 'L_true', L, 'U_true', U, 'S_true', S);

    % Add algorithm-specific parameters
    switch algo_name
        case 'WNNM_MC'
            parameters.C = 1.0; % Default value
        case 'FedWNNM_MC'
            parameters.r = base_params.r_est;
            parameters.p = base_params.p;
            parameters.C = 1.0; % Default value
            parameters.p_over = base_params.p_over;
            parameters.rho = base_params.rho;
            parameters.q = 1; % Default value
    end
end

function result = run_specific_algorithm(algo_name, data, mask, parameters)
    % Unified interface for calling algorithms.
    switch algo_name
        case 'WNNM_MC'
            result = WNNM_MC(data, mask, parameters);
        case 'FedWNNM_MC'
            result = FedWNNM_MC(data, mask, parameters);
        case {'AltGD', 'altGDMin_T', 'altGDMinCntrl_T', 'altMinCntrl_T', 'altMinPrvt_T'}
            % Assuming these functions have a similar calling convention.
            % Placeholder for actual function calls if they were to be used in this script.
            func_handle = str2func(algo_name);
            result = func_handle(data, mask, parameters);
        otherwise
            error('Unknown algorithm name: "%s".', algo_name);
    end
end