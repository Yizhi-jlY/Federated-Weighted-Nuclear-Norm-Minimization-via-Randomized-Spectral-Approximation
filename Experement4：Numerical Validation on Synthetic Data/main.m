%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                   Main Function: Federated Low-Rank Matrix Completion Algorithms Comparison
%
%   Description:
%       This script executes the core performance and convergence analysis.
%       It supports two modes:
%       1. 'run_and_analyze': Runs a new experiment and analyzes its results.
%       2. 'analyze_only':    Loads and analyzes results from a specified folder without running a new experiment.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1. Initialize Environment
clc;
clear;
close all;
fprintf('============================================================\n');
fprintf('        Experiment 1: Core Performance and Convergence Analysis\n');
fprintf('============================================================\n\n');

% Set random seed for reproducibility
rng(2024, 'twister');

%% 2. Path Management
% -------------------- Dynamic Path Management --------------------
% This script adds necessary paths for algorithms at runtime and removes them afterward.
% This helps maintain a clean MATLAB workspace.

% Get the root path of the 'code' folder.
codeFolderPath0 = fileparts(fileparts(mfilename('fullpath'))); 
codeFolderPath = fullfile(codeFolderPath0,'code');
fprintf('Code root directory located: %s\n', codeFolderPath);
fprintf('Path management policy: Add paths at runtime, remove after use.\n\n');

% Add utility functions path
utils_path = fullfile(codeFolderPath, 'utils');
addpath(genpath(utils_path));
fprintf('[Path Management] Added utility functions path: %s\n\n', utils_path);

%% 3. Experiment Configuration
% -------------------- Execution Mode Control --------------------
% 'run_and_analyze': Run experiment and analyze new results.
% 'analyze_only':    Only load and analyze previous results.
ANALYSIS_MODE = 'analyze_only'; 
new = 0;

% In 'analyze_only' mode, specify the target results folder.
% Example: TARGET_RESULTS_DIR = 'results/Experiment1/20250728_123000';
TARGET_RESULTS_DIR = '.\results\Experiment4';

% -------------------- Data Generation Parameters --------------------
params.m = 100;        % Number of matrix rows
params.n = 100;        % Number of matrix columns
params.r = 5;          % True rank
params.p_obs = 0.3;    % Observation rate
params.mc = 20;         % Number of Monte Carlo simulations

% -------------------- General Algorithm Parameters --------------------
params.maxiter = 350;    % Maximum number of iterations
params.tol = 1e-5;     % Convergence tolerance

% -------------------- Federated Learning Parameters --------------------
params.p = 5;          % Number of federated clients

% -------------------- Randomized SVD Parameters --------------------
params.r_est = params.r + 20; % Estimated rank for randomized SVD
params.p_over = 10;           % Oversampling parameter
params.q = 0;                 % Number of power iterations

%% 4. Algorithm Run Flags (1: Run, 0: Skip)
% --- Federated Algorithms ---
run_flags.FedWNNM_MC     = 1;
run_flags.altGDMin_T     = 1; % Federated AltGD
run_flags.altGDMinCntrl_T = 1;
run_flags.altMinCntrl_T   = 1;
run_flags.altMinPrvt_T   = 1;

% --- Centralized Baseline Algorithms ---
run_flags.WNNM_MC        = 1;
run_flags.AltGD          = 1;

%% 5, 6, 7: Experiment Execution (only in 'run_and_analyze' mode)
if strcmpi(ANALYSIS_MODE, 'run_and_analyze')
    
    % --- 5. Create Results Directory ---
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    if new
     results_dir = fullfile('results', 'Experiment4', timestamp);
    else 
        results_dir =  TARGET_RESULTS_DIR;
    end
    if ~exist(results_dir, 'dir')
        mkdir(results_dir);
    end
    fprintf('All new results will be saved in: %s\n\n', results_dir);

    % --- 6 & 7. Monte Carlo Simulation Loop: Generate Data and Run Algorithms ---
    fprintf('============================================================\n');
    fprintf('                  Starting Monte Carlo Simulations\n');
    fprintf('============================================================\n\n');
    
    for mc_iter = 1:params.mc
        fprintf('Monte Carlo Simulation: %d of %d\n', mc_iter, params.mc);
        
        % Generate synthetic data
        fprintf('Generating synthetic data...\n');
        fprintf('Matrix dimensions: %d x %d, Rank: %d, Observation rate: %.2f\n', params.m, params.n, params.r, params.p_obs);
        
        U_true = randn(params.m, params.r);
        V_true = randn(params.n, params.r);
        L_true = U_true * V_true';
        L_min = min(L_true(:)); L_max = max(L_true(:));
        L_true = (L_true - L_min) / (L_max - L_min);
        
        [U_true_orth, S_true_diag, ~] = svd(L_true, 'econ');
        U_true = U_true_orth(:, 1:params.r);
        S_true = diag(S_true_diag); S_true = S_true(1:params.r);
        
        num_observed = round(params.m * params.n * params.p_obs);
        omega = randperm(params.m * params.n, num_observed);
        mask = zeros(params.m, params.n);
        mask(omega) = 1;
        data = L_true .* mask;
        
        data_filename = fullfile(results_dir, sprintf('synthetic_data_mc%d.mat', mc_iter));
        save(data_filename, 'data', 'mask', 'L_true', 'U_true', 'S_true', 'params');
        fprintf('Synthetic data generated and saved (MC %d).\n\n', mc_iter);

        % Run algorithms and collect results
        fprintf('Executing matrix completion algorithms...\n');
        % Define the execution order to ensure consistency if needed
        algorithm_execution_order = {    
            'AltGD', 'altGDMinCntrl_T', 'altGDMin_T', 'altMinCntrl_T', 'altMinPrvt_T', ...
            'WNNM_MC', 'FedWNNM_MC'
        };

        algorithms_to_run = fieldnames(run_flags);
        disp('Algorithms to run:');
        for i = 1:length(algorithms_to_run)
            if run_flags.(algorithms_to_run{i})
                disp(['- ', algorithms_to_run{i}]);
            end
        end
        fprintf('\n');

        for i = 1:length(algorithm_execution_order)
            algo_name = algorithm_execution_order{i};
            if isfield(run_flags, algo_name) && run_flags.(algo_name)
                result_filename = fullfile(results_dir, sprintf('result_%s_mc%d.mat', algo_name, mc_iter));
                
                % Dynamic path management
                algo_path = '';
                switch algo_name
                    case {'AltGD', 'altGDMinCntrl_T', 'altGDMin_T', 'altMinCntrl_T', 'altMinPrvt_T'}
                        folder_name = 'AltGD';
                    case 'FedWNNM_MC'
                        folder_name = 'Fed_WNNM';
                    case 'WNNM_MC'
                        folder_name = 'WNNM';
                    otherwise
                        warning('No folder mapping found for %s. No specific path will be added.', algo_name);
                        folder_name = '';
                end
                
                if ~isempty(folder_name)
                    algo_path = fullfile(codeFolderPath, folder_name);
                    addpath(genpath(algo_path));
                    fprintf('[Path Management] Temporarily added path for %s: %s\n', algo_name, algo_path);
                end
                
                fprintf('--- Running [%s]... ---\n', algo_name);
                
                % Configure specific parameters for each algorithm
                switch algo_name
                    case 'AltGD'
                        parameters = struct(...
                            'm', params.m, ...
                            'n', params.n, ...
                            'rank', params.r, ...
                            'p_obs', params.p_obs, ...
                            'maxiter', params.maxiter, ...
                            'step_const', 0.75, ...
                            'tol', params.tol, ...
                            'L_true', L_true, ...
                            'U_true', U_true, ...
                            'S_true', S_true ...
                        );
                    case 'altGDMinCntrl_T'
                        parameters = struct(...
                            'r', params.r, ...
                            'eta_c', 1.0, ...
                            'p_obs',params.p_obs,...
                            'maxiter', params.maxiter, ...
                            'tol', params.tol, ...
                            'L_true', L_true, ...
                            'U_true', U_true, ...
                            'S_true', S_true ...
                        );
                    case 'altGDMin_T'
                        parameters = struct(...
                            'r', params.r, ...
                            'p', params.p, ...
                            'eta_c', 1.0, ...
                            'maxiter', params.maxiter, ...
                            'tol', params.tol, ...
                            'Tsvd', 15, ...
                            'L_true', L_true, ...
                            'U_true', U_true, ...
                            'S_true', S_true ...
                        );
                    case 'altMinCntrl_T'
                        parameters = struct(...
                            'm', params.m, ...
                            'n', params.n, ...
                            'r', params.r, ...
                            'p', params.p, ...
                            'T', params.maxiter, ...
                            'Tsvd', 15, ...
                            'L_true', L_true, ...
                            'U_true', U_true ...
                        );
                    case 'altMinPrvt_T'
                        parameters = struct(...
                            'm', params.m, ...
                            'n', params.n, ...
                            'p', params.p, ...
                            'rank', params.r, ...
                            'p_obs', params.p_obs, ...
                            'maxiter', params.maxiter, ...
                            'T_inner', 10, ...
                            'Tsvd', 15, ...
                            'tol', params.tol, ...
                            'L_true', L_true, ...
                            'U_true', U_true, ...
                            'S_true', S_true ...
                        );
                    case 'WNNM_MC'
                        parameters = struct(...
                            'C', 1* sqrt(params.m * params.n), ...
                            'myeps', 1e-6, ...
                            'tol', params.tol, ...
                            'maxiter', params.maxiter, ...
                            'L_true', L_true, ...
                            'U_true', U_true, ...
                            'S_true', S_true ...
                        );
                    case 'FedWNNM_MC'
                        parameters = struct(...
                            'm', params.m, ...
                            'n', params.n, ...
                            'p', params.p, ...
                            'C', 1* sqrt(params.m * params.n), ...
                            'myeps', 1e-6, ...
                            'tol', params.tol, ...
                            'maxiter', params.maxiter, ...
                            'p_over', params.p_over, ...
                            'rho', 0.1, ...
                            'q', params.q, ...
                            'L_true', L_true, ...
                            'U_true', U_true, ...
                            'S_true', S_true ,...
                            'r_est',params.r_est...
                        );
                end
                
                % Call each function explicitly using a switch statement
                switch algo_name
                    case 'AltGD',           result = AltGD(data, mask, parameters);
                    case 'altGDMinCntrl_T', result = altGDMinCntrl_T(data, mask, parameters);
                    case 'altGDMin_T',      result = altGDMin_T(data, mask, parameters);
                    case 'altMinCntrl_T',   result = altMinCntrl_T(data, mask, parameters);
                    case 'altMinPrvt_T',    result = altMinPrvt_T(data, mask, parameters);
                    case 'WNNM_MC',         result = WNNM_MC(data, mask, parameters);
                    case 'FedWNNM_MC',      result = FedWNNM_MC(data, mask, parameters);
                    otherwise
                        error('Unknown algorithm name: "%s". Please add its function call to the switch-case statement.', algo_name);
                end
           
                fprintf('--- [%s] finished. Total time: %.2f seconds ---\n', algo_name, result.total_time);
                save(result_filename, 'result');
                fprintf('--- [%s] result saved.\n', algo_name);
                
                % Remove temporary path
                if ~isempty(algo_path)
                    rmpath(genpath(algo_path));
                    fprintf('[Path Management] Removed temporary path for %s: %s\n\n', algo_name, algo_path);
                else
                    fprintf('\n');
                end
            end
        end
        fprintf('Monte Carlo Simulation %d completed.\n\n', mc_iter);
    end
end

%% 8. Aggregate, Display, and Plot Results
fprintf('============================================================\n');
fprintf('                 Loading, Aggregating, and Analyzing Results\n');
fprintf('============================================================\n\n');

% Determine the results directory to analyze
if strcmpi(ANALYSIS_MODE, 'run_and_analyze')
    results_dir_for_analysis = results_dir; % Use the newly generated directory
    fprintf('Mode: [Run and Analyze]. Analyzing results from:\n%s\n\n', results_dir_for_analysis);
else % 'analyze_only' mode
    if isempty(TARGET_RESULTS_DIR) || ~exist(TARGET_RESULTS_DIR, 'dir')
        error(['Error: In "analyze_only" mode, please set a valid TARGET_RESULTS_DIR in Section 3.\n' ...
               'Example: TARGET_RESULTS_DIR = ''results/Experiment1/yyyymmdd_HHMMSS'';']);
    end
    results_dir_for_analysis = TARGET_RESULTS_DIR;
    fprintf('Mode: [Analyze Only]. Loading results from:\n%s\n\n', results_dir_for_analysis);
end

% Add utility functions path again in case of 'analyze_only' mode
utils_path = fullfile(codeFolderPath, 'utils');
addpath(genpath(utils_path));
fprintf('[Path Management] Added utility functions path: %s\n\n', utils_path);

% Load results from all Monte Carlo simulations
all_results = struct();
for mc_iter = 1:params.mc
    result_files = dir(fullfile(results_dir_for_analysis, sprintf('result_*_mc%d.mat', mc_iter)));
    for k = 1:length(result_files)
        filename = result_files(k).name;
        algo_name = extractBetween(filename, 'result_', sprintf('_mc%d.mat', mc_iter));
        if ~isempty(algo_name)
            algo_name = algo_name{1};
            fprintf('Loading: %s\n', filename);
            loaded_data = load(fullfile(results_dir_for_analysis, filename));
            if ~isfield(all_results, algo_name)
                all_results.(algo_name) = {};
            end
            all_results.(algo_name){end+1} = loaded_data.result;
        end
    end
end

% Calculate average results
average_results = struct();
for algo_name = fieldnames(all_results)'
    algo_name = algo_name{1};
    results = all_results.(algo_name);
    num_mc = length(results);
    if num_mc > 0
        % Initialize arrays to store metrics from each simulation
        all_A_hats = zeros(params.m, params.n, num_mc);
        all_iteration_counts = zeros(1, num_mc);
        all_convergeds = zeros(1, num_mc);
        all_total_times = zeros(1, num_mc);
        all_relative_errors = zeros(1, num_mc);
        all_psnr_values = zeros(1, num_mc);
        all_ssim_values = zeros(1, num_mc);
        
        % Initialize matrices for vector data, using NaN for padding
        all_residuals = nan(params.maxiter+1, num_mc);
        all_iteration_times = nan(params.maxiter+1, num_mc);
        all_wall_clock_times = nan(params.maxiter+1, num_mc);
        
        % Loop through each Monte Carlo run to collect data
        for i = 1:num_mc
            current_run_result = results{i};
            
            % Store matrix result
            all_A_hats(:, :, i) = current_run_result.A_hat;
            
            % Store scalar metrics
            all_iteration_counts(i) = current_run_result.iteration_count;
            % all_convergeds(i) = current_run_result.converged; % Assuming this field exists
            all_total_times(i) = current_run_result.total_time;
            all_relative_errors(i) = current_run_result.relative_error;
            all_psnr_values(i) = current_run_result.psnr_value;
            all_ssim_values(i) = current_run_result.ssim_value;
            
            % Store vector metrics
            num_iter = length(current_run_result.residuals);
            all_residuals(1:num_iter, i) = abs(current_run_result.residuals);
            all_iteration_times(1:num_iter, i) = current_run_result.iteration_times;
            all_wall_clock_times(1:num_iter, i) = current_run_result.wall_clock_times(1:num_iter);
        end
        
        % Calculate averages, ignoring NaNs for vectors
        avg_A_hat = mean(all_A_hats, 3);
        avg_iteration_count = mean(all_iteration_counts);
        avg_converged = mean(all_convergeds);
        avg_total_time = mean(all_total_times);
        avg_relative_error = mean(all_relative_errors);
        avg_psnr_value = mean(all_psnr_values);
        avg_ssim_value = mean(all_ssim_values);
        
        avg_residuals = mean(all_residuals, 2, 'omitnan');
        avg_iteration_times = mean(all_iteration_times, 2, 'omitnan');
        avg_wall_clock_times = mean(all_wall_clock_times, 2, 'omitnan');
        
        % Store averaged results
        average_results.(algo_name).name = algo_name;
        average_results.(algo_name).avg_A_hat = avg_A_hat;
        average_results.(algo_name).avg_iteration_count = avg_iteration_count;
        average_results.(algo_name).avg_converged = avg_converged;
        average_results.(algo_name).avg_total_time = avg_total_time;
        average_results.(algo_name).avg_residuals = avg_residuals;
        average_results.(algo_name).avg_iteration_times = avg_iteration_times;
        average_results.(algo_name).avg_wall_clock_times = avg_wall_clock_times;
        average_results.(algo_name).avg_relative_error = avg_relative_error;
        average_results.(algo_name).avg_psnr_value = avg_psnr_value;
        average_results.(algo_name).avg_ssim_value = avg_ssim_value;
        average_results.(algo_name).num_runs = num_mc;
    end
end

% Create a list of averaged results for plotting and display
executed_algos = fieldnames(average_results);
results_list = {};
for i = 1:length(executed_algos)
    algo_name = executed_algos{i};
    results_list{end+1} = average_results.(algo_name);
end

if isempty(results_list)
    disp('No valid algorithm results were loaded. Cannot generate report.');
    return;
end

% Load data from the last MC run for image comparison
L_true = []; data = [];
data_filename = fullfile(results_dir_for_analysis, sprintf('synthetic_data_mc%d.mat', params.mc));
if exist(data_filename, 'file')
    load(data_filename, 'L_true', 'data', 'params');
    fprintf('\nSuccessfully loaded original data for comparison (MC %d).\n', params.mc);
else
    warning('Warning: Could not find synthetic_data_mc%d.mat. Image comparison plot will not be generated.', params.mc);
end

% 8.1. Display average results table in the command line
fprintf('\nDisplaying average results table (based on %d Monte Carlo runs):\n', params.mc);
display_results_table_and_plots(results_list, results_dir_for_analysis);

% 8.2. Plot and save convergence curves (using averaged results)
plot_convergence_curves_1(results_list, results_dir_for_analysis);

fprintf('\nAll analysis tasks are complete!\n');