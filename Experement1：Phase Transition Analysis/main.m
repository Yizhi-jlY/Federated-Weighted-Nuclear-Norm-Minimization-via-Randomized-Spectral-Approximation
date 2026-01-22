%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1. Initialize Environment
clc;
clear;
close all;
fprintf('============================================================\n');
fprintf('============================================================\n\n');

% Set random seed for reproducible results
rng(6190, 'twister');
num_computer = 0;

%% 2. Path Management (Modified as required)
% -------------------- Smart Path Management --------------------
% This script uses a dynamic path management strategy. Instead of pre-loading
% all folders, it adds the necessary path only when a specific algorithm or
% function is called, and removes it after use. This helps maintain a clean
% MATLAB working environment.

% Get the folder path of the current main script, which is the root
% directory of the 'code' folder. This is the base path for locating all
% algorithm and utility functions.
codeFolderPath0 = fileparts(fileparts(mfilename('fullpath'))); 
codeFolderPath = fullfile(codeFolderPath0,'code');
fprintf('Code root directory located: %s\n', codeFolderPath);
fprintf('Path management strategy: Dynamically add on run, remove after use.\n\n');

utils_path = fullfile(codeFolderPath, 'utils');
addpath(genpath(utils_path));
fprintf('[Path Management] Utility function path added: %s\n\n', utils_path);

%% 3. Experiment Parameter Configuration
% -------------------- Execution Mode Control --------------------
% 'run_and_analyze': Run experiments and analyze new results.
% 'analyze_only':    Load and analyze results from a previous experiment.
ANALYSIS_MODE = 'run_and_analyze'; 
new = 1;

% In 'analyze_only' mode, the path to the target results folder must be specified.
% Example: TARGET_RESULTS_DIR = 'results/Experiment1/20250728_123000';
TARGET_RESULTS_DIR = '';

% -------------------- Data Generation Parameters --------------------
params.m = 100;        % Number of matrix rows
params.n = 100;        % Number of matrix columns
params.mc = 20;        % Number of Monte Carlo simulations

% New: Ranges for missing rates and ranks
missing_rates = 0.9:-0.1:0.1;  % Missing rates from 0.1 to 0.9
ranks = 10:10:90;             % Ranks from 10 to 90
success_threshold = 0.17;     % Success threshold: Final relative error < this value is considered a success (adjust as needed)

% New: C_scaler mapping for FedWNNM_MC based on missing rate
c_scalers = containers.Map({'0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'}, ...
                           {12.2781, 11.2532, 7.3112, 12.4142, 19.6809, 43.9287, 46.4541, 46.4007, 45.2524}, ...
                           'UniformValues', true);

% -------------------- General Algorithm Parameters --------------------
params.maxiter = 200;   % Maximum number of iterations
params.tol = 1e-5;      % Convergence tolerance

% -------------------- Federated Learning Parameters --------------------
params.p = 5;           % Number of federated clients

% -------------------- Randomized SVD Parameters --------------------
params.r_est = 20;      % Estimated rank for Randomized SVD (dynamically adjusted to r + 20 in the loop)
params.p_over = 10;     % Oversampling parameter
params.q = 0;           % Number of power iterations


run_flags.FedWNNM_MC     = 1;
run_flags.altGDMin_T     = 1; % Federated AltGD
run_flags.altMinPrvt_T   = 1; % Federated AltMin
run_flags.AltGD          = 1;
run_flags.altGDMinCntrl_T= 1;
run_flags.altMinCntrl_T  = 1;

% --- Centralized Baseline Algorithms ---
run_flags.WNNM_MC        = 1;


%% 5, 6, 7: Experiment Execution Section (Runs only in 'run_and_analyze' mode)
if strcmpi(ANALYSIS_MODE, 'run_and_analyze')
    
    % --- 5. Create Directory for Storing Results ---
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    if new
        results_dir = fullfile('results', 'Experiment_phase_trasition', timestamp);
    else 
        results_dir = TARGET_RESULTS_DIR;
    end
    if ~exist(results_dir, 'dir')
        mkdir(results_dir);
    end
    fprintf('All new results will be saved in: %s\n\n', results_dir);

    % --- 6 & 7. Outer Loop: Varying Missing Rates and Ranks, Inner Monte Carlo Simulations ---
    fprintf('============================================================\n');
    fprintf('                  Starting Monte Carlo Simulations with Varying Missing Rates and Ranks\n');
    fprintf('============================================================\n\n');
    
    for mr_idx = 1:length(missing_rates)
        missing_rate = missing_rates(mr_idx);
        params.p_obs = 1 - missing_rate;  % Observation rate = 1 - missing rate
        
        % Get the C_scaler for the current missing rate
        c_scaler_key = num2str(missing_rate, '%.1f');
        if ~isKey(c_scalers, c_scaler_key)
            error('C_scaler value for missing rate %.1f not found.', missing_rate);
        end
        current_c_scaler = c_scalers(c_scaler_key);
        
        for r_idx = 1:length(ranks)
            params.r = ranks(r_idx);
            params.r_est = params.r + 20;  % Update estimated rank
            
            fprintf('Current configuration: Missing Rate=%.1f, Rank=%d\n', missing_rate, params.r);
            
            for mc_iter = num_computer+1:params.mc+num_computer
                fprintf('  Monte Carlo Simulation Iteration %d of %d\n', mc_iter, params.mc);
                
                % Generate synthetic data
                fprintf('  Generating synthetic data...\n');
                fprintf('  Matrix dimensions: %d x %d, Rank: %d, Observation Rate: %.2f\n', params.m, params.n, params.r, params.p_obs);
                
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
                
                data_filename = fullfile(results_dir, sprintf('synthetic_data_mr%.1f_r%d_mc%d.mat', missing_rate, params.r, mc_iter));
                save(data_filename, 'data', 'mask', 'L_true', 'U_true', 'S_true', 'params');
                fprintf('  Synthetic data generated and saved (Missing Rate %.1f, Rank %d, MC %d).\n\n', missing_rate, params.r, mc_iter);

                % Run algorithms and collect results
                fprintf('  Executing recovery algorithms...\n');
                algorithm_execution_order = {    
                    'factGDNew', 'AltGD', 'altGDMinCntrl_T', 'altGDMin_T', 'altMinCntrl_T', 'altMinParfor_T', 'altMinPrvt_T', ...
                    'SVP_MC', 'SVT', 'SVT_Rand', 'FedSVT_MC', ...    
                    'WNNM_MC', 'FedWNNM_MC'
                };

                algorithms_to_run = fieldnames(run_flags);
                disp('  Algorithms to be executed:');
                for i = 1:length(algorithms_to_run)
                    if run_flags.(algorithms_to_run{i})
                        disp(['  - ', algorithms_to_run{i}]);
                    end
                end
                fprintf('\n');

                for i = 1:length(algorithm_execution_order)
                    algo_name = algorithm_execution_order{i};
                    if isfield(run_flags, algo_name) && run_flags.(algo_name)
                        result_filename = fullfile(results_dir, sprintf('result_%s_mr%.1f_r%d_mc%d.mat', algo_name, missing_rate, params.r, mc_iter));
                        
                        % Dynamic Path Management
                        algo_path = '';
                        switch algo_name
                            case {'AltGD', 'altGDMinCntrl_T', 'altGDMin_T', 'altMinCntrl_T', 'altMinParfor_T', 'altMinPrvt_T', 'factGDNew'}
                                folder_name = 'AltGD';
                            case 'FedSVT_MC'
                                folder_name = 'Fed_SVT';
                            case 'FedWNNM_MC'
                                folder_name = 'Fed_WNNM';
                            case 'SVP_MC'
                                folder_name = 'SVP';
                            case 'SVT'
                                folder_name = 'SVT';
                            case 'SVT_Rand'
                                folder_name = 'SVT_rand';
                            case 'WNNM_MC'
                                folder_name = 'WNNM';
                            otherwise
                                warning('No folder mapping found for %s. No specific path will be added.', algo_name);
                                folder_name = '';
                        end
                        
                        if ~isempty(folder_name)
                            algo_path = fullfile(codeFolderPath, folder_name);
                            addpath(genpath(algo_path));
                            fprintf('[Path Management] Temporary path added for algorithm %s: %s\n', algo_name, algo_path);
                        end
                        
                        fprintf('  --- [%s] Running... ---\n', algo_name);
                        
                        % Configure specific parameters for different algorithms
                        switch algo_name
                            case 'factGDNew'
                                parameters = struct(...
                                    'm', params.m, ...
                                    'n', params.n, ...
                                    'r', params.r, ...
                                    'p', params.p, ...
                                    'maxiter', params.maxiter, ...
                                    'tol', params.tol, ...
                                    'Tsvd', 15, ...
                                    'step_const', 0.75, ...
                                    'sampling_rate', params.p_obs, ...
                                    'L_true', L_true, ...
                                    'U_true', U_true, ...
                                    'S_true', S_true ...
                                );
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
                            case 'altMinParfor_T'
                                parameters = struct(...
                                    'm', params.m, ...
                                    'n', params.n, ...
                                    'r', params.r, ...
                                    'p', params.p, ...
                                    'maxiter', params.maxiter, ...
                                    'tol', params.tol, ...
                                    'Tsvd', 15, ...
                                    'p_obs', params.p_obs, ...
                                    'L_true', L_true, ...
                                    'U_true', U_true, ...
                                    'S_true', S_true ...
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
                            case 'SVP_MC'
                                parameters = struct(...
                                    'k', params.r + 40, ...
                                    'step', 1 / params.p_obs, ...
                                    'maxiter', params.maxiter, ...
                                    'tol', params.tol, ...
                                    'L_true', L_true, ...
                                    'U_true', U_true, ...
                                    'S_true', S_true ...
                                );
                            case 'SVT'
                                parameters = struct(...
                                    'm', params.m, ...
                                    'n', params.n, ...
                                    'tao', sqrt(params.m * params.n) / 4, ...
                                    'step', 1.2, ...
                                    'tol', params.tol, ...
                                    'maxiter', params.maxiter, ...
                                    'L_true', L_true, ...
                                    'U_true', U_true, ...
                                    'S_true', S_true ...
                                );
                            case 'SVT_Rand'
                                parameters = struct(...
                                    'tao', 0.843*sqrt(params.m * params.n), ...
                                    'step', 1.5475, ...
                                    'r', params.r_est, ...
                                    'p_over', params.p_over, ...
                                    'q', params.q, ...
                                    'p_clients', params.p, ...
                                    'maxiter', params.maxiter, ...
                                    'tol', params.tol, ...
                                    'use_rand_svd', true, ...
                                    'L_true', L_true, ...
                                    'U_true', U_true, ...
                                    'S_true', S_true ...
                                );
                            case 'FedSVT_MC'
                                parameters = struct(...
                                    'm', params.m, ...
                                    'n', params.n, ...
                                    'r', params.r_est, ...
                                    'p', params.p, ...
                                    'tau', 3.014 * sqrt(params.m * params.n), ...
                                    'delta0',9.8824, ...
                                    'gamma', 0.7202, ...
                                    'tol', params.tol, ...
                                    'maxiter', params.maxiter, ...
                                    'p_over', params.p_over, ...
                                    'rho', 1, ...
                                    'q', params.q, ...
                                    'L_true', L_true, ...
                                    'U_true', U_true, ...
                                    'S_true', S_true ...
                                );
                            case 'WNNM_MC'
                                parameters = struct(...
                                    'C', 5.252* sqrt(params.m * params.n), ...
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
                                    'C', 5.252 * sqrt(params.m * params.n), ...
                                    'myeps', 1e-6, ...
                                    'tol', params.tol, ...
                                    'maxiter', params.maxiter, ...
                                    'p_over', params.p_over, ...
                                    'rho', 0.2, ...
                                    'q', params.q, ...
                                    'L_true', L_true, ...
                                    'U_true', U_true, ...
                                    'S_true', S_true ...
                                );
                        end
                        
                        % Use a switch statement to explicitly call each function
                        switch algo_name
                            case 'factGDNew',       result = factGDNew(data, mask, parameters);
                            case 'AltGD',           result = AltGD(data, mask, parameters);
                            case 'altGDMinCntrl_T', result = altGDMinCntrl_T(data, mask, parameters);
                            case 'altGDMin_T',      result = altGDMin_T(data, mask, parameters);
                            case 'altMinCntrl_T',   result = altMinCntrl_T(data, mask, parameters);
                            case 'altMinParfor_T',  result = altMinParfor_T(data, mask, parameters);
                            case 'altMinPrvt_T',    result = altMinPrvt_T(data, mask, parameters);
                            case 'SVP_MC',          result = SVP_MC(data, mask, parameters);
                            case 'SVT',             result = SVT(data, mask, parameters);
                            case 'SVT_Rand',        result = SVT_Rand(data, mask, parameters);
                            case 'FedSVT_MC',       result = FedSVT_MC(data, mask, parameters);
                            case 'WNNM_MC',         result = WNNM_MC(data, mask, parameters);
                            case 'FedWNNM_MC',      result = FedWNNM_MC(data, mask, parameters);
                            otherwise
                                error('Unknown algorithm name: "%s". Please add a call for this algorithm in the main function switch-case statement.', algo_name);
                        end
                   
                        fprintf('  --- [%s] Execution finished, total time: %.2f seconds ---\n', algo_name, result.total_time);
                        save(result_filename, 'result');
                        fprintf('  --- [%s] Result saved.\n', algo_name);
                        
                        % Remove temporary path
                        if ~isempty(algo_path)
                            rmpath(genpath(algo_path));
                            fprintf('[Path Management] Temporary path for algorithm %s removed: %s\n\n', algo_name, algo_path);
                        else
                            fprintf('\n');
                        end
                    end
                end
                fprintf('  Monte Carlo simulation iteration %d finished.\n\n', mc_iter);
            end
        end
    end
end


fprintf('============================================================\n');
fprintf('                 Main Flow for Experiment Result Analysis\n');
fprintf('============================================================\n\n');

% Determine the results directory
if strcmpi(ANALYSIS_MODE, 'run_and_analyze')
    % In 'run_and_analyze' mode, results_dir should be generated by the execution script
    % For demonstration, we assume it exists here.
    % results_dir_for_analysis = results_dir; 
    error('This script is currently configured for "analyze_only" mode.');
else
    if isempty(TARGET_RESULTS_DIR) || ~exist(TARGET_RESULTS_DIR, 'dir')
        error('Please set a valid target folder path (TARGET_RESULTS_DIR).');
    end
    results_dir_for_analysis = TARGET_RESULTS_DIR;
    fprintf('Mode: [Analyze Only]. Analyzing directory:\n%s\n\n', results_dir_for_analysis);
end

% Create a folder to save figures
figures_dir = fullfile(results_dir_for_analysis, 'figures');
if ~exist(figures_dir, 'dir'), mkdir(figures_dir); end
fprintf('[Path Management] Figures will be saved to: %s\n\n', figures_dir);


codeFolderPath = fileparts(fileparts(mfilename('fullpath')));
fprintf('Code root directory located: %s\n', codeFolderPath);

utils_path = fullfile(codeFolderPath, 'utils');
addpath(genpath(utils_path));
fprintf('[Path Management] Utility function path added: %s\n\n', utils_path);

%% Step 1: Load All Experiment Result Data
fprintf('------------------------------------------------------------\n');
fprintf('Step 1: Loading all results from disk...\n');
fprintf('------------------------------------------------------------\n');
% Use a workspace "exist" check to avoid redundant loading, which is more intuitive
if ~exist('results_cache', 'var')
    results_cache = load_all_results(results_dir_for_analysis, run_flags, missing_rates, ranks, params);
end
fprintf('All result data loaded.\n\n');



%% Step 3: Generate Phase Transition Plots (Original Functionality)
fprintf('------------------------------------------------------------\n');
fprintf('Step 3: Generating phase transition plots...\n');
fprintf('------------------------------------------------------------\n');

% Use different thresholds for quick analysis
% You can modify the threshold here and rerun only this section
% (using "Run Section" in the MATLAB editor)
% =======================================================
%                 Modify Threshold and Rerun Here
% =======================================================
analyze_and_plot_results(results_cache, success_threshold, figures_dir);

fprintf('Phase transition plots generated.\n\n');


%% 9. Absolute Success Rate Difference Plot (WNNM vs. FedWNNM)
% =========================================================================
%   This section calculates the ABSOLUTE difference in success rates 
%   between WNNM and FedWNNM and visualizes it using a blue-to-yellow
%   heatmap, where blue means no difference and yellow means max difference.
% =========================================================================

fprintf('\n============================================================\n');
fprintf('    Generating Absolute Difference Plot |WNNM - FedWNNM|\n');
fprintf('============================================================\n\n');

% --- 9.1. Select the Two Algorithms to Compare ---
algo1_name = 'WNNM_MC';    % The centralized benchmark
algo2_name = 'FedWNNM_MC'; % The main federated algorithm

fprintf('Calculating absolute difference between:\n');
fprintf('  Algorithm 1: %s\n', algo1_name);
fprintf('  and Algorithm 2: %s\n\n', algo2_name);

% Find the index of each algorithm in the results.
idx_alg1 = find(strcmp(executed_algos, algo1_name));
idx_alg2 = find(strcmp(executed_algos, algo2_name));

if isempty(idx_alg1) || isempty(idx_alg2)
    error('One or both of the selected algorithms were not found. Please check the names.');
end

% --- 9.2. Calculate the Absolute Success Rate Difference ---
% Extract the success rate matrices for both algorithms.
rates_alg1 = squeeze(success_rates(:, :, idx_alg1));
rates_alg2 = squeeze(success_rates(:, :, idx_alg2));

% Calculate the ABSOLUTE difference.
abs_difference_matrix = abs(rates_alg1 - rates_alg2);

fprintf('Absolute difference matrix calculated.\n');
fprintf('Values range from 0 (no difference) to 1 (max difference).\n');

% --- 9.3. Plotting and Visualization ---
figure('Name', 'Absolute Success Rate Difference |WNNM vs. FedWNNM|', 'NumberTitle', 'off');

% --- Centralized Font Size Configuration ---
fontSizeOptions = struct(...
    'title', 16, ...
    'labels', 24, ...
    'ticks', 25, ...
    'legend', 18 ...
);

% --- Draw the Heatmap of the Absolute Difference ---
imagesc(ranks, missing_rates, abs_difference_matrix);

% --- Axis configuration using the fontSizeOptions struct ---
axis on;
axis xy; % Set Y-axis to normal direction (starts from bottom)
xlabel(' Rank (r)', 'FontSize', fontSizeOptions.labels);
ylabel('Missing Rate', 'FontSize', fontSizeOptions.labels);
%title('Absolute Success Rate Difference |WNNM - FedWNNM|', 'FontSize', fontSizeOptions.title);
set(gca, 'FontSize', fontSizeOptions.ticks); % Sets font size for axis tick numbers

% --- Window and background color ---
set(gcf, 'color', 'w');
set(gcf, 'unit', 'normalized', 'position', [0.2, 0.2, 0.4, 0.55]);

% --- MODIFICATION: Use a Blue-to-Yellow colormap ---
% The 'parula' colormap is MATLAB's default and transitions smoothly
% from dark blue (for low values) to bright yellow (for high values).
colormap(parula);

% --- Colorbar Configuration ---
cb = colorbar;
cb.FontSize = fontSizeOptions.legend;
ylabel(cb, 'Absolute Difference Magnitude', 'FontSize', fontSizeOptions.legend-2); 

% --- Set color limits from 0 to 1 ---
% This ensures that a difference of 0 corresponds to the start of the
% colormap (blue) and 1 corresponds to the end (yellow).
caxis([0, 1]);

% --- Save the Figure ---
% MODIFICATION: Updated filename to reflect new color scheme
img_filename_diff = fullfile(results_dir_for_analysis, 'absolute_difference_blue_yellow_plot.png');
saveas(gcf, img_filename_diff);
fprintf('Blue-yellow difference plot saved to: %s\n', img_filename_diff);

fprintf('\nDifference plot generation completed!\n');

fprintf('============================================================\n');
fprintf('                      All Analysis Tasks Completed\n');
fprintf('============================================================\n');