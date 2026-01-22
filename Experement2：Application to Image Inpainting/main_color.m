%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                 Main Function: Real-World Dataset Application (Experiment 4 - Image Inpainting)
%                               [Batch Processing & Mean Analysis Version - Color Image Support]
%
%   Description:
%         This script performs image inpainting experiments on all color images in the cbsd68 dataset.
%         It iterates through each image in the folder, applies an artificial mask, and then
%         runs a series of federated and centralized matrix completion algorithms on each color channel (R, G, B) separately for recovery.
%         Finally, the script calculates the average performance metrics
%         (PSNR, SSIM, etc.) for each algorithm across the entire dataset, displays them in a table, and saves the restored
%         color images.
%
%   Supported Modes:
%         1. 'run_and_analyze': Run a new experiment and analyze its results.
%         2. 'analyze_only':     Only load and analyze past results from a specified folder.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1. Environment Initialization
clc;
clear;
close all;
fprintf('============================================================\n');
fprintf('    Experiment 4: Real-World Dataset Application (Image Inpainting) - Batch Processing & Color Version\n');
fprintf('============================================================\n\n');

% Set the random seed to ensure reproducibility (especially for mask generation)
rng(2025, 'twister');

%% 2. Path Management
codeFolderPath0 = fileparts(fileparts(mfilename('fullpath'))); 
codeFolderPath = fullfile(codeFolderPath0,'code');

fprintf('Code root directory located: %s\n', codeFolderPath);
utils_path = fullfile(codeFolderPath, 'utils');
addpath(genpath(utils_path));
fprintf('[Path Management] Utility functions path added: %s\n\n', utils_path);

%% 3. Experiment Parameter Configuration
% -------------------- Execution Mode Control --------------------
ANALYSIS_MODE = 'run_and_analyze';
% In 'analyze_only' mode, specify the target results folder
% Example: TARGET_RESULTS_DIR = 'results/Experiment4_ImageInpainting/20250807_103000';
TARGET_RESULTS_DIR = '';

for ms_rate = [ 0.3 0.5 0.7]

% -------------------- Dataset and Mask Type Configuration --------------------
% Assuming the 'datasets' folder is at the same level as the 'code' folder
params.dataset_path = fullfile(codeFolderPath, '..', 'datasets', 'cbsd68t'); % Dataset path now points to cbsd68
params.mask_type = 'random';       % Options: 'text' or 'random'
params.missing_rate = ms_rate;         % Effective only when mask_type = 'random', e.g., [0.1, 0.2, ..., 0.9]
params.text_to_mask = 'CONFIDENTIAL'; % Effective only when mask_type = 'text'

% -------------------- General Algorithm Parameters --------------------
params.maxiter = 350;      % Number of iterations (can be adjusted as needed)
params.tol = 1e-4;         % Tolerance
params.r = 50;             % Estimated image rank (this is a key hyperparameter)

% -------------------- Federated Learning Parameters --------------------
params.p = 40;             % Number of federated clients

% -------------------- Randomized SVD Related Parameters --------------------
params.r_est = params.r + 10; % Estimated rank for Randomized SVD
params.p_over = 10;        % Oversampling parameter
params.q = 1;              % Number of power iterations (for images, q=1 or 2 is usually sufficient)

% -------------------- [New] FedWNNM_MC Specific Parameter Lookup Table --------------------
% Based on the relationship between missing rate and optimal C_scaler value
fedwnnm_param_table = containers.Map(...
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], ... % Keys: missing_rate
    [4.0047, 4.0012, 4.0010, 11.1182, 27.4829, 45.8091, 49.9929, 49.9944, 49.9995] ... % Values: C_scaler
);
% Add this lookup table to the params struct for easy passing and calling within the loop
params.fedwnnm_param_table = fedwnnm_param_table;
fprintf('[Parameter Config] Optimal C parameter lookup table for FedWNNM_MC loaded.\n\n');


%% 4. Algorithm Execution Switches (1: Run, 0: Skip)
run_flags.FedSVT_MC      = 0;
run_flags.FedWNNM_MC     = 1;
run_flags.factGDNew      = 0; % Federated AltGD
run_flags.altGDMin_T     = 1; % Federated AltGD
run_flags.altMinPrvt_T   = 1; % Federated AltMin
run_flags.AltGD          = 1;
run_flags.altGDMinCntrl_T= 1;
run_flags.altMinCntrl_T  = 1;

% --- Centralized Baseline Algorithms ---
run_flags.SVT            = 0;
run_flags.WNNM_MC        = 1;

%% 5, 6, 7: Experiment Execution (Only in 'run_and_analyze' mode)
if strcmpi(ANALYSIS_MODE, 'run_and_analyze')
    
    % --- 5. Create Directory for Storing Results ---
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    results_dir = fullfile('G:\FR_wnnm\results', 'Experiment4_ImageInpainting_Batch_Color', sprintf('missing_rate_%.1f', params.missing_rate), timestamp);
    if ~exist(results_dir, 'dir')
        mkdir(results_dir);
    end
    fprintf('All new results will be saved in: %s\n\n', results_dir);

    % --- 6. Get All Image Files from the Dataset ---
    image_files = dir(fullfile(params.dataset_path, '*.png'));
    if isempty(image_files)
        error('No .png image files found in the path: %s', params.dataset_path);
    end
    num_images = length(image_files);
    fprintf('%d images found in the dataset for processing.\n\n', num_images);
    
    % --- 7. Initialize Struct to Store All Run Results ---
    algorithms_to_run = fieldnames(run_flags);
    all_run_results = struct();
    
    disp('Algorithms to be executed:');
    for i = 1:length(algorithms_to_run)
        algo_name = algorithms_to_run{i};
        if run_flags.(algo_name)
            disp(['- ', algo_name]);
            % Initialize an empty cell array for each algorithm to be run, to store the results of each image
            all_run_results.(algo_name) = {};
        end
    end
    fprintf('\n');

    % Define algorithm execution order and folder mapping
    algorithm_map = containers.Map(...
        {'factGDNew', 'AltGD', 'altGDMinCntrl_T', 'altGDMin_T', 'altMinCntrl_T', 'altMinParfor_T', 'altMinPrvt_T', 'FedSVT_MC', 'FedWNNM_MC', 'SVT', 'WNNM_MC'}, ...
        {'AltGD',     'AltGD', 'AltGD',          'AltGD',      'AltGD',        'AltGD',          'AltGD',        'Fed_SVT',   'Fed_WNNM',   'SVT', 'WNNM'} ...
    );
    algorithm_execution_order = keys(algorithm_map);

    % --- Start Loop to Process Each Image ---
    for img_idx = 1:num_images
        current_image_info = image_files(img_idx);
        fprintf('============================================================\n');
        fprintf('>>> Processing image: %d / %d  (%s)\n', img_idx, num_images, current_image_info.name);
        fprintf('============================================================\n');

        % [Color Mod] Load image, do not convert to grayscale
        image_full_path = fullfile(params.dataset_path, current_image_info.name);
        L_true_rgb_uint8 = imread(image_full_path);
        
        % Ensure the image is a 3-channel color image, otherwise skip
        if size(L_true_rgb_uint8, 3) ~= 3
            fprintf('   >>> Warning: Image %s is not a 3-channel color image, skipping.\n', current_image_info.name);
            continue;
        end
        L_true_rgb = im2double(L_true_rgb_uint8);
        
        [m, n, ~] = size(L_true_rgb);
        current_params = params; % Create a copy of the parameters
        current_params.m = m;
        current_params.n = n;
        
        % Generate a mask for the current image (this mask will be applied to all three channels)
        mask = ones(m, n);
        if strcmpi(params.mask_type, 'text')
            % (Code for generating text mask for color images is omitted here, logic is similar to grayscale but needs to handle three channels)
            % For simplicity, only the random mask is kept here
            error('Text mask is not currently supported in color image mode. Please select "random" type.');
        elseif strcmpi(params.mask_type, 'random')
            num_missing = round(m * n * params.missing_rate);
            omega_missing = randperm(m * n, num_missing);
            mask(omega_missing) = 0;
        end
        
        % [Color Mod] Apply the mask to each channel of the color image
        data_rgb = L_true_rgb .* repmat(mask, [1, 1, 3]);
        current_params.p_obs = mean(mask(:));
        
        % --- Run All Selected Algorithms ---
        for algo_idx = 1:length(algorithm_execution_order)
            algo_name = algorithm_execution_order{algo_idx};
            if isfield(run_flags, algo_name) && run_flags.(algo_name)
                
                % Dynamic path management
                folder_name = algorithm_map(algo_name);
                algo_path = fullfile(codeFolderPath, folder_name);
                addpath(genpath(algo_path));
                
                fprintf('--- [%s] Running... ---\n', algo_name);
                
                % [Color Mod] Initialize variables to store the three recovered channels and total time
                L_est_rgb = zeros(m, n, 3);
                total_time_for_algo = 0;
                
                % [Color Mod] Process the R, G, B channels separately
                for channel = 1:3
                    fprintf('    > Processing channel: %d\n', channel);
                    
                    % Extract data for the current channel
                    L_true_channel = L_true_rgb(:, :, channel);
                    data_channel = data_rgb(:, :, channel);
                    
                    % Get the true singular values/vectors of the current channel for evaluation (Note: This is only for internal evaluation in some algorithms; final evaluation is based on PSNR/SSIM)
                    [U_true_orth, S_true_diag, ~] = svd(L_true_channel, 'econ');
                    U_true = U_true_orth(:, 1:params.r);
                    S_true = diag(S_true_diag); S_true = S_true(1:params.r);
                
                    % Configure specific parameters for different algorithms
                    parameters = current_params;
                    parameters.L_true = L_true_channel;
                    parameters.U_true = U_true;
                    parameters.S_true = S_true;
                    
                    % (The switch-case block for parameter settings, identical to the original code, is omitted here)
                    switch algo_name
                        % ... (The switch-case block for parameter settings, identical to the original code, is omitted here)
                        case 'factGDNew',       parameters.step_const = 0.75; parameters.Tsvd = 15; parameters.maxiter = params.maxiter;
                        case 'AltGD',           parameters.rank = params.r; parameters.step_const = 0.75; parameters.maxiter = params.maxiter;
                        case 'altGDMinCntrl_T', parameters.r = params.r; parameters.eta_c = 1.0; parameters.p_obs = current_params.p_obs; parameters.maxiter = params.maxiter;
                        case 'altGDMin_T',      parameters.r = params.r; parameters.eta_c = 1.0; parameters.Tsvd = 15; parameters.maxiter = params.maxiter;
                        case 'altMinCntrl_T',   parameters.T = params.maxiter; parameters.Tsvd = 15;
                        case 'altMinParfor_T',  parameters.Tsvd = 15; parameters.p_obs = current_params.p_obs; parameters.maxiter = params.maxiter;
                        case 'altMinPrvt_T',    parameters.rank = params.r; parameters.T_inner = 10; parameters.Tsvd = 15; parameters.maxiter = params.maxiter;
                        case 'SVT',             parameters.tao = 2.5 * sqrt(m * n); parameters.step = 1.2; parameters.maxiter = params.maxiter;
                        case 'FedSVT_MC',       parameters.tau =  sqrt(m*n)/3133.03; parameters.delta0 = 1.9992; parameters.gamma = 0.92647; parameters.maxiter = params.maxiter;
                        case 'WNNM_MC',         parameters.C = sqrt(max(m, n))/4.381; parameters.myeps = 1e-6; parameters.maxiter = params.maxiter;
                        case 'FedWNNM_MC'
                            current_rate_key = round(params.missing_rate, 1); 
                            if isKey(parameters.fedwnnm_param_table, current_rate_key)
                                selected_C = parameters.fedwnnm_param_table(current_rate_key);
                                fprintf('      > FedWNNM_MC: For missing rate %.1f, selected preset optimal C = %.4f\n', current_rate_key, selected_C);
                                parameters.C = selected_C;
                            else
                                default_C = sqrt(max(m, n)) / 4.381;
                                fprintf('      > FedWNNM_MC: Warning! Missing rate %.2f not in the preset list, using default C = %.4f\n', params.missing_rate, default_C);
                                parameters.C = default_C;
                            end
                            parameters.myeps = 1e-6; parameters.maxiter = params.maxiter;
                    end
                
                    % Unified function call
                    switch algo_name
                        case 'factGDNew',       result_channel = factGDNew(data_channel, mask, parameters);
                        case 'AltGD',           result_channel = AltGD(data_channel, mask, parameters);
                        case 'altGDMinCntrl_T', result_channel = altGDMinCntrl_T(data_channel, mask, parameters);
                        case 'altGDMin_T',      result_channel = altGDMin_T(data_channel, mask, parameters);
                        case 'altMinCntrl_T',   result_channel = altMinCntrl_T(data_channel, mask, parameters);
                        case 'altMinParfor_T',  result_channel = altMinParfor_T(data_channel, mask, parameters);
                        case 'altMinPrvt_T',    result_channel = altMinPrvt_T(data_channel, mask, parameters);
                        case 'SVP_MC',          result_channel = SVP_MC(data_channel, mask, parameters);
                        case 'SVT',             result_channel = SVT(data_channel, mask, parameters);
                        case 'SVT_Rand',        result_channel = SVT_Rand(data_channel, mask, parameters);
                        case 'FedSVT_MC',       result_channel = FedSVT_MC(data_channel, mask, parameters);
                        case 'WNNM_MC',         result_channel = WNNM_MC(data_channel, mask, parameters);
                        case 'FedWNNM_MC',      result_channel = FedWNNM_MC(data_channel, mask, parameters);
                    end
                    
                    % [Color Mod] Store the recovery result of the current channel and accumulate time
                    L_est_rgb(:, :, channel) = result_channel.A_hat;
                    total_time_for_algo = total_time_for_algo + result_channel.total_time;
                end % --- End of channel loop ---

                % [Color Mod] Calculate performance metrics for the entire color image
                psnr_val = psnr(L_est_rgb, L_true_rgb);
                ssim_val = ssim(L_est_rgb, L_true_rgb);

                fprintf('--- [%s] Completed. Total time: %.2f s, PSNR (RGB): %.2f, SSIM (RGB): %.4f\n', ...
                    algo_name, total_time_for_algo, psnr_val, ssim_val);
                
                % [Color Mod] Construct the final result struct for storage
                result.L_est = L_est_rgb; % Store the restored color image
                result.total_time = total_time_for_algo;
                result.psnr_value = psnr_val;
                result.ssim_value = ssim_val;
                % Keep the convergence information of one channel as a reference
                if exist('result_channel', 'var') && isfield(result_channel, 'iter_psnr')
                    result.iter_psnr = result_channel.iter_psnr; 
                end
                
                % Store the result of this run into the main results struct
                all_run_results.(algo_name){end+1} = result;
                
                % [Color Mod] Save the restored color image to a file
                restored_img_filename = sprintf('%s_%s.png', current_image_info.name(1:end-4), algo_name);
                imwrite(L_est_rgb, fullfile(results_dir, restored_img_filename));
                fprintf('    > Restored color image saved to: %s\n', restored_img_filename);
                
                % Remove path
                rmpath(genpath(algo_path));
            end
        end
        fprintf('\nImage %s processing complete.\n\n', current_image_info.name);

        % [New] Save the original and masked images for easy comparison
        imwrite(L_true_rgb, fullfile(results_dir, sprintf('%s_original.png', current_image_info.name(1:end-4))));
        imwrite(data_rgb, fullfile(results_dir, sprintf('%s_masked_%.1f.png', current_image_info.name(1:end-4), params.missing_rate)));

        % ... (Section for saving single image results to .mat file remains unchanged) ...
        single_image_results = struct();
        for i_algo = 1:length(algorithms_to_run)
            algo_name = algorithms_to_run{i_algo};
            if isfield(all_run_results, algo_name) && ~isempty(all_run_results.(algo_name))
                single_image_results.(algo_name) = all_run_results.(algo_name){end};
            end
        end
        single_result_filename = sprintf('result_img%03d_%s.mat', img_idx, current_image_info.name(1:end-4));
        single_result_filepath = fullfile(results_dir, single_result_filename);
        save(single_result_filepath, 'single_image_results', 'current_image_info', 'params');
        fprintf('All algorithm results for this image have been saved to: %s\n', single_result_filepath);
    end % --- End of image loop ---
    
    % --- Save a single .mat file containing all results ---
    fprintf('All images processed. Saving aggregated results...\n');
    full_results_filename = fullfile(results_dir, 'full_experiment_results.mat');
    save(full_results_filename, 'all_run_results', 'params', '-v7.3');
    fprintf('Aggregated results saved to: %s\n\n', full_results_filename);
    
end

%% 8. Aggregate Results, Calculate Averages, and Display
% (This section of code does not need modification as it directly processes psnr_value and ssim_value from all_run_results,
%  which are now already calculated based on color images)
fprintf('============================================================\n');
fprintf('               Starting to aggregate results and calculate average performance across all images\n');
fprintf('============================================================\n\n');

% Determine the results directory to analyze
if strcmpi(ANALYSIS_MODE, 'run_and_analyze')
    results_dir_for_analysis = results_dir;
    fprintf('Mode: [Run and Analyze]. Analyzing results from the current run:\n%s\n\n', results_dir_for_analysis);
else
    if isempty(TARGET_RESULTS_DIR) || ~exist(TARGET_RESULTS_DIR, 'dir')
        error('Error: In "analyze_only" mode, please set a valid target folder path (TARGET_RESULTS_DIR).');
    end
    results_dir_for_analysis = TARGET_RESULTS_DIR;
    fprintf('Mode: [Analyze Only]. Loading results from the specified directory:\n%s\n\n', results_dir_for_analysis);
end

% Load the .mat file containing all results
full_results_file = fullfile(results_dir_for_analysis, 'full_experiment_results.mat');
if ~exist(full_results_file, 'file')
    error('full_experiment_results.mat not found in the target directory: %s.', results_dir_for_analysis);
end
load(full_results_file, 'all_run_results');
fprintf('Successfully loaded the aggregated results file.\n\n');

% --- Calculate Average Metrics for Each Algorithm ---
executed_algos = fieldnames(all_run_results);
final_metrics = struct();

for i = 1:length(executed_algos)
    algo_name = executed_algos{i};
    results_for_algo = all_run_results.(algo_name);
    
    if isempty(results_for_algo) || ~iscell(results_for_algo) || isempty(results_for_algo{1})
        continue;
    end
    
    num_runs = length(results_for_algo);
    psnr_vec = zeros(1, num_runs);
    ssim_vec = zeros(1, num_runs);
    time_vec = zeros(1, num_runs);
    
    for j = 1:num_runs
        if isstruct(results_for_algo{j})
            psnr_vec(j) = results_for_algo{j}.psnr_value;
            ssim_vec(j) = results_for_algo{j}.ssim_value;
            time_vec(j) = results_for_algo{j}.total_time;
        end
    end
    
    final_metrics.(algo_name).psnr_mean = mean(psnr_vec);
    final_metrics.(algo_name).psnr_std = std(psnr_vec);
    final_metrics.(algo_name).ssim_mean = mean(ssim_vec);
    final_metrics.(algo_name).ssim_std = std(ssim_vec);
    final_metrics.(algo_name).time_mean = mean(time_vec);
    final_metrics.(algo_name).time_std = std(time_vec);
end

% --- Display the Final Average Results in a Table in the Command Window and a .txt File ---
fprintf('--------------------------------------------------------------------------------------\n');
fprintf('         Experiment 4: Color Image Inpainting - Average Performance Summary (Dataset: cbsd68)\n');
fprintf('--------------------------------------------------------------------------------------\n');
fprintf('%-15s | %-20s | %-20s | %-20s\n', ...
    'Algorithm', 'PSNR (Mean ± Std)', 'SSIM (Mean ± Std)', 'Time (s) (Mean ± Std)');
fprintf('--------------------------------------------------------------------------------------\n');

summary_txt_path = fullfile(results_dir_for_analysis, 'experiment_summary.txt');
fid = fopen(summary_txt_path, 'w');
fprintf(fid, '--------------------------------------------------------------------------------------\n');
fprintf(fid, '         Experiment 4: Color Image Inpainting - Average Performance Summary (Dataset: cbsd68)\n');
fprintf(fid, '--------------------------------------------------------------------------------------\n');
fprintf(fid, '%-15s | %-20s | %-20s | %-20s\n', ...
    'Algorithm', 'PSNR (Mean ± Std)', 'SSIM (Mean ± Std)', 'Time (s) (Mean ± Std)');
fprintf(fid, '--------------------------------------------------------------------------------------\n');

algo_names_sorted = fieldnames(final_metrics);
for i = 1:length(algo_names_sorted)
    algo_name = algo_names_sorted{i};
    metrics = final_metrics.(algo_name);

    psnr_str = sprintf('%.2f ± %.2f', metrics.psnr_mean, metrics.psnr_std);
    ssim_str = sprintf('%.4f ± %.4f', metrics.ssim_mean, metrics.ssim_std);
    time_str = sprintf('%.2f ± %.2f', metrics.time_mean, metrics.time_std);

    fprintf('%-15s | %-20s | %-20s | %-20s\n', algo_name, psnr_str, ssim_str, time_str);
    fprintf(fid, '%-15s | %-20s | %-20s | %-20s\n', algo_name, psnr_str, ssim_str, time_str);
end
fprintf('--------------------------------------------------------------------------------------\n\n');
fprintf(fid, '--------------------------------------------------------------------------------------\n\n');
fclose(fid);
fprintf('Average experiment performance has been saved to: %s\n\n', summary_txt_path);

fprintf('All analysis tasks are complete! The final average performance is displayed in the table above.\n');
fprintf('Detailed run data and restored color images are saved in: %s\n', results_dir_for_analysis);
end