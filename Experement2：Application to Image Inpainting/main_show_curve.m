%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                 Main Function: Real-World Dataset Application (Experiment 4 - Image Inpainting)
%                               [Batch Processing & Mean Analysis Version]
%
%   Description:
%         This script performs image inpainting experiments on all images in the cbsd68 dataset.
%         It iterates through each image, applies an artificial mask, and then runs a series
%         of federated and centralized matrix completion algorithms for recovery.
%         Finally, the script calculates and tabulates the average performance metrics
%         (PSNR, SSIM, etc.) for each algorithm across the entire dataset.
%         [New Feature]: Generates a separate convergence curve comparison plot for each processed image.
%
%   Supported Modes:
%         1. 'run_and_analyze':    Run a new experiment and analyze its results.
%         2. 'analyze_only':         Only load and analyze past results from a specified folder (calculates averages).
%         3. 'replot_only':          [New] Only regenerate convergence curves for each image from the latest past results.
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1. Environment Initialization
clc;
clear;
close all;
fprintf('============================================================\n');
fprintf('    Experiment 4: Real-World Dataset Application (Image Inpainting) - Batch Processing Version\n');
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
% Options: 'run_and_analyze', 'analyze_only', 'replot_only'
ANALYSIS_MODE = 'analyze_only'; 

% --- For 'analyze_only' mode, specify the target results folder ---
% TARGET_RESULTS_DIR = 'C:\FR_wnnm\results\Experiment4_ImageInpainting_Batch_Summary\missing_rate_0.7\20250916_192050';
TARGET_RESULTS_DIR = 'C:\Users\22920\Desktop\FedWNNM\代码\Supplementary Materials\Experement2：Application to Image Inpainting\Experiment4_ImageInpainting_Batch_Color\missing_rate_0.7';

% --- [New] For 'replot_only' mode, specify the root directory containing all missing rate folders ---
BASE_RESULTS_PATH = './Experiment4_ImageInpainting_Batch_Color';


% --- Loop through different missing rates ---
% When you only want to replot, you can also set this to a specific missing rate you care about, e.g., for ms_rate = [0.7]
for ms_rate = [ 0.7 0.5 0.3]

% -------------------- Dataset and Mask Type Configuration --------------------
% Assuming the 'datasets' folder is at the same level as the 'code' folder
params.dataset_path = fullfile(codeFolderPath, '..', 'datasets', 'cbsd68t');
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
fedwnnm_param_table = containers.Map(...
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], ... % Keys: missing_rate
    [4.0047, 4.0012, 4.0010, 11.1182, 27.4829, 45.8091, 49.9929, 49.9944, 49.9995] ... % Values: C_scaler
);
params.fedwnnm_param_table = fedwnnm_param_table;


%% 4. Algorithm Execution Switches (1: Run, 0: Skip) - Effective only in 'run_and_analyze' mode
run_flags.FedWNNM_MC     = 1;
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
    results_dir = fullfile(BASE_RESULTS_PATH, sprintf('missing_rate_%.1f', params.missing_rate), timestamp);
    if ~exist(results_dir, 'dir')
        mkdir(results_dir);
    end
    fprintf('All new results will be saved in: %s\n\n', results_dir);

    % --- 6. Get All Image Files from the Dataset ---
    image_files = dir(fullfile(params.dataset_path, '*.png'));
    if isempty(image_files)
        error('No .png image files found in the path: %s.', params.dataset_path);
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
            all_run_results.(algo_name) = {};
        end
    end
    fprintf('\n');

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

        image_full_path = fullfile(params.dataset_path, current_image_info.name);
        L_true_rgb = imread(image_full_path);
        if size(L_true_rgb, 3) == 3, L_true = im2double(rgb2gray(L_true_rgb)); else, L_true = im2double(L_true_rgb); end
        [m, n] = size(L_true);
        current_params = params; current_params.m = m; current_params.n = n;
        mask = ones(m, n);
        if strcmpi(params.mask_type, 'random')
            num_missing = round(m * n * params.missing_rate);
            omega_missing = randperm(m * n, num_missing);
            mask(omega_missing) = 0;
        end
        data = L_true .* mask;
        current_params.p_obs = mean(mask(:));
        [U_true_orth, S_true_diag, ~] = svd(L_true, 'econ');
        U_true = U_true_orth(:, 1:params.r);
        S_true = diag(S_true_diag); S_true = S_true(1:params.r);
        
        single_image_results = struct();

        % --- Run All Selected Algorithms ---
        for algo_idx = 1:length(algorithm_execution_order)
            algo_name = algorithm_execution_order{algo_idx};
            if isfield(run_flags, algo_name) && run_flags.(algo_name)
                
                folder_name = algorithm_map(algo_name);
                algo_path = fullfile(codeFolderPath, folder_name);
                addpath(genpath(algo_path));
                
                fprintf('--- [%s] Running... ---\n', algo_name);
                
                parameters = current_params;
                parameters.L_true = L_true; parameters.U_true = U_true; parameters.S_true = S_true;
                
                switch algo_name
                    case 'factGDNew', parameters.step_const = 0.75; parameters.Tsvd = 15;
                    case 'AltGD', parameters.rank = params.r; parameters.step_const = 0.75;
                    case 'altGDMinCntrl_T', parameters.r = params.r; parameters.eta_c = 1.0; parameters.p_obs = current_params.p_obs;
                    case 'altGDMin_T', parameters.r = params.r; parameters.eta_c = 1.0; parameters.Tsvd = 15;
                    case 'altMinCntrl_T', parameters.T = params.maxiter; parameters.Tsvd = 15;
                    case 'altMinParfor_T', parameters.Tsvd = 15; parameters.p_obs = current_params.p_obs;
                    case 'altMinPrvt_T', parameters.rank = params.r; parameters.T_inner = 10; parameters.Tsvd = 15;
                    case 'SVT', parameters.tao = 2.5 * sqrt(m * n); parameters.step = 1.2;
                    case 'FedSVT_MC', parameters.tau =  sqrt(m * n)/3133.03; parameters.delta0 = 1.9992; parameters.gamma = 0.92647;
                    case 'WNNM_MC', parameters.C = sqrt(max(m, n))/4.381; parameters.myeps = 1e-6;
                    case 'FedWNNM_MC'
                        current_rate_key = round(params.missing_rate, 1); 
                        if isKey(parameters.fedwnnm_param_table, current_rate_key)
                            selected_C = parameters.fedwnnm_param_table(current_rate_key);
                            fprintf('  > FedWNNM_MC: For missing rate %.1f, selected preset optimal C = %.4f\n', current_rate_key, selected_C);
                            parameters.C = selected_C;
                        else
                            default_C = sqrt(max(m, n)) / 4.381;
                            fprintf('  > FedWNNM_MC: Warning! Missing rate %.2f not in preset list, using default C = %.4f\n', params.missing_rate, default_C);
                            parameters.C = default_C;
                        end
                        parameters.myeps = 1e-6;
                end
                parameters.maxiter = params.maxiter;
                
                switch algo_name
                    case 'factGDNew',       result = factGDNew(data, mask, parameters);
                    case 'AltGD',           result = AltGD(data, mask, parameters);
                    case 'altGDMinCntrl_T', result = altGDMinCntrl_T(data, mask, parameters);
                    case 'altGDMin_T',      result = altGDMin_T(data, mask, parameters);
                    case 'altMinCntrl_T',   result = altMinCntrl_T(data, mask, parameters);
                    case 'altMinParfor_T',  result = altMinParfor_T(data, mask, parameters);
                    case 'altMinPrvt_T',    result = altMinPrvt_T(data, mask, parameters);
                    case 'SVT',             result = SVT(data, mask, parameters);
                    case 'FedSVT_MC',       result = FedSVT_MC(data, mask, parameters);
                    case 'WNNM_MC',         result = WNNM_MC(data, mask, parameters);
                    case 'FedWNNM_MC',      result = FedWNNM_MC(data, mask, parameters);
                end
                
                fprintf('--- [%s] Completed. Time: %.2f s, PSNR: %.2f, SSIM: %.4f\n', ...
                    algo_name, result.total_time, result.psnr_value, result.ssim_value);
                
                all_run_results.(algo_name){end+1} = result;
                single_image_results.(algo_name) = result;
                
                rmpath(genpath(algo_path));
            end
        end
        fprintf('\nImage %s processing complete.\n\n', current_image_info.name);

        single_result_filename = sprintf('result_img%03d_%s.mat', img_idx, current_image_info.name);
        single_result_filepath = fullfile(results_dir, single_result_filename);
        save(single_result_filepath, 'single_image_results', 'current_image_info', 'params');
        fprintf('All algorithm results for this image have been saved to: %s\n', single_result_filepath);

        fprintf('>>> Generating convergence curve plot for image "%s"...\n', current_image_info.name);
        results_list_for_plot = {};
        executed_algos_on_this_image = fieldnames(single_image_results);
        for i = 1:length(executed_algos_on_this_image)
            algo_name = executed_algos_on_this_image{i};
            result_struct = single_image_results.(algo_name);
            if isfield(result_struct, 'residuals') && isfield(result_struct, 'wall_clock_times')
                result_struct.name = algo_name;
                average_results = struct();
                average_results.avg_residuals = result_struct.residuals;
                average_results.name = algo_name;
                results_list_for_plot{end+1} = average_results;
            else
                fprintf('  -> Warning: Algorithm %s is missing convergence data and will not be plotted.\n', algo_name);
            end
        end
        if ~isempty(results_list_for_plot)
            image_name_no_ext = extractBefore(current_image_info.name, '.png');
            plot_subdir = fullfile(results_dir, sprintf('plots_img_%03d_%s', img_idx, image_name_no_ext));
            if ~exist(plot_subdir, 'dir'), mkdir(plot_subdir); end
            plot_convergence_curves_4(results_list_for_plot, plot_subdir);
            fprintf('  -> Convergence curve plot successfully generated and saved to:\n     %s\n\n', plot_subdir);
        else
            fprintf('  -> No convergence plot was generated for this image due to a lack of valid convergence data.\n\n');
        end

    end % --- End of image loop ---
    
    fprintf('All images processed. Saving aggregated results...\n');
    full_results_filename = fullfile(results_dir, 'full_experiment_results.mat');
    save(full_results_filename, 'all_run_results', 'params', '-v7.3');
    fprintf('Aggregated results saved to: %s\n\n', full_results_filename);
    
end

%% 8. [New] Regenerate Plots from Existing Results (Only in 'replot_only' mode)
if strcmpi(ANALYSIS_MODE, 'replot_only')
    fprintf('============================================================\n');
    fprintf('       Mode: replot_only - Regenerating convergence curves from existing results\n');
    fprintf('============================================================\n\n');
    
    % --- 1. Locate the folder for the current missing rate ---
    current_rate_path = fullfile(BASE_RESULTS_PATH, sprintf('missing_rate_%.1f', ms_rate));
    if ~exist(current_rate_path, 'dir')
        fprintf('Warning: Path %s not found, skipping current missing rate.\n\n', current_rate_path);
        continue; % Skip to the next ms_rate in the for loop
    end
    
    % --- 2. Automatically find the latest experiment results subfolder ---
    all_subdirs = dir(current_rate_path);
    dir_flags = [all_subdirs.isdir] & ~ismember({all_subdirs.name}, {'.', '..'});
    experiment_folders = all_subdirs(dir_flags);
    
    if isempty(experiment_folders)
        fprintf('Warning: No experiment result folders found in %s, skipping.\n\n', current_rate_path);
        continue;
    end
    
    % Sort by date to find the newest folder
    [~, sorted_idx] = sort([experiment_folders.datenum], 'descend');
    latest_folder_name = experiment_folders(sorted_idx(1)).name;
    results_dir_to_plot = fullfile(current_rate_path, latest_folder_name);
    
    fprintf('Automatically selected the latest results folder for plotting:\n  %s\n\n', results_dir_to_plot);

    % --- 3. Find all single-image result .mat files in this folder ---
    result_files = dir(fullfile(results_dir_to_plot, 'result_img*.mat'));
    if isempty(result_files)
        fprintf('Warning: No result files in "result_img*.mat" format found in %s.\n\n', results_dir_to_plot);
        continue;
    end
    
    fprintf('Found %d image result files. Starting to generate convergence curves for each...\n\n', length(result_files));
    
    % --- 4. Loop through each .mat file, load it, and execute plotting ---
    for file_idx = 1:length(result_files)
        mat_file_info = result_files(file_idx);
        mat_file_path = fullfile(results_dir_to_plot, mat_file_info.name);
        
        fprintf('------------------------------------------------------------\n');
        fprintf('>>> Loading: %s\n', mat_file_info.name);
        
        % Load variables from the .mat file
        loaded_data = load(mat_file_path);
        
        % Check if necessary variables exist
        if ~isfield(loaded_data, 'single_image_results') || ~isfield(loaded_data, 'current_image_info')
            fprintf('  -> Warning: This .mat file is missing necessary variables (single_image_results or current_image_info), skipping.\n');
            continue;
        end
        
        single_image_results = loaded_data.single_image_results;
        current_image_info = loaded_data.current_image_info;
        
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        %               [[[ CORE PLOTTING LOGIC: REPRODUCED HERE ]]]
        %
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        fprintf('>>> Generating convergence curve plot for image "%s"...\n', current_image_info.name);
        
        % 1. Prepare data in the format required by the plotting function (cell array)
        results_list_for_plot = {};
        executed_algos_on_this_image = fieldnames(single_image_results);
        
        for i = 1:length(executed_algos_on_this_image)
            algo_name = executed_algos_on_this_image{i};
            result_struct = single_image_results.(algo_name);
            
            % Ensure the result contains the data needed for plotting
            if isfield(result_struct, 'residuals') % && isfield(result_struct, 'wall_clock_times')
                average_results = struct();
                average_results.residuals = result_struct.residuals;
                average_results.name = algo_name;
                average_results.wall_clock_times = result_struct.wall_clock_times;
                % Note: To match the input format of the plot_convergence_curves function,
                % we use the original residuals and wall_clock_times directly.
                % The naming might use avg_... to be consistent with a potential
                % structure in the original code.
                
                results_list_for_plot{end+1} = average_results;
            else
                fprintf('  -> Warning: Algorithm %s is missing convergence data and will not be plotted.\n', algo_name);
            end
        end
        
        % 2. Call the plotting function
        if ~isempty(results_list_for_plot)
            % Extract filename without extension from the image info
            [~, image_name_no_ext, ~] = fileparts(current_image_info.name);
            
            % Extract image index number (e.g., '003' from 'result_img003_...')
            img_idx_str = regexp(mat_file_info.name, 'img(\d+)', 'tokens');
            if ~isempty(img_idx_str)
                img_idx = str2double(img_idx_str{1}{1});
            else
                img_idx = file_idx; % If matching fails, use the file order as index
            end

            plot_subdir = fullfile(results_dir_to_plot, sprintf('plots_img_%03d_%s', img_idx, image_name_no_ext));
            if ~exist(plot_subdir, 'dir')
                mkdir(plot_subdir);
            end
            
            % Assuming plot_convergence_curves_4 is on the MATLAB path
            plot_convergence_curves_4(results_list_for_plot, plot_subdir);
            fprintf('  -> Convergence curve plot successfully generated and saved to:\n     %s\n\n', plot_subdir);
        else
            fprintf('  -> No convergence plot was generated for this image due to a lack of valid convergence data.\n\n');
        end
    end % --- End of .mat file loop ---
    
    fprintf('All image convergence curves for missing rate %.1f have been generated.\n\n', ms_rate);
end


%% 9. Aggregate Results, Calculate Averages, and Display (in 'run_and_analyze' and 'analyze_only' modes)
if strcmpi(ANALYSIS_MODE, 'run_and_analyze') || strcmpi(ANALYSIS_MODE, 'analyze_only')
    fprintf('============================================================\n');
    fprintf('               Starting to aggregate results and calculate average performance across all images\n');
    fprintf('============================================================\n\n');

    if strcmpi(ANALYSIS_MODE, 'run_and_analyze')
        results_dir_for_analysis = results_dir; % Use the newly generated folder
    else % 'analyze_only'
        results_dir_for_analysis = TARGET_RESULTS_DIR; % Use the manually specified folder
    end
    
    fprintf('Analyzing folder: %s\n', results_dir_for_analysis);

    full_results_file = fullfile(results_dir_for_analysis, 'full_experiment_results.mat');
    if ~exist(full_results_file, 'file'), error('full_experiment_results.mat not found in the specified path'); end
    load(full_results_file, 'all_run_results');
    fprintf('Successfully loaded aggregated results file.\n\n');

    executed_algos = fieldnames(all_run_results);
    final_metrics = struct();
    for i = 1:length(executed_algos)
        algo_name = executed_algos{i};
        results_for_algo = all_run_results.(algo_name);
        if isempty(results_for_algo), continue; end
        num_runs = length(results_for_algo);
        psnr_vec=zeros(1,num_runs); ssim_vec=zeros(1,num_runs); nrmse_vec=zeros(1,num_runs); time_vec=zeros(1,num_runs);
        for j = 1:num_runs
            psnr_vec(j) = results_for_algo{j}.psnr_value;
            ssim_vec(j) = results_for_algo{j}.ssim_value;
            if isfield(results_for_algo{j}, 'final_nrmse'), nrmse_vec(j) = results_for_algo{j}.final_nrmse;
            elseif isfield(results_for_algo{j}, 'relative_error'), nrmse_vec(j) = results_for_algo{j}.relative_error; end
            time_vec(j) = results_for_algo{j}.total_time;
        end
        final_metrics.(algo_name).psnr_mean = mean(psnr_vec); final_metrics.(algo_name).psnr_std = std(psnr_vec);
        final_metrics.(algo_name).ssim_mean = mean(ssim_vec); final_metrics.(algo_name).ssim_std = std(ssim_vec);
        final_metrics.(algo_name).nrmse_mean = mean(nrmse_vec); final_metrics.(algo_name).nrmse_std = std(nrmse_vec);
        final_metrics.(algo_name).time_mean = mean(time_vec); final_metrics.(algo_name).time_std = std(time_vec);
    end

    summary_txt_path = fullfile(results_dir_for_analysis, 'experiment_summary.txt');
    fid = fopen(summary_txt_path, 'w');
    fprintf(fid, '--------------------------------------------------------------------------------------------------\n');
    fprintf(fid, '                     Experiment 4: Image Inpainting - Average Performance Summary (Dataset: cbsd68)\n');
    fprintf(fid, '--------------------------------------------------------------------------------------------------\n');
    fprintf(fid, '%-15s | %-20s | %-20s | %-20s | %-20s\n', 'Algorithm', 'PSNR (Mean ± Std)', 'SSIM (Mean ± Std)', 'NRMSE (Mean ± Std)', 'Time (s) (Mean ± Std)');
    fprintf(fid, '--------------------------------------------------------------------------------------------------\n');
    fprintf('--------------------------------------------------------------------------------------------------\n');
    fprintf('                     Experiment 4: Image Inpainting - Average Performance Summary (Dataset: cbsd68)\n');
    fprintf('--------------------------------------------------------------------------------------------------\n');
    fprintf('%-15s | %-20s | %-20s | %-20s | %-20s\n', 'Algorithm', 'PSNR (Mean ± Std)', 'SSIM (Mean ± Std)', 'NRMSE (Mean ± Std)', 'Time (s) (Mean ± Std)');
    fprintf('--------------------------------------------------------------------------------------------------\n');

    algo_names_sorted = fieldnames(final_metrics);
    for i = 1:length(algo_names_sorted)
        algo_name = algo_names_sorted{i}; metrics = final_metrics.(algo_name);
        psnr_str = sprintf('%.2f ± %.2f', metrics.psnr_mean, metrics.psnr_std);
        ssim_str = sprintf('%.4f ± %.4f', metrics.ssim_mean, metrics.ssim_std);
        nrmse_str = sprintf('%.4f ± %.4f', metrics.nrmse_mean, metrics.nrmse_std);
        time_str = sprintf('%.2f ± %.2f', metrics.time_mean, metrics.time_std);
        fprintf('%-15s | %-20s | %-20s | %-20s | %-20s\n', algo_name, psnr_str, ssim_str, nrmse_str, time_str);
        fprintf(fid, '%-15s | %-20s | %-20s | %-20s | %-20s\n', algo_name, psnr_str, ssim_str, nrmse_str, time_str);
    end
    fprintf('--------------------------------------------------------------------------------------------------\n\n');
    fprintf(fid, '--------------------------------------------------------------------------------------------------\n\n');
    fclose(fid);
    fprintf('Average experiment performance has been saved to: %s\n\n', summary_txt_path);
    fprintf('Analysis tasks complete!\n');
end

% --- End of Loop Marker ---
fprintf('============================================================\n');
fprintf('       All tasks for missing rate %.1f have been completed.\n', ms_rate);
fprintf('============================================================\n\n');

end % --- End of missing rate for loop ---