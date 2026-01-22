% File: load_full_results.m
function results_cache = load_full_results(results_dir, run_flags, missing_rates, ranks, params)
% LOAD_FULL_RESULTS - Loads the complete result structures from all experiments into memory.
%
%   This version differs from load_all_results in that it loads and stores
%   the complete 'result' struct from each .mat file, not just the final error.
%   This allows for more detailed subsequent analysis, such as plotting convergence curves.
%
% Inputs:
%   results_dir   - The root directory containing the .mat result files.
%   run_flags     - A struct containing flags indicating which algorithms were run.
%   missing_rates - A vector of the missing rates used in the experiments.
%   ranks         - A vector of the matrix ranks used in the experiments.
%   params        - A parameters struct containing the number of Monte Carlo runs (mc).
%
% Outputs:
%   results_cache - A struct where each field is an algorithm's name. Each field
%                   contains a {number_of_missing_rates x number_of_ranks} cell array.
%                   Each cell within this array is itself a cell array containing the
%                   'result' struct from every Monte Carlo run.
%                   Example: results_cache.FedWNNM_MC{mr_idx, r_idx}{mc_iter}

fprintf('============================================================\n');
fprintf('           Starting to load [complete] results for all experiments...\n');
fprintf('============================================================\n\n');

% Determine which algorithms were executed
algorithms_to_run = fieldnames(run_flags);
executed_algos = {};
for i = 1:length(algorithms_to_run)
    if run_flags.(algorithms_to_run{i})
        executed_algos{end+1} = algorithms_to_run{i};
    end
end
num_algos = length(executed_algos);
num_mr = length(missing_rates);
num_r = length(ranks);

% Initialize a struct to store all the results
results_cache = struct();
for algo_idx = 1:num_algos
    algo_name = executed_algos{algo_idx};
    % Create a {num_mr x num_r} cell array for each algorithm
    results_cache.(algo_name) = cell(num_mr, num_r);
end

% Start loading the data
total_files_to_check = num_mr * num_r * num_algos * params.mc;
loaded_count = 0;
fprintf('A total of %d potential result files will be checked.\n', total_files_to_check);

% ======================= [CODE CORRECTION] =======================
% Added 'Interpreter', 'none' to prevent TeX interpreter errors due to '_'
% characters in the filenames.
progress_bar = waitbar(0, 'Loading result files...', 'Interpreter', 'none');
% ===============================================================

for mr_idx = 1:num_mr
    missing_rate = missing_rates(mr_idx);
    for r_idx = 1:num_r
        r = ranks(r_idx);
        for algo_idx = 1:num_algos
            algo_name = executed_algos{algo_idx};
            
            % Initialize a cell array for the current (mr, r, algo) combination
            mc_results_list = cell(1, params.mc);
            
            for mc_iter = 1:params.mc
                result_filename = fullfile(results_dir, sprintf('result_%s_mr%.1f_r%d_mc%d.mat', algo_name, missing_rate, r, mc_iter));
                
                if exist(result_filename, 'file')
                    try
                        loaded_data = load(result_filename, 'result'); % Load only the 'result' variable
                        mc_results_list{mc_iter} = loaded_data.result; % Store the complete result struct
                        loaded_count = loaded_count + 1;
                    catch
                        warning('File is corrupt or could not be read, skipped: %s', result_filename);
                        mc_results_list{mc_iter} = []; % Mark as empty
                    end
                else
                    % A file not being found is a normal occurrence, so this warning
                    % can be suppressed to keep the command line clean.
                    % warning('File not found: %s', result_filename);
                    mc_results_list{mc_iter} = []; % Mark as empty
                end
                
                % Update the progress bar
                current_progress = ((mr_idx-1)*num_r*num_algos*params.mc + ...
                                    (r_idx-1)*num_algos*params.mc + ...
                                    (algo_idx-1)*params.mc + mc_iter) / total_files_to_check;
                
                % When updating the waitbar, check if the handle is still valid
                % (in case the user manually closed the window).
                if ishandle(progress_bar)
                    waitbar(current_progress, progress_bar, sprintf('Loading: %s', result_filename));
                else
                    fprintf('\nProgress bar window was closed. Loading continues in the background...\n');
                    % Create a flag to prevent this message from repeating.
                    if ~exist('progress_bar_closed', 'var')
                        progress_bar_closed = true;
                    end
                end
            end
            
            % Store this list of Monte Carlo results in the main cache
            results_cache.(algo_name){mr_idx, r_idx} = mc_results_list;
        end
    end
end

if ishandle(progress_bar)
    close(progress_bar);
end
fprintf('\nLoading complete! Successfully loaded %d files.\n\n', loaded_count);

end