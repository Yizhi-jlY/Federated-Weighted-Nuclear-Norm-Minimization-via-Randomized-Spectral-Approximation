% File: load_all_results.m
function results_data = load_all_results(results_dir, run_flags, missing_rates, ranks, params)
% LOAD_ALL_RESULTS - Loads the final error values from all experiments into memory at once.
%
% Inputs:
%   results_dir   - The root directory containing the .mat result files.
%   run_flags     - A struct containing flags indicating which algorithms were run.
%   missing_rates - A vector of the missing rates used in the experiments.
%   ranks         - A vector of the matrix ranks used in the experiments.
%   params        - A parameters struct containing the number of Monte Carlo runs (mc).
%
% Outputs:
%   results_data  - A struct containing all the loaded data, with the following fields:
%                   .all_final_errors (4D-array): Stores the final error for each experiment.
%                   .executed_algos   (cell):     Algorithm names.
%                   .missing_rates    (vector):   Missing rates.
%                   .ranks            (vector):   Ranks.
%                   .mc_trials        (scalar):   Number of Monte Carlo trials.

fprintf('============================================================\n');
fprintf('                 Starting to load all experiment results...\n');
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

% Initialize a 4D matrix to store all final error values
% Dimensions: (missing_rate, rank, algorithm, monte_carlo_iteration)
% Initialize with NaN to be able to detect missing files
all_final_errors = NaN(num_mr, num_r, num_algos, params.mc);

% Start loading the data
total_files = num_mr * num_r * num_algos * params.mc;
loaded_count = 0;
fprintf('A total of %d result files need to be loaded.\n', total_files);

for mr_idx = 1:num_mr
    missing_rate = missing_rates(mr_idx);
    for r_idx = 1:num_r
        r = ranks(r_idx);
        for algo_idx = 1:num_algos
            algo_name = executed_algos{algo_idx};
            for mc_iter = 1:params.mc
                result_filename = fullfile(results_dir, sprintf('result_%s_mr%.1f_r%d_mc%d.mat', algo_name, missing_rate, r, mc_iter));
                
                if exist(result_filename, 'file')
                    loaded_data = load(result_filename, 'result'); % Load only the required variable
                    result = loaded_data.result;
                    
                    final_error = NaN; % Default to NaN
                    if isfield(result, 'relative_error')
                        final_error = result.relative_error;
                    elseif isfield(result, 'residuals') && ~isempty(result.residuals)
                        final_error = result.residuals(end);
                    else
                        warning('Algorithm %s (mr=%.1f, r=%d, mc=%d) has no valid error field and has been marked as NaN.', algo_name, missing_rate, r, mc_iter);
                    end
                    
                    all_final_errors(mr_idx, r_idx, algo_idx, mc_iter) = final_error;
                    loaded_count = loaded_count + 1;
                else
                    warning('File not found, marked as NaN: %s', result_filename);
                end
            end
        end
    end
end

fprintf('\nLoading complete! Successfully loaded %d / %d files.\n\n', loaded_count, total_files);

% Package all data into a struct to return
results_data.all_final_errors = all_final_errors;
results_data.executed_algos = executed_algos;
results_data.missing_rates = missing_rates;
results_data.ranks = ranks;
results_data.mc_trials = params.mc;

end