function [avg_results, std_results] = aggregate_and_average_comm_cost(all_runs_results)
% aggregate_and_average_comm_cost - Aggregates and averages communication cost results 
%                                   from multiple Monte Carlo (MC) simulations.
%
%   Inputs:
%       all_runs_results: A cell array, where each cell contains the results of one MC run.
%                         Each cell is a struct array with results from multiple algorithms.
%
%   Outputs:
%       avg_results: Struct with the average results for all algorithms.
%       std_results: Struct with the standard deviation of results for all algorithms.
%
%   Result struct fields:
%       name:        Name of the algorithm.
%       comm_cost:   Average communication cost over all MC runs.
%       final_error: Average final error over all MC runs.

    % Check for empty input
    if isempty(all_runs_results)
        warning('Input results cell array is empty. Cannot aggregate.');
        avg_results = [];
        std_results = [];
        return;
    end

    % Use the first MC run as a baseline
    first_run_results = all_runs_results{1};
    num_algos = length(first_run_results);
    num_mc_runs = length(all_runs_results);
    
    % Initialize matrices to store all results
    % Dimensions: [algo_index, mc_run_index]
    comm_costs_matrix = zeros(num_algos, num_mc_runs);
    errors_matrix = zeros(num_algos, num_mc_runs);
    
    % Get algorithm names
    algo_names = cell(1, num_algos);
    for i = 1:num_algos
        algo_names{i} = first_run_results{i}.name;
    end

    % Loop over all MC runs to extract results
    for i = 1:num_mc_runs
        current_run_results = all_runs_results{i};
        
        % Ensure the number of algorithms is consistent across runs
        if length(current_run_results) ~= num_algos
            error('Mismatch in the number of algorithms in MC run %d. Check data integrity.', i);
        end
        
        % Loop over each algorithm to fill the matrices
        for j = 1:num_algos
            % Check for consistent algorithm order
            if ~strcmp(current_run_results{j}.name, algo_names{j})
                error('Inconsistent algorithm order in MC run %d.', i);
            end
            
            comm_costs_matrix(j, i) = current_run_results{j}.total_communication;
            errors_matrix(j, i) = current_run_results{j}.relative_error;
        end
    end
    
    % Initialize structs for average and standard deviation results
    avg_results = struct('name', cell(1, num_algos), 'comm_cost', zeros(1, num_algos), 'final_error', zeros(1, num_algos));
    std_results = struct('name', cell(1, num_algos), 'comm_cost', zeros(1, num_algos), 'final_error', zeros(1, num_algos));
    
    % Calculate mean and standard deviation for each algorithm
    for j = 1:num_algos
        avg_results(j).name = algo_names{j};
        avg_results(j).comm_cost = mean(comm_costs_matrix(j, :));
        avg_results(j).final_error = mean(errors_matrix(j, :));
        
        std_results(j).name = algo_names{j};
        std_results(j).comm_cost = std(comm_costs_matrix(j, :));
        std_results(j).final_error = std(errors_matrix(j, :));
    end

    fprintf('Successfully aggregated and averaged results from %d Monte Carlo simulations.\n', num_mc_runs);
end