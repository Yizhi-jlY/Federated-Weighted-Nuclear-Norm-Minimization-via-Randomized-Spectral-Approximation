function display_averaged_communication_table(avg_results, std_results)
% display_averaged_communication_table - Displays the averaged communication cost and error in a table format.
%
%   Inputs:
%       avg_results: Struct array containing the average results for each algorithm.
%                    Expected fields: .name, .comm_cost, .final_error
%       std_results: Struct array containing the standard deviations for each algorithm.
%                    Expected fields: .name, .comm_cost, .final_error

    if isempty(avg_results)
        fprintf('No communication cost results to display.\n');
        return;
    end

    fprintf('\n============================================================\n');
    fprintf('         Averaged Communication Cost Analysis (MC Simulations)\n');
    fprintf('============================================================\n');
    
    % Define table column widths and formats
    col1_width = 20; % Algorithm Name
    col2_width = 20; % Communication Cost
    col3_width = 20; % NRMSE Error
    
    % Print table header
    fprintf('%-*s | %-*s | %-*s\n', col1_width, 'Algorithm', col2_width, 'Comm. Cost (bytes)', col3_width, 'NRMSE');
    fprintf('%s\n', repmat('-', 1, col1_width + col2_width + col3_width + 6));
    
    % Print each row of results
    for i = 1:length(avg_results)
        algo_name = avg_results(i).name;
        avg_cost = avg_results(i).comm_cost;
        std_cost = std_results(i).comm_cost;
        avg_error = avg_results(i).final_error;
        std_error = std_results(i).final_error;
        
        % Format the cost and error strings to include standard deviation
        cost_str = sprintf('%.2e +/- %.2e', avg_cost, std_cost);
        error_str = sprintf('%.4f +/- %.4f', avg_error, std_error);
        
        fprintf('%-*s | %-*s | %-*s\n', col1_width, algo_name, col2_width, cost_str, col3_width, error_str);
    end
    
    fprintf('============================================================\n\n');

end