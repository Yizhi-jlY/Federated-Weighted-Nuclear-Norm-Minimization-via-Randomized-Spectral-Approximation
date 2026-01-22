function display_results_table(results_list)
% display_results_table - Displays a summary of the final results in a table format in the command window.
%
% Input:
%   results_list - A cell array where each element is a struct of an algorithm's averaged results.

if isempty(results_list)
    disp('The results list is empty, cannot generate the table.');
    return;
end

fprintf('----------------------------------------------------------------------\n');
fprintf('                Experiment Results Summary (Averaged Performance)\n');
fprintf('----------------------------------------------------------------------\n');
fprintf('%-20s | %-12s | %-18s | %-12s\n', ...
    'Algorithm', 'Valid Runs', 'Avg Rel. Error', 'Avg Total Time (s)');
fprintf('----------------------------------------------------------------------\n');

for i = 1:length(results_list)
    res = results_list{i};

    % Access the aggregated average data fields from the main function
    avg_rel_error = res.avg_relative_error;
    avg_total_time = res.avg_total_time;
    num_runs = res.num_runs;

    fprintf('%-20s | %-12d | %-18.6f | %-12.4f\n', ...
        strrep(res.name, '_', ' '), num_runs, avg_rel_error, avg_total_time);
end

fprintf('----------------------------------------------------------------------\n\n');

end