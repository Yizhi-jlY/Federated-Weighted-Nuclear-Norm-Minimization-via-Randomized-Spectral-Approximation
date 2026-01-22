function display_communication_table(results_list)
% display_communication_table - Displays a summary table of communication costs in the command window.
%
%   Input:
%       results_list: A cell array where each cell contains a struct 
%                     with the results of a single algorithm run.
%                     Expected fields: .name, .total_communication, 
%                                      .relative_error, .total_time

fprintf('\n--- Communication Cost Summary Table ---\n');
fprintf('%-20s | %-25s | %-25s | %-15s\n', 'Algorithm', 'Total Communication (MB)', 'Final Relative Error', 'Total Time (s)');
fprintf(repmat('-', 1, 94));
fprintf('\n');

for i = 1:length(results_list)
    res = results_list{i};
    % Convert total communication from bytes to megabytes for display
    total_comm_mb = res.total_communication / (1024^2);
    
    fprintf('%-20s | %-25.2f | %-25.4e | %-15.2f\n', ...
        strrep(res.name, '_', '-'), total_comm_mb, res.relative_error, res.total_time);
end
fprintf('\n');
end