function plot_error_vs_communication(results_list, save_dir)
% Plots the relative error vs. total communication volume curve.
figure('Name', 'Error vs Communication', 'NumberTitle', 'off', 'Position', [100, 100, 800, 600]);
hold on;
grid on;

legends = {};
for i = 1:length(results_list)
    res = results_list{i};
    if isfield(res, 'communication_volumes') && ~isempty(res.communication_volumes) && res.total_communication > 0
        % Convert cumulative communication from Bytes to Megabytes
        comm_cumulative_mb = cumsum(res.communication_volumes) / (1024^2);
        
        % Use relative error history; if not available, use residuals.
        if isfield(res, 'relative_error_trajectory') && ~isempty(res.relative_error_trajectory)
            error_hist = res.relative_error_trajectory;
        else
            error_hist = exp(res.residuals); % Residuals are in log form.
        end
        
        % Ensure dimensions match
        len = min(length(comm_cumulative_mb), length(error_hist));
        plot(comm_cumulative_mb(1:len), log10(error_hist(1:len)), 'LineWidth', 2, 'Marker', 'o', 'MarkerSize', 4);
        legends{end+1} = strrep(res.name, '_', '-');
    end
end

title('Convergence vs. Communication Cost');
xlabel('Total Communication Volume (MB)');
ylabel('Log_{10}(Relative Recovery Error)');
legend(legends, 'Location', 'northeast');
hold off;

filename = fullfile(save_dir, 'plot_error_vs_communication.png');
saveas(gcf, filename);
fprintf('Communication cost convergence plot saved to: %s\n', filename);
end