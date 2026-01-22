function plot_averaged_error_vs_communication(avg_results, std_results, save_dir)
% plot_averaged_error_vs_communication - Plots a scatter plot of average error vs. 
%                                        communication cost with error bars.
%
%   Inputs:
%       avg_results: Struct array containing the average results for each algorithm.
%                    Expected fields: .name, .comm_cost, .final_error
%       std_results: Struct array containing the standard deviations for each algorithm.
%                    Expected fields: .name, .comm_cost, .final_error
%       save_dir:    Directory path to save the figure.

    if isempty(avg_results)
        fprintf('No communication cost results to plot.\n');
        return;
    end

    % Create a new figure window
    h_fig = figure;
    set(h_fig, 'Position', [100, 100, 900, 600]);
    
    hold on; % Allow multiple plots on the same axes
    
    % Define markers and colors for different algorithms
    markers = {'o', 's', 'd', '^', 'v', '>', '<', 'p', 'h', '*'};
    colors = lines(length(avg_results));
    
    % Extract and organize data for plotting
    avg_costs = [avg_results.comm_cost];
    std_costs = [std_results.comm_cost];
    avg_errors = [avg_results.final_error];
    std_errors = [std_results.final_error];
    
    % Plot scatter points with error bars
    for i = 1:length(avg_results)
        % Use the errorbar function to plot points with error bars
        h_plot(i) = errorbar(avg_costs(i), avg_errors(i), ...
                             std_errors(i), std_errors(i), ...
                             std_costs(i), std_costs(i), ...
                             'o', 'LineWidth', 1.5, ...
                             'MarkerFaceColor', colors(i,:), ...
                             'MarkerEdgeColor', colors(i,:), ...
                             'MarkerSize', 8, ...
                             'Color', colors(i,:));
        
        % Set marker style
        set(h_plot(i), 'Marker', markers{mod(i-1, length(markers)) + 1});
    end
    
    % Set the title and labels for the plot
    title('Average NRMSE vs. Communication Cost', 'FontSize', 14);
    xlabel('Communication Cost (bytes)', 'FontSize', 12);
    ylabel('NRMSE', 'FontSize', 12);
    
    % Add a legend
    legend({avg_results.name}, 'Location', 'bestoutside', 'FontSize', 10);
    
    % Optimize axis display
    set(gca, 'XScale', 'log'); % Use a log scale for the x-axis as communication costs can vary widely
    grid on;
    box on;
    
    hold off;
    
    % Attempt to save the figure
    if ~exist(save_dir, 'dir')
        mkdir(save_dir);
    end
    
    filename = fullfile(save_dir, 'averaged_error_vs_communication.png');
    fprintf('Saving communication cost comparison plot to: %s\n', filename);
    saveas(h_fig, filename);
    
    fprintf('Communication cost comparison plot finished.\n');
end