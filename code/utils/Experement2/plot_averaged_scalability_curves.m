function plot_averaged_scalability_curves(scalability_stats, save_dir)
% PLOT_AVERAGED_SCALABILITY_CURVES Plots the averaged results of the scalability analysis.
%
%   Usage:
%       plot_averaged_scalability_curves(scalability_stats, save_dir)
%
%   Inputs:
%       scalability_stats - A struct where each field is an algorithm name. Each field 
%                           contains another struct with: p_range, nrmse_mean, nrmse_std,
%                           time_mean, and time_std.
%       save_dir          - A string representing the directory path to save the plots.
%
%   Description:
%       This function generates two plots:
%       1. Final NRMSE vs. Number of Clients
%       2. Total Runtime vs. Number of Clients
%       Each plot shows the average performance curves for all algorithms, with shaded 
%       areas representing the standard deviation. The plots are automatically saved
%       to the specified directory.

% Get algorithm list and plotting styles
algo_names = fieldnames(scalability_stats);
if isempty(algo_names)
    fprintf('No scalability data available for plotting.\n');
    return;
end
num_algos = length(algo_names);
colors = lines(num_algos); % Generate a unique color for each algorithm
markers = {'-o', '-s', '-d', '-^', '-v', '-x', '-+'}; % Define marker styles
legend_names = strrep(algo_names, '_', ' '); % Replace underscores with spaces for the legend

% --- Plot 1: NRMSE vs. Number of Clients ---
h_fig1 = figure('Name', 'Scalability: NRMSE vs. Clients', 'Position', [100, 100, 800, 600], 'Visible', 'on');
hold on;
plot_handles_nrmse = [];

for i = 1:num_algos
    algo_name = algo_names{i};
    data = scalability_stats.(algo_name);
    
    p = data.p_range;
    y_mean = data.nrmse_mean;
    y_std = data.nrmse_std;
    
    % Create coordinates (x and y) for the standard deviation area
    x_fill = [p, fliplr(p)];
    y_fill = [y_mean - y_std, fliplr(y_mean + y_std)];
    
    % Plot a semi-transparent shaded area for the standard deviation
    fill(x_fill, y_fill, colors(i,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none');
    
    % Plot the mean curve on top of the shaded area
    h = plot(p, y_mean, markers{mod(i-1, length(markers))+1}, ...
             'Color', colors(i,:), 'LineWidth', 1.5, 'MarkerSize', 8);
    plot_handles_nrmse(end+1) = h;
end

% Configure plot properties
title('Scalability Analysis: NRMSE vs. Number of Clients', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Number of Clients (p)', 'FontSize', 12);
ylabel('Final NRMSE (mean ± std)', 'FontSize', 12);
legend(plot_handles_nrmse, legend_names, 'Location', 'northwest', 'FontSize', 10);
grid on;
set(gca, 'FontSize', 12, 'XScale', 'log', 'XTick', scalability_stats.(algo_names{1}).p_range); % Use log scale for X-axis and set specific ticks
hold off;

% Save the figure
nrmse_filename = fullfile(save_dir, 'scalability_nrmse_vs_clients');
fprintf('Saving NRMSE scalability plot to: %s.(png/fig)\n', nrmse_filename);
saveas(h_fig1, [nrmse_filename, '.png']);
savefig(h_fig1, [nrmse_filename, '.fig']);


% --- Plot 2: Runtime vs. Number of Clients ---
h_fig2 = figure('Name', 'Scalability: Time vs. Clients', 'Position', [950, 100, 800, 600], 'Visible', 'on');
hold on;
plot_handles_time = [];

for i = 1:num_algos
    algo_name = algo_names{i};
    data = scalability_stats.(algo_name);
    
    p = data.p_range;
    y_mean = data.time_mean;
    y_std = data.time_std;
    
    % Create coordinates (x and y) for the standard deviation area
    x_fill = [p, fliplr(p)];
    y_fill = [y_mean - y_std, fliplr(y_mean + y_std)];
    
    % Plot a semi-transparent shaded area for the standard deviation
    fill(x_fill, y_fill, colors(i,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none');
    
    % Plot the mean curve on top of the shaded area
    h = plot(p, y_mean, markers{mod(i-1, length(markers))+1}, ...
             'Color', colors(i,:), 'LineWidth', 1.5, 'MarkerSize', 8);
    plot_handles_time(end+1) = h;
end

% Configure plot properties
title('Scalability Analysis: Runtime vs. Number of Clients', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Number of Clients (p)', 'FontSize', 12);
ylabel('Total Runtime (s) (mean ± std)', 'FontSize', 12);
legend(plot_handles_time, legend_names, 'Location', 'northwest', 'FontSize', 10);
grid on;
set(gca, 'FontSize', 12, 'XScale', 'log', 'XTick', scalability_stats.(algo_names{1}).p_range); % Use log scale for X-axis
hold off;

% Save the figure
time_filename = fullfile(save_dir, 'scalability_time_vs_clients');
fprintf('Saving Runtime scalability plot to: %s.(png/fig)\n', time_filename);
saveas(h_fig2, [time_filename, '.png']);
savefig(h_fig2, [time_filename, '.fig']);

end