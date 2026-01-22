function plot_convergence_curves(results_list, save_dir, fontSizeOptions)
% plot_convergence_curves - Plots convergence curves for multiple algorithms (Enhanced Version).
%
% This version enhances visual distinction and provides detailed font size control, 
% making it suitable for academic publications.
%
% Inputs:
%   results_list    - A cell array, where each element is a result struct from an algorithm.
%   save_dir        - The directory path to save the figures.
%   fontSizeOptions - (Optional) A struct to control font sizes, with the following fields:
%                     .title  - Title font size (default: 16)
%                     .labels - Axis label font size (default: 14)
%                     .ticks  - Axis tick font size (default: 12)
%                     .legend - Legend font size (default: 12)
%
% Outputs:
%   Saves two high-quality convergence curve plots to the specified directory.

% --- Input Parameter Checks ---
if nargin < 2
    error('A results list and a save directory must be provided.');
end
if isempty(results_list)
    disp('The results list is empty. Cannot create plots.');
    return;
end

% --- Set Font Sizes (use defaults if not provided) ---
if nargin < 3 || isempty(fontSizeOptions)
    fontSizeOptions = struct(...
        'title', 16, ...
        'labels', 24, ...
        'ticks', 16, ...
        'legend', 18 ...
    );
end

fprintf('Generating convergence curve plots...\n');

% --- Define Plotting Colors and Styles (optimized for high contrast and distinction) ---
% 1. Define a set of visually distinct colors (RGB triplets)
colors = [
    0, 0.4470, 0.7410;      % Blue
    0.8500, 0.3250, 0.0980;  % Orange
    0.9290, 0.6940, 0.1250;  % Yellow
    0.4940, 0.1840, 0.5560;  % Purple
    0.4660, 0.6740, 0.1880;  % Green
    0.3010, 0.7450, 0.9330;  % Cyan
    0.6350, 0.0780, 0.1840;  % Maroon
    0, 0, 0;                 % Black
];

% 2. Define different line styles
lineStyles = {'-', '--', ':', '-.'};
lineStyles = {'-'};

% 3. Define different markers
markers = {'o', 's', 'd', '^', 'v', 'p', 'h', '*'};
markers = {'none'};


%% 1. Relative Error vs. Wall-Clock Time (log-linear scale)
h1 = figure('Name', 'Convergence: Error vs. Time', 'NumberTitle', 'off', 'Position', [100 100 800 600]);
hold on;
legend_entries = {};

for i = 1:length(results_list)
    res = results_list{i};
    
    % Get the color, line style, and marker for the current loop
    color = colors(mod(i-1, size(colors, 1)) + 1, :);
    lineStyle = lineStyles{mod(i-1, length(lineStyles)) + 1};
    marker = markers{mod(i-1, length(markers)) + 1};
    
    % Determine the data for plotting
    if isfield(res, 'wall_clock_times') && isfield(res, 'relative_error_trajectory')
        x_data = res.wall_clock_times;
        y_data = res.relative_error_trajectory;
        legend_entries{end+1} = strrep(res.name, '_', '\_');
    elseif isfield(res, 'avg_wall_clock_times') && isfield(res, 'avg_residuals')
        x_data = res.avg_wall_clock_times;
        y_data = res.avg_residuals;
        legend_entries{end+1} = strrep(res.name, '_', '\_');
    else
        continue; % Skip if no data is available
    end
    
    % To avoid overly dense markers, sample the points for plotting
    num_points = length(x_data);
    marker_indices = unique(round(linspace(1, num_points, 15))); % Display at most 15 markers
    
    % Plot the log-scale curve combining lines and markers
    semilogy(x_data, y_data, ...
        'LineStyle', lineStyle, ...
        'Marker', marker, ...
        'MarkerIndices', marker_indices, ...
        'MarkerEdgeColor', color, ...
        'MarkerFaceColor', 'none', ... % Can be set to 'color' to fill it
        'MarkerSize', 7, ...
        'Color', color, ...
        'LineWidth', 1.5);
end
hold off;
grid on;

% --- Apply fonts and labels ---
%title('Convergence: Relative Error vs. Wall-Clock Time', 'FontSize', fontSizeOptions.title);
xlabel('Wall-Clock Time (seconds)', 'FontSize', fontSizeOptions.labels);
ylabel('Relative Error (log scale)', 'FontSize', fontSizeOptions.labels);
legend(legend_entries, 'Location', 'best', 'Interpreter', 'tex', 'FontSize', fontSizeOptions.legend);
set(gca, 'FontSize', fontSizeOptions.ticks, 'YMinorGrid', 'on');

% --- Save the figure ---
fig_filename_time = fullfile(save_dir, 'convergence_vs_time.png');
saveas(h1, fig_filename_time);
% % --- Recommended: Use the print function to save higher quality images for papers ---
print(h1, fullfile(save_dir, 'convergence_vs_time.eps'), '-depsc'); % Save as an EPS vector graphic
print(h1, fullfile(save_dir, 'convergence_vs_time_high_res.png'), '-dpng', '-r300'); % Save as a 300 DPI PNG

fprintf('Convergence curve plot (vs. Time) saved to: %s\n', fig_filename_time);


%% 2. Relative Error vs. Iterations (log-linear scale)
h2 = figure('Name', 'Convergence: Error vs. Iterations', 'NumberTitle', 'off', 'Position', [950 100 800 600]);
hold on;
legend_entries = {}; % Reset legend entries

for i = 1:length(results_list)
    res = results_list{i};
    
    % Get the color, line style, and marker for the current loop
    color = colors(mod(i-1, size(colors, 1)) + 1, :);
    lineStyle = lineStyles{mod(i-1, length(lineStyles)) + 1};
    marker = markers{mod(i-1, length(markers)) + 1};
    
    % Determine the data for plotting
    if isfield(res, 'relative_error_trajectory')
        y_data = res.relative_error_trajectory;
        legend_entries{end+1} = strrep(res.name, '_', '\_');
    elseif isfield(res, 'avg_residuals')
        y_data = res.avg_residuals;
        legend_entries{end+1} = strrep(res.name, '_', '\_');
    else
        continue; % Skip if no data is available
    end
    
    iters = 1:length(y_data);
    
    % To avoid overly dense markers, sample the points for plotting
    num_points = length(iters);
    marker_indices = unique(round(linspace(1, num_points, 15))); % Display at most 15 markers
    
    % Use semilogy to directly plot the log-scale graph, combining lines and markers
    semilogy(iters, y_data, ...
        'LineStyle', lineStyle, ...
        'Marker', marker, ...
        'MarkerIndices', marker_indices, ...
        'MarkerEdgeColor', color, ...
        'MarkerFaceColor', 'none', ...
        'MarkerSize', 7, ...
        'Color', color, ...
        'LineWidth', 1.5);
end
hold off;
grid on;

% --- Apply fonts and labels ---
%title('Convergence: Relative Error vs. Iterations', 'FontSize', fontSizeOptions.title);
xlabel('Iteration', 'FontSize', fontSizeOptions.labels);
ylabel('Relative Error (log scale)', 'FontSize', fontSizeOptions.labels);
legend(legend_entries, 'Location', 'best', 'Interpreter', 'tex', 'FontSize', fontSizeOptions.legend);
set(gca, 'FontSize', fontSizeOptions.ticks, 'YMinorGrid', 'on');

% --- Save the figure ---
fig_filename_iter = fullfile(save_dir, 'convergence_vs_iterations.png');
saveas(h2, fig_filename_iter);
% % --- Recommended: Use the print function to save higher quality images for papers ---
print(h2, fullfile(save_dir, 'convergence_vs_iterations.eps'), '-depsc'); % Save as an EPS vector graphic
print(h2, fullfile(save_dir, 'convergence_vs_iterations_high_res.png'), '-dpng', '-r300'); % Save as a 300 DPI PNG

fprintf('Convergence curve plot (vs. Iterations) saved to: %s\n', fig_filename_iter);

end