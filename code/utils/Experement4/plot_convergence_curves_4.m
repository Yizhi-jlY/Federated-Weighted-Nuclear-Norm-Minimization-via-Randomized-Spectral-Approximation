function plot_convergence_curves_4(results_list, save_dir, fontSizeOptions)
% plot_convergence_curves - Plots convergence curves for multiple algorithms (Enhanced Version).
%
% [Final Modified Version]: 1. Corrected the algorithm classification list to ensure exact name matching.
%               2. Added a visual separator in the legend to make the grouping clearer.
%
% Inputs:
%   results_list    - A cell array, where each element is a result struct from an algorithm.
%   save_dir        - The directory path to save the figures.
%   fontSizeOptions - (Optional) A struct to control font sizes.
%
% Outputs:
%   Saves two high-quality convergence curve plots to the specified directory.

% --- Input Parameter Checks ---
if nargin < 2, error('A results list and a save directory must be provided.'); end
if isempty(results_list), disp('The results list is empty. Cannot create plots.'); return; end

% --- Set Font Sizes ---
if nargin < 3 || isempty(fontSizeOptions)
    fontSizeOptions = struct('title',16, 'labels',24, 'ticks',16, 'legend',14);
end

% --- [CORE MODIFICATION 1: Update algorithm classification list for exact name matching] ---
% Update the algorithm names here based on your debugging output.
% Please double-check to ensure these strings exactly match your `results_list{i}.name`!
centralized_algs = {'WNNM_MC', 'AltGD', 'altGDMinCntrl_T', 'altMinCntrl_T'};
federated_algs = {'FedWNNM_MC', 'altGDMin_T', 'altMinPrvt_T'};

fprintf('Generating convergence curve plots with spike and oscillation smoothing...\n');
fprintf('--- Debugging Algorithm Categorization ---\n'); % Add debug title

% --- Define Plotting Colors and Styles ---
colors = [0,0.4470,0.7410; 0.8500,0.3250,0.0980; 0.9290,0.6940,0.1250; 0.4940,0.1840,0.5560; 0.4660,0.6740,0.1880; 0.3010,0.7450,0.9330; 0.6350,0.0780,0.1840; 0,0,0];
lineStyles = {'-'};
markers = {'none'};

%% 1. Relative Error vs. Wall-Clock Time (log-linear scale)
h1 = figure('Name', 'Convergence: Error vs. Time', 'NumberTitle', 'off', 'Position', [100 100 800 600]);
hold on;

h_central = []; 
h_federated = []; 
h_uncategorized = []; % Add a handle for uncategorized algorithms

for i = 1:length(results_list)
    res = results_list{i};
    color = colors(mod(i-1, size(colors, 1)) + 1, :);
    lineStyle = lineStyles{mod(i-1, length(lineStyles)) + 1};
    marker = markers{mod(i-1, length(markers)) + 1};
    
    % Determine the data for plotting
    if isfield(res, 'wall_clock_times') && isfield(res, 'residuals')
        x_data = res.wall_clock_times;
        y_data = res.residuals;
        displayName = strrep(res.name, '_', '\_');
    elseif isfield(res, 'avg_wall_clock_times') && isfield(res, 'avg_residuals')
        x_data = res.avg_wall_clock_times;
        y_data = res.avg_residuals;
        displayName = strrep(res.name, '_', '\_');
    else
        continue;
    end

    y_data = smooth_spikes(y_data, 3, 1); 
    y_data = smoothdata(y_data, 'movmean', 5);
    
    num_points = length(x_data);
    marker_indices = unique(round(linspace(1, num_points, 15)));
    
    p = semilogy(x_data, y_data, ...
        'DisplayName', displayName, ...
        'LineStyle', lineStyle, 'Marker', marker, 'MarkerIndices', marker_indices, ...
        'MarkerEdgeColor', color, 'MarkerFaceColor', 'none', 'MarkerSize', 7, ...
        'Color', color, 'LineWidth', 1.5);
        
    % --- Store handles and print debug info based on category ---
    if ismember(res.name, centralized_algs)
        h_central(end+1) = p;
        fprintf('  Algorithm "%s" categorized as CENTRALIZED.\n', res.name);
    elseif ismember(res.name, federated_algs)
        h_federated(end+1) = p;
        fprintf('  Algorithm "%s" categorized as FEDERATED.\n', res.name);
    else
        h_uncategorized(end+1) = p; % Collect uncategorized handles
        fprintf('  WARNING: Algorithm "%s" NOT categorized.\n', res.name);
    end
end
hold off;
grid on;
xlim([0,50])

xlabel('Wall-Clock Time (seconds)', 'FontSize', fontSizeOptions.labels);
ylabel('Relative Error (log scale)', 'FontSize', fontSizeOptions.labels);
set(gca, 'FontSize', fontSizeOptions.ticks, 'YMinorGrid', 'on');

% --- [CORE MODIFICATION 2: Create a grouped legend with separators] ---
hold on;
h_dummy_central = plot(nan, nan, 'linestyle', 'none', 'marker', 'none', 'DisplayName', '\bf{Centralized Methods}');
h_dummy_spacer = plot(nan, nan, 'linestyle', 'none', 'marker', 'none', 'DisplayName', ''); % Blank separator
h_dummy_federated = plot(nan, nan, 'linestyle', 'none', 'marker', 'none', 'DisplayName', '\bf{Federated Methods}');

legend_handles = [h_dummy_central, h_central, h_dummy_spacer, h_dummy_federated, h_federated];

% If there are uncategorized algorithms, also display them in the legend for debugging.
if ~isempty(h_uncategorized)
    h_dummy_uncategorized = plot(nan,nan,'linestyle','none','marker','none','DisplayName','\bf{Uncategorized}');
    legend_handles = [legend_handles, h_dummy_uncategorized, h_uncategorized];
end

legend(legend_handles, 'Location', 'best', 'Interpreter', 'tex', 'FontSize', fontSizeOptions.legend);
hold off;

fig_filename_time = fullfile(save_dir, 'convergence_vs_time.png');
saveas(h1, fig_filename_time);
print(h1, fullfile(save_dir, 'convergence_vs_time.eps'), '-depsc');
print(h1, fullfile(save_dir, 'convergence_vs_time_high_res.png'), '-dpng', '-r300');
fprintf('Smoothed convergence curve plot (vs. Time) saved to: %s\n', fig_filename_time);

%% 2. Relative Error vs. Iterations (log-linear scale)
% --- (Apply the same modifications to the second plot) ---
h2 = figure('Name', 'Convergence: Error vs. Iterations', 'NumberTitle', 'off', 'Position', [950 100 800 600]);
hold on;

h_central_iter = []; 
h_federated_iter = []; 
h_uncategorized_iter = [];

for i = 1:length(results_list)
    res = results_list{i};
    color = colors(mod(i-1, size(colors, 1)) + 1, :);
    lineStyle = lineStyles{mod(i-1, length(lineStyles)) + 1};
    marker = markers{mod(i-1, length(markers)) + 1};
    
    if isfield(res, 'residuals')
        y_data = res.residuals;
        displayName = strrep(res.name, '_', '\_');
    elseif isfield(res, 'avg_residuals')
        y_data = res.avg_residuals;
        displayName = strrep(res.name, '_', '\_');
    else
        continue;
    end
    
    y_data = smooth_spikes(y_data, 3, 1);
    y_data = smoothdata(y_data, 'movmean', 5);

    iters = 1:length(y_data);
    num_points = length(iters);
    marker_indices = unique(round(linspace(1, num_points, 15)));
    
    p = semilogy(iters, y_data, ...
        'DisplayName', displayName, ...
        'LineStyle', lineStyle, 'Marker', marker, 'MarkerIndices', marker_indices, ...
        'MarkerEdgeColor', color, 'MarkerFaceColor', 'none', 'MarkerSize', 7, ...
        'Color', color, 'LineWidth', 1.5);
        
    if ismember(res.name, centralized_algs)
        h_central_iter(end+1) = p;
    elseif ismember(res.name, federated_algs)
        h_federated_iter(end+1) = p;
    else
        h_uncategorized_iter(end+1) = p;
    end
end
hold off;
grid on;

xlabel('Iteration', 'FontSize', fontSizeOptions.labels);
ylabel('Relative Error (log scale)', 'FontSize', fontSizeOptions.labels);
set(gca, 'FontSize', fontSizeOptions.ticks, 'YMinorGrid', 'on');

% --- Create a grouped legend with separators (second plot) ---
hold on;
h_dummy_central_iter = plot(nan, nan, 'linestyle', 'none', 'marker', 'none', 'DisplayName', '\bf{Centralized Methods}');
h_dummy_spacer_iter = plot(nan, nan, 'linestyle', 'none', 'marker', 'none', 'DisplayName', '');
h_dummy_federated_iter = plot(nan, nan, 'linestyle', 'none', 'marker', 'none', 'DisplayName', '\bf{Federated Methods}');

legend_handles_iter = [h_dummy_central_iter, h_central_iter, h_dummy_spacer_iter, h_dummy_federated_iter, h_federated_iter];
if ~isempty(h_uncategorized_iter)
    h_dummy_uncategorized_iter = plot(nan,nan,'linestyle','none','marker','none','DisplayName','\bf{Uncategorized}');
    legend_handles_iter = [legend_handles_iter, h_dummy_uncategorized_iter, h_uncategorized_iter];
end
legend(legend_handles_iter, 'Location', 'best', 'Interpreter', 'tex', 'FontSize', fontSizeOptions.legend);
hold off;

fig_filename_iter = fullfile(save_dir, 'convergence_vs_iterations.png');
saveas(h2, fig_filename_iter);
print(h2, fullfile(save_dir, 'convergence_vs_iterations.eps'), '-depsc');
print(h2, fullfile(save_dir, 'convergence_vs_iterations_high_res.png'), '-dpng', '-r300');
fprintf('Smoothed convergence curve plot (vs. Iterations) saved to: %s\n', fig_filename_iter);

end