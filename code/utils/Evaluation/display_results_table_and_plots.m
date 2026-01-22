function display_results_table_and_plots(results_list, save_dir, fontSizeOptions, options)
% display_results_table_and_plots - Generates a dual y-axis, grouped bar chart
%                                   for visually comparing metrics using their original scales.
%                                   The plotting order of algorithms is fixed.
%
% Input:
%   results_list    - A cell array where each element is a struct of an algorithm's averaged results.
%   save_dir        - (Optional) The directory path to save the generated figure.
%   fontSizeOptions - (Optional) A struct to control font sizes for the plot.
%   options         - (Optional) A struct for advanced options.
%                       options.errorScale: 'linear' or 'log' (default).
%                       options.timeScale:  'linear' or 'log' (default).

%% 0. Handle Optional 'options' Argument
if nargin < 4, options = struct(); end
if ~isfield(options, 'timeScale'), options.timeScale = 'log'; end
if ~isfield(options, 'errorScale'), options.errorScale = 'log'; end

if isempty(results_list)
    disp('The results list is empty, cannot generate the table or plots.');
    return;
end

fprintf('Reordering results to match the specified plot order...\n');

% --- 1. Define fixed algorithm plotting order (consistent with LaTeX table) ---
ordered_names = {'AltGD', 'altGDMinCntrl', 'altMinCntrl', 'WNNM', ...
                 'altGDMin', 'altMinPrvt', 'FedWNNM'};

% --- 2. Extract original algorithm names from input 'results_list' for matching ---
original_names = cell(1, length(results_list));
for i = 1:length(results_list)
    res = results_list{i};
    % Using the same name cleaning logic as the original code
    clean_name = extractBefore(res.name, '_');
    if isempty(clean_name), clean_name = res.name; end
    original_names{i} = clean_name;
end

% --- 3. Reorder 'results_list' according to 'ordered_names' ---
results_list_ordered = {};
% Find algorithms present in the input list and arrange them in the specified fixed order
for i = 1:length(ordered_names)
    current_name = ordered_names{i};
    % Look up the index of the current ordered name in the original names list
    original_idx = find(strcmp(current_name, original_names), 1);
    
    % If found, add the corresponding result to the new ordered list
    if ~isempty(original_idx)
        results_list_ordered{end+1} = results_list{original_idx};
    end
end

if isempty(results_list_ordered)
    disp('None of the specified algorithms in ''ordered_names'' were found in the results list.');
    return;
end
% ======================================================================================


%% 1. Display Summary Table and Extract Data

fprintf('----------------------------------------------------------------------\n');
fprintf('                Experiment Results Summary (Averaged Performance)\n');
fprintf('----------------------------------------------------------------------\n');
fprintf('%-20s | %-12s | %-18s | %-12s\n', ...
    'Algorithm', 'Valid Runs', 'Avg Rel. Error', 'Avg Total Time (s)');
fprintf('----------------------------------------------------------------------\n');

% --- Prepare data for plotting from the NEW ORDERED list ---
alg_names = {};
avg_errors = [];
avg_times = [];

% Now iterating through the sorted 'results_list_ordered'
for i = 1:length(results_list_ordered)
    res = results_list_ordered{i};

    clean_name = extractBefore(res.name, '_');
    if isempty(clean_name)
        clean_name=res.name;
    end
    avg_rel_error = res.avg_relative_error;
    avg_total_time = res.avg_total_time;
    num_runs = res.num_runs;
    
    fprintf('%-20s | %-12d | %-18.6f | %-12.4f\n', ...
        clean_name, num_runs, avg_rel_error, avg_total_time);
    
    % Store sorted data into plotting variables
    alg_names{end+1} = clean_name;
    avg_errors(end+1) = avg_rel_error;
    avg_times(end+1) = avg_total_time;
end

fprintf('----------------------------------------------------------------------\n\n');

%% 2. Generate Dual-Axis Bar Chart with Original Data Scales

fprintf('Generating dual-axis bar chart with original data scales...\n');

% --- Font and Color Definitions ---
if nargin < 3 || isempty(fontSizeOptions)
    fontSizeOptions = struct('title',20, 'labels',50, 'ticks',29, 'legend',20, 'text',9);
end
color_error = [0, 0.4470, 0.7410]; % Blue
color_time = [0.8500, 0.3250, 0.0980]; % Orange

% --- Create Figure ---
h_fig = figure('Name', 'Dual-Axis Performance Comparison', 'NumberTitle', 'off', 'Position', [100 100 1200 900]);

x_pos = 1:length(alg_names);
bar_width = 0.35;

% --- Plot Left Y-axis (Error) ---
yyaxis left;
b1 = bar(x_pos - bar_width/2, avg_errors, bar_width, 'FaceColor', color_error);
ylabel('Average Relative Error', 'FontSize', fontSizeOptions.labels);
ax = gca;
ax.YColor = color_error;

if strcmpi(options.errorScale, 'log')
    set(gca, 'YScale', 'log');
    fprintf('Using LOGARITHMIC scale for the left Y-axis (Error).\n');
    if max(avg_errors) > 0
        ax.YLim(2) = ax.YLim(2) * 2;
    end
else
    set(gca, 'YScale', 'linear');
    max_error = max(avg_errors);
    ylim([0, max_error * 1.2 + eps]);
    ax.YAxis(1).Exponent = 0;
    fprintf('Using LINEAR scale for the left Y-axis (Error).\n');
end

% --- Plot Right Y-axis (Time) ---
yyaxis right;
b2 = bar(x_pos + bar_width/2, avg_times, bar_width, 'FaceColor', color_time);
ylabel('Average Total Time (s)', 'FontSize', fontSizeOptions.labels);
ax = gca;
ax.YColor = color_time;

if strcmpi(options.timeScale, 'log')
    set(gca, 'YScale', 'log');
    fprintf('Using LOGARITHMIC scale for the right Y-axis (Time).\n');
    if max(avg_times) > 0
        ax.YLim(2) = ax.YLim(2) * 2;
    end
else
    set(gca, 'YScale', 'linear');
    max_time = max(avg_times);
    ylim([0, max_time * 1.2 + eps]);
    fprintf('Using LINEAR scale for the right Y-axis (Time).\n');
end

% --- Customize Shared X-axis ---
set(gca, 'XTick', x_pos, 'XTickLabel', alg_names, 'FontSize', fontSizeOptions.ticks);
xtickangle(30);
grid on;

% --- Add Text Annotations for Original Data Values ---
yyaxis left;
x_coords_error = b1.XEndPoints;
y_coords_error = b1.YEndPoints;
valid_indices_error = y_coords_error > 0;
text(x_coords_error(valid_indices_error), y_coords_error(valid_indices_error), ...
     num2str(avg_errors(valid_indices_error)', '%.1e'), ...
     'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
     'FontSize', fontSizeOptions.text, 'Color', 'k');

yyaxis right;
x_coords_time = b2.XEndPoints;
y_coords_time = b2.YEndPoints;
valid_indices_time = y_coords_time > 0;
text(x_coords_time(valid_indices_time), y_coords_time(valid_indices_time), ...
     num2str(avg_times(valid_indices_time)', '%.1e'), ...
     'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
     'FontSize', fontSizeOptions.text, 'Color', 'k');

% --- Create Legend ---
yyaxis left;
legend([b1, b2], {'Avg Rel. Error', 'Avg Total Time'}, 'Location', 'northwest', 'FontSize', fontSizeOptions.legend);

%% 3. Save Figure (Optional)

if nargin > 1 && ~isempty(save_dir)
    if ~exist(save_dir, 'dir')
        mkdir(save_dir);
        fprintf('Created save directory: %s\n', save_dir);
    end
    
    fig_filename = fullfile(save_dir, 'comparison_dual_axis_performance');
    print(h_fig, [fig_filename '.png'], '-dpng', '-r300');
    print(h_fig, [fig_filename '.eps'], '-depsc');
    fprintf('Dual-axis performance comparison plot saved to: %s.(png/eps)\n', fig_filename);
end

fprintf('Done.\n\n');

end