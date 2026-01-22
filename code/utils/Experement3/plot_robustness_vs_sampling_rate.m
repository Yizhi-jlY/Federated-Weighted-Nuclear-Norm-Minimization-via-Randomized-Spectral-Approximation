function plot_robustness_vs_sampling_rate(aggregated_results)
% PLOT_ROBUSTNESS_VS_SAMPLING_RATE - Plots the PSNR vs. sampling rate curves.
%
% Inputs:
%   aggregated_results - Struct containing the results for various algorithms.
%   save_dir           - Directory to save the figure.

    % --- Define font size options ---
    fontSizeOptions = struct(...
        'title', 16, ...
        'labels', 30, ...
        'ticks', 16, ...
        'legend', 18 ...
    );

    % --- Create figure and get its handle for saving ---
    h = figure('Name', 'Robustness to Sampling Rate', 'NumberTitle', 'off', 'Position', [100, 100, 800, 600]);
    hold on;
    
    colors = lines(length(fieldnames(aggregated_results)));
    markers = {'-o', '-s', '-^', '-d', '-x', '-*'};
    
    algo_names = fieldnames(aggregated_results);
    legend_entries = {};
    
    for i = 1:length(algo_names)
        algo_name = algo_names{i};
        data = aggregated_results.(algo_name);
        
        % Ensure data points are sorted by p_obs
        [sorted_p_obs, sort_idx] = sort(data.p_obs, 'ascend');
        sorted_psnr = data.psnr(sort_idx);
        
        plot(sorted_p_obs, sorted_psnr, markers{mod(i-1, length(markers))+1}, ...
            'Color', colors(i,:), 'LineWidth', 1.5, 'MarkerSize', 8);
        
        legend_entries{end+1} = strrep(algo_name, '_', '\_');
    end
    
    hold off;
    grid on;
    box on;
    
    % --- Apply specified font sizes ---
    xlabel('Sampling Rate (p_{obs})', 'FontSize', fontSizeOptions.labels);
    ylabel('PSNR (dB)', 'FontSize', fontSizeOptions.labels);
    title('Algorithm Robustness to Sampling Rate', 'FontSize', fontSizeOptions.title);
    legend(legend_entries, 'Location', 'southeast', 'Interpreter', 'latex', 'FontSize', fontSizeOptions.legend);
    set(gca, 'FontSize', fontSizeOptions.ticks);
    
    % --- Save the figure with specified resolution ---
    img_filename = fullfile( 'robustness_vs_sampling_rate_high_res.png'); % Updated filename
    print(h, img_filename, '-dpng', '-r300'); % Save as a 300 DPI PNG
    
    fprintf('Figure saved to: %s\n', img_filename);
end