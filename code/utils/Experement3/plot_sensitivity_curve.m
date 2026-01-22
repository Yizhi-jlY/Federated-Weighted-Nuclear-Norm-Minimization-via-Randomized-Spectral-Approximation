function plot_sensitivity_curve(param_vals, metric_vals, param_name, algo_name, save_dir, plot_type)
% PLOT_SENSITIVITY_CURVE - Plots a parameter sensitivity curve (e.g., PSNR vs. parameter).
%
% Inputs:
%   param_vals  - Vector of parameter values.
%   metric_vals - Vector of performance metric values (e.g., PSNR).
%   param_name  - String of the parameter name (for labels).
%   algo_name   - String of the algorithm name (for title/filename).
%   save_dir    - Directory to save the figure.
%   plot_type   - 'linear' or 'semilogx'.

    % --- Define font size options ---
    fontSizeOptions = struct(...
        'title', 16, ...
        'labels', 30, ...
        'ticks', 25, ...
        'legend', 18 ... % Included for consistency, though no legend is in this plot
    );

    if nargin < 6, plot_type = 'linear'; end

    % --- Create figure and get its handle for saving ---
    h = figure('Name', ['Sensitivity to ' param_name], 'NumberTitle', 'off', 'Position', [200, 200, 800, 600]);
    
    if strcmpi(plot_type, 'semilogx')
        semilogx(param_vals, metric_vals, 'o-', 'LineWidth', 3, 'MarkerSize', 8);
    else
        plot(param_vals, metric_vals, 'o-', 'LineWidth', 3, 'MarkerSize', 8);
    end
    ylim([-0.012, 0.012])
    grid on;
    box on;
    
    % --- Apply specified font sizes ---
    xlabel(['Parameter: ' param_name], 'Interpreter', 'tex', 'FontSize', fontSizeOptions.labels);
    ylabel('Average relative error', 'FontSize', fontSizeOptions.labels);
    %title(['Sensitivity of ' strrep(algo_name, '_', '\_') ' to ' param_name], 'Interpreter', 'tex', 'FontSize', fontSizeOptions.title);
    set(gca, 'FontSize', fontSizeOptions.ticks);
    
    % --- Save the figure with specified resolution ---
    
    % Create a valid filename from the parameter name
    param_name_for_file = param_name;
    if startsWith(param_name_for_file, '\')
        param_name_for_file = param_name_for_file(2:end); % e.g., get 'tau' from '\tau'
    end
    
    img_filename = fullfile( sprintf('sensitivity_%s_vs_%s_high_res.png', lower(algo_name), lower(param_name_for_file)));
    print(h, img_filename, '-dpng', '-r300'); % Save as a 300 DPI PNG
    
    fprintf('Figure saved to: %s\n', img_filename);
end