function plot_sensitivity_pover(pover_vals, psnr_vals, time_vals, results_dir)
% PLOT_SENSITIVITY_POVER - Plots the sensitivity to the over-sampling parameter.
%
% Inputs:
%   pover_vals  - Vector of over-sampling parameter values.
%   psnr_vals   - Vector of corresponding PSNR values.
%   time_vals   - Vector of corresponding time values (not used in this plot).
%   results_dir - Directory to save the figure.

    % --- Define font size options ---
    fontSizeOptions = struct(...
        'title', 16, ...
        'labels', 30, ...
        'ticks', 25 ...
    );

    % The figure handle is already named 'fig'
    fig = figure('Position', [200, 200, 800, 600]);

    plot(pover_vals, psnr_vals, '-o', 'LineWidth', 3, 'MarkerSize', 8);
    
    % --- Apply specified font sizes ---
    ylabel('Average relative error', 'FontSize', fontSizeOptions.labels);
    xlabel('Over-sampling Parameter p_{over}', 'FontSize', fontSizeOptions.labels);
    %title('Sensitivity to p_{over}', 'FontSize', fontSizeOptions.title); % Enabled and formatted title
    set(gca, 'FontSize', fontSizeOptions.ticks);
    ylim([-0.012, 0.012])
    grid on;

    % --- Save the figure with specified resolution ---
    if ~exist(results_dir, 'dir'), mkdir(results_dir); end
    img_filename = fullfile( 'sensitivity_frsvd_pover_high_res.png');
    print(fig, img_filename, '-dpng', '-r300'); % Save as a 300 DPI PNG
    
    fprintf('Figure saved to: %s\n', img_filename);
    
    % The 'close(fig)' command is left commented out as in the original
    % close(fig);
end