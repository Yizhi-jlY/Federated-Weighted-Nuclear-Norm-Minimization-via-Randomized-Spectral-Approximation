function plot_sensitivity_q(q_vals, psnr_vals, time_vals, save_dir)
% PLOT_SENSITIVITY_Q - Plots the sensitivity to the number of power iterations, q.
%
% Inputs:
%   q_vals    - Vector of q values.
%   psnr_vals - Vector of PSNR values.
%   time_vals - Vector of total time values (not used in this plot).
%   save_dir  - Directory to save the figure.

    % --- Define font size options ---
    fontSizeOptions = struct(...
        'title', 16, ...
        'labels', 24, ...
        'ticks', 16 ...
    );

    % --- Create figure and get its handle for saving ---
    h = figure('Name', 'Sensitivity to Power Iteration q', 'NumberTitle', 'off', 'Position', [300, 300, 800, 600]);
    
    % Plot PSNR vs. q
    plot(q_vals, psnr_vals, '-o', 'LineWidth', 3, 'MarkerSize', 8);
    
    % --- Apply specified font sizes ---
    xlabel('Power Iteration Count (q)', 'FontSize', fontSizeOptions.labels);
    ylabel('Average relative error', 'FontSize', fontSizeOptions.labels);
    title('Sensitivity to Power Iteration q', 'FontSize', fontSizeOptions.title); % Enabled and formatted title
    set(gca, 'FontSize', fontSizeOptions.ticks);
    
    grid on;
    box on;
    
    % --- Save the figure with specified resolution ---
    img_filename = fullfile( 'sensitivity_frsvd_vs_q_high_res.png');
    print(h, img_filename, '-dpng', '-r300'); % Save as a 300 DPI PNG
    
    fprintf('Figure saved to: %s\n', img_filename);
end