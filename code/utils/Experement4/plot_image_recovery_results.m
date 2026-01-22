function plot_image_recovery_results(results_list, L_true, data, params, save_dir)
% plot_image_recovery_results - Plots and saves a visual comparison of image recovery results.
%
% Inputs:
%   results_list - A cell array containing the results from all algorithms.
%   L_true       - The original, complete image matrix.
%   data         - The observed image matrix with missing values.
%   params       - A struct containing experiment parameters (used to get mask info).
%   save_dir     - The folder path to save the images.

    fprintf('Generating image recovery comparison plot...\n');

    num_plots = length(results_list) + 2; % Original + Masked + results from each algorithm
    
    % Dynamically calculate subplot layout
    cols = ceil(sqrt(num_plots));
    rows = ceil(num_plots / cols);
    
    figure('Name', 'Image Recovery Results Comparison', 'NumberTitle', 'off', 'Position', [50, 50, 350*cols, 300*rows]);
    
    % 1. Plot the original image
    subplot(rows, cols, 1);
    imshow(L_true, []);
    title('Original Image', 'FontSize', 12);
    
    % 2. Plot the masked image
    subplot(rows, cols, 2);
    imshow(data, []);
    if strcmpi(params.mask_type, 'text')
        mask_info = sprintf('Mask Type: Text');
    else
        mask_info = sprintf('Mask Type: Random (%.0f%% Missing)', (1-params.p_obs)*100);
    end
    title({'Masked Image', mask_info}, 'FontSize', 12);
    
    % 3. Loop through and plot the recovery result for each algorithm
    for i = 1:length(results_list)
        res = results_list{i};
        subplot(rows, cols, i + 2);
        
        if isfield(res, 'A_hat') && ~isempty(res.A_hat)
            imshow(res.A_hat, []);
            
            % Safely get PSNR and SSIM values
            psnr_val = -1; ssim_val = -1;
            if isfield(res, 'psnr_value') && ~isempty(res.psnr_value), psnr_val = res.psnr_value; end
            if isfield(res, 'ssim_value') && ~isempty(res.ssim_value), ssim_val = res.ssim_value; end
            
            title_str = {strrep(res.name, '_', '\_'), ...
                         sprintf('PSNR: %.2f dB, SSIM: %.4f', psnr_val, ssim_val)};
            title(title_str, 'FontSize', 10);
        else
            text(0.5, 0.5, 'Result not found', 'HorizontalAlignment', 'center');
            title(strrep(res.name, '_', '\_'), 'FontSize', 10);
        end
    end
    
    % Adjust layout and save
    sgtitle('Image Inpainting Results Comparison for All Algorithms', 'FontSize', 16, 'FontWeight', 'bold');
    
    img_filename = fullfile(save_dir, 'visual_comparison_results.png');
    try
        saveas(gcf, img_filename);
        fprintf('Image recovery comparison plot saved to: %s\n', img_filename);
    catch save_err
        fprintf('Warning: An error occurred while saving the image: %s\n', save_err.message);
    end
    close(gcf);
end