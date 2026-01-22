function display_results_table_images(results_list)
% display_results_table_images - Displays a summary table of image recovery results in the command window.
%
% Input:
%   results_list - A cell array where each cell contains a struct with algorithm results.

    fprintf('\n===================================================================================================\n');
    fprintf('                                 Image Recovery Performance Summary\n');
    fprintf('===================================================================================================\n');
    fprintf('%-20s | %-15s | %-12s | %-12s | %-12s | %-10s\n', ...
            'Algorithm Name', 'Relative Error', 'PSNR (dB)', 'SSIM', 'Total Time (s)', 'Converged?');
    fprintf('---------------------------------------------------------------------------------------------------\n');

    for i = 1:length(results_list)
        res = results_list{i};
        
        % Safely get field values, using default 'N/A' if a field does not exist.
        rel_err_str = 'N/A';
        if isfield(res, 'relative_error') && ~isempty(res.relative_error)
            rel_err_str = sprintf('%.5f', res.relative_error);
        end
        
        psnr_str = 'N/A';
        if isfield(res, 'psnr_value') && ~isempty(res.psnr_value)
            psnr_str = sprintf('%.2f', res.psnr_value);
        end
        
        ssim_str = 'N/A';
        if isfield(res, 'ssim_value') && ~isempty(res.ssim_value)
            ssim_str = sprintf('%.4f', res.ssim_value);
        end
        
        time_str = 'N/A';
        if isfield(res, 'total_time') && ~isempty(res.total_time)
            time_str = sprintf('%.2f', res.total_time);
        end
        
        conv_str = 'N/A';
        if isfield(res, 'converged')
            if res.converged
                conv_str = 'Yes';
            else
                conv_str = 'No';
            end
        end

        fprintf('%-20s | %-15s | %-12s | %-12s | %-12s | %-10s\n', ...
                res.name, rel_err_str, psnr_str, ssim_str, time_str, conv_str);
    end
    fprintf('===================================================================================================\n\n');
end