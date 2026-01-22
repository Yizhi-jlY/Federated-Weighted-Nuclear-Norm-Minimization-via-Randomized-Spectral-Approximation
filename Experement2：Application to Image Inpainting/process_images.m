% process_images_side_by_side_configurable_zoom.m
% -------------------------------------------------------------------------
% Description:
% This script is for batch processing image inpainting results. It iterates
% through folders, calculates PSNR and SSIM, and saves a new composite image.
%
% The final image has a side-by-side layout:
% 1. Left Panel: The restored image with PSNR/SSIM values (at the bottom)
%    and a box indicating the magnified region.
% 2. Right Panel: Two vertically stacked magnified views, each with a border:
%    - Top: The magnified region from the restored image.
%    - Bottom: The corresponding residual heatmap (difference).
%
% Prerequisites:
% - MATLAB's Image Processing Toolbox must be installed.
% -------------------------------------------------------------------------

% --- Configuration Parameters ---
baseDir = './Experiment4_ImageInpainting_Batch_Color';
outputBaseDir = fullfile(baseDir, 'results_composite_side_layout');
fontSize = 60;
fontColor = 'yellow';

% --- Per-Image Magnification Configuration ---
zoomRegions = containers.Map();
zoomRegions('default') = struct('pos', [100, 150], 'size', [80, 80]);
zoomRegions('0000') = struct('pos', [174, 116], 'size', [57, 62]);
zoomRegions('0010') = struct('pos', [50, 120], 'size', [90, 90]);
zoomRegions('0013') = struct('pos', [196, 25], 'size', [96, 87]);
zoomRegions('0018') = struct('pos', [186, 58], 'size', [70, 67]);
zoomRegions('0027') = struct('pos', [297, 13], 'size', [83, 83]);
zoomRegions('0046') = struct('pos', [220, 180], 'size', [95, 95]);

% --- Visual Style Configuration ---
zoom_border_color = 'red';
zoom_border_width = 4;
right_panel_border_width = 8;
top_zoom_border_color = 'red';
bottom_zoom_border_color = 'green';
right_panel_width_ratio = 0.4;

% --- Main Program ---
fprintf('Starting to process experiment images...\n');

rateFolders = dir(fullfile(baseDir, 'missing_rate_*'));
rateFolders = rateFolders([rateFolders.isdir]);

if isempty(rateFolders)
    error('No folders matching the format "missing_rate_*" were found in the directory: %s', baseDir);
end

for i = 1:length(rateFolders)
    currentRateFolder = fullfile(baseDir, rateFolders(i).name);
    fprintf('\nProcessing folder: %s\n', currentRateFolder);

    outputRateFolder = fullfile(outputBaseDir, rateFolders(i).name);
    if ~exist(outputRateFolder, 'dir')
        mkdir(outputRateFolder);
        fprintf('Created output directory: %s\n', outputRateFolder);
    end

    originalImages = dir(fullfile(currentRateFolder, '*_original.png'));

    for j = 1:length(originalImages)
        [~, name, ~] = fileparts(originalImages(j).name);
        imgPrefix = strrep(name, '_original', '');
        fprintf('  - Processing image set: %s\n', imgPrefix);

        if isKey(zoomRegions, imgPrefix)
            zoom_region = zoomRegions(imgPrefix);
        else
            zoom_region = zoomRegions('default');
        end
        zoom_region_pos = zoom_region.pos;
        zoom_region_size = zoom_region.size;

        originalImgPath = fullfile(currentRateFolder, originalImages(j).name);
        originalImg = imread(originalImgPath);
        if size(originalImg, 3) == 1
            originalImg = cat(3, originalImg, originalImg, originalImg);
        end

        algorithmImages = dir(fullfile(currentRateFolder, [imgPrefix, '_*.png']));

        for k = 1:length(algorithmImages)
            imgFileName = algorithmImages(k).name;
            if contains(imgFileName, '_original.png')
                continue;
            end
            
            currentImgPath = fullfile(currentRateFolder, imgFileName);
            restoredImg = imread(currentImgPath);
            if size(restoredImg, 3) == 1
                restoredImg = cat(3, restoredImg, restoredImg, restoredImg);
            end
            
            if all(size(restoredImg) == size(originalImg))
                % --- Step 1: Calculate PSNR and SSIM ---
                psnrValue = psnr(restoredImg, originalImg);
                ssimValue = ssim(restoredImg, originalImg);
                textString = sprintf('%.2f/%.2f', psnrValue, ssimValue);

                % --- Step 2: Create the Left Panel (Main Image with Text/Box) ---
                % ==================== MODIFICATION START ====================
                % Description: Changed position and anchor point to move text to the bottom-right.
                [imgHeight, imgWidth, ~] = size(restoredImg);
                position = [imgWidth, imgHeight]; % Set position to the bottom-right corner

                % Adjusted shadow offset for better visibility at the bottom
                shadow_offset = [-3, -3];
                shadow_color = 'black';
                
                % Add text with shadow, using 'RightBottom' as the anchor point
                imgWithShadow = insertText(restoredImg, position + shadow_offset, textString, ...
                                         'FontSize', fontSize, 'TextColor', shadow_color, ...
                                         'BoxOpacity', 0, 'AnchorPoint', 'RightBottom');
                imgWithText = insertText(imgWithShadow, position, textString, ...
                                       'FontSize', fontSize, 'TextColor', fontColor, ...
                                       'BoxOpacity', 0, 'AnchorPoint', 'RightBottom');
                % ===================== MODIFICATION END =====================

                left_panel = insertShape(imgWithText, 'Rectangle', [zoom_region_pos, zoom_region_size], ...
                                         'Color', zoom_border_color, 'LineWidth', zoom_border_width);

                % --- Step 3: Create the Right Panel (Magnified Views) ---
                [r, c, ~] = size(left_panel);
                
                original_patch = imcrop(originalImg, [zoom_region_pos, zoom_region_size-1]);
                restored_patch = imcrop(restoredImg, [zoom_region_pos, zoom_region_size-1]);
                
                residual_rgb = imabsdiff(original_patch, restored_patch);
                residual_gray = rgb2gray(residual_rgb);
                [indexed_img, ~] = gray2ind(residual_gray, 256);
                residual_heatmap_rgb = ind2rgb(indexed_img, jet(256));
                residual_heatmap = im2uint8(residual_heatmap_rgb);
                
                panel_width = floor(c * right_panel_width_ratio);
                panel_height = floor(r / 2);
                
                enlarged_top_patch = imresize(restored_patch, [panel_height, panel_width]);
                enlarged_residual = imresize(residual_heatmap, [panel_height, panel_width]);
                
                border_inset = ceil(right_panel_border_width / 2);
                border_rect = [border_inset, border_inset, ...
                               panel_width - 2*border_inset, panel_height - 2*border_inset];

                enlarged_top_patch_bordered = insertShape(enlarged_top_patch, 'Rectangle', border_rect, ...
                                                 'Color', top_zoom_border_color, 'LineWidth', right_panel_border_width);
                enlarged_residual_bordered = insertShape(enlarged_residual, 'Rectangle', border_rect, ...
                                                'Color', bottom_zoom_border_color, 'LineWidth', right_panel_border_width);

                right_panel = [enlarged_top_patch_bordered; enlarged_residual_bordered];
                
                if size(right_panel, 1) ~= r
                    right_panel = imresize(right_panel, [r, panel_width]);
                end
                
                % --- Step 4: Combine Left and Right Panels into a final image ---
                final_image = [left_panel, right_panel];
                                  
                outputFilePath = fullfile(outputRateFolder, [imgPrefix, '_', extractAfter(imgFileName, [imgPrefix, '_'])]);
                imwrite(final_image, outputFilePath);
            end
        end
    end
end

fprintf('\nProcessing complete! All composite images have been saved to the "%s" directory.\n', outputBaseDir);