% select_zoom_region.m
% -------------------------------------------------------------------------
% Description:
% This is an interactive helper script to find the coordinates for a zoom region.
% It allows you to visually select an area on a sample image and then
% prints the required parameter lines to the command window.
%
% How to use:
% 1. Run this script.
% 2. A file dialog will open. Select a representative image from your dataset.
% 3. The image will be displayed. Your mouse cursor will become a crosshair.
% 4. Click and drag on the image to draw a rectangle over the desired zoom area.
% 5. You can move or resize the rectangle after drawing it.
% 6. When you are satisfied with the position and size, DOUBLE-CLICK the rectangle.
% 7. The script will print the coordinate lines in the MATLAB Command Window.
% 8. Copy those lines and paste them into your main batch processing script.
% -------------------------------------------------------------------------

clear;
clc;
close all;

% --- Step 1: Ask the user to select a sample image ---
disp('Please select a sample image to define the zoom region...');
[file, path] = uigetfile({'*.png;*.jpg;*.jpeg;*.bmp', 'Image Files (*.png, *.jpg, *.bmp)'}, 'Select a sample image');

% Check if the user cancelled the selection
if isequal(file, 0)
    disp('User cancelled the operation.');
    return;
end

% Construct the full path to the image
fullImagePath = fullfile(path, file);

% --- Step 2: Display the image and let the user draw a rectangle ---
try
    % Read and display the image
    img = imread(fullImagePath);
    fig = figure('Name', 'Select Zoom Region', 'NumberTitle', 'off');
    imshow(img);
    title('Draw a rectangle, then DOUBLE-CLICK it to confirm.', 'FontSize', 12);
    
    % Let the user draw an interactive rectangle
    % imrect is part of the Image Processing Toolbox
    h_rect = imrect; 
    
    % Pause the script and wait for the user to double-click the rectangle
    wait(h_rect);
    
    % --- Step 3: Get the position and size of the final rectangle ---
    pos = getPosition(h_rect);
    
    % Round the values to get integer pixel coordinates
    pos = round(pos);
    
    % The format of pos is [xmin, ymin, width, height]
    zoom_pos_x = pos(1);
    zoom_pos_y = pos(2);
    zoom_size_w = pos(3);
    zoom_size_h = pos(4);
    
    % --- Step 4: Display the results in a copy-paste friendly format ---
    fprintf('\n=========================================================\n');
    fprintf('Region selection complete!\n');
    fprintf('Copy the following two lines into your main script:\n\n');
    
    % This is the exact format needed for the main script
    fprintf('zoom_region_pos = [%d, %d];\n', zoom_pos_x, zoom_pos_y);
    fprintf('zoom_region_size = [%d, %d];\n', zoom_size_w, zoom_size_h);
    
    fprintf('\n=========================================================\n');
    
catch ME
    % In case of an error (e.g., file is not an image)
    fprintf(2, 'An error occurred: %s\n', ME.message);
    disp('Please ensure you selected a valid image file.');
end