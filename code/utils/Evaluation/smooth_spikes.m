function cleaned_residuals = smooth_spikes(residuals, window_size, threshold_factor)
% smooth_spikes: Detects and smooths abnormal spikes in data using a moving window median method.
%
% Inputs:
%   residuals (vector):         The original vector of error data.
%   window_size (integer):      The half-width of the moving window (e.g., 5 means a total window size of 2*5+1=11).
%   threshold_factor (double):  The threshold factor for identifying a spike. A point is
%                               considered a spike if its value > factor * local median.
%
% Outputs:
%   cleaned_residuals (vector): The data vector after smoothing the abnormal spikes.

    if nargin < 2
        window_size = 5;      % Default window half-width
    end
    if nargin < 3
        threshold_factor = 10; % Default threshold factor; considers values 10x higher than local median as abnormal
    end

    n = length(residuals);
    cleaned_residuals = residuals; % Create a copy of the original data to modify

    for i = 1:n
        % Define the window boundaries for the current point, handling array boundary cases
        start_idx = max(1, i - window_size);
        end_idx = min(n, i + window_size);
        
        % Extract data from the window, excluding the current point itself to calculate the baseline
        window_data = [residuals(start_idx:i-1); residuals(i+1:end_idx)];
        
        % If there is not enough data in the window, skip to the next point
        if isempty(window_data)
            continue;
        end
        
        % Calculate the median of the data in the window, which serves as our robust baseline
        local_median = median(window_data(~isnan(window_data))); % Ignore NaN values
        
        % Check if the current point is a significant spike
        % Also, ensure the median is not zero to avoid division-by-zero errors
        if local_median > 0 && residuals(i) > threshold_factor * local_median
            % If it is a spike, replace it with the local median
            cleaned_residuals(i) = local_median;
        end
    end
end