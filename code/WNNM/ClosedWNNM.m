%% Helper function: ClosedWNNM
function [tempDiagS, svp] = ClosedWNNM(diagS, w_lambda, myeps)
% ClosedWNNM: Closed-form solution for Weighted Nuclear Norm Minimization (WNNM).
% Inputs:
%   diagS    - A vector of singular values.
%   w_lambda - The weight parameter.
%   myeps    - A small constant to avoid division by zero.
% Outputs:
%   tempDiagS - The thresholded singular values.
%   svp       - The number of non-zero singular values after thresholding.

    % Sort singular values in descending order
    [diagS, idx] = sort(diagS, 'descend');
    [n, ~] = size(diagS);
    
    % Calculate the weights for each singular value
    weights = zeros(n, 1);
    for i = 1:n
        weights(i) = w_lambda / (diagS(i) + myeps);
    end
    
    % Apply the weighted soft-thresholding
    tempDiagS = max(diagS - weights, 0);
    
    % Count the number of non-zero singular values
    svp = sum(tempDiagS > 0);
    
    % Restore the original order of the singular values
    tempDiagS(idx) = tempDiagS;
    
    % Keep only the non-zero singular values
    tempDiagS = tempDiagS(1:svp);
end