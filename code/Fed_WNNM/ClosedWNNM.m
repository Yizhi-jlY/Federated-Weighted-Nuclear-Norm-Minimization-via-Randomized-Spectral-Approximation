%% Helper Function: ClosedWNNM
function [tempDiagS,svp] = ClosedWNNM(diagS, w_lambda, myeps)
% ClosedWNNM: Closed-form solution for Weighted Nuclear Norm Minimization.
%
% Inputs:
%   diagS    - Vector of singular values.
%   w_lambda - The weighting parameter.
%   myeps    - A small epsilon parameter to avoid division by zero.
%
% Outputs:
%   tempDiagS - The thresholded singular values.
%   svp       - The number of effective (non-zero) singular values.

    % Sort singular values in descending order
    [diagS, idx] = sort(diagS, 'descend');
    [n, ~] = size(diagS);
    
    % Calculate weights
    weights = zeros(n, 1);
    for i = 1:n
        weights(i) = w_lambda / (diagS(i) + myeps);
    end
    
    % Apply weighted soft-thresholding
    tempDiagS = max(diagS - weights, 0);
    
    % Count the number of effective singular values
    svp = sum(tempDiagS > 0);
    
    % Restore the original order
    tempDiagS(idx) = tempDiagS;
    
    % Trim to the number of effective singular values
    tempDiagS = tempDiagS(1:svp);
end