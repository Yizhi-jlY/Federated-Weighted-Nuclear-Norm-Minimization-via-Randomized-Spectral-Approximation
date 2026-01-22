function result = WNNM_MC(data, mask, parameters)
% WNNM_MC - Weighted Nuclear Norm Minimization for Matrix Completion (Centralized)
%           using the Inexact Augmented Lagrange Multiplier (IALM) method.
%
% This is a centralized implementation based on the WNNM-RPCA algorithm 
% proposed by S. Gu et al. in "Weighted nuclear norm minimization and its applications".
% The function has been adapted for detailed experimental evaluation.
%
% Input Arguments:
%   data - The observed matrix (m x n) with missing entries, typically filled with 0.
%   mask - The observation mask matrix (m x n), where 1 indicates an observed entry and 0 indicates a missing one.
%   parameters - A struct containing optional parameters:
%     .C       - Weight parameter for WNNM (default: 1.0).
%     .myeps   - A small constant in WNNM weight calculation to prevent division by zero (default: 1e-6).
%     .tol     - Tolerance for algorithm convergence (default: 1e-7).
%     .maxiter - Maximum number of iterations (default: 500).
%     .L_true  - The ground truth complete matrix (optional), for calculating recovery error, PSNR, and SSIM.
%     .U_true  - The ground truth left singular vectors (optional), for calculating subspace distance.
%     .S_true  - The ground truth singular values (optional), for calculating singular value error.
%
% Output:
%   result - A struct containing the results:
%     .A_hat                 - The recovered low-rank matrix.
%     .E_hat                 - The recovered sparse error matrix (represents unobserved entries).
%     .iteration_count       - The total number of iterations performed.
%     .converged             - A boolean flag indicating if the algorithm converged.
%     .total_time            - Total execution time in seconds.
%     .residuals             - Residuals at each iteration.
%     .iteration_times       - Time taken for each iteration.
%     .wall_clock_times      - Cumulative wall-clock time at each iteration.
%     .rank_trajectory       - The rank of the matrix at each iteration.
%     .relative_error        - Relative recovery error (if L_true is provided).
%     .psnr_value            - PSNR of the recovered image (if L_true is provided).
%     .ssim_value            - SSIM of the recovered image (if L_true is provided).
%     .subspace_distance     - Subspace distance to the true subspace (if U_true is provided).
%     .singular_value_error  - Error in singular values (if S_true is provided).
%     .convergence_curve_iter - Convergence data (iteration vs. residual).
%     .convergence_curve_time - Convergence data (time vs. residual).

%% 1. Parameter Setup and Default Values
if nargin < 3, parameters = struct(); end

% Get parameters from the 'parameters' struct or set default values.
% Defaults are inferred from the demo and original WNNM_MC code.
if ~isfield(parameters, 'C'),       parameters.C = 1.0; end
if ~isfield(parameters, 'myeps'),   parameters.myeps = 1e-6; end
if ~isfield(parameters, 'tol'),     parameters.tol = 1e-7; end
if ~isfield(parameters, 'maxiter'), parameters.maxiter = 500; end

% Extract parameters to local variables for easier access.
C = parameters.C;
myeps = parameters.myeps;
tol = parameters.tol;
maxiter = parameters.maxiter;

%% 2. Initialization
[m, n] = size(data);

% Ensure the mask is a logical matrix.
mask = logical(mask);

% Initialize the result struct.
result = struct();

% Initialize arrays for performance monitoring.
residuals = zeros(maxiter + 1, 1);
iteration_times = zeros(maxiter + 1, 1);
wall_clock_times = zeros(maxiter + 1, 1);
rank_trajectory = zeros(maxiter + 1, 1);
relative_error_iter = zeros(maxiter + 1, 1);
relative_error_iter(1) = relative_error(data, parameters.L_true);

% Initialize core algorithm variables.
Y = data;
norm_two = lansvd(Y, 1, 'L');
norm_inf = norm(Y(:), inf);
dual_norm = max(norm_two, norm_inf);
Y = Y / dual_norm; % Normalize the Lagrangian multiplier.

A_hat = zeros(m, n); % Initialize the low-rank component.
E_hat = zeros(m, n); % Initialize the sparse component.

% Initialize ALM-related parameters.
mu = 1 / norm_two;     % Penalty parameter.
mu_bar = mu * 1e7;     % Upper bound for mu.
rho = 1.05;            % Update rate for mu.
d_norm = norm(data, 'fro'); % Frobenius norm of the original data for normalizing residuals.

iter = 0;
converged = false;
sv = 10; % Initial number of singular values to compute in SVD.

%% 3. Log Initial State (Iteration 0)
total_start_time = tic; % Start the total timer.

Z_init = data - A_hat - E_hat;
residuals(1) = norm(Z_init, 'fro') / d_norm;
rank_trajectory(1) = 0;
iteration_times(1) = 0;
wall_clock_times(1) = 0;

fprintf('Starting Centralized WNNM Matrix Completion...\n');
fprintf('Iter 0/%d: Initial residual = %.6e\n', maxiter, residuals(1));

%% 4. Main Iteration Loop
while ~converged && iter < maxiter
    iter = iter + 1;
    iter_start_time = tic;

    % --- Core Algorithm Logic Starts ---

    % Step 1: Update the sparse component E_hat.
    % E_hat corresponds to the unobserved entries.
    E_temp = data - A_hat + (1/mu) * Y;
    E_hat = E_temp;
    E_hat(mask) = 0; % The sparse error is zero at observed locations.

    % Step 2: Update the low-rank component A_hat.
    % Solve the WNNM problem using SVD.
    Z_svd = data - E_hat + (1/mu) * Y;
    if choosvd(n, sv) == 1
        [U, S, V] = lansvd(Z_svd, sv, 'L');
    else
        [U, S, V] = svd(Z_svd, 'econ');
    end

    diagS = diag(S);
    % Apply the closed-form solution for WNNM (soft-thresholding).
    [tempDiagS, svp] = ClosedWNNM(diagS, C/mu, myeps);
    
    % Reconstruct the low-rank matrix A_hat.
    A_hat = U(:, 1:svp) * diag(tempDiagS) * V(:, 1:svp)';

    % Dynamically adjust the number of singular values for the next SVD.
    if svp < sv
        sv = min(svp + 1, n);
    else
        sv = min(svp + round(0.05*n), n);
    end

    % Step 3: Update the Lagrangian multiplier Y and the penalty parameter mu.
    Z = data - A_hat - E_hat;
    Y = Y + mu * Z;
    mu = min(mu * rho, mu_bar);

    % --- Core Algorithm Logic Ends ---

    %% 5. Record Performance Metrics
    % Record time for the current iteration and cumulative time.
    current_iter_time = toc(iter_start_time);
    iteration_times(iter + 1) = current_iter_time;
    wall_clock_times(iter + 1) = toc(total_start_time);

    % Calculate and record the residual.
    stopCriterion = norm(Z, 'fro') / d_norm;
    residuals(iter + 1) = stopCriterion;
    
    if isfield(parameters, 'L_true') && ~isempty(parameters.L_true)
        % Relative recovery error
        relative_error_iter(iter + 1) = relative_error(A_hat, parameters.L_true);
    end

    % Record the current rank.
    rank_trajectory(iter + 1) = svp;
    
    % Print iteration status.
    fprintf('  Iter %d/%d: residual = %.6e, rank = %d, time = %.3fs\n', ...
            iter, maxiter, stopCriterion, svp, current_iter_time);

    %% 6. Check for Convergence
    if stopCriterion < tol
        fprintf('Algorithm converged: Final residual %.2e is less than tolerance %.2e.\n', stopCriterion, tol);
        converged = true;
    end
    
    if ~converged && iter >= maxiter
        fprintf('Reached maximum number of iterations %d.\n', maxiter);
        converged = true; % Force exit
    end
end

%% 7. Construct and Output Final Results
total_time = toc(total_start_time);

% Trim the log data to the actual number of iterations.
actual_iters = iter + 1;
result.residuals = relative_error_iter(1:actual_iters);
result.iteration_times = iteration_times(1:actual_iters);
result.wall_clock_times = wall_clock_times(1:actual_iters);
result.rank_trajectory = rank_trajectory(1:actual_iters);

% Store the final results.
result.A_hat = A_hat;
result.E_hat = E_hat;
result.iteration_count = iter;
result.converged = (residuals(actual_iters) < tol);
result.total_time = total_time;

% Format data for convergence curves.
result.convergence_curve_iter = [(0:iter)', result.residuals];
result.convergence_curve_time = [result.wall_clock_times, result.residuals];

%% 8. Calculate Final Evaluation Metrics (if ground truth is provided)
fprintf('\nCalculating final evaluation metrics...\n');
if isfield(parameters, 'L_true') && ~isempty(parameters.L_true)
    L_true = parameters.L_true;
    
    % Relative recovery error
    result.relative_error = relative_error(result.A_hat, L_true);
    fprintf('  Relative recovery error: %.6f\n', result.relative_error);
    
    % PSNR
    result.psnr_value = psnr_image(result.A_hat, L_true);
    fprintf('  PSNR: %.2f dB\n', result.psnr_value);
    
    % SSIM
    result.ssim_value = ssim_image(result.A_hat, L_true);
    fprintf('  SSIM: %.4f\n', result.ssim_value);
else
    fprintf('  Ground truth matrix L_true not provided. Skipping error, PSNR, and SSIM calculation.\n');
end

% Calculate metrics for federated analysis (based on SVD of the final result).
[U_final, S_final, ~] = svd(result.A_hat, 'econ');
if isfield(parameters, 'U_true') && ~isempty(parameters.U_true)
    U_true = parameters.U_true;
    rank_true = size(U_true, 2);
    result.subspace_distance = subspace_distance(U_true, U_final(:, 1:rank_true));
    fprintf('  Subspace distance: %.6f\n', result.subspace_distance);
else
    result.subspace_distance = NaN;
end

if isfield(parameters, 'S_true') && ~isempty(parameters.S_true)
    S_true = parameters.S_true;
    rank_true = size(S_true, 1);
    S_approx = S_final(1:min(end, rank_true), 1:min(end, rank_true));
    result.singular_value_error = singular_value_error(S_true, S_approx);
    fprintf('  Singular value error: %.6f\n', result.singular_value_error);
else
     result.singular_value_error = NaN;
end

fprintf('\nCentralized WNNM execution finished:\n');
fprintf('  Total iterations: %d\n', result.iteration_count);
fprintf('  Final residual: %.4f\n', result.residuals(end));
fprintf('  Total runtime: %.3f seconds\n', result.total_time);
fprintf('  Rank of final recovered matrix: %d\n\n', result.rank_trajectory(end));

end