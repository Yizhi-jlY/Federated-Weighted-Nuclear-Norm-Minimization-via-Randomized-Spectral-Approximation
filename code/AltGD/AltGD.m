function result = AltGD(data, mask, parameters)
% AltGD - Centralized Alternating Gradient Descent for Matrix Completion
%
% This algorithm recovers a matrix by performing alternating gradient descent 
% on the low-rank factors U and V. This is a centralized implementation.
%
% Inputs:
%   data       - The observed matrix with missing values (m x n).
%   mask       - The observation mask, 1 for observed, 0 for missing (m x n).
%   parameters - A struct containing the following fields:
%     .m         - Number of rows (default: size(data,1)).
%     .n         - Number of columns (default: size(data,2)).
%     .rank      - The rank of the target matrix (default: 10).
%     .p_obs     - Observation rate (default: inferred from mask).
%     .maxiter   - Maximum number of iterations (default: 50).
%     .step_const- Step size constant (default: 0.75).
%     .tol       - Convergence tolerance (default: 1e-5).
%     .L_true    - The true low-rank matrix for error calculation (optional).
%     .U_true    - The true left singular vectors for subspace distance (optional).
%     .S_true    - The true singular values for singular value error (optional).
%
% Outputs:
%   result - A struct containing the following fields:
%     .A_hat                  - The recovered low-rank matrix.
%     .iteration_count        - The actual number of iterations performed.
%     .converged              - A boolean indicating if the algorithm converged.
%     .total_time             - Total execution time in seconds.
%     .residuals              - Residuals at each iteration.
%     .iteration_times        - Time taken for each iteration.
%     .wall_clock_times       - Cumulative wall-clock time.
%     .relative_error         - Relative recovery error (if L_true is provided).
%     .psnr_value             - PSNR value (if L_true is provided).
%     .ssim_value             - SSIM value (if L_true is provided).
%     .communication_volumes  - Communication volume per round (0 for this algorithm).
%     .total_communication    - Total communication volume (0 for this algorithm).
%     .subspace_distances     - Subspace distance at each iteration (if U_true is provided).
%     .singular_value_errors  - Singular value error at each iteration (if S_true is provided).
%     .rank_trajectory        - Rank at each iteration.
%     .convergence_curve_iter - Convergence curve data (iterations vs. residual).
%     .convergence_curve_time - Convergence curve data (time vs. residual).

%% 1. Parameter Setup and Default Values
% -------------------------------------------------------------------------
if nargin < 3, parameters = struct(); end

% Get matrix dimensions
[m, n] = size(data);

% Set default parameters
if ~isfield(parameters, 'm'), parameters.m = m; end
if ~isfield(parameters, 'n'), parameters.n = n; end
if ~isfield(parameters, 'rank'), parameters.rank = 10; end % r
if ~isfield(parameters, 'maxiter'), parameters.maxiter = 50; end % T
if ~isfield(parameters, 'step_const'), parameters.step_const = 0.75; end
if ~isfield(parameters, 'tol'), parameters.tol = 1e-5; end

% Infer observation rate p_obs from mask if not provided
if ~isfield(parameters, 'p_obs')
    parameters.p_obs = nnz(mask) / (m * n);
end

% Extract parameters for use in the code
r = parameters.rank;
maxiter = parameters.maxiter;
step_const = parameters.step_const;
tol = parameters.tol;
p_obs = parameters.p_obs;

%% 2. Initialize Result Structure and Logging Arrays
% -------------------------------------------------------------------------
result = struct();

% Initialize performance monitoring arrays, length maxiter+1 to store initial state (iter=0)
residuals = zeros(maxiter + 1, 1);
relative_error_iter = zeros(maxiter + 1, 1);
iteration_times = zeros(maxiter + 1, 1);
wall_clock_times = zeros(maxiter + 1, 1);
subspace_distances = zeros(maxiter + 1, 1);
singular_value_errors = zeros(maxiter + 1, 1);
% The rank is fixed in this algorithm
rank_trajectory = r * ones(maxiter + 1, 1); 

%% 3. Data Preprocessing and Algorithm Initialization
% -------------------------------------------------------------------------
fprintf('Initializing Centralized Alternating Gradient Descent (AltGD)...\n');

% Find indices of unobserved entries
unobserved_idx = find(mask == 0);

% --- Algorithm Initialization ---
total_start_time = tic; % Start total timer
iter_start_time = tic;  % Start timer for the first iteration

% Initialization using SVD
fprintf('Performing SVD initialization...\n');
[U_init, Sig_init, V_init] = svds(data / p_obs, r);

% Core Logic: Factor Initialization
U = U_init(:, 1:r) * sqrt(Sig_init(1:r, 1:r));
V = V_init(:, 1:r) * sqrt(Sig_init(1:r, 1:r));

% Core Logic: Step size and projection parameter calculation
steplength = step_const / Sig_init(1, 1);
norm_U = max(vecnorm(U')) * sqrt(m / r);
norm_V = max(vecnorm(V')) * sqrt(n / r);
mu = max(norm_U, norm_V);
const1 = sqrt(4 * mu * r / m) * Sig_init(1, 1);
const2 = sqrt(4 * mu * r / n) * Sig_init(1, 1);

% Core Logic: Initial Projection
U = U .* repmat(min(ones(m, 1), const1 ./ vecnorm(U, 2, 2)), 1, r);
V = V .* repmat(min(ones(n, 1), const2 ./ vecnorm(V, 2, 2)), 1, r);

A_hat_prev = U * V'; % Used for checking convergence

% --- Calculate initial performance metrics (iter = 0) ---
iteration_times(1) = toc(iter_start_time);
wall_clock_times(1) = 0;
residuals(1) = 1; % Initial residual is defined as 1

% Calculate initial subspace distance
if isfield(parameters, 'U_true') && ~isempty(parameters.U_true)
    [U_orth, ~] = qr(U, 'econ');
    subspace_distances(1) = subspace_distance(parameters.U_true, U_orth);
else
    subspace_distances(1) = NaN;
end

% Calculate initial singular value error
if isfield(parameters, 'S_true') && ~isempty(parameters.S_true)
    [~, S_approx, ~] = svd(A_hat_prev, 'vector');
    S_approx = S_approx(S_approx>tol);
    singular_value_errors(1) = singular_value_error(parameters.S_true, S_approx);
else
    singular_value_errors(1) = NaN;
end

relative_error_iter(1) = relative_error(data, parameters.L_true);

history = struct();
history.U = cell(maxiter + 1, 1);
history.V = cell(maxiter + 1, 1);
history.U_before_proj = cell(maxiter, 1);
history.V_before_proj = cell(maxiter, 1);
history.U(1) = {U};
history.V(1) = {V};
result.history = history; % Store history handle in result struct

%% 4. Main Iteration Loop
% -------------------------------------------------------------------------
fprintf('Starting AltGD iterations...\n');
converged = false;
iter = 0;

while ~converged && iter < maxiter
    iter = iter + 1;
    iter_start_time = tic;
    
    fprintf('Iteration %d/%d...\n', iter, maxiter);
    
    % --- Core Algorithm Logic ---
    
    % Step 1: Compute gradient
    gradM = U * V' - data;
    gradM(unobserved_idx) = 0;
    
    % Step 2: Gradient descent update for U and V
    % Note: The divisor p is the observation rate, here p_obs
    gradU_term = (gradM * V) / p_obs;
    gradV_term = (gradM' * U) / p_obs;
    
    % Regularization term to encourage balanced norms of U and V
    regul_term_U = U * (U' * U - V' * V);
    regul_term_V = V * (V' * V - U' * U);
    
    U_new = U - steplength * gradU_term - (steplength / 16) * regul_term_U;
    V_new = V - steplength * gradV_term - (steplength / 16) * regul_term_V;
    
    result.history.U_before_proj(iter) = {U_new};
    result.history.V_before_proj(iter) = {V_new};

    % Step 3: Projection operation
    U_new = U_new .* repmat(min(ones(m, 1), const1 ./ vecnorm(U_new, 2, 2)), 1, r);
    V_new = V_new .* repmat(min(ones(n, 1), const2 ./ vecnorm(V_new, 2, 2)), 1, r);
    
    % Update factors
    U = U_new;
    V = V_new;

    result.history.U(iter+1) = {U};
    result.history.V(iter+1) = {V};
    
    % --- End of Core Algorithm Logic ---

    %% 5. Performance Evaluation and Logging (after each iteration)
    % ---------------------------------------------------------------------
    
    % Calculate communication volume (zero for this centralized algorithm)
    communication_volumes(iter) = 0;
    
    % Reconstruct the current recovered matrix
    A_hat = U * V';
    
    % Calculate residual (change in recovered matrix between iterations)
    stop_criterion = norm(A_hat - A_hat_prev, 'fro') / (norm(A_hat_prev, 'fro') + eps);
    residuals(iter + 1) = stop_criterion;
    A_hat_prev = A_hat; % Update the matrix from the previous iteration
    
    % Calculate subspace distance
    if isfield(parameters, 'U_true') && ~isempty(parameters.U_true)
        [U_orth, ~] = qr(U, 'econ');
        subspace_distances(iter + 1) = subspace_distance(parameters.U_true, U_orth);
    else
        subspace_distances(iter+1) = NaN;
    end
    
    % Calculate singular value error
    if isfield(parameters, 'S_true') && ~isempty(parameters.S_true)
        [~, S_approx_iter, ~] = svd(A_hat, 'vector');
        S_approx_iter = S_approx_iter(S_approx_iter>tol);
        
        % Pad the shorter singular value vector with zeros for comparison
        if length(parameters.S_true) ~= length(S_approx_iter)
            len_diff = abs(length(parameters.S_true) - length(S_approx_iter));
            if length(parameters.S_true) < length(S_approx_iter)
                parameters.S_true = [parameters.S_true; zeros(len_diff,1)];
            else
                S_approx_iter = [S_approx_iter; zeros(len_diff,1)];
            end
        end
        singular_value_errors(iter + 1) = singular_value_error(parameters.S_true, S_approx_iter);
    else
        singular_value_errors(iter+1) = NaN;
    end

    % Relative recovery error
    if isfield(parameters, 'L_true') && ~isempty(parameters.L_true)
        relative_error_iter(iter + 1) = relative_error(A_hat, parameters.L_true);
    end
    
    % Record time
    current_iter_time = toc(iter_start_time);
    iteration_times(iter + 1) = current_iter_time;
    wall_clock_times(iter + 1) = toc(total_start_time);
    
    % Print progress
    fprintf('  Residual: %.6e, Subspace Distance: %.4f, Time: %.3f sec\n', ...
            residuals(iter + 1), subspace_distances(iter + 1), current_iter_time);
    
    % Check for convergence
    if stop_criterion < tol
        fprintf('Converged: Residual %.2e < Tolerance %.2e\n', stop_criterion, tol);
        converged = true;
    end
end

%% 6. Construct Final Result
% -------------------------------------------------------------------------
total_time = toc(total_start_time);

% Basic results
result.A_hat = U * V';
result.iteration_count = iter;
result.converged = converged;
result.total_time = total_time;

% Performance trajectory (trimmed to actual number of iterations)
final_iter_count = iter + 1;
result.residuals = relative_error_iter(1:final_iter_count);
result.iteration_times = iteration_times(1:final_iter_count);
result.wall_clock_times = wall_clock_times(1:final_iter_count);
result.rank_trajectory = rank_trajectory(1:final_iter_count);

% Communication-related
result.communication_volumes = zeros(iter, 1); % Explicitly zero
result.total_communication = 0;
result.history = result.history;

% Subspace and singular value errors
if any(~isnan(subspace_distances))
    result.subspace_distances = subspace_distances(1:final_iter_count);
end
if any(~isnan(singular_value_errors))
    result.singular_value_errors = singular_value_errors(1:final_iter_count);
end

% Convergence curve data
result.convergence_curve_iter = [(0:iter)', result.residuals];
result.convergence_curve_time = [result.wall_clock_times, result.residuals];

% Final performance metrics calculation (if true matrix is provided)
if isfield(parameters, 'L_true') && ~isempty(parameters.L_true)
    L_true = parameters.L_true;
    result.relative_error = relative_error(result.A_hat, L_true);
    result.psnr_value = psnr_image(result.A_hat, L_true);
    result.ssim_value = ssim_image(result.A_hat, L_true);
    
    fprintf('\nFinal Performance Metrics (compared to true matrix):\n');
    fprintf('  Relative Error: %.6f\n', result.relative_error);
    fprintf('  PSNR: %.2f dB\n', result.psnr_value);
    fprintf('  SSIM: %.4f\n', result.ssim_value);
end

fprintf('\nAltGD Complete:\n');
fprintf('  Total Iterations: %d\n', result.iteration_count);
fprintf('  Final Residual: %.6e\n', residuals(result.iteration_count + 1));
fprintf('  Total Runtime: %.3f sec\n', result.total_time);
fprintf('  Total Communication: %.2f MB\n', result.total_communication);
fprintf('  Final Rank: %d\n', result.rank_trajectory(end));

end