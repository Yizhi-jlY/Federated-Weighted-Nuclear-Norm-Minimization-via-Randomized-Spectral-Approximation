function result = altGDMinCntrl_T(data, mask, parameters)
% altGDMinCntrl_T - Centralized Alternating Gradient Descent (Control Version)
%
% This function is another centralized implementation for matrix completion,
% serving as a control or baseline for comparison with federated methods.
% The output format is aligned with AltGD and altGDMin_T for consistency.
%
% Inputs:
%   data       - The observed matrix with missing values (m x n).
%   mask       - The observation mask, 1 for observed, 0 for missing (m x n).
%   parameters - A struct containing the following fields:
%     .r         - Rank of the target low-rank matrix (default: 10).
%     .p_obs     - (Recommended) Observation rate of the matrix (0 to 1).
%                  If not provided, it's calculated from the mask.
%     .maxiter   - Maximum number of iterations (default: 100).
%     .tol       - Convergence tolerance (default: 1e-5).
%     .L_true    - (Optional) The true complete matrix for error calculation.
%     .U_true    - (Optional) The true left singular vectors for subspace distance.
%     .S_true    - (Optional) The true singular values for error calculation.
%
% Outputs:
%   result - A struct containing all specified performance and log metrics.

%% 1. Parameter Setup and Default Values
if nargin < 3, parameters = struct(); end
[m, n] = size(data);
if isfield(parameters, 'r'), r = parameters.r; else, r = 10; end
if isfield(parameters, 'maxiter'), maxiter = parameters.maxiter; else, maxiter = 100; end
if isfield(parameters, 'tol'), tol = parameters.tol; else, tol = 1e-5; end
if isfield(parameters, 'p_obs'), p_obs = parameters.p_obs; else, p_obs = nnz(mask) / numel(mask); end

% Check if ground truth data is provided for evaluation
has_L_true = isfield(parameters, 'L_true') && ~isempty(parameters.L_true);
has_U_true = isfield(parameters, 'U_true') && ~isempty(parameters.U_true);
has_S_true = isfield(parameters, 'S_true') && ~isempty(parameters.S_true);

%% 2. Initialize Result Structure and Logging Arrays
result = struct();

% Initialize monitoring arrays (length maxiter+1 for initial state)
residuals = zeros(maxiter + 1, 1);
iteration_times = zeros(maxiter + 1, 1);
wall_clock_times = zeros(maxiter + 1, 1);
subspace_distances = nan(maxiter + 1, 1);
singular_value_errors = zeros(maxiter + 1, 1);
relative_error_iter = zeros(maxiter + 1, 1);
rank_trajectory = r * ones(maxiter + 1, 1); % Rank is fixed

% Initialize history struct for storing factors
history = struct();
history.U = cell(maxiter + 1, 1);
history.V = cell(maxiter + 1, 1); % B is renamed to V for consistency

%% 3. Data Preprocessing
total_start_time = tic;
[row, col] = find(mask);
observed_values = data(logical(mask));
rowIdx = cell(n, 1);
Xcol = cell(n, 1);
parfor j = 1:n
    rowIdx{j} = row(col == j);
    Xcol{j} = data(rowIdx{j}, j);
end

%% 4. Algorithm Core Initialization
[U_init, S_init, ~] = svds(data / p_obs, r);

% Added row normalization step to match original implementation
% Note: The 'n' in the original paper corresponds to 'm' here (number of rows)
mu = min(vecnorm(U_init, 2, 2)) * sqrt(m / r);
const1 = mu * sqrt(r / m);
% vecnorm is the modern and clearer way to calculate row norms
U_init_normalized = U_init .* repmat(const1 ./ vecnorm(U_init, 2, 2), 1, r);
U = orth(U_init_normalized); % Orthogonalize the *normalized* matrix

V = zeros(r, n); % B is renamed to V
if S_init(1,1) > 0, eta = 1 / (S_init(1,1)^2 * p_obs); else, eta = 1.0; end
A_prev = zeros(m, n);

% Record initial state (iter=0)
history.U{1} = U;
history.V{1} = V;
iteration_times(1) = 0;
wall_clock_times(1) = 0;
residuals(1) = 1; % Define initial residual as 1

if has_U_true
    % Subspace distance calculated as the projection error
    subspace_distances(1) = norm((eye(m) - U * U') * parameters.U_true, 'fro');
end
if has_S_true
    % Initial matrix is zero, so singular values are zero
    singular_value_errors(1) = singular_value_error(parameters.S_true, 0);
end
if has_L_true
    relative_error_iter(1) = relative_error(A_prev, parameters.L_true);
end

%% 5. Main Iteration Loop
converged = false;
iter = 0;
while ~converged && iter < maxiter
    iter = iter + 1;
    iter_start_time = tic;

    % Core algorithm logic (V was formerly B)
    parfor j = 1:n
        if ~isempty(rowIdx{j})
            V(:, j) = U(rowIdx{j}, :) \ Xcol{j};
        end
    end
    diff_vec = sum(U(row, :) .* V(:, col)', 2) - observed_values;
    grad_sparse = sparse(row, col, diff_vec, m, n);
    grad_U = grad_sparse * V';
    U = U - eta * grad_U;
    [U, ~] = qr(U, 'econ');
    
    % Store current factors in history
    history.U{iter + 1} = U;
    history.V{iter + 1} = V;
    
    % --- Performance Evaluation ---
    A_hat = U * V;
    norm_A_prev = norm(A_prev, 'fro');
    if norm_A_prev > 0
        current_residual = norm(A_hat - A_prev, 'fro') / norm_A_prev;
    else
        current_residual = norm(A_hat, 'fro');
    end
    residuals(iter + 1) = current_residual;
    A_prev = A_hat;
    
    if has_U_true
        subspace_distances(iter + 1) = norm((eye(m) - U * U') * parameters.U_true, 'fro');
    end

    if has_S_true
        n_singular_values = length(parameters.S_true);
        [~, S_approx_iter, ~] = svd(A_hat, 'vector');
        S_approx_iter = S_approx_iter(S_approx_iter > tol);
        % Pad with zeros to match the length of true singular values
        if length(S_approx_iter) < n_singular_values
            S_approx_iter(end+1 : n_singular_values) = 0;
        end
        singular_value_errors(iter + 1) = singular_value_error(parameters.S_true, S_approx_iter);
    end

    if has_L_true
        relative_error_iter(iter + 1) = relative_error(A_hat, parameters.L_true);
    end
    
    % Record time
    iteration_times(iter + 1) = toc(iter_start_time);
    wall_clock_times(iter + 1) = toc(total_start_time);
    
    if current_residual < tol, converged = true; end
end

%% 6. Construct Final Result
total_time = toc(total_start_time);
actual_len = iter + 1;

% Basic results
result.A_hat = U * V;
result.iteration_count = iter;
result.converged = converged;
result.total_time = total_time;
result.history = history;
% Trim history cells to actual number of iterations
result.history.U = history.U(1:actual_len);
result.history.V = history.V(1:actual_len);


% Performance trajectories
result.residuals = relative_error_iter(1:actual_len);
result.iteration_times = iteration_times(1:actual_len);
result.wall_clock_times = wall_clock_times(1:actual_len);
result.rank_trajectory = rank_trajectory(1:actual_len);

% Communication (zero for centralized algorithms)
result.communication_volumes = zeros(iter, 1);
result.total_communication = 0;

% Convergence curve data
result.convergence_curve_iter = [(0:iter)', result.residuals];
result.convergence_curve_time = [result.wall_clock_times, result.residuals];

if has_U_true
    result.subspace_distances = subspace_distances(1:actual_len);
end
if has_S_true
    result.singular_value_errors = singular_value_errors(1:actual_len);
end

% Final metrics if ground truth is available
if has_L_true
    result.relative_error = relative_error(result.A_hat, parameters.L_true);
    result.psnr_value = psnr_image(result.A_hat, parameters.L_true);
    result.ssim_value = ssim_image(result.A_hat, parameters.L_true);
    
    fprintf('\nFinal Performance Metrics:\n');
    fprintf('  Relative Error: %.6f\n', result.relative_error);
    fprintf('  PSNR: %.2f dB\n', result.psnr_value);
    fprintf('  SSIM: %.4f\n', result.ssim_value);
end

fprintf('\naltGDMin-Cntrl Complete:\n');
fprintf('  Total Iterations: %d\n', result.iteration_count);
fprintf('  Final Relative Error: %.6e\n', result.residuals(end));
fprintf('  Total Runtime: %.3f sec\n', result.total_time);
fprintf('  Total Communication: %.2f MB\n', result.total_communication);
fprintf('  Final Rank: %d\n', result.rank_trajectory(end));
end