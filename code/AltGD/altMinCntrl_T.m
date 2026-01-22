function result = altMinCntrl_T(data, mask, parameters)
% altMinCntrl_T - Alternating Minimization for Matrix Completion (Control Version)
%
% This function implements a centralized alternating minimization algorithm.
% The output structure is aligned with the other algorithms for consistent comparison.
%
% Inputs:
%   data       - The observed matrix with missing values (m x n).
%   mask       - The observation mask, 1 for observed, 0 for missing (m x n).
%   parameters - A struct containing the following fields:
%     .m         - Number of rows (default: size(data,1)).
%     .n         - Number of columns (default: size(data,2)).
%     .r         - The rank of the target matrix (default: 10).
%     .T         - Maximum number of iterations (default: 25).
%     .tol       - Convergence tolerance (default: 1e-5).
%     .L_true    - (Optional) The true low-rank matrix for error calculation.
%     .U_true    - (Optional) The true left singular vectors for subspace distance.
%     .S_true    - (Optional) The true singular values for error calculation.
%
% Outputs:
%   result - A struct containing all experiment results and logs, consistent
%            with the other implemented algorithms.

%% 1. Parameter Setup and Default Values
if nargin < 3, parameters = struct(); end

% Get matrix dimensions
[m, n] = size(data);

% Set default parameters
if ~isfield(parameters, 'm'), parameters.m = m; end
if ~isfield(parameters, 'n'), parameters.n = n; end
if ~isfield(parameters, 'r'), parameters.r = 10; end
if ~isfield(parameters, 'T'), parameters.T = 25; end
if ~isfield(parameters, 'tol'), parameters.tol = 1e-5; end
if ~isfield(parameters, 'p_obs'), parameters.p_obs = nnz(mask) / (m * n); end

% Extract parameters
r = parameters.r;
T = parameters.T;
tol = parameters.tol;
p_obs = parameters.p_obs; % Observation rate

% Check if ground truth is available for evaluation
has_L_true = isfield(parameters, 'L_true') && ~isempty(parameters.L_true);
has_U_true = isfield(parameters, 'U_true') && ~isempty(parameters.U_true);
has_S_true = isfield(parameters, 'S_true') && ~isempty(parameters.S_true);


%% 2. Initialize Result Structure and Logging Arrays
result = struct();

% Initialize monitoring arrays (length T+1 for the initial state)
stop_criterion_log = zeros(T + 1, 1);
iteration_times = zeros(T + 1, 1);
wall_clock_times = zeros(T + 1, 1);
subspace_distances = nan(T + 1, 1);
singular_value_errors = zeros(T + 1, 1);
relative_error_iter = zeros(T + 1, 1);
rank_trajectory = r * ones(T + 1, 1);

% Initialize history struct for storing factors
history = struct();
history.U = cell(T + 1, 1);
history.V = cell(T + 1, 1);

%% 3. Data Preprocessing
fprintf('Initializing altMinCntrl_T...\n');
total_start_time = tic;

% Ensure mask is logical
if ~islogical(mask)
    if size(mask, 2) == 1
        Omega_idx = mask;
        Omega = false(m, n);
        Omega(Omega_idx) = true;
    else
        Omega = logical(mask);
    end
else
    Omega = mask;
end

% Create cell arrays for row/column indices and values for efficient access
[row, col] = find(Omega);
Xcol = cell(n, 1);
Xrow = cell(m, 1);
rowIdx = cell(n, 1);
colIdx = cell(m, 1);
parfor j = 1:n
    rowIdx{j} = row(col == j);
    Xcol{j} = data(rowIdx{j}, j);
end
parfor i = 1:m
    colIdx{i} = col(row == i);
    Xrow{i} = data(i, colIdx{i})';
end

%% 4. Algorithm Initialization
% SVD Initialization
[U_init, ~, ~] = svds(data / p_obs, r);

% Row normalization (consistent with other methods)
mu = min(vecnorm(U_init')) * sqrt(m / r);
const1 = mu * sqrt(r / m);
U_init = U_init .* repmat(const1 ./ vecnorm(U_init,2,2), 1, r);
U = orth(U_init);
V = zeros(r, n);
A_hat_prev = zeros(m, n);

% Record initial state (iter=0)
iter_start_time = tic;
history.U{1} = U;
history.V{1} = V;
iteration_times(1) = toc(iter_start_time);
wall_clock_times(1) = 0;
stop_criterion_log(1) = 1;

if has_U_true
    subspace_distances(1) = subspace_distance(parameters.U_true, U);
end
if has_S_true
    singular_value_errors(1) = singular_value_error(parameters.S_true, 0);
end
if has_L_true
    relative_error_iter(1) = relative_error(A_hat_prev, parameters.L_true);
end

%% 5. Main Iteration Loop
fprintf('Starting altMinCntrl_T iterations...\n');
converged = false;
iter = 0;

while ~converged && iter < T
    iter = iter + 1;
    iter_start_time = tic;
    
    % --- Core Algorithm Logic ---
    % V update
    parfor j = 1:n
        if ~isempty(rowIdx{j})
            V(:, j) = U(rowIdx{j}, :) \ Xcol{j};
        else
            V(:, j) = zeros(r, 1);
        end
    end
    
    % U update
    parfor j = 1:m
        if ~isempty(colIdx{j})
            U(j, :) = V(:, colIdx{j})' \ Xrow{j};
        else
            U(j, :) = zeros(1, r);
        end
    end
    % --- End of Core Algorithm Logic ---
    
    history.U{iter + 1} = U;
    history.V{iter + 1} = V;

    % --- Performance Evaluation ---
    A_hat = U * V;
    
    % Calculate stopping criterion (relative change in recovered matrix)
    stop_criterion = norm(A_hat - A_hat_prev, 'fro') / (norm(A_hat_prev, 'fro') + eps);
    stop_criterion_log(iter + 1) = stop_criterion;
    A_hat_prev = A_hat;
    
    if has_U_true
        [U_orth, ~] = qr(U, 'econ');
        subspace_distances(iter + 1) = subspace_distance(parameters.U_true, U_orth);
    end
    
    if has_S_true
        [~, S_approx_iter, ~] = svd(A_hat, 'vector');
        S_approx_iter = S_approx_iter(S_approx_iter > tol);
        % Pad with zeros to match the length of true singular values
        if length(S_approx_iter) < length(parameters.S_true)
            S_approx_iter(end+1 : length(parameters.S_true)) = 0;
        end
        singular_value_errors(iter + 1) = singular_value_error(parameters.S_true, S_approx_iter(1:length(parameters.S_true)));
    end

    if has_L_true
        relative_error_iter(iter + 1) = relative_error(A_hat, parameters.L_true);
    end
    
    % Record time
    iteration_times(iter + 1) = toc(iter_start_time);
    wall_clock_times(iter + 1) = toc(total_start_time);

    fprintf('  Iteration %d/%d: Stop Criterion: %.6e, Relative Error: %.6f, Time: %.3fs\n', ...
            iter, T, stop_criterion, relative_error_iter(iter+1), iteration_times(iter + 1));

    % Check for convergence
    if stop_criterion < tol
        fprintf('Converged: Stop criterion %.2e < Tolerance %.2e\n', stop_criterion, tol);
        converged = true;
    end
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

fprintf('\naltMinCntrl_T Complete:\n');
fprintf('  Total Iterations: %d\n', result.iteration_count);
fprintf('  Final Relative Error: %.6e\n', result.residuals(end));
fprintf('  Total Runtime: %.3f sec\n', result.total_time);
fprintf('  Total Communication: %.2f MB\n', result.total_communication);
fprintf('  Final Rank: %d\n', result.rank_trajectory(end));
end