function result = altGDMin_T(data, mask, parameters)
% altGDMin_T - Federated Alternating Gradient Descent for Matrix Completion
%
% This function implements the altGDMin_T algorithm. The output format is
% aligned with the centralized AltGD algorithm for direct comparison.
%
% Inputs:
%   data       - The observed matrix with missing values (m x n), where missing entries are 0.
%   mask       - The observation mask (m x n), 1 for observed, 0 for missing.
%   parameters - A struct containing all algorithm parameters.
%
% Outputs:
%   result - A struct containing all results and logs, with fields and
%            indexing consistent with the AltGD algorithm.

%% 1. Parameter Setup and Default Values
if nargin < 3, parameters = struct(); end

% Get matrix dimensions from input data
[m, n] = size(data);

% Set default parameters based on the demo code and common usage
if ~isfield(parameters, 'm'), parameters.m = m; end
if ~isfield(parameters, 'n'), parameters.n = n; end
if ~isfield(parameters, 'r'), parameters.r = 10; end % rank
if ~isfield(parameters, 'p'), parameters.p = 8; end % number of clients (workers)
if ~isfield(parameters, 'maxiter'), parameters.maxiter = 25; end % max iterations
if ~isfield(parameters, 'tol'), parameters.tol = 1e-5; end % convergence tolerance
if ~isfield(parameters, 'eta_c'), parameters.eta_c = 1.0; end % learning rate constant
if ~isfield(parameters, 'Tsvd'), parameters.Tsvd = 15; end % federated SVD iterations
if ~isfield(parameters, 'p_obs'), parameters.p_obs = nnz(mask) / (m * n); end

% Extract parameters into local variables for easier access
r = parameters.r;
num_clients = parameters.p;
maxiter = parameters.maxiter;
tol = parameters.tol;
eta_c = parameters.eta_c;
Tsvd = parameters.Tsvd;

% Check if the number of columns is divisible by the number of clients
if mod(n, num_clients) ~= 0
    error('Number of columns `n` must be divisible by the number of clients `p`.');
end

%% 2. Initialize Result Structure and Logging Arrays (aligned with AltGD)
% -------------------------------------------------------------------------
result = struct();

% Initialize monitoring arrays with length maxiter+1 to store the initial state (iter=0)
residuals = zeros(maxiter + 1, 1);
iteration_times = zeros(maxiter + 1, 1);
wall_clock_times = zeros(maxiter + 1, 1);
subspace_distances = zeros(maxiter + 1, 1);
singular_value_errors = zeros(maxiter + 1, 1);
communication_volumes = zeros(maxiter, 1); % Communication occurs during iterations, so length is maxiter
rank_trajectory = r * ones(maxiter + 1, 1); % Added rank_trajectory

%% 3. Data Preprocessing and Algorithm Initialization
% -------------------------------------------------------------------------
fprintf('FedAltGDMin: Initializing and preparing data...\n');
total_start_time = tic; % Start total timer
iter_start_time = tic;  % Start timer for the first iteration

% Ensure missing entries are zero
observed_data = data;
observed_data(~mask) = 0;
sampling_rate = parameters.p_obs;

% Partition data for each client (column-wise)
fprintf('Partitioning data for %d clients...\n', num_clients);
cols_per_client = n / num_clients;
[row, col, val] = find(observed_data);
global_rowIdx = cell(n, 1);
global_Xcol = cell(n, 1);
for j = 1:n
    indices = (col == j);
    global_rowIdx{j} = row(indices);
    global_Xcol{j} = val(indices);
end

client_row_indices = cell(num_clients, cols_per_client);
client_col_values = cell(num_clients, cols_per_client);
for j = 1:num_clients
    offset = (j - 1) * cols_per_client;
    for k = 1:cols_per_client
        global_col_idx = offset + k;
        client_row_indices{j, k} = global_rowIdx{global_col_idx};
        client_col_values{j, k} = global_Xcol{global_col_idx};
    end
end

% --- Algorithm Initialization ---
fprintf('Performing initial SVD...\n');
[U_init, S_init, ~] = svds(observed_data / sampling_rate, r);

% Normalize initial U (original algorithm logic)
mu_inc = min(vecnorm(U_init')) * sqrt(m / r);
const1 = mu_inc * sqrt(r / m);
U_init = U_init .* repmat(const1 ./ vecnorm(U_init, 2, 2), 1, r);
[U, ~, ~] = svd(U_init, 'econ'); % Orthogonalize

% Calculate learning rate eta
eta = eta_c / (S_init(1,1)^2 * sampling_rate);

% Initialize V and A_prev for initial residual calculation
V = zeros(r, n);
parfor j = 1:n
    if ~isempty(global_rowIdx{j})
        V(:, j) = U(global_rowIdx{j}, :) \ global_Xcol{j};
    end
end
A_prev = U * V;

%% 4. Calculate Initial Performance Metrics (iter = 0) (aligned with AltGD)
% -------------------------------------------------------------------------
iteration_times(1) = toc(iter_start_time);
wall_clock_times(1) = 0; % Wall-clock time starts accumulating after the first iteration
residuals(1) = 1; % MODIFICATION: Initial residual is defined as 1

% Calculate initial subspace distance
if isfield(parameters, 'U_true') && ~isempty(parameters.U_true)
    subspace_distances(1) = subspace_distance(parameters.U_true, U);
else
    subspace_distances(1) = NaN;
end

% Calculate initial singular value error
if isfield(parameters, 'S_true') && ~isempty(parameters.S_true)
    [~, S_approx, ~] = svd(A_prev, 'vector');
    S_approx_iter = S_approx(S_approx > tol);

    % Pad the shorter singular value vector with zeros for comparison
    if length(parameters.S_true) ~= length(S_approx_iter)
        len_diff = abs(length(parameters.S_true) - length(S_approx_iter));
        if length(parameters.S_true) < length(S_approx_iter)
            parameters.S_true = [parameters.S_true; zeros(len_diff,1)];
        else
            S_approx_iter = [S_approx_iter; zeros(len_diff,1)];
        end
    end
    singular_value_errors(1) = singular_value_error(parameters.S_true, S_approx_iter);
else
    singular_value_errors(1) = NaN;
end

relative_error_iter = zeros(maxiter + 1, 1);
relative_error_iter(1) = relative_error(data, parameters.L_true);

% MODIFICATION: Initialize history struct to align with AltGD
history = struct();
history.U = cell(maxiter + 1, 1);
history.V = cell(maxiter + 1, 1);
history.global_grad = cell(maxiter, 1);
history.U_before_proj = cell(maxiter, 1);
history.U{1} = U;
history.V{1} = V; % Store initial V
result.history = history;

%% 5. Main Iteration Loop
% -------------------------------------------------------------------------
fprintf('Starting FedAltGDMin iterations...\n');
converged = false;
iter = 0;

while ~converged && iter < maxiter
    iter = iter + 1;
    iter_start_time = tic;
    
    fprintf('Iteration %d/%d...\n', iter, maxiter);
    
    % --- Communication: Server -> Clients (Broadcast U) ---
    server_to_client_comm = {[m, r]};
    
    % --- Client-side Parallel Computation ---
    local_gradients = zeros(m, r, num_clients);
    parfor j = 1:num_clients
        local_rows = client_row_indices(j, :);
        local_vals = client_col_values(j, :);
        V_local = zeros(r, cols_per_client);
        
        diff_vals = []; diff_rows = []; diff_cols = [];
        current_pos = 0;
        
        for k = 1:cols_per_client
            row_jk = local_rows{k};
            if ~isempty(row_jk)
                num_rows = length(row_jk);
                V_local(:, k) = U(row_jk, :) \ local_vals{k};
                diff = U(row_jk, :) * V_local(:, k) - local_vals{k};
                
                diff_vals(current_pos + 1 : current_pos + num_rows) = diff;
                diff_rows(current_pos + 1 : current_pos + num_rows) = row_jk;
                diff_cols(current_pos + 1 : current_pos + num_rows) = k;
                current_pos = current_pos + num_rows;
            end
        end
        M_j = sparse(diff_rows, diff_cols, diff_vals, m, cols_per_client);
        local_gradients(:, :, j) = M_j * V_local';
    end
    
    % --- Communication: Clients -> Server (Upload Gradients) ---
    client_to_server_comm = cell(num_clients, 1);
    for j = 1:num_clients, client_to_server_comm{j} = [m, r]; end
    communication_volumes(iter) = communication_volume([server_to_client_comm; client_to_server_comm]);
    
    % --- Server-side Aggregation and Update ---
    global_gradient = sum(local_gradients, 3);
    result.history.global_grad{iter} = global_gradient;
    
    % Update U using gradient descent
    U_new = U - eta * global_gradient;
    result.history.U_before_proj{iter} = U_new; % MODIFICATION: Record U before projection
    
    % Orthogonalize U (projection step)
    [U, ~] = qr(U_new, 'econ');
    result.history.U{iter + 1} = U;

    %% 6. Performance Evaluation and Logging (after each iteration)
    % ---------------------------------------------------------------------
    
    % To check convergence, the full V and A are needed
    parfor j = 1:n
        if ~isempty(global_rowIdx{j})
            V(:, j) = U(global_rowIdx{j}, :) \ global_Xcol{j};
        end
    end
    result.history.V{iter + 1} = V; % MODIFICATION: Record V for the current iteration
    A_current = U * V;
    
    % Calculate residual for this iteration (relative change in the recovered matrix)
    stop_criterion = norm(A_current - A_prev, 'fro') / (norm(A_prev, 'fro') + eps);
    residuals(iter + 1) = stop_criterion; % MODIFICATION: Store the raw residual value
    A_prev = A_current;
    
    % Calculate subspace distance
    if isfield(parameters, 'U_true') && ~isempty(parameters.U_true)
        subspace_distances(iter + 1) = subspace_distance(parameters.U_true, U);
    else
        subspace_distances(iter+1) = NaN;
    end

    % Relative recovery error
    if isfield(parameters, 'L_true') && ~isempty(parameters.L_true)
        relative_error_iter(iter + 1) = relative_error(A_current, parameters.L_true);
    end
    
    % Standardized calculation for singular value error
    if isfield(parameters, 'S_true') && ~isempty(parameters.S_true)
        [~, S_approx_iter, ~] = svd(A_current, 'vector');
        S_approx_iter = S_approx_iter(S_approx_iter > tol);

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
    
    % Record time
    current_iter_time = toc(iter_start_time);
    iteration_times(iter + 1) = current_iter_time;
    wall_clock_times(iter + 1) = toc(total_start_time);
    
    fprintf('  Residual: %.6e, Subspace Distance: %.4f, Communication: %.3f MB, Time: %.3fs\n', ...
            stop_criterion, subspace_distances(iter + 1), communication_volumes(iter), current_iter_time);
            
    % Check for convergence
    if stop_criterion < tol
        fprintf('Converged: Residual %.2e < Tolerance %.2e\n', stop_criterion, tol);
         converged = true; % Can be uncommented for early termination
    end
end

if ~converged && iter == maxiter
    fprintf('Algorithm reached the maximum number of iterations %d.\n', maxiter);
end

%% 7. Construct Final Output (aligned with AltGD)
% -------------------------------------------------------------------------
total_time = toc(total_start_time);
final_iter_count = iter + 1; % Total number of records, including the initial state

% Basic results
result.A_hat = A_prev;
result.iteration_count = iter;
result.converged = converged;
result.total_time = total_time;

% Trim log data to the actual number of iterations
result.residuals = relative_error_iter(1:final_iter_count);
result.iteration_times = iteration_times(1:final_iter_count);
result.wall_clock_times = wall_clock_times(1:final_iter_count);
result.rank_trajectory = rank_trajectory(1:final_iter_count);
result.communication_volumes = communication_volumes(1:iter);
result.total_communication = sum(result.communication_volumes);

if any(~isnan(subspace_distances))
    result.subspace_distances = subspace_distances(1:final_iter_count);
end
if any(~isnan(singular_value_errors))
    result.singular_value_errors = singular_value_errors(1:final_iter_count);
end

% Convergence curve data
result.convergence_curve_iter = [(0:iter)', result.residuals];
result.convergence_curve_time = [result.wall_clock_times, result.residuals];

%% 8. Calculate Final Performance Metrics (if true matrix is provided)
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

fprintf('\nFedAltGDMin Complete:\n');
fprintf('  Total Iterations: %d\n', result.iteration_count);
fprintf('  Final Relative Error: %.6e\n', result.residuals(end));
fprintf('  Total Runtime: %.3f sec\n', result.total_time);
fprintf('  Total Communication: %.2f MB\n', result.total_communication);
fprintf('  Final Rank: %d\n', result.rank_trajectory(end));
fprintf('=========================================\n');

end