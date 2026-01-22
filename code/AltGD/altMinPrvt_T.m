function result = altMinPrvt_T(data, mask, parameters)
% altMinPrvt_T - Federated Alternating Minimization for Matrix Completion (Privatized/Local Update Version)
%
% This function implements a federated matrix recovery method based on alternating minimization.
% A key feature is the inner iteration loop (T_inner) when updating the U matrix,
% which can be viewed as clients performing multiple local update steps,
% potentially reducing the number of communication rounds.
%
% Arguments:
%   data - The observed matrix with missing values (m x n).
%   mask - The observation mask, where 1 indicates an observed entry and 0 indicates a missing entry (m x n).
%   parameters - A struct containing the following fields:
%     .m - Number of rows in the matrix (default: size(data,1)).
%     .n - Number of columns in the matrix (default: size(data,2)).
%     .p - Number of federated clients (default: 8).
%     .rank - The target rank of the matrix (default: 10).
%     .p_obs - The observation rate (default: inferred from mask).
%     .maxiter - Maximum number of outer iterations (T) (default: 25).
%     .T_inner - Number of inner iterations for the U matrix update (default: 10).
%     .Tsvd - Number of iterations for fedSvd (default: 15).
%     .tol - Convergence tolerance (default: 1e-5).
%     .L_true - The true low-rank matrix, for calculating recovery error (optional).
%     .U_true - The true left singular vectors, for calculating subspace distance (optional).
%     .S_true - The true singular values, for calculating singular value error (optional).
%
% Returns:
%   result - A struct containing the following fields:
%     .A_hat - The recovered low-rank matrix.
%     .iteration_count - The actual number of iterations performed.
%     .converged - A boolean indicating if the algorithm converged.
%     .total_time - Total execution time in seconds.
%     .residuals - The residual at each iteration (in log scale).
%     .iteration_times - Time taken for each iteration.
%     .wall_clock_times - Cumulative wall-clock time.
%     .relative_error - Relative recovery error (if L_true is provided).
%     .psnr_value - Peak Signal-to-Noise Ratio (if L_true is provided).
%     .ssim_value - Structural Similarity Index (if L_true is provided).
%     .communication_volumes - Communication volume per round (in MB).
%     .total_communication - Total communication volume (in MB).
%     .subspace_distances - Subspace distance (if U_true is provided).
%     .singular_value_errors - Singular value error (if S_true is provided).
%     .rank_trajectory - The rank at each iteration.
%     .convergence_curve_iter - Convergence curve data (iterations vs. log residual).
%     .convergence_curve_time - Convergence curve data (time vs. log residual).

%% 1. Parameter Setup and Default Values
% -------------------------------------------------------------------------
if nargin < 3, parameters = struct(); end

% Get matrix dimensions
[m, n] = size(data);

% Set default parameters (inferred from demo code)
if ~isfield(parameters, 'm'), parameters.m = m; end
if ~isfield(parameters, 'n'), parameters.n = n; end
if ~isfield(parameters, 'p'), parameters.p = 8; end % numWrkrs
if ~isfield(parameters, 'rank'), parameters.rank = 10; end % r
if ~isfield(parameters, 'maxiter'), parameters.maxiter = 25; end % T
if ~isfield(parameters, 'T_inner'), parameters.T_inner = 10; end
if ~isfield(parameters, 'Tsvd'), parameters.Tsvd = 15; end
if ~isfield(parameters, 'tol'), parameters.tol = 1e-5; end

% Infer observation rate p_obs from the mask if not provided
if ~isfield(parameters, 'p_obs')
    parameters.p_obs = nnz(mask) / (m * n);
end

% Extract parameters for use in the code
p = parameters.p;
r = parameters.rank;
maxiter = parameters.maxiter;
T_inner = parameters.T_inner;
Tsvd = parameters.Tsvd;
tol = parameters.tol;
p_obs = parameters.p_obs;

%% 2. Initialize Result Struct and Logging Arrays
% -------------------------------------------------------------------------
result = struct();

% Initialize performance monitoring arrays, with length maxiter+1 to store the initial state (iter=0)
residuals = zeros(maxiter + 1, 1);
iteration_times = zeros(maxiter + 1, 1);
wall_clock_times = zeros(maxiter + 1, 1);
communication_volumes = zeros(maxiter, 1);
subspace_distances = zeros(maxiter + 1, 1);
singular_value_errors = zeros(maxiter + 1, 1);
% The rank is fixed in this algorithm
rank_trajectory = r * ones(maxiter + 1, 1); 

%% 3. Data Preprocessing
% -------------------------------------------------------------------------
fprintf('Initializing FedAltMinPrivate Federated Learning...\n');

% Convert input data to the format required by the algorithm (partitioned by columns)
% The original code uses a parfor j=1:q loop, indicating column-wise parallelization.
[row, col, ~] = find(mask);
rowIdx = cell(n, 1);
Xcol = cell(n, 1);
% Use a for loop (or parfor) to prepare data for each "client" (here, each column)
for j = 1:n
    rowIdx{j} = row(col == j);
    Xcol{j} = data(rowIdx{j}, j);
end

%% 4. Algorithm Core Logic
% -------------------------------------------------------------------------

% --- Algorithm Initialization ---
total_start_time = tic; % Start total timer
iter_start_time = tic;  % Start timer for the first iteration

% Initialize using Federated SVD
% Note: The original code was fedSvd(Xzeros/p, ...), where p is the observation rate.
% We follow this logic, scaling by p_obs.
% fedSvd is an external function, assumed to be in the path.
fprintf('Executing Federated SVD Initialization...\n');
[U_init, S_init] = fedSvd(data / p_obs, r, Tsvd, p);

% Initialize and project the U matrix
mu = min(vecnorm(U_init')) * sqrt(m / r);
const1 = mu * sqrt(r / m);
U_init = U_init .* repmat(const1 ./ sqrt(sum(U_init.^2,2)), 1, r);
U = orth(U_init);
U = U(:, 1:r);

% Initialize the V matrix
V = zeros(r, n);
A_hat_prev = zeros(m, n); % Used for convergence check

% --- Calculate Initial Performance Metrics (iter = 0) ---
iteration_times(1) = toc(iter_start_time);
wall_clock_times(1) = 0;
residuals(1) = 1; % Initial residual is defined as 1

% Calculate initial subspace distance
if isfield(parameters, 'U_true') && ~isempty(parameters.U_true)
    subspace_distances(1) = subspace_distance(parameters.U_true, U);
else
    subspace_distances(1) = NaN;
end
relative_error_iter = zeros(maxiter + 1, 1);
relative_error_iter(1) = relative_error(data, parameters.L_true);

% Calculate initial singular value error
if isfield(parameters, 'S_true') && ~isempty(parameters.S_true)
    S_approx = zeros(r,1);
    S_init_ = diag(S_init);
    S_approx(1:length(S_init_)) = S_init_;
    singular_value_errors(1) = singular_value_error(parameters.S_true, S_approx);
else
    singular_value_errors(1) = NaN;
end

etaIn = 1 / (S_init(1,1)^2 * p_obs);
tmp = cell(n, 1);

% --- Main Iteration Loop ---
fprintf('Starting FedAltMinPrivate Federated Learning Iterations...\n');
converged = false;
iter = 0;
% Add a history field to the result struct
result.history = struct();
result.history.U = cell(maxiter, 1);
result.history.V = cell(maxiter, 1);

while ~converged && iter < maxiter
    iter = iter + 1;
    iter_start_time = tic;
    
    fprintf('Iteration %d/%d...\n', iter, maxiter);
    
    % --- Core Algorithm Logic ---
    
    % Step 1: V Update (Client-side local computation)
    % Each client (handling one or more columns) uses the global U matrix to compute its local V vector.
    % A parfor loop is used here to simulate parallel computation.
    parfor j = 1:n
        V(:, j) = U(rowIdx{j}, :) \ Xcol{j};
    end
    
    % Step 2: U Update (with inner iterations)
    % The learning rate etaIn is calculated based on the initial SVD result.
    for tIn = 1:T_inner
        % Parallel computation of gradient terms
        parfor j = 1:n
            tmp{j} = (U(rowIdx{j}, :) * V(:, j) - Xcol{j}) * V(:, j)';
        end
        % Sequentially update the U matrix to avoid parallel conflicts.
        % The original code's structure of parallel tmp computation followed by serial U update is preserved.
        for j = 1:n
            U(rowIdx{j}, :) = U(rowIdx{j}, :) - etaIn * tmp{j};
        end
    end
    
    % Step 3: U Matrix Orthogonalization (Server-side aggregation and processing)
    [U_proj, ~] = qr(U, 'econ');
    U = U_proj; % Update U to its orthogonalized version
    
    % --- Record Intermediate Variables ---
    % Record the updated U (before projection) and V to match the original algorithm's behavior.
    result.history.U{iter+1} = U; 
    result.history.V{iter+1} = V; % V was computed at the beginning of this iteration.
    % ------------------------------------

    % --- End of Core Algorithm Logic ---

    %% 5. Performance Evaluation and Logging (after each iteration)
    % ---------------------------------------------------------------------
    
    % Calculate communication volume
    % Assumption: In each round, the server broadcasts U (m*r), and clients upload their updated U (m*r).
    % This is a simplified model that captures the main data exchange.
    comm_data_sizes = {[m, r], [m, r]}; % {Download, Upload}
    communication_volumes(iter) = communication_volume(comm_data_sizes);
    
    % Reconstruct the currently recovered matrix
    A_hat = U * V;
    
    % Calculate the residual (change in the recovered matrix between two iterations)
    stop_criterion = norm(A_hat - A_hat_prev, 'fro') / (norm(A_hat_prev, 'fro') + eps);
    residuals(iter + 1) = stop_criterion;
    A_hat_prev = A_hat; % Update the matrix from the previous iteration
    
    % Calculate subspace distance
    if isfield(parameters, 'U_true') && ~isempty(parameters.U_true)
        subspace_distances(iter + 1) = subspace_distance(parameters.U_true, U_proj);
    else
        subspace_distances(iter+1) = NaN;
    end
    
    % Calculate singular value error
    % (This part is commented out but can be enabled if needed)
    % if isfield(parameters, 'S_true') && ~isempty(parameters.S_true)
    %     [~, S_approx_iter, ~] = svd(A_hat, 'vector');
    %     S_approx_iter = S_approx_iter(S_approx_iter>tol);
    %     singular_value_errors(iter + 1) = singular_value_error(parameters.S_true, S_approx_iter);
    % else
    %     singular_value_errors(iter+1) = NaN;
    % end

    if isfield(parameters, 'L_true') && ~isempty(parameters.L_true)
        % Relative recovery error
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
       converged = true;
    end
end

%% 6. Construct Final Result
% -------------------------------------------------------------------------
total_time = toc(total_start_time);

% Basic results
result.A_hat = U * V; % Reconstruct with the final U and V
result.iteration_count = iter;
result.converged = converged;
result.total_time = total_time;

% Performance trajectories (trimmed to the actual number of iterations)
final_iter_count = iter + 1;
result.residuals = relative_error_iter(1:final_iter_count);
result.iteration_times = iteration_times(1:final_iter_count);
result.wall_clock_times = wall_clock_times(1:final_iter_count);
result.rank_trajectory = rank_trajectory(1:final_iter_count);

% Communication-related results
result.communication_volumes = communication_volumes(1:iter);
result.total_communication = sum(result.communication_volumes);

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

% Final performance metrics (if the true matrix is provided)
if isfield(parameters, 'L_true') && ~isempty(parameters.L_true)
    L_true = parameters.L_true;
    result.relative_error = relative_error(result.A_hat, L_true);
    result.psnr_value = psnr_image(result.A_hat, L_true);
    result.ssim_value = ssim_image(result.A_hat, L_true);
    
    fprintf('\nFinal Performance Metrics (vs. True Matrix):\n');
    fprintf('  Relative Error: %.6f\n', result.relative_error);
    fprintf('  PSNR: %.2f dB\n', result.psnr_value);
    fprintf('  SSIM: %.4f\n', result.ssim_value);
end

fprintf('\nFedAltMinPrivate Finished:\n');
fprintf('  Total Iterations: %d\n', result.iteration_count);
fprintf('  Final Residual: %.6e\n', 10^result.residuals(end));
fprintf('  Total Runtime: %.3f sec\n', result.total_time);
fprintf('  Total Communication: %.2f MB\n', result.total_communication);
fprintf('  Final Rank: %d\n', result.rank_trajectory(end));

end