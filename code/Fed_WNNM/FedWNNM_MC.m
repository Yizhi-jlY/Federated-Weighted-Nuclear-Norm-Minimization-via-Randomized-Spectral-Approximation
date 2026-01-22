function result = FedWNNM_MC(data, mask, parameters)
% FedWNNM_MC - Federated Weighted Nuclear Norm Minimization for Matrix Completion
%
% Inputs:
%   data - m x n observed matrix with missing values
%   mask - m x n binary mask (1 for observed, 0 for missing)
%   parameters - Struct containing the following fields:
%     .m - Number of rows (default: size(data,1))
%     .n - Number of columns (default: size(data,2))
%     .p - Number of federated clients (default: 4)
%     .C - WNNM weight parameter (default: 1)
%     .myeps - Small epsilon for WNNM weight calculation (default: 1e-6)
%     .tol - Convergence tolerance (default: 1e-7)
%     .maxiter - Maximum number of iterations (default: 500)
%     .p_over - Oversampling parameter for randomized SVD (default: 10)
%     .rho - Diagonal decay factor for randomized SVD (default: 1)
%     .q - Number of power iterations for randomized SVD (default: 20)
%     .L_true - Ground truth low-rank matrix for error calculation (optional)
%     .U_true - Ground truth left singular vectors for subspace distance (optional)
%     .S_true - Ground truth singular values for singular value error (optional)
%
% Outputs:
%   result - Struct containing the following fields:
%     .A_hat - Recovered low-rank matrix
%     .E_hat - Recovered sparse component
%     .iteration_count - Number of iterations performed
%     .converged - Boolean flag for convergence
%     .total_time - Total execution time in seconds
%     .residuals - Residuals at each iteration
%     .iteration_times - Time taken for each iteration
%     .wall_clock_times - Cumulative wall-clock time
%     .relative_error - Relative recovery error (if L_true is provided)
%     .psnr_value - PSNR value (if L_true is provided)
%     .ssim_value - SSIM value (if L_true is provided)
%     .communication_volumes - Communication volume per round (MB)
%     .total_communication - Total communication volume (MB)
%     .subspace_distances - Subspace distance (if U_true is provided)
%     .singular_value_errors - Singular value error (if S_true is provided)
%     .rank_trajectory - Rank at each iteration
%     .convergence_curve_iter - Convergence curve data (iterations vs. error)
%     .convergence_curve_time - Convergence curve data (time vs. error)

%% Parameter setup and default values
if nargin < 3, parameters = struct(); end

% Get matrix dimensions
[m, n] = size(data);

% Set default parameters
if ~isfield(parameters, 'm'), parameters.m = m; end
if ~isfield(parameters, 'n'), parameters.n = n; end
if ~isfield(parameters, 'p'), parameters.p = 4; end
if ~isfield(parameters, 'C'), parameters.C = 1; end
if ~isfield(parameters, 'myeps'), parameters.myeps = 1e-6; end
if ~isfield(parameters, 'tol'), parameters.tol = 1e-7; end
if ~isfield(parameters, 'maxiter'), parameters.maxiter = 500; end
if ~isfield(parameters, 'p_over'), parameters.p_over = 10; end
if ~isfield(parameters, 'rho'), parameters.rho = 1; end
if ~isfield(parameters, 'q'), parameters.q = 20; end
if ~isfield(parameters, 'dp_enable'),  parameters.dp_enable = false; end
if ~isfield(parameters, 'dp_clip_C'),  parameters.dp_clip_C = 1.0; end
if ~isfield(parameters, 'dp_epsilon'), parameters.dp_epsilon = 1.0; end
if ~isfield(parameters, 'dp_delta'),   parameters.dp_delta = 1e-5; end
if ~isfield(parameters, 'dp_seed'),    parameters.dp_seed = 2025; end

% Extract parameters
p = parameters.p;
C = parameters.C;
myeps = parameters.myeps;
tol = parameters.tol;
maxiter = parameters.maxiter;
p_over = parameters.p_over;
rho = parameters.rho;
q = parameters.q;
dp_opts = struct( ...
    'enable',  parameters.dp_enable, ...
    'clip_C',  parameters.dp_clip_C, ...
    'epsilon', parameters.dp_epsilon, ...
    'delta',   parameters.dp_delta, ...
    'seed',    parameters.dp_seed ...
);

% Validate the number of clients
if mod(m, p) ~= 0
    error('Number of rows m must be divisible by the number of clients p');
end
rows_per_client = m / p;

%% Initialize result struct
result = struct();

% Initialize performance monitoring arrays
residuals = zeros(maxiter + 1, 1);
iteration_times = zeros(maxiter + 1, 1);
wall_clock_times = zeros(maxiter + 1, 1);
communication_volumes = zeros(maxiter, 1);
subspace_distances = zeros(maxiter + 1, 1);
singular_value_errors = zeros(maxiter + 1, 1);
rank_trajectory = zeros(maxiter + 1, 1);

%% Data preprocessing and distribution
fprintf('Initializing FedWNNM Federated Learning...\n');

% Convert observation indices
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

% Distribute data to clients
D_clients = cell(p, 1);
A_clients = cell(p, 1);
E_clients = cell(p, 1);
Y_clients = cell(p, 1);
Omega_clients = cell(p, 1);

for i = 1:p
    start_row = (i-1) * rows_per_client + 1;
    end_row = i * rows_per_client;
    
    D_clients{i} = data(start_row:end_row, :);
    Omega_clients{i} = Omega(start_row:end_row, :);
    A_clients{i} = zeros(rows_per_client, n);
    E_clients{i} = zeros(rows_per_client, n);
    Y_clients{i} = D_clients{i};
end

%% Algorithm parameter initialization
Y_full = data;
norm_two = lansvd(Y_full, 1, 'L');
norm_inf = norm(Y_full(:), inf);
dual_norm = max(norm_two, norm_inf);

% Normalize Lagrange multipliers
for i = 1:p
    Y_clients{i} = Y_clients{i} / dual_norm;
end

mu = 1 / norm_two;
mu_bar = mu * 1e7;
rho_alm = 1.05;
d_norm = norm(data, 'fro');
sv = 10;

%% Calculate initial performance metrics
A_full = vertcat(A_clients{:});
E_full = vertcat(E_clients{:});
Z_full = data;

residuals(1) = 1;
relative_error_iter = zeros(maxiter + 1, 1);
relative_error_iter(1) = relative_error(data, parameters.L_true);
rank_trajectory(1) = 0;

% Initial subspace distance
if isfield(parameters, 'U_true') && ~isempty(parameters.U_true)
    subspace_distances(1) = subspace_distance(parameters.U_true, zeros(m, 1));
else
    subspace_distances(1) = NaN;
end

% Initial singular value error
if isfield(parameters, 'S_true') && ~isempty(parameters.S_true)
    singular_value_errors(1) = norm(diag(parameters.S_true), 2);
else
    singular_value_errors(1) = NaN;
end

% Start timer
total_start_time = tic;
wall_clock_times(1) = 0;
iteration_times(1) = 0;

%% Main iteration loop
fprintf('Starting FedWNNM Federated Learning iterations...\n');

converged = false;
iter = 0;

while ~converged && iter < maxiter
    iter = iter + 1;
    iter_start_time = tic;
    
    fprintf('Iteration %d/%d...\n', iter, maxiter);
    
    %% Step 1: Update sparse component E and construct Z_clients
    Z_clients = cell(p, 1);
    comm_data_sizes = {}; % Record communication data size
    
    for i = 1:p
        % Update sparse component E
        E_temp = D_clients{i} - A_clients{i} + (1/mu) * Y_clients{i};
        E_temp(Omega_clients{i}) = 0;
        E_clients{i} = E_temp;
        
        % Construct matrix for SVD
        Z_clients{i} = D_clients{i} - E_clients{i} + (1/mu) * Y_clients{i};
        
        % Record communication data size (sending Z_clients{i} to server)
        comm_data_sizes{end+1} = size(Z_clients{i});
    end
    
    %% Step 2: Federated Randomized SVD
    [U_clients, S, V] = FR_svd_parallel(Z_clients, sv, p_over, rho, q, dp_opts);
    
    % Record communication data size (server sends V back to clients)
    comm_data_sizes{end+1} = size(V);
    
    %% Step 3: WNNM Thresholding
    diagS = diag(S);
    [tempDiagS, svp] = ClosedWNNM(diagS, C/mu, myeps);
    rank_trajectory(iter + 1) = svp;
    
    %% Step 4: Reconstruct low-rank component and update Lagrange multipliers
    if svp > 0
        S_thresh = diag(tempDiagS);
        V_thresh = V(:, 1:svp);
        
        for i = 1:p
            % Reconstruct low-rank matrix
            U_thresh_local = U_clients{i}(:, 1:svp);
            A_clients{i} = U_thresh_local * S_thresh * V_thresh';
            
            % Update Lagrange multiplier
            Z_local = D_clients{i} - A_clients{i} - E_clients{i};
            Y_clients{i} = Y_clients{i} + mu * Z_local;
        end
        
        % Record communication data size (sending thresholded singular values)
        comm_data_sizes{end+1} = [svp, 1];
    else
        for i = 1:p
            A_clients{i} = zeros(rows_per_client, n);
            Z_local = D_clients{i} - A_clients{i} - E_clients{i};
            Y_clients{i} = Y_clients{i} + mu * Z_local;
        end
    end
    
    %% Calculate communication volume
    communication_volumes(iter) = communication_volume(comm_data_sizes);
    
    % Adaptively adjust the number of singular values
    if svp < sv
        sv = min(svp + 1, n);
    else
        sv = min(svp + round(0.05*n), n);
    end
    
    % Update penalty parameter
    mu = min(mu * rho_alm, mu_bar);
    
    %% Performance evaluation
    A_pre =  A_full;
    E_pre = E_full;
    A_full = vertcat(A_clients{:});
    E_full = vertcat(E_clients{:});
    
    Z_full = data -  A_full- E_full;
    
    % Calculate relative error residual
    residuals(iter + 1) = norm(Z_full, 'fro') / d_norm;
    
    % Calculate subspace distance
    if isfield(parameters, 'U_true') && ~isempty(parameters.U_true)
        if svp > 0
            [U_current, ~, ~] = svd(A_full, 'econ');
            if size(U_current, 2) >= size(parameters.U_true, 2)
                subspace_distances(iter + 1) = subspace_distance(parameters.U_true, U_current(:, 1:size(parameters.U_true, 2)));
            else
                subspace_distances(iter + 1) = norm(parameters.U_true, 'fro');
            end
        else
            subspace_distances(iter + 1) = norm(parameters.U_true, 'fro');
        end
    else
        subspace_distances(iter + 1) = NaN;
    end
    
    % Calculate singular value error
    if isfield(parameters, 'S_true') && ~isempty(parameters.S_true) && svp > 0
        S_current = diag(S);
        min_len = min(length(diag(parameters.S_true)), length(S_current));
        if min_len > 0
            singular_value_errors(iter + 1) = singular_value_error(parameters.S_true(1:min_len), diag(S_current(1:min_len)));
        else
            singular_value_errors(iter + 1) = norm(diag(parameters.S_true), 2);
        end
    else
        singular_value_errors(iter + 1) = NaN;
    end
    
    % Relative recovery error
    if isfield(parameters, 'L_true') && ~isempty(parameters.L_true)
        relative_error_iter(iter + 1) = relative_error(A_full, parameters.L_true);
    end

    % Record time
    current_iter_time = toc(iter_start_time);
    iteration_times(iter + 1) = current_iter_time;
    wall_clock_times(iter + 1) = toc(total_start_time);
    
    % Print progress
    fprintf('  Residual: %.6e, Rank: %d, Time: %.3fs\n', residuals(iter + 1), svp, current_iter_time);
    
    %% Convergence check
    if residuals(iter + 1) < tol
        fprintf('Converged: Residual %.2e < Tolerance %.2e\n', residuals(iter + 1), tol);
        converged = true;
    end
end

%% Construct final results
total_time = toc(total_start_time);

% Basic results
result.A_hat = vertcat(A_clients{:});
result.E_hat = vertcat(E_clients{:});
result.iteration_count = iter;
result.converged = converged;
result.total_time = total_time;

% Performance trajectory (truncated to actual iterations)
%result.residuals = log10(residuals(1:iter+1)+eps); % logarithmic form
result.residuals = relative_error_iter(1:iter+1);
result.iteration_times = iteration_times(1:iter+1);
result.wall_clock_times = wall_clock_times(1:iter+1);
result.rank_trajectory = rank_trajectory(1:iter+1);

% Communication related
result.communication_volumes = communication_volumes(1:iter);
result.total_communication = sum(communication_volumes(1:iter));

% Subspace and singular value errors
if any(~isnan(subspace_distances(1:iter+1)))
    result.subspace_distances = subspace_distances(1:iter+1);
end
if any(~isnan(singular_value_errors(1:iter+1)))
    result.singular_value_errors = singular_value_errors(1:iter+1);
end

% Convergence curve data
result.convergence_curve_iter = [(0:iter)', result.residuals];
result.convergence_curve_time = [result.wall_clock_times, result.residuals];

%% Final performance metrics calculation
if isfield(parameters, 'L_true') && ~isempty(parameters.L_true)
    % Relative recovery error
    result.relative_error = relative_error(result.A_hat, parameters.L_true);
    
    % PSNR calculation
    result.psnr_value = psnr_image(result.A_hat, parameters.L_true);
    
    % SSIM calculation
    result.ssim_value = ssim_image(result.A_hat, parameters.L_true);
    
    fprintf('Final Performance Metrics:\n');
    fprintf('  Relative Error: %.6f\n', result.relative_error);
    fprintf('  PSNR: %.2f dB\n', result.psnr_value);
    fprintf('  SSIM: %.4f\n', result.ssim_value);
end

fprintf('\nFedWNNM Finished:\n');
fprintf('  Iterations: %d\n', result.iteration_count);
fprintf('  Final Residual: %.6e\n', 10^result.residuals(end));
fprintf('  Total Time: %.3fs\n', result.total_time);
fprintf('  Total Communication: %.2f MB\n', result.total_communication);
fprintf('  Final Rank: %d\n', result.rank_trajectory(end));

end

%% Helper Functions

function err = relative_error(A_hat, L)
% Relative recovery error
    err = norm(A_hat - L, 'fro') / norm(L, 'fro');
end

function psnr_val = psnr_image(img_rec, img_true)
% PSNR calculation (assumes images are normalized to [0,1])
    mse = mean((img_rec(:) - img_true(:)).^2);
    if mse == 0
        psnr_val = Inf;
    else
        psnr_val = 10 * log10(1 / mse);
    end
end

function ssim_val = ssim_image(img_rec, img_true)
% SSIM calculation (using MATLAB's built-in function)
    ssim_val = ssim(img_rec, img_true);
end

function volume = communication_volume(data_sizes)
% Communication volume calculation (MB)
    % data_sizes is a cell array of dimensions of data sent, e.g., {[m1, k], [m2, k], ...}
    total_bytes = 0;
    for i = 1:length(data_sizes)
        sz = data_sizes{i};
        total_bytes = total_bytes + prod(sz) * 8; % double precision float
    end
    volume = total_bytes / (1024^2); % Convert to MB
end

function dist = subspace_distance(U_true, U_approx)
% Subspace distance
    P = U_approx * U_approx';
    dist = norm((eye(size(P)) - P) * U_true, 'fro');
end

function err = singular_value_error(S_true, S_approx)
% Singular value error
    err = norm(diag(S_true) - diag(S_approx), 2) / norm(diag(S_true), 2);
end