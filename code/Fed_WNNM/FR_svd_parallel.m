function [U_k_clients, S_k, V_k] = FR_svd_parallel(A_clients, k, p_over, rho, q, dp_opts)
% FR_svd_parallel - Main function for the Federated Randomized SVD algorithm.
% Implements randomized SVD in a federated learning environment.
%
% Inputs:
%   A_clients - Cell array of client data matrices, where each element is a 
%               local data matrix (a row-wise partition of the original matrix).
%   k         - Target rank (dimension of the low-rank approximation).
%   p_over    - Oversampling parameter to improve stability.
%   rho       - Diagonal decay factor for the ill-conditioned projector.
%   q         - Number of power iterations.
%
% Outputs:
%   U_k_clients - Cell array containing the local left singular vectors for each client.
%   S_k         - Diagonal matrix of singular values.
%   V_k         - Matrix of right singular vectors.

    % --- SERVER: Initialization ---
    % Get the number of clients and data dimensions.
    num_clients = length(A_clients); % Number of clients
    [~, n] = size(A_clients{1});     % Number of columns
    d = min(k + p_over, size(A_clients{1},1) * num_clients); % Target dimension for the random projection.

    % Differential Privacy options (optional)
    if nargin < 6 || isempty(dp_opts), dp_opts = struct(); end
    if ~isfield(dp_opts, 'enable'),   dp_opts.enable = false; end
    if ~isfield(dp_opts, 'clip_C'),   dp_opts.clip_C = 1.0; end
    if ~isfield(dp_opts, 'epsilon'),  dp_opts.epsilon = 1.0; end
    if ~isfield(dp_opts, 'delta'),    dp_opts.delta = 1e-5; end
    if ~isfield(dp_opts, 'seed'),     dp_opts.seed = 2025; end

    dp_sigma = 0;
    if dp_opts.enable
        if dp_opts.epsilon <= 0 || dp_opts.delta <= 0
            error('dp_opts.epsilon and dp_opts.delta must be positive when dp_opts.enable is true.');
        end
        dp_sigma = dp_opts.clip_C * sqrt(2 * log(1.25 / dp_opts.delta)) / dp_opts.epsilon;
    end

    % Stage 1: Federated Range Finding with Destructive Projection
    % The core idea is to create a low-dimensional sketch of the data using
    % a random projector. An ill-conditioned projector Omega_ill is used
    % here, which also helps in preserving privacy.

    % 1. Server: Construct and distribute the ill-conditioned projector.
    [U_omega, ~] = qr(randn(n, n), 0);       % Generate an n x n random orthogonal matrix U_omega.
    [V_omega, ~] = qr(randn(d, d), 0);       % Generate a d x d random orthogonal matrix V_omega.
    D_omega = diag(rho.^(0:d-1));           % Construct the ill-conditioned diagonal matrix with decay factor rho.
    Omega_ill = U_omega(:, 1:d) * D_omega * V_omega'; % Construct the ill-conditioned projector.


    % --- CLIENTS (Parallel): Initial Sketch Calculation ---
    % 2. Clients: Calculate local data sketches.
    % Each client projects its local data A_i onto a lower-dimensional space
    % by computing Y_i = A_i * Omega_ill. This creates a local 'sketch' of the data.
    Y_clients = cell(num_clients, 1);       % Initialize cell array for client sketches.
    parfor i = 1:num_clients
        Y_clients{i} = A_clients{i} * Omega_ill; % Client i computes its local sketch.
    end

    % --- SERVER & CLIENTS: Power Iteration ---
    % 3. Power Iteration.
    % Power iteration is used to improve the approximation of the column space
    % of the global matrix A. This is done by iteratively multiplying by A and A'.
    for iter = 1:q
        % --- CLIENTS (Parallel): Calculate Local Gradients ---
        % Clients compute local gradients.
        G_clients = cell(num_clients, 1);   % Initialize cell array for client gradients.
        parfor i = 1:num_clients
            G_clients{i} = A_clients{i}' * Y_clients{i}; % Client i computes its local gradient.
        end

        % --- SERVER: Aggregate Global Gradient ---
        % Server aggregates the global gradient.
        G = zeros(n, d);                    % Initialize the global gradient matrix.
        for i = 1:num_clients
            G = G + G_clients{i};           % Aggregate gradients from all clients.
        end

        % --- CLIENTS (Parallel): Update Local Sketches ---
        % Clients update their local sketches.
        parfor i = 1:num_clients
            Y_clients{i} = A_clients{i} * G; % Update local sketch using the global gradient.
        end
    end

    % --- CLIENTS (Parallel): Local Orthogonalization and Projection ---
    % 4. Clients: Local Orthogonalization.
    % Each client performs a QR decomposition on its local sketch Y_i to obtain
    % an orthonormal basis Q_Y_i for the range of Y_i.
    
    % 6. Clients: Compute local projected components.
    % (This step is combined with step 4 in the parfor loop below).
    % Each client then projects its local data A_i onto this basis: B_prime_i = Q_Y_i' * A_i.
    
    Q_Y_clients = cell(num_clients, 1);     % Initialize cell array for client orthonormal bases.
    R_Y_clients = cell(num_clients, 1);     % Initialize cell array for client upper triangular matrices.
    B_prime_clients = cell(num_clients, 1); % Initialize cell array for client projected components.
    m = zeros(num_clients, 1);              % Track block sizes for slicing

    % Pre-create streams for reproducible DP noise under parfor
    streams = cell(num_clients, 1);
    if dp_opts.enable
        for i = 1:num_clients
            streams{i} = RandStream('CombRecursive', 'Seed', dp_opts.seed + i);
        end
    end
    clip_scales = zeros(num_clients, 1);
    noise_norms = zeros(num_clients, 1);
    
    parfor i = 1:num_clients
        % Step 4: Client i performs QR decomposition.
        [Q_Y_clients{i}, R_Y_clients{i}] = qr(Y_clients{i}, 0); 
        m(i) = size(R_Y_clients{i}, 1);
        % Step 6: Client i computes its local projected component.
        B_local = Q_Y_clients{i}' * A_clients{i};

        if dp_opts.enable
            norm_B = norm(B_local, 'fro');
            if norm_B > 0
                scale = min(1, dp_opts.clip_C / norm_B);
            else
                scale = 1;
            end
            B_clipped = B_local * scale;
            noise = dp_sigma * randn(streams{i}, size(B_local));
            B_local = B_clipped + noise;
            clip_scales(i) = scale;
            noise_norms(i) = norm(noise, 'fro');
        end

        B_prime_clients{i} = B_local; 
    end

    % --- SERVER: Aggregation and Central SVD ---
    % 5. Server: Aggregate and perform global orthogonalization.
    % The server stacks the R_Y_i matrices from all clients and performs another
    % QR decomposition. This step combines the local bases into a global orthonormal basis.
    R_prime_Y = vertcat(R_Y_clients{:});    % Vertically stack all client R_Y_i matrices.
    [Q_R, R_agg] = qr(R_prime_Y, 0);        % Perform QR decomposition on the stacked matrix.

    % Split Q_R into blocks for each client.
    Q_R_clients = cell(num_clients, 1);     % Initialize cell array for client Q_R blocks.
    start_idx = 1;
    for i = 1:num_clients
        end_idx = start_idx + m(i) - 1;
        Q_R_clients{i} = Q_R(start_idx:end_idx, :); % Assign a block of Q_R to client i.
        start_idx = end_idx + 1;
    end

    % Stage 2: Federated Projection and Centralized SVD
    % The server computes the core low-dimensional matrix B by aggregating the
    % projected components from all clients. A final SVD is performed on this small matrix B.
    
    % 7. Server: Construct and decompose the low-dimensional matrix.
    B = zeros(d, n);                        % Initialize the low-dimensional matrix B.
    for i = 1:num_clients
        B = B + Q_R_clients{i}' * B_prime_clients{i}; % Aggregate projected components from all clients.
    end

    % Perform SVD on B.
    [U_tilde_k, S_k, V_k] = svd(B, 'econ'); % 'econ' for economy-size SVD.


    % Stage 3: Client-side SVD Correction
    % The final left singular vectors U are reconstructed locally at each client.
    
    % --- CLIENTS (Parallel): Calculate Local U contributions ---
    U_k_clients = cell(num_clients, 1);     % Initialize cell array for client left singular vectors.

    parfor i = 1:num_clients
        % 8. Clients: Compute local left singular vector slices.
        U_k_clients{i} = Q_Y_clients{i} * Q_R_clients{i} * U_tilde_k; % Client i computes its local left singular vectors.
    end

end