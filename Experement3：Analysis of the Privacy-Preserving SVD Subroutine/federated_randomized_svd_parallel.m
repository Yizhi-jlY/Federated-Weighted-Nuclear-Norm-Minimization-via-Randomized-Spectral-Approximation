% Federated Randomized SVD Algorithm Main Function (Parallel Version)
% Implements randomized SVD in a federated learning environment using parallel computation.
% Inputs:
%     A_clients: Cell array of client data matrices, where each element is a client's local data matrix
%     params: Struct containing the following fields:
%       - k: Target rank (low-rank approximation dimension for SVD)
%       - p_over: Oversampling parameter, used to increase the stability of random projection
%       - rho: Diagonal decay factor for the ill-conditioned projection
%       - q: Number of power iterations
%       - enable_client_correction: Whether to enable client-side correction
%       - upload_corrected_sv: Whether to upload client-corrected singular values
% Outputs:
%     U_k: Global left singular vector matrix
%     S_k: Diagonal matrix of singular values
%     V_k: Right singular vector matrix
%     privacy_info: Analysis results of privacy protection effect

function [U_k, S_k, V_k, privacy_info] = federated_randomized_svd_parallel(A_clients, params)
    % --- SERVER: Initialization ---
    % Get the number of clients and data matrix dimensions
    num_clients = length(A_clients); % Number of clients
    [~, n] = size(A_clients{1}); % Number of columns in the data matrix
    d = params.k + params.p_over; % Target dimension of the random projection (target rank + oversampling)

    % --- Differential Privacy parameters (defaults keep legacy behavior) ---
    if ~isfield(params, 'dp_enable'),        params.dp_enable = false; end
    if ~isfield(params, 'dp_clip_C'),        params.dp_clip_C = 1.0; end
    if ~isfield(params, 'dp_epsilon'),       params.dp_epsilon = 1.0; end
    if ~isfield(params, 'dp_delta'),         params.dp_delta = 1e-5; end
    if ~isfield(params, 'dp_seed'),          params.dp_seed = 1234; end

    dp_sigma = 0;
    if params.dp_enable
        if params.dp_epsilon <= 0 || params.dp_delta <= 0
            error('dp_epsilon and dp_delta must be positive when dp_enable is true.');
        end
        dp_sigma = params.dp_clip_C * sqrt(2 * log(1.25 / params.dp_delta)) / params.dp_epsilon;
    end

    % Stage 1: Federated Range Exploration and Destructive Projection
    % Simplified Algorithm Comment:
    % An ill-conditioned projector, Omega_ill, is used to create low-dimensional sketches 
    % of client data. This projection enhances privacy by introducing a decay factor rho.

    % 1. Server: Construct and distribute the ill-conditioned projector
    [U_omega, ~] = qr(randn(n, n), 0); % Generate n x n random orthogonal matrix U_omega
    [V_omega, ~] = qr(randn(d, d), 0); % Generate d x d random orthogonal matrix V_omega
    D_omega = diag(params.rho.^(0:d-1)); % Construct the ill-conditioned diagonal matrix with decay factor rho
    Omega_ill = U_omega(:, 1:d) * D_omega * V_omega'; % Construct the ill-conditioned projector

    %Omega_ill = eye(n,d);

    % --- CLIENTS (Parallel): Initial Sketch Calculation ---
    % 2. Clients: Compute local data sketches
    % Simplified Algorithm Comment:
    % Each client computes Y_i = A_i * Omega_ill. Y_i is a low-dimensional sketch of the local data.
    Y_clients = cell(num_clients, 1); % Initialize cell array for client sketches
    parfor i = 1:num_clients
        Y_clients{i} = A_clients{i} * Omega_ill; % Client i computes its local sketch
    end

    % --- SERVER & CLIENTS: Power Iteration ---
    % 3. Power Iteration
    % Simplified Algorithm Comment:
    % Power iteration is used to better approximate the dominant subspace of A.
    % The global gradient G = sum(A_i' * Y_i) is aggregated on the server.
    for iter = 1:params.q
        % --- CLIENTS (Parallel): Calculate Local Gradients ---
        % Clients compute local gradients
        G_clients = cell(num_clients, 1); % Initialize cell array for client gradients
        parfor i = 1:num_clients
            G_clients{i} = A_clients{i}' * Y_clients{i}; % Client i computes its local gradient
        end

        % --- SERVER: Aggregate Global Gradient ---
        % Server aggregates the global gradient
        G = zeros(n, d); % Initialize global gradient matrix
        for i = 1:num_clients
            G = G + G_clients{i}; % Aggregate gradients from all clients
        end

        % --- CLIENTS (Parallel): Update Local Sketches ---
        % Clients update their local sketches
        parfor i = 1:num_clients
            Y_clients{i} = A_clients{i} * G; % Update local sketch using the global gradient
        end
    end

    % --- CLIENTS (Parallel): Local Orthogonalization and Projection ---
    % 4. Clients: Local Orthogonalization
    % Simplified Algorithm Comment:
    % Each client performs QR decomposition on its Y_i to get an orthogonal basis Q_Y_i.
    Q_Y_clients = cell(num_clients, 1); % Initialize cell array for client orthogonal bases
    R_Y_clients = cell(num_clients, 1); % Initialize cell array for client upper triangular matrices

    % 6. Clients: Compute local projected components
    % (This step is combined with step 4 in the parfor loop below)
    % Simplified Algorithm Comment:
    % Clients compute their local projected components B_prime_i = Q_Y_i' * A_i.
    B_prime_clients = cell(num_clients, 1); % Initialize cell array for client projected components

    % Pre-create streams for reproducible DP noise under parfor
    streams = cell(num_clients, 1);
    if params.dp_enable
        for i = 1:num_clients
            streams{i} = RandStream('CombRecursive', 'Seed', params.dp_seed + i);
        end
    end
    clip_scales = zeros(num_clients, 1);
    noise_norms = zeros(num_clients, 1);

    parfor i = 1:num_clients
        % Step 4: Client i performs QR decomposition
        [Q_Y_clients{i}, R_Y_clients{i}] = qr(Y_clients{i}, 0); 
        
        % Step 6: Client i computes its local projected component
        B_local = Q_Y_clients{i}' * A_clients{i};

        if params.dp_enable
            % Clip to enforce bounded sensitivity
            norm_B = norm(B_local, 'fro');
            if norm_B > 0
                scale = min(1, params.dp_clip_C / norm_B);
            else
                scale = 1;
            end
            B_clipped = B_local * scale;

            % Add Gaussian noise for (epsilon, delta)-DP
            noise = dp_sigma * randn(streams{i}, size(B_local));
            B_noisy = B_clipped + noise;

            B_prime_clients{i} = B_noisy;
            clip_scales(i) = scale;
            noise_norms(i) = norm(noise, 'fro');
        else
            B_prime_clients{i} = B_local;
        end
    end

    % --- SERVER: Aggregation and Central SVD ---
    % 5. Server: Aggregation and Global Orthogonalization
    % Simplified Algorithm Comment:
    % The server stacks all R_Y_i, performs a QR decomposition to get a global orthogonal basis,
    % and then splits it back into blocks for each client.
    R_prime_Y = vertcat(R_Y_clients{:}); % Stack all client R_Y_i matrices
    [Q_R, R_agg] = qr(R_prime_Y, 0); % Perform QR decomposition on the stacked matrix

    % Split Q_R into blocks for each client
    Q_R_clients = cell(num_clients, 1); % Initialize cell array for client Q_R blocks
    start_idx = 1;
    for i = 1:num_clients
        end_idx = start_idx + d - 1;
        Q_R_clients{i} = Q_R(start_idx:end_idx, :); % Assign Q_R block for client i
        start_idx = end_idx + 1;
    end

    % Stage 2: Federated Projection and Centralized SVD
    % Simplified Algorithm Comment:
    % The server aggregates client projections into a small matrix B and then 
    % performs SVD on B to get the core SVD components.
    
    % 7. Server: Construct and decompose the low-dimensional matrix
    B = zeros(d, n); % Initialize the low-dimensional matrix B
    for i = 1:num_clients
        B = B + Q_R_clients{i}' * B_prime_clients{i}; % Aggregate projected components from all clients
    end

    % Perform SVD on B
    [U_tilde_k, S_k, V_k] = svd(B, 'econ'); % Economy-size SVD
    U_tilde_k = U_tilde_k(:, 1:params.k); % Truncate to the top k left singular vectors
    S_k = S_k(1:params.k, 1:params.k); % Truncate to the top k singular values
    V_k = V_k(:, 1:params.k); % Truncate to the top k right singular vectors

    % Stage 3: Client-side SVD Correction
    % Simplified Algorithm Comment:
    % If enabled, clients compute their local shares of the left singular vectors. 
    % They can also optionally correct the singular values using their local data.
    if params.enable_client_correction
        % --- CLIENTS (Parallel): Calculate Local U and S contributions ---
        U_k_clients = cell(num_clients, 1); % Initialize cell array for client left singular vectors
        S_k_clients = cell(num_clients, 1); % This is needed for parfor classification

        parfor i = 1:num_clients
            % 8. Clients: Compute local left singular vector segments
            U_k_clients{i} = Q_Y_clients{i} * Q_R_clients{i} * U_tilde_k; % Client i computes its local left singular vectors
            
            % 9. Clients: Correct singular values using local full data
            if params.upload_corrected_sv
                S_k_clients{i} = U_k_clients{i}' * A_clients{i} * V_k; % Client i computes its local singular value contribution
            end
        end

        % --- SERVER: Aggregate and Finalize ---
        if params.upload_corrected_sv
            % 10. Server: Aggregate local singular value contributions
            S_k = zeros(params.k, params.k); % Initialize global singular value matrix
            for i = 1:num_clients
                S_k = S_k + S_k_clients{i}; % Aggregate singular value contributions from all clients
            end
        end

        % Construct the global left singular vectors
        U_k = vertcat(U_k_clients{:}); % Stack the left singular vectors from all clients
    else
        % --- CLIENTS (Parallel) & SERVER: Construct Global U without correction ---
        % Without client-side correction, directly construct the global left singular vectors
        
        % Client-side calculation of slices
        U_k_clients = cell(num_clients, 1);
        parfor i = 1:num_clients
             U_k_clients{i} = Q_Y_clients{i} * Q_R_clients{i} * U_tilde_k; % Calculate uncorrected left singular vector segments
        end
        
        % Server-side assembly
        U_k = vertcat(U_k_clients{:}); % Stack the uncorrected left singular vectors
    end

    % --- SERVER: Privacy Analysis ---
    % Privacy Protection Analysis
    % Call the analyze_privacy_protection function to analyze the effect of ill-conditioned projection and randomization on privacy
    privacy_info = analyze_privacy_protection(A_clients, Omega_ill, R_Y_clients, B, params);

    % Attach DP bookkeeping for downstream plotting/reporting
    privacy_info.dp = struct( ...
        'enabled', params.dp_enable, ...
        'clip_C', params.dp_clip_C, ...
        'epsilon', params.dp_epsilon, ...
        'delta', params.dp_delta, ...
        'sigma', dp_sigma, ...
        'clip_scales', clip_scales, ...
        'noise_fro_norms', noise_norms ...
    );
end