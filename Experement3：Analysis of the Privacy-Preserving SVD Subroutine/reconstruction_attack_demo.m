function reconstruction_attack_demo()
% reconstruction_attack_demo - Visual reconstruction attack illustration
% Shows original image vs. non-DP reconstruction vs. DP-protected reconstruction.

    rng(123);

    % --- Load image ---
    img_path = fullfile('..', 'datasets', 'cbsd68t', '0000.png');
    if ~isfile(img_path)
        error('Image not found: %s', img_path);
    end
    I = im2double(imread(img_path));
    if size(I, 3) > 1
        I = rgb2gray(I); % use luminance for simplicity
    end

    % --- Prepare federated slices ---
    [m_total, n] = size(I);
    num_clients = 4;
    rows_per_client = floor(m_total / num_clients);
    A_clients = cell(num_clients, 1);
    for i = 1:num_clients
        s = (i - 1) * rows_per_client + 1;
        if i == num_clients
            e = m_total; % last client takes remainder
        else
            e = i * rows_per_client;
        end
        A_clients{i} = I(s:e, :);
    end

    % --- Shared params ---
    base_params = struct('k', 40, 'p_over', 8, 'rho', 0.1, 'q', 1, ...
                         'enable_client_correction', false, 'upload_corrected_sv', false);

    % No-protection reconstruction
    params_no_dp = base_params;
    params_no_dp.dp_enable = false;
    [U0, S0, V0] = federated_randomized_svd_parallel(A_clients, params_no_dp);
    recon_no_dp = U0 * S0 * V0';

    % DP-protected reconstruction
    params_dp = base_params;
    params_dp.dp_enable = true;
    params_dp.dp_clip_C = 5;      % clipping bound
    params_dp.dp_epsilon = 0.5;   % stronger privacy -> more noise
    params_dp.dp_delta = 1e-5;
    params_dp.dp_seed = 999;
    [Udp, Sdp, Vdp] = federated_randomized_svd_parallel(A_clients, params_dp);
    recon_dp = Udp * Sdp * Vdp';

    % Clip back to [0,1] for display
    recon_no_dp = min(max(recon_no_dp, 0), 1);
    recon_dp = min(max(recon_dp, 0), 1);

    % --- Plot results ---
    h = figure('Color', 'w');
    set(gcf, 'Position', [160, 160, 1200, 360]);
    subplot(1,3,1); imshow(I); title('Original');
    subplot(1,3,2); imshow(recon_no_dp); title('No protection');
    subplot(1,3,3); imshow(recon_dp); title('(\epsilon,\delta)-DP protection');

    exportgraphics(h, 'reconstruction_attack.png', 'Resolution', 300);
    exportgraphics(h, 'reconstruction_attack.pdf');
    fprintf('Saved reconstruction attack figure to reconstruction_attack.png/pdf\n');
end
