function parameters = get_algorithm_parameters(algo_name, base_params, L_true, U_true, S_true)
% Generates algorithm-specific parameter struct based on the algorithm name and base parameters.
parameters = struct('L_true', L_true, 'U_true', U_true, 'S_true', S_true, ...
                    'maxiter', base_params.maxiter, 'tol', base_params.tol);
parameters.m = base_params.m;
parameters.n = base_params.n;

switch algo_name
    case 'factGDNew'
        parameters.r = base_params.r;
        parameters.p = base_params.p_default;
        parameters.Tsvd = 15;
        parameters.step_const = 0.75;
        parameters.sampling_rate = base_params.p_obs;
    case 'altGDMin_T'
        parameters.r = base_params.r;
        parameters.p = base_params.p_default;
        parameters.eta_c = 1.0;
        parameters.Tsvd = 15;
    case 'altMinPrvt_T'
        parameters.p = base_params.p_default;
        parameters.rank = base_params.r;
        parameters.p_obs = base_params.p_obs;
        parameters.T_inner = 10;
        parameters.Tsvd = 15;
    case 'SVT_Rand'
        parameters.tao = 0.1519 * sqrt(base_params.m * base_params.n);
        parameters.step = 1.2;
        parameters.r = base_params.r + 20;
        parameters.p_over = base_params.p_over;
        parameters.q = base_params.q;
        parameters.use_rand_svd = true;
    case 'FedSVT_MC'
        parameters.r = base_params.r + 20;
        parameters.p = base_params.p_default;
        parameters.tau = 2.9409 * sqrt(base_params.m * base_params.n);
        parameters.delta0 = 4.8049;
        parameters.gamma = 0.72687;
        parameters.p_over = base_params.p_over;
        parameters.rho = 1;
        parameters.q = base_params.q;
    case 'FedWNNM_MC'
        parameters.p = base_params.p_default;
        parameters.C = 0.773 * sqrt(base_params.m * base_params.n);
        parameters.myeps = 1e-6;
        parameters.p_over = base_params.p_over;
        parameters.rho = 1;
        parameters.q = base_params.q;
    otherwise
        % Add configurations for other algorithms...
end
end