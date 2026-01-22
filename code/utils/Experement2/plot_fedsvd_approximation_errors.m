function plot_fedsvd_approximation_errors(svd_errors, save_dir)
% Plots the approximation error of the Federated SVD.
figure('Name', 'FedSVD Approximation Error', 'NumberTitle', 'off', 'Position', [100, 100, 1200, 500]);

% Plot 1: vs. Number of Power Iterations (q)
subplot(1, 2, 1);
yyaxis left;
plot(svd_errors.q_range, svd_errors.singular_value_error_vs_q, '-o', 'LineWidth', 2, 'DisplayName', 'Singular Value Error');
ylabel('Relative Singular Value Error');
ylim([0, max(svd_errors.singular_value_error_vs_q) * 1.2]);

yyaxis right;
plot(svd_errors.q_range, svd_errors.subspace_distance_U_vs_q, '-s', 'LineWidth', 2, 'DisplayName', 'Subspace Distance');
ylabel('Subspace Distance ||(I-UU^T)U_{true}||_F');
ylim([0, max(svd_errors.subspace_distance_U_vs_q) * 1.2]);

grid on;
title('FedSVD Approximation Error vs. Power Iterations (q)');
xlabel('Number of Power Iterations (q)');
legend('show');

% Plot 2: vs. Oversampling Parameter p_over (if data is valid)
subplot(1, 2, 2);
if ~all(isnan(svd_errors.singular_value_error_vs_p_over))
    % (Plotting code reserved for future expansion)
    title('FedSVD Approximation Error vs. Oversampling (p_{over})');
    xlabel('Oversampling Parameter (p_{over})');
else
    text(0.5, 0.5, 'p_{over} analysis was not performed', 'HorizontalAlignment', 'center', 'FontSize', 12);
    axis off;
end

filename = fullfile(save_dir, 'plot_fedsvd_approximation_errors.png');
saveas(gcf, filename);
fprintf('Federated SVD approximation error plot saved to: %s\n', filename);
end