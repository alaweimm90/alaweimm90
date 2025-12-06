function physics_informed_advanced()
% PHYSICS_INFORMED_ADVANCED - Advanced Physics-Informed Neural Networks
%
% This example demonstrates advanced physics-informed neural networks (PINNs)
% for solving partial differential equations in scientific computing.
% We'll solve heat and wave equations with complex boundary conditions.
%
% Learning Objectives:
%   - Understand physics-informed neural networks
%   - Learn PDE residual computation
%   - Implement boundary and initial conditions
%   - Analyze solution accuracy and convergence
%   - Apply to multiple PDE types
%
% Author: Berkeley SciComp Team
% Date: 2024
fprintf('🌌 Berkeley SciComp: Advanced Physics-Informed Networks\n');
fprintf('==================================================\n');
fprintf('Solving PDEs with physics-informed neural networks\n');
fprintf('We''ll tackle heat and wave equations with complex physics.\n\n');
try
    % Solve heat equation with PINN
    fprintf('🔥 Solving Heat Equation with PINN...\n');
    heat_results = solve_heat_equation_pinn();
    % Solve wave equation with PINN
    fprintf('\n🌊 Solving Wave Equation with PINN...\n');
    wave_results = solve_wave_equation_pinn();
    % Compare with analytical solutions
    compare_with_analytical(heat_results, wave_results);
    % Advanced techniques demonstration
    demonstrate_advanced_techniques();
    % Physics insights
    physics_insights();
    fprintf('\n✨ Advanced example completed successfully!\n');
    fprintf('\nKey Takeaways:\n');
    fprintf('• PINNs can solve complex PDEs without traditional discretization\n');
    fprintf('• Physics constraints improve solution accuracy\n');
    fprintf('• Automatic differentiation enables gradient computation\n');
    fprintf('• Neural networks provide continuous solution representations\n');
    fprintf('\nNext Steps:\n');
    fprintf('• Explore inverse problem solving with PINNs\n');
    fprintf('• Try multi-physics coupled problems\n');
    fprintf('• Implement uncertainty quantification\n');
catch ME
    fprintf('❌ Error occurred: %s\n', ME.message);
    fprintf('Please check your installation and try again.\n');
end
end
function results = solve_heat_equation_pinn()
% Solve 1D heat equation using PINN
%
% PDE: ∂u/∂t = α ∂²u/∂x²
% Domain: x ∈ [0, 1], t ∈ [0, 0.5]
% BC: u(0, t) = u(1, t) = 0
% IC: u(x, 0) = sin(πx)
fprintf('\nSetting up heat equation PINN...\n');
% Problem parameters
alphaᵯ = 1.0;  % Thermal diffusivity
xDomain = [0, 1];
tDomain = [0, 0.5];
% Create PINN
layers = [2, 50, 50, 50, 1];  % [x, t] -> u
pinn = physics_informed.PINN(layers, ...
                            'Activation', 'tanh', ...
                            'PDEWeight', 1.0, ...
                            'BCWeight', 10.0, ...
                            'ICWeight', 10.0, ...
                            'LearningRate', 0.001);
% Custom boundary and initial conditions
pinn.setBoundaryConditions(@(x, t) heat_boundary_conditions(x, t));
pinn.setInitialConditions(@(x) heat_initial_condition(x));
% Train PINN
fprintf('Training PINN for heat equation...\n');
training_results = pinn.train(xDomain, tDomain, ...
                             'EquationType', 'heat', ...
                             'PDEParams', struct('Diffusivity', alpha), ...
                             'NPDEPoints', 10000, ...
                             'NBCPoints', 200, ...
                             'NICPoints', 200, ...
                             'Epochs', 2000, ...
                             'Verbose', true);
% Generate solution on grid
fprintf('Generating solution grid...\n');
nx = 101;
nt = 51;
x_grid = linspace(xDomain(1), xDomain(2), nx);
t_grid = linspace(tDomain(1), tDomain(2), nt);
[X_grid, T_grid] = meshgrid(x_grid, t_grid);
% Predict solution
X_flat = X_grid(:);
T_flat = T_grid(:);
U_pred = pinn.predict(X_flat, T_flat);
U_grid = reshape(U_pred, size(X_grid));
% Analytical solution for comparison
U_analytical = analytical_heat_solution(X_grid, T_grid, alpha);
% Compute error
error = abs(U_grid - U_analytical);
relative_error = error ./ (abs(U_analytical) + 1e-8);
fprintf('Solution computed successfully!\n');
fprintf('Mean absolute error: %.6f\n', mean(error(:)));
fprintf('Max absolute error: %.6f\n', max(error(:)));
fprintf('Mean relative error: %.4f%%\n', mean(relative_error(:)) * 100);
% Visualize results
visualize_heat_solution(X_grid, T_grid, U_grid, U_analytical, error, training_results);
% Store results
results = struct();
results.type = 'heat';
results.X_grid = X_grid;
results.T_grid = T_grid;
results.U_pred = U_grid;
results.U_analytical = U_analytical;
results.error = error;
results.training_results = training_results;
results.pinn = pinn;
end
function results = solve_wave_equation_pinn()
% Solve 1D wave equation using PINN
%
% PDE: ∂²u/∂t² = c² ∂²u/∂x²
% Domain: x ∈ [0, 1], t ∈ [0, 1]
% BC: u(0, t) = u(1, t) = 0
% IC: u(x, 0) = sin(πx), ∂u/∂t(x, 0) = 0
fprintf('\nSetting up wave equation PINN...\n');
% Problem parameters
c = 1.0;  % Wave speed
xDomain = [0, 1];
tDomain = [0, 1];
% Create PINN
layers = [2, 64, 64, 64, 1];  % [x, t] -> u
pinn = physics_informed.PINN(layers, ...
                            'Activation', 'tanh', ...
                            'PDEWeight', 1.0, ...
                            'BCWeight', 10.0, ...
                            'ICWeight', 10.0, ...
                            'LearningRate', 0.001);
% Custom boundary and initial conditions
pinn.setBoundaryConditions(@(x, t) wave_boundary_conditions(x, t));
pinn.setInitialConditions(@(x) wave_initial_conditions(x));
% Train PINN
fprintf('Training PINN for wave equation...\n');
training_results = pinn.train(xDomain, tDomain, ...
                             'EquationType', 'wave', ...
                             'PDEParams', struct('WaveSpeed', c), ...
                             'NPDEPoints', 15000, ...
                             'NBCPoints', 300, ...
                             'NICPoints', 300, ...
                             'Epochs', 3000, ...
                             'Verbose', true);
% Generate solution on grid
fprintf('Generating solution grid...\n');
nx = 101;
nt = 101;
x_grid = linspace(xDomain(1), xDomain(2), nx);
t_grid = linspace(tDomain(1), tDomain(2), nt);
[X_grid, T_grid] = meshgrid(x_grid, t_grid);
% Predict solution
X_flat = X_grid(:);
T_flat = T_grid(:);
U_pred = pinn.predict(X_flat, T_flat);
U_grid = reshape(U_pred, size(X_grid));
% Analytical solution for comparison
U_analytical = analytical_wave_solution(X_grid, T_grid, c);
% Compute error
error = abs(U_grid - U_analytical);
relative_error = error ./ (abs(U_analytical) + 1e-8);
fprintf('Solution computed successfully!\n');
fprintf('Mean absolute error: %.6f\n', mean(error(:)));
fprintf('Max absolute error: %.6f\n', max(error(:)));
fprintf('Mean relative error: %.4f%%\n', mean(relative_error(:)) * 100);
% Visualize results
visualize_wave_solution(X_grid, T_grid, U_grid, U_analytical, error, training_results);
% Store results
results = struct();
results.type = 'wave';
results.X_grid = X_grid;
results.T_grid = T_grid;
results.U_pred = U_grid;
results.U_analytical = U_analytical;
results.error = error;
results.training_results = training_results;
results.pinn = pinn;
end
function bc = heat_boundary_conditions(x, t)
% Boundary conditions for heat equation: u(0, t) = u(1, t) = 0
bc.left = zeros(size(t));
bc.right = zeros(size(t));
end
function ic = heat_initial_condition(x)
% Initial condition for heat equation: u(x, 0) = sin(πx)
ic = sin(pi * x);
end
function bc = wave_boundary_conditions(x, t)
% Boundary conditions for wave equation: u(0, t) = u(1, t) = 0
bc.left = zeros(size(t));
bc.right = zeros(size(t));
end
function ic = wave_initial_conditions(x)
% Initial conditions for wave equation
% u(x, 0) = sin(πx), ∂u/∂t(x, 0) = 0
ic.u = sin(pi * x);
ic.ut = zeros(size(x));
end
function U = analytical_heat_solution(X, T, alpha)
% Analytical solution for heat equation
% u(x, t) = sin(πx) * exp(-π² * α * t)
U = sin(pi * X) .* exp(-pi^2 * alpha * T);
end
function U = analytical_wave_solution(X, T, c)
% Analytical solution for wave equation
% u(x, t) = sin(πx) * cos(πct)
U = sin(pi * X) .* cos(pi * c * T);
end
function visualize_heat_solution(X, T, U_pred, U_analytical, error, training_results)
% Visualize heat equation results
% Berkeley colors
berkeleyBlue = [0, 50, 98]/255;
californiaGold = [253, 181, 21]/255;
figure('Position', [100, 100, 1600, 1200]);
% Predicted solution
subplot(2, 3, 1);
contourf(X, T, U_pred, 50, 'LineStyle', 'none');
colorbar;
xlabel('x');
ylabel('t');
title('PINN Solution');
% Analytical solution
subplot(2, 3, 2);
contourf(X, T, U_analytical, 50, 'LineStyle', 'none');
colorbar;
xlabel('x');
ylabel('t');
title('Analytical Solution');
% Error
subplot(2, 3, 3);
contourf(X, T, error, 50, 'LineStyle', 'none');
colormap(gca, 'hot');
colorbar;
xlabel('x');
ylabel('t');
title('Absolute Error');
% Solution at different times
subplot(2, 3, 4);
t_snapshots = [0, 0.1, 0.2, 0.3, 0.4, 0.5];
colors = lines(length(t_snapshots));
for i = 1:length(t_snapshots)
    t_idx = find(T(:, 1) >= t_snapshots(i), 1);
    if ~isempty(t_idx)
        plot(X(1, :), U_pred(t_idx, :), 'Color', colors(i, :), 'LineWidth', 2, ...
             'DisplayName', sprintf('t = %.1f', t_snapshots(i)));
        hold on;
        plot(X(1, :), U_analytical(t_idx, :), '--', 'Color', colors(i, :), 'LineWidth', 1);
    end
end
xlabel('x');
ylabel('u(x, t)');
title('Solution Snapshots (solid: PINN, dashed: analytical)');
legend('Location', 'best');
grid on;
% Training loss
subplot(2, 3, 5);
semilogy(training_results.lossHistory, 'Color', berkeleyBlue, 'LineWidth', 2);
hold on;
semilogy(training_results.pdeLossHistory, 'Color', californiaGold, 'LineWidth', 2);
semilogy(training_results.bcLossHistory, 'Color', 'red', 'LineWidth', 2);
semilogy(training_results.icLossHistory, 'Color', 'green', 'LineWidth', 2);
xlabel('Epoch');
ylabel('Loss');
title('Training Loss Components');
legend({'Total', 'PDE', 'BC', 'IC'}, 'Location', 'best');
grid on;
% Error statistics
subplot(2, 3, 6);
error_stats = [mean(error(:)), std(error(:)), max(error(:)), min(error(:))];
stat_names = {'Mean', 'Std', 'Max', 'Min'};
bar(1:4, error_stats, 'FaceColor', berkeleyBlue);
set(gca, 'XTickLabel', stat_names);
ylabel('Absolute Error');
title('Error Statistics');
grid on;
sgtitle('Heat Equation PINN Results', 'FontSize', 16, 'FontWeight', 'bold');
end
function visualize_wave_solution(X, T, U_pred, U_analytical, error, training_results)
% Visualize wave equation results
% Berkeley colors
berkeleyBlue = [0, 50, 98]/255;
californiaGold = [253, 181, 21]/255;
figure('Position', [200, 100, 1600, 1200]);
% Predicted solution
subplot(2, 3, 1);
contourf(X, T, U_pred, 50, 'LineStyle', 'none');
colorbar;
xlabel('x');
ylabel('t');
title('PINN Solution');
% Analytical solution
subplot(2, 3, 2);
contourf(X, T, U_analytical, 50, 'LineStyle', 'none');
colorbar;
xlabel('x');
ylabel('t');
title('Analytical Solution');
% Error
subplot(2, 3, 3);
contourf(X, T, error, 50, 'LineStyle', 'none');
colormap(gca, 'hot');
colorbar;
xlabel('x');
ylabel('t');
title('Absolute Error');
% Solution at different times
subplot(2, 3, 4);
t_snapshots = [0, 0.25, 0.5, 0.75, 1.0];
colors = lines(length(t_snapshots));
for i = 1:length(t_snapshots)
    t_idx = find(T(:, 1) >= t_snapshots(i), 1);
    if ~isempty(t_idx)
        plot(X(1, :), U_pred(t_idx, :), 'Color', colors(i, :), 'LineWidth', 2, ...
             'DisplayName', sprintf('t = %.2f', t_snapshots(i)));
        hold on;
        plot(X(1, :), U_analytical(t_idx, :), '--', 'Color', colors(i, :), 'LineWidth', 1);
    end
end
xlabel('x');
ylabel('u(x, t)');
title('Wave Snapshots (solid: PINN, dashed: analytical)');
legend('Location', 'best');
grid on;
% Training loss
subplot(2, 3, 5);
semilogy(training_results.lossHistory, 'Color', berkeleyBlue, 'LineWidth', 2);
hold on;
semilogy(training_results.pdeLossHistory, 'Color', californiaGold, 'LineWidth', 2);
semilogy(training_results.bcLossHistory, 'Color', 'red', 'LineWidth', 2);
semilogy(training_results.icLossHistory, 'Color', 'green', 'LineWidth', 2);
xlabel('Epoch');
ylabel('Loss');
title('Training Loss Components');
legend({'Total', 'PDE', 'BC', 'IC'}, 'Location', 'best');
grid on;
% Wave energy analysis
subplot(2, 3, 6);
% Compute energy at different times
t_indices = 1:10:size(T, 1);
energy_pred = zeros(size(t_indices));
energy_analytical = zeros(size(t_indices));
for i = 1:length(t_indices)
    idx = t_indices(i);
    % Approximate energy as integral of u^2
    energy_pred(i) = trapz(X(1, :), U_pred(idx, :).^2);
    energy_analytical(i) = trapz(X(1, :), U_analytical(idx, :).^2);
end
plot(T(t_indices, 1), energy_pred, 'Color', berkeleyBlue, 'LineWidth', 2, 'DisplayName', 'PINN');
hold on;
plot(T(t_indices, 1), energy_analytical, '--', 'Color', californiaGold, 'LineWidth', 2, 'DisplayName', 'Analytical');
xlabel('Time');
ylabel('Energy');
title('Wave Energy Conservation');
legend('Location', 'best');
grid on;
sgtitle('Wave Equation PINN Results', 'FontSize', 16, 'FontWeight', 'bold');
end
function compare_with_analytical(heat_results, wave_results)
% Compare PINN solutions with analytical solutions
fprintf('\n📊 Comparing PINN vs Analytical Solutions\n');
fprintf('==========================================\n');
% Heat equation comparison
heat_error = heat_results.error;
heat_mean_error = mean(heat_error(:));
heat_max_error = max(heat_error(:));
heat_rel_error = mean(heat_error(:) ./ (abs(heat_results.U_analytical(:)) + 1e-8)) * 100;
fprintf('Heat Equation Results:\n');
fprintf('  Mean absolute error: %.6f\n', heat_mean_error);
fprintf('  Max absolute error: %.6f\n', heat_max_error);
fprintf('  Mean relative error: %.4f%%\n', heat_rel_error);
% Wave equation comparison
wave_error = wave_results.error;
wave_mean_error = mean(wave_error(:));
wave_max_error = max(wave_error(:));
wave_rel_error = mean(wave_error(:) ./ (abs(wave_results.U_analytical(:)) + 1e-8)) * 100;
fprintf('\nWave Equation Results:\n');
fprintf('  Mean absolute error: %.6f\n', wave_mean_error);
fprintf('  Max absolute error: %.6f\n', wave_max_error);
fprintf('  Mean relative error: %.4f%%\n', wave_rel_error);
% Convergence analysis
figure('Position', [300, 100, 1200, 800]);
% Heat equation convergence
subplot(2, 2, 1);
semilogy(heat_results.training_results.lossHistory, 'Color', [0, 50, 98]/255, 'LineWidth', 2);
xlabel('Epoch');
ylabel('Total Loss');
title('Heat Equation Convergence');
grid on;
% Wave equation convergence
subplot(2, 2, 2);
semilogy(wave_results.training_results.lossHistory, 'Color', [253, 181, 21]/255, 'LineWidth', 2);
xlabel('Epoch');
ylabel('Total Loss');
title('Wave Equation Convergence');
grid on;
% Error comparison
subplot(2, 2, 3);
error_comparison = [heat_mean_error, wave_mean_error; ...
                   heat_max_error, wave_max_error];
bar(error_comparison);
set(gca, 'XTickLabel', {'Mean Error', 'Max Error'});
ylabel('Absolute Error');
title('Error Comparison');
legend({'Heat', 'Wave'}, 'Location', 'best');
grid on;
% Relative error comparison
subplot(2, 2, 4);
rel_error_comparison = [heat_rel_error, wave_rel_error];
bar(rel_error_comparison);
set(gca, 'XTickLabel', {'Heat', 'Wave'});
ylabel('Relative Error (%)');
title('Relative Error Comparison');
grid on;
sgtitle('PINN vs Analytical Solution Comparison', 'FontSize', 14, 'FontWeight', 'bold');
end
function demonstrate_advanced_techniques()
% Demonstrate advanced PINN techniques
fprintf('\n🚀 Advanced PINN Techniques\n');
fprintf('============================\n');
fprintf('Demonstrating advanced techniques:\n');
fprintf('• Adaptive loss weighting\n');
fprintf('• Transfer learning\n');
fprintf('• Multi-task learning\n');
fprintf('• Uncertainty quantification\n');
fprintf('\n');
fprintf('These techniques can significantly improve PINN performance:\n');
fprintf('• Adaptive weighting balances loss components automatically\n');
fprintf('• Transfer learning leverages solutions from similar problems\n');
fprintf('• Multi-task learning solves related PDEs simultaneously\n');
fprintf('• Uncertainty quantification provides solution confidence\n');
end
function physics_insights()
% Provide physics insights about PINNs
fprintf('\n🌌 Physics Insights\n');
fprintf('==================\n');
fprintf('Physics-Informed Neural Networks (PINNs):\n');
fprintf('\n');
fprintf('Key Advantages:\n');
fprintf('• Mesh-free solution of PDEs\n');
fprintf('• Continuous solution representation\n');
fprintf('• Automatic incorporation of physics laws\n');
fprintf('• Capability to handle complex geometries\n');
fprintf('• Data-efficient for inverse problems\n');
fprintf('\n');
fprintf('Heat Equation Physics:\n');
fprintf('• Describes diffusion processes\n');
fprintf('• Temperature distribution in materials\n');
fprintf('• Concentration gradients in chemistry\n');
fprintf('• Financial option pricing models\n');
fprintf('\n');
fprintf('Wave Equation Physics:\n');
fprintf('• Describes oscillatory phenomena\n');
fprintf('• Sound and electromagnetic waves\n');
fprintf('• Seismic wave propagation\n');
fprintf('• Quantum mechanical wave functions\n');
fprintf('\n');
fprintf('Future Directions:\n');
fprintf('• Multi-physics coupled problems\n');
fprintf('• Inverse parameter identification\n');
fprintf('• Real-time PDE solutions\n');
fprintf('• Quantum computing integration\n');
end