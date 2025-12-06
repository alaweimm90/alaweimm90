%% Basic PDE Examples - Beginner Level
%  Introduction to solving PDEs with the Berkeley SciComp framework
%
%  Author: Berkeley SciComp Team
%  Date: 2024
%% Setup
clear; close all; clc;
% Add the ODE_PDE core directory to path
addpath('../../core');
% Berkeley colors
berkeley_blue = [0, 50, 98]/255;
california_gold = [253, 181, 21]/255;
berkeley_light_blue = [59, 126, 161]/255;
fprintf('=== Basic PDE Examples - Beginner Level ===\n\n');
%% Example 1: Steady Heat Equation (Poisson Problem)
fprintf('Example 1: Steady Heat Equation\n');
fprintf('-------------------------------\n');
fprintf('Equation: -α∇²u = f(x)\n');
fprintf('Domain: x ∈ [0, 1]\n');
fprintf('Boundary conditions: u(0) = 0, u(1) = 0\n');
fprintf('Source: f(x) = π²sin(πx)\n');
fprintf('Analytical solution: u(x) = sin(πx)/α\n\n');
% Setup domain
x = linspace(0, 1, 51);
domain.x = x;
% Boundary conditions: u(0) = 0, u(1) = 0
bc_steady.dirichlet = containers.Map({0, 50}, {0, 0});
% Thermal diffusivity
alpha = 0.1;
% Create solver
heat_solver = HeatEquationSolver(domain, bc_steady, alpha);
% Source term: f(x) = π²sin(πx)
source_steady = @(x) pi^2 * sin(pi * x);
% Solve steady problem
result_steady = heat_solver.solve_steady(source_steady);
% Analytical solution
u_analytical_steady = sin(pi * x) / alpha;
% Plot results
figure('Position', [100, 100, 800, 500]);
subplot(2, 1, 1);
plot(x, u_analytical_steady, 'k--', 'LineWidth', 2, 'DisplayName', 'Analytical');
hold on;
plot(x, result_steady.u, 'o-', 'Color', berkeley_blue, 'LineWidth', 1.5, ...
    'MarkerSize', 4, 'DisplayName', 'Numerical');
hold off;
title('Steady Heat Equation Solution', 'FontSize', 14, 'Color', berkeley_blue);
xlabel('x', 'FontSize', 12);
ylabel('Temperature u(x)', 'FontSize', 12);
legend('show', 'Location', 'best');
grid on;
% Error analysis
error_steady = abs(result_steady.u' - u_analytical_steady);
subplot(2, 1, 2);
semilogy(x, error_steady, 'Color', california_gold, 'LineWidth', 2);
title('Absolute Error', 'FontSize', 12, 'Color', berkeley_blue);
xlabel('x', 'FontSize', 11);
ylabel('|Error|', 'FontSize', 11);
grid on;
max_error_steady = max(error_steady);
fprintf('Maximum error in steady solution: %.6e\n', max_error_steady);
%% Example 2: Transient Heat Equation
fprintf('\nExample 2: Transient Heat Equation\n');
fprintf('----------------------------------\n');
fprintf('Equation: ∂u/∂t = α∇²u\n');
fprintf('Domain: x ∈ [0, 1], t ∈ [0, T]\n');
fprintf('Boundary conditions: u(0,t) = 0, u(1,t) = 0\n');
fprintf('Initial condition: u(x,0) = sin(πx)\n');
fprintf('Analytical solution: u(x,t) = sin(πx)exp(-π²αt)\n\n');
% Setup (reuse domain and boundary conditions)
% Initial condition
initial_condition = @(x) sin(pi * x);
% Time parameters
T_final = 0.5;
dt = 0.001;
% Solve transient problem
result_transient = heat_solver.solve_transient(initial_condition, ...
    'time_span', [0, T_final], 'dt', dt);
if result_transient.success
    % Analytical solution for comparison
    [X_mesh, T_mesh] = meshgrid(x, result_transient.t);
    U_analytical = sin(pi * X_mesh) .* exp(-pi^2 * alpha * T_mesh);
    % Plot surface
    figure('Position', [150, 150, 1200, 400]);
    subplot(1, 3, 1);
    surf(X_mesh, T_mesh, result_transient.u, 'EdgeColor', 'none');
    colormap([linspace(1, berkeley_blue(1), 64)', ...
              linspace(1, berkeley_blue(2), 64)', ...
              linspace(1, berkeley_blue(3), 64)']);
    title('Numerical Solution', 'FontSize', 12, 'Color', berkeley_blue);
    xlabel('x', 'FontSize', 11);
    ylabel('Time', 'FontSize', 11);
    zlabel('Temperature', 'FontSize', 11);
    view(45, 30);
    subplot(1, 3, 2);
    surf(X_mesh, T_mesh, U_analytical, 'EdgeColor', 'none');
    colormap([linspace(1, california_gold(1), 64)', ...
              linspace(1, california_gold(2), 64)', ...
              linspace(1, california_gold(3), 64)']);
    title('Analytical Solution', 'FontSize', 12, 'Color', berkeley_blue);
    xlabel('x', 'FontSize', 11);
    ylabel('Time', 'FontSize', 11);
    zlabel('Temperature', 'FontSize', 11);
    view(45, 30);
    % Error surface
    subplot(1, 3, 3);
    error_transient = abs(result_transient.u - U_analytical);
    surf(X_mesh, T_mesh, error_transient, 'EdgeColor', 'none');
    title('Absolute Error', 'FontSize', 12, 'Color', berkeley_blue);
    xlabel('x', 'FontSize', 11);
    ylabel('Time', 'FontSize', 11);
    zlabel('|Error|', 'FontSize', 11);
    view(45, 30);
    colorbar;
    % Time snapshots
    figure('Position', [200, 200, 800, 500]);
    t_snapshots = [0, 0.1, 0.2, 0.3, 0.5];
    colors = [berkeley_blue; california_gold; berkeley_light_blue;
              [196, 130, 14]/255; [0, 176, 218]/255];
    hold on;
    for i = 1:length(t_snapshots)
        t_snap = t_snapshots(i);
        % Find closest time index
        [~, t_idx] = min(abs(result_transient.t - t_snap));
        plot(x, result_transient.u(t_idx, :), 'o-', 'Color', colors(i, :), ...
            'LineWidth', 1.5, 'MarkerSize', 4, ...
            'DisplayName', sprintf('t = %.1f', t_snap));
        % Analytical comparison
        u_analytical_snap = sin(pi * x) * exp(-pi^2 * alpha * t_snap);
        plot(x, u_analytical_snap, '--', 'Color', colors(i, :), ...
            'LineWidth', 1, 'HandleVisibility', 'off');
    end
    hold off;
    title('Heat Diffusion: Time Evolution', 'FontSize', 14, 'Color', berkeley_blue);
    xlabel('x', 'FontSize', 12);
    ylabel('Temperature', 'FontSize', 12);
    legend('show', 'Location', 'best');
    grid on;
    max_error_transient = max(error_transient(:));
    fprintf('Maximum error in transient solution: %.6e\n', max_error_transient);
    fprintf('Number of time steps: %d\n', length(result_transient.t));
else
    fprintf('Transient solution failed: %s\n', result_transient.message);
end
%% Example 3: Different Boundary Conditions
fprintf('\nExample 3: Different Boundary Conditions\n');
fprintf('----------------------------------------\n');
fprintf('Comparing Dirichlet vs mixed boundary conditions\n\n');
% Case 1: Dirichlet BC (already done above)
fprintf('Case 1: Dirichlet BC - u(0) = 0, u(1) = 0\n');
% Case 2: Mixed BC - u(0) = 0, du/dx(1) = 0 (Neumann at right)
fprintf('Case 2: Mixed BC - u(0) = 0, du/dx(1) = 0\n');
bc_mixed.dirichlet = containers.Map({0}, {0});
bc_mixed.neumann = containers.Map({50}, {0});  % Zero flux at right boundary
heat_solver_mixed = HeatEquationSolver(domain, bc_mixed, alpha);
% Source term (same as before)
result_mixed = heat_solver_mixed.solve_steady(source_steady);
% Case 3: Non-zero Dirichlet BC
fprintf('Case 3: Non-zero Dirichlet BC - u(0) = 1, u(1) = 2\n');
bc_nonzero.dirichlet = containers.Map({0, 50}, {1, 2});
heat_solver_nonzero = HeatEquationSolver(domain, bc_nonzero, alpha);
% Modified source term to have interesting solution
source_nonzero = @(x) ones(size(x));  % Constant source
result_nonzero = heat_solver_nonzero.solve_steady(source_nonzero);
% Plot comparison
figure('Position', [250, 250, 1000, 400]);
subplot(1, 2, 1);
plot(x, result_steady.u, 'Color', berkeley_blue, 'LineWidth', 2, ...
    'DisplayName', 'Dirichlet: u(0)=0, u(1)=0');
hold on;
if result_mixed.success
    plot(x, result_mixed.u, 'Color', california_gold, 'LineWidth', 2, ...
        'DisplayName', 'Mixed: u(0)=0, du/dx(1)=0');
end
if result_nonzero.success
    plot(x, result_nonzero.u, 'Color', berkeley_light_blue, 'LineWidth', 2, ...
        'DisplayName', 'Dirichlet: u(0)=1, u(1)=2');
end
hold off;
title('Different Boundary Conditions', 'FontSize', 14, 'Color', berkeley_blue);
xlabel('x', 'FontSize', 12);
ylabel('Temperature', 'FontSize', 12);
legend('show', 'Location', 'best');
grid on;
% Zoom in on boundary regions
subplot(1, 2, 2);
idx_left = 1:10;    % Left boundary region
idx_right = 42:51;  % Right boundary region
plot(x(idx_left), result_steady.u(idx_left), 'o-', 'Color', berkeley_blue, ...
    'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Left: Dirichlet');
hold on;
plot(x(idx_right), result_steady.u(idx_right), 's-', 'Color', berkeley_blue, ...
    'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Right: Dirichlet');
if result_mixed.success
    plot(x(idx_right), result_mixed.u(idx_right), '^-', 'Color', california_gold, ...
        'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Right: Neumann');
end
hold off;
title('Boundary Behavior Detail', 'FontSize', 12, 'Color', berkeley_blue);
xlabel('x', 'FontSize', 11);
ylabel('Temperature', 'FontSize', 11);
legend('show', 'Location', 'best');
grid on;
%% Example 4: Convergence Study
fprintf('\nExample 4: Grid Convergence Study\n');
fprintf('---------------------------------\n');
fprintf('Testing solution accuracy with different grid resolutions\n\n');
% Test different grid sizes
grid_sizes = [21, 41, 81, 161];
errors = zeros(size(grid_sizes));
fprintf('Grid Size\tMax Error\tConv. Rate\n');
fprintf('-------------------------------\n');
for i = 1:length(grid_sizes)
    n = grid_sizes(i);
    x_test = linspace(0, 1, n);
    domain_test.x = x_test;
    % Boundary conditions (adjust indices for different grid sizes)
    bc_test.dirichlet = containers.Map({0, n-1}, {0, 0});
    % Create solver for this grid
    solver_test = HeatEquationSolver(domain_test, bc_test, alpha);
    % Solve
    result_test = solver_test.solve_steady(source_steady);
    if result_test.success
        % Analytical solution
        u_analytical_test = sin(pi * x_test) / alpha;
        % Calculate error
        error = max(abs(result_test.u' - u_analytical_test));
        errors(i) = error;
        % Calculate convergence rate
        if i > 1
            h_old = 1 / (grid_sizes(i-1) - 1);
            h_new = 1 / (grid_sizes(i) - 1);
            rate = log(errors(i-1) / errors(i)) / log(h_old / h_new);
            fprintf('%d\t\t%.2e\t%.2f\n', n, error, rate);
        else
            fprintf('%d\t\t%.2e\t--\n', n, error);
        end
    else
        errors(i) = NaN;
        fprintf('%d\t\tFAILED\t--\n', n);
    end
end
% Plot convergence
figure('Position', [300, 300, 600, 500]);
h_values = 1 ./ (grid_sizes - 1);  % Grid spacing
valid_idx = ~isnan(errors) & errors > 0;
loglog(h_values(valid_idx), errors(valid_idx), 'o-', 'Color', berkeley_blue, ...
    'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', berkeley_blue);
hold on;
% Plot theoretical O(h²) convergence line
if sum(valid_idx) >= 2
    loglog(h_values(valid_idx), errors(valid_idx(1)) * (h_values(valid_idx) / h_values(find(valid_idx, 1))).^2, ...
        '--', 'Color', california_gold, 'LineWidth', 2, 'DisplayName', 'O(h²)');
end
hold off;
title('Grid Convergence Study', 'FontSize', 14, 'Color', berkeley_blue);
xlabel('Grid Spacing h', 'FontSize', 12);
ylabel('Maximum Error', 'FontSize', 12);
legend({'Numerical Error', 'O(h²) Reference'}, 'Location', 'best');
grid on;
%% Summary
fprintf('\n=== PDE Examples Summary ===\n');
fprintf('1. Steady heat equation: Basic elliptic PDE with known analytical solution\n');
fprintf('2. Transient heat equation: Parabolic PDE showing time evolution\n');
fprintf('3. Boundary conditions: Impact of different BC types on solution\n');
fprintf('4. Convergence study: Verification of second-order spatial accuracy\n\n');
fprintf('Key observations:\n');
fprintf('- Finite difference methods achieve O(h²) spatial accuracy\n');
fprintf('- Boundary conditions significantly affect solution behavior\n');
fprintf('- Time-dependent problems require stability considerations\n');
fprintf('- Grid refinement studies verify implementation correctness\n\n');
fprintf('All PDE examples completed successfully!\n');
fprintf('Check the generated figures for visual results.\n');