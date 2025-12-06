%% Simple ODE Examples - Beginner Level
%  Basic demonstrations of ODE solving with the Berkeley SciComp framework
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
fprintf('=== Simple ODE Examples - Beginner Level ===\n\n');
%% Example 1: Exponential Growth/Decay
fprintf('Example 1: Exponential Growth and Decay\n');
fprintf('---------------------------------------\n');
fprintf('Equation: dy/dt = λy\n');
fprintf('For λ > 0: exponential growth\n');
fprintf('For λ < 0: exponential decay\n\n');
% Define the ODE: dy/dt = λy
lambda_growth = 0.5;   % Growth rate
lambda_decay = -1.0;   % Decay rate
ode_growth = @(t, y) lambda_growth * y;
ode_decay = @(t, y) lambda_decay * y;
% Analytical solutions for comparison
analytical_growth = @(t) exp(lambda_growth * t);
analytical_decay = @(t) exp(lambda_decay * t);
% Initial condition
y0 = 1.0;
t_span = [0, 3];
dt = 0.1;
% Solve using different methods
euler_solver = ExplicitEuler();
rk4_solver = RungeKutta4();
% Growth case
result_growth_euler = euler_solver.solve(ode_growth, y0, t_span, dt);
result_growth_rk4 = rk4_solver.solve(ode_growth, y0, t_span, dt);
% Decay case
result_decay_euler = euler_solver.solve(ode_decay, y0, t_span, dt);
result_decay_rk4 = rk4_solver.solve(ode_decay, y0, t_span, dt);
% Plot results
figure('Position', [100, 100, 1200, 400]);
% Growth subplot
subplot(1, 2, 1);
t_fine = linspace(0, 3, 300);
plot(t_fine, analytical_growth(t_fine), 'k--', 'LineWidth', 2, 'DisplayName', 'Analytical');
hold on;
plot(result_growth_euler.t, result_growth_euler.y, 'o-', 'Color', berkeley_blue, ...
    'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', 'Euler');
plot(result_growth_rk4.t, result_growth_rk4.y, 's-', 'Color', california_gold, ...
    'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', 'RK4');
hold off;
title('Exponential Growth (λ = 0.5)', 'FontSize', 14, 'Color', berkeley_blue);
xlabel('Time', 'FontSize', 12);
ylabel('y(t)', 'FontSize', 12);
legend('show', 'Location', 'northwest');
grid on;
% Decay subplot
subplot(1, 2, 2);
plot(t_fine, analytical_decay(t_fine), 'k--', 'LineWidth', 2, 'DisplayName', 'Analytical');
hold on;
plot(result_decay_euler.t, result_decay_euler.y, 'o-', 'Color', berkeley_blue, ...
    'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', 'Euler');
plot(result_decay_rk4.t, result_decay_rk4.y, 's-', 'Color', california_gold, ...
    'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', 'RK4');
hold off;
title('Exponential Decay (λ = -1.0)', 'FontSize', 14, 'Color', berkeley_blue);
xlabel('Time', 'FontSize', 12);
ylabel('y(t)', 'FontSize', 12);
legend('show', 'Location', 'northeast');
grid on;
% Calculate and display errors
error_growth_euler = abs(result_growth_euler.y(end) - analytical_growth(t_span(2)));
error_growth_rk4 = abs(result_growth_rk4.y(end) - analytical_growth(t_span(2)));
error_decay_euler = abs(result_decay_euler.y(end) - analytical_decay(t_span(2)));
error_decay_rk4 = abs(result_decay_rk4.y(end) - analytical_decay(t_span(2)));
fprintf('Growth case - Final time errors:\n');
fprintf('  Euler: %.6f\n', error_growth_euler);
fprintf('  RK4:   %.6f\n', error_growth_rk4);
fprintf('Decay case - Final time errors:\n');
fprintf('  Euler: %.6f\n', error_decay_euler);
fprintf('  RK4:   %.6f\n', error_decay_rk4);
%% Example 2: Simple Harmonic Oscillator
fprintf('\nExample 2: Simple Harmonic Oscillator\n');
fprintf('-------------------------------------\n');
fprintf('Equation: d²y/dt² + ω²y = 0\n');
fprintf('Convert to system: dy₁/dt = y₂, dy₂/dt = -ω²y₁\n');
fprintf('Analytical solution: y(t) = A cos(ωt) + B sin(ωt)\n\n');
% Parameters
omega = 2.0;  % Angular frequency
A = 1.0;      % Amplitude for cosine
B = 0.5;      % Amplitude for sine
% System of ODEs: [y₁, y₂] = [position, velocity]
harmonic_ode = @(t, y) [y(2); -omega^2 * y(1)];
% Initial conditions: y(0) = A, dy/dt(0) = B*ω
y0_harmonic = [A; B * omega];
% Analytical solution
analytical_position = @(t) A * cos(omega * t) + B * sin(omega * t);
analytical_velocity = @(t) -A * omega * sin(omega * t) + B * omega * cos(omega * t);
% Solve for one complete period
T = 2 * pi / omega;  % Period
t_span_harmonic = [0, T];
dt_harmonic = T / 100;
result_harmonic = rk4_solver.solve(harmonic_ode, y0_harmonic, t_span_harmonic, dt_harmonic);
% Plot results
figure('Position', [150, 150, 1200, 400]);
% Position vs time
subplot(1, 3, 1);
t_fine = linspace(0, T, 1000);
plot(t_fine, analytical_position(t_fine), 'k--', 'LineWidth', 2, 'DisplayName', 'Analytical');
hold on;
plot(result_harmonic.t, result_harmonic.y(:, 1), 'o-', 'Color', berkeley_blue, ...
    'LineWidth', 1.5, 'MarkerSize', 3, 'DisplayName', 'Numerical');
hold off;
title('Position vs Time', 'FontSize', 12, 'Color', berkeley_blue);
xlabel('Time', 'FontSize', 11);
ylabel('Position', 'FontSize', 11);
legend('show', 'Location', 'best');
grid on;
% Velocity vs time
subplot(1, 3, 2);
plot(t_fine, analytical_velocity(t_fine), 'k--', 'LineWidth', 2, 'DisplayName', 'Analytical');
hold on;
plot(result_harmonic.t, result_harmonic.y(:, 2), 'o-', 'Color', california_gold, ...
    'LineWidth', 1.5, 'MarkerSize', 3, 'DisplayName', 'Numerical');
hold off;
title('Velocity vs Time', 'FontSize', 12, 'Color', berkeley_blue);
xlabel('Time', 'FontSize', 11);
ylabel('Velocity', 'FontSize', 11);
legend('show', 'Location', 'best');
grid on;
% Phase portrait
subplot(1, 3, 3);
plot(result_harmonic.y(:, 1), result_harmonic.y(:, 2), 'Color', berkeley_blue, 'LineWidth', 2);
hold on;
plot(result_harmonic.y(1, 1), result_harmonic.y(1, 2), 'o', 'Color', california_gold, ...
    'MarkerSize', 8, 'MarkerFaceColor', california_gold, 'DisplayName', 'Start');
plot(result_harmonic.y(end, 1), result_harmonic.y(end, 2), 's', 'Color', [1, 0, 0], ...
    'MarkerSize', 8, 'MarkerFaceColor', [1, 0, 0], 'DisplayName', 'End');
hold off;
title('Phase Portrait', 'FontSize', 12, 'Color', berkeley_blue);
xlabel('Position', 'FontSize', 11);
ylabel('Velocity', 'FontSize', 11);
legend('show', 'Location', 'best');
grid on;
axis equal;
% Energy conservation check
kinetic_energy = 0.5 * result_harmonic.y(:, 2).^2;
potential_energy = 0.5 * omega^2 * result_harmonic.y(:, 1).^2;
total_energy = kinetic_energy + potential_energy;
energy_variation = std(total_energy) / mean(total_energy);
fprintf('Energy conservation check:\n');
fprintf('  Relative energy variation: %.6f%%\n', energy_variation * 100);
%% Example 3: First-Order Linear ODE with Forcing
fprintf('\nExample 3: First-Order Linear ODE with Forcing\n');
fprintf('----------------------------------------------\n');
fprintf('Equation: dy/dt + ay = f(t)\n');
fprintf('Example: dy/dt + 2y = sin(t)\n');
fprintf('Analytical solution involves integrating factor method\n\n');
% Parameters
a = 2.0;
forcing_freq = 1.0;
% ODE with sinusoidal forcing
forced_ode = @(t, y) -a * y + sin(forcing_freq * t);
% Analytical solution (derived using integrating factor)
% For dy/dt + ay = sin(t), solution is:
% y(t) = (sin(t) - a*cos(t))/(1 + a²) + C*exp(-at)
% With y(0) = y0, C = y0 - (-a)/(1 + a²)
y0_forced = 0.5;
C = y0_forced + a / (1 + a^2);
analytical_forced = @(t) (sin(forcing_freq * t) - a * cos(forcing_freq * t)) / (1 + a^2) + ...
                         C * exp(-a * t);
% Solve
t_span_forced = [0, 5];
dt_forced = 0.05;
result_forced = rk4_solver.solve(forced_ode, y0_forced, t_span_forced, dt_forced);
% Plot
figure('Position', [200, 200, 800, 600]);
subplot(2, 1, 1);
t_fine = linspace(0, 5, 1000);
plot(t_fine, analytical_forced(t_fine), 'k--', 'LineWidth', 2, 'DisplayName', 'Analytical');
hold on;
plot(result_forced.t, result_forced.y, 'o-', 'Color', berkeley_blue, ...
    'LineWidth', 1.5, 'MarkerSize', 3, 'DisplayName', 'Numerical');
plot(t_fine, sin(forcing_freq * t_fine) / (1 + a^2), ':', 'Color', california_gold, ...
    'LineWidth', 2, 'DisplayName', 'Steady State');
hold off;
title('Forced Oscillation: dy/dt + 2y = sin(t)', 'FontSize', 14, 'Color', berkeley_blue);
xlabel('Time', 'FontSize', 12);
ylabel('y(t)', 'FontSize', 12);
legend('show', 'Location', 'best');
grid on;
% Error analysis
subplot(2, 1, 2);
error_forced = abs(result_forced.y - analytical_forced(result_forced.t)');
semilogy(result_forced.t, error_forced, 'Color', berkeley_blue, 'LineWidth', 2);
title('Absolute Error', 'FontSize', 12, 'Color', berkeley_blue);
xlabel('Time', 'FontSize', 11);
ylabel('|Error|', 'FontSize', 11);
grid on;
max_error_forced = max(error_forced);
fprintf('Maximum absolute error: %.6e\n', max_error_forced);
%% Example 4: Population Growth Models
fprintf('\nExample 4: Population Growth Models\n');
fprintf('----------------------------------\n');
fprintf('Compare exponential vs logistic growth\n');
fprintf('Exponential: dP/dt = rP\n');
fprintf('Logistic:    dP/dt = rP(1 - P/K)\n\n');
% Parameters
r = 0.3;        % Growth rate
K = 100;        % Carrying capacity
P0 = 5;         % Initial population
% ODEs
exponential_growth = @(t, P) r * P;
logistic_growth = @(t, P) r * P * (1 - P / K);
% Analytical solutions
analytical_exp = @(t) P0 * exp(r * t);
analytical_logistic = @(t) K * P0 * exp(r * t) ./ (K + P0 * (exp(r * t) - 1));
% Solve both models
t_span_pop = [0, 15];
dt_pop = 0.1;
result_exp = rk4_solver.solve(exponential_growth, P0, t_span_pop, dt_pop);
result_logistic = rk4_solver.solve(logistic_growth, P0, t_span_pop, dt_pop);
% Plot comparison
figure('Position', [250, 250, 800, 500]);
t_fine = linspace(0, 15, 1000);
plot(t_fine, analytical_exp(t_fine), '--', 'Color', berkeley_blue, 'LineWidth', 2, ...
    'DisplayName', 'Exponential (Analytical)');
hold on;
plot(t_fine, analytical_logistic(t_fine), '--', 'Color', california_gold, 'LineWidth', 2, ...
    'DisplayName', 'Logistic (Analytical)');
plot(result_exp.t, result_exp.y, 'o', 'Color', berkeley_blue, 'MarkerSize', 4, ...
    'DisplayName', 'Exponential (Numerical)');
plot(result_logistic.t, result_logistic.y, 's', 'Color', california_gold, 'MarkerSize', 4, ...
    'DisplayName', 'Logistic (Numerical)');
% Add carrying capacity line
yline(K, 'r:', 'LineWidth', 2, 'DisplayName', 'Carrying Capacity');
hold off;
title('Population Growth Models', 'FontSize', 14, 'Color', berkeley_blue);
xlabel('Time', 'FontSize', 12);
ylabel('Population', 'FontSize', 12);
legend('show', 'Location', 'best');
grid on;
% Calculate errors
error_exp = max(abs(result_exp.y - analytical_exp(result_exp.t)'));
error_logistic = max(abs(result_logistic.y - analytical_logistic(result_logistic.t)'));
fprintf('Population model errors:\n');
fprintf('  Exponential: %.6e\n', error_exp);
fprintf('  Logistic:    %.6e\n', error_logistic);
%% Summary
fprintf('\n=== Example Summary ===\n');
fprintf('1. Exponential growth/decay: Basic first-order linear ODEs\n');
fprintf('2. Harmonic oscillator: Second-order ODE converted to system\n');
fprintf('3. Forced oscillation: First-order with time-dependent forcing\n');
fprintf('4. Population models: Nonlinear growth with carrying capacity\n\n');
fprintf('Key observations:\n');
fprintf('- RK4 is generally more accurate than Euler for the same step size\n');
fprintf('- Energy conservation in harmonic oscillator tests solver quality\n');
fprintf('- Forcing terms require careful handling of time dependence\n');
fprintf('- Nonlinear models (logistic) can exhibit rich behavior\n\n');
fprintf('All examples completed successfully!\n');
fprintf('Check the generated figures for visual results.\n');