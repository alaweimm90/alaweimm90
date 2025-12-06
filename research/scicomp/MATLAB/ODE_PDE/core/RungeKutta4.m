classdef RungeKutta4 < ODESolver
    %RUNGEKUTTA4 Fourth-order Runge-Kutta ODE solver
    %   Classical RK4 method with fourth-order accuracy
    %
    %   Author: Berkeley SciComp Team
    %   Date: 2024
    methods
        function obj = RungeKutta4(varargin)
            %RUNGEKUTTA4 Constructor
            obj@ODESolver(varargin{:});
            obj.name = 'Runge-Kutta 4';
            obj.order = 4;
        end
        function result = solve(obj, func, y0, t_span, dt, varargin)
            %SOLVE Solve ODE using RK4 method
            %   result = solve(obj, func, y0, t_span, dt)
            %
            %   Example:
            %       solver = RungeKutta4();
            %       dydt = @(t, y) [y(2); -y(1)];  % Harmonic oscillator
            %       result = solver.solve(dydt, [1; 0], [0, 2*pi], 0.1);
            obj.reset();
            result = obj.solve_generic(func, y0, t_span, dt, varargin{:});
            if result.success
                result.method_info = struct(...
                    'name', 'Runge-Kutta 4', ...
                    'order', 4, ...
                    'type', 'explicit', ...
                    'stability', 'conditionally stable', ...
                    'description', 'Fourth-order Runge-Kutta method'...
                );
            end
        end
        function y_new = step(obj, func, t, y, dt)
            %STEP Single step of RK4 method
            %   y_new = step(obj, func, t, y, dt)
            %
            %   Implements classical RK4:
            %   k1 = h*f(t, y)
            %   k2 = h*f(t + h/2, y + k1/2)
            %   k3 = h*f(t + h/2, y + k2/2)
            %   k4 = h*f(t + h, y + k3)
            %   y_new = y + (k1 + 2*k2 + 2*k3 + k4)/6
            y = y(:);  % Ensure column vector
            % RK4 stages
            k1 = dt * func(t, y);
            k1 = k1(:);
            k2 = dt * func(t + dt/2, y + k1/2);
            k2 = k2(:);
            k3 = dt * func(t + dt/2, y + k2/2);
            k3 = k3(:);
            k4 = dt * func(t + dt, y + k3);
            k4 = k4(:);
            % Combine stages
            y_new = y + (k1 + 2*k2 + 2*k3 + k4)/6;
            % Check for NaN or Inf
            if any(~isfinite(y_new))
                error('RungeKutta4:NumericalError', ...
                    'Non-finite values detected in solution');
            end
        end
    end
    methods (Static)
        function demo()
            %DEMO Demonstration of RK4 solver
            fprintf('Runge-Kutta 4 Method Demo\n');
            fprintf('=========================\n\n');
            % Problem 1: Van der Pol oscillator
            fprintf('Problem 1: Van der Pol oscillator\n');
            fprintf('d²x/dt² - μ(1-x²)dx/dt + x = 0\n');
            fprintf('Convert to system: dx/dt = y, dy/dt = μ(1-x²)y - x\n\n');
            mu = 2;  % Nonlinearity parameter
            van_der_pol = @(t, z) [z(2); mu*(1-z(1)^2)*z(2) - z(1)];
            solver = RungeKutta4();
            result = solver.solve(van_der_pol, [2; 0], [0, 20], 0.01);
            if result.success
                figure;
                subplot(2, 2, 1);
                plot(result.t, result.y(:, 1), 'Color', [0, 50, 98]/255, 'LineWidth', 2);
                title('Position vs Time', 'FontSize', 12);
                xlabel('Time', 'FontSize', 11);
                ylabel('x(t)', 'FontSize', 11);
                grid on;
                subplot(2, 2, 2);
                plot(result.t, result.y(:, 2), 'Color', [253, 181, 21]/255, 'LineWidth', 2);
                title('Velocity vs Time', 'FontSize', 12);
                xlabel('Time', 'FontSize', 11);
                ylabel('dx/dt', 'FontSize', 11);
                grid on;
                subplot(2, 2, [3, 4]);
                plot(result.y(:, 1), result.y(:, 2), 'Color', [59, 126, 161]/255, 'LineWidth', 2);
                title('Phase Portrait', 'FontSize', 12);
                xlabel('Position', 'FontSize', 11);
                ylabel('Velocity', 'FontSize', 11);
                grid on;
                axis equal;
                sgtitle(sprintf('Van der Pol Oscillator (μ = %.1f)', mu), 'FontSize', 14);
            end
            % Problem 2: Lorenz system
            fprintf('Problem 2: Lorenz system (chaotic dynamics)\n');
            fprintf('dx/dt = σ(y - x)\n');
            fprintf('dy/dt = x(ρ - z) - y\n');
            fprintf('dz/dt = xy - βz\n\n');
            % Lorenz parameters
            sigma = 10;
            rho = 28;
            beta = 8/3;
            lorenz = @(t, xyz) [sigma*(xyz(2) - xyz(1));
                               xyz(1)*(rho - xyz(3)) - xyz(2);
                               xyz(1)*xyz(2) - beta*xyz(3)];
            result_lorenz = solver.solve(lorenz, [1; 1; 1], [0, 25], 0.001);
            if result_lorenz.success
                figure;
                % 3D trajectory
                subplot(2, 2, 1);
                plot3(result_lorenz.y(:, 1), result_lorenz.y(:, 2), result_lorenz.y(:, 3), ...
                    'Color', [0, 50, 98]/255, 'LineWidth', 1);
                title('3D Trajectory', 'FontSize', 12);
                xlabel('x', 'FontSize', 11);
                ylabel('y', 'FontSize', 11);
                zlabel('z', 'FontSize', 11);
                grid on;
                view(45, 30);
                % Time series
                subplot(2, 2, 2);
                plot(result_lorenz.t, result_lorenz.y(:, 1), 'Color', [0, 50, 98]/255, 'LineWidth', 1);
                title('x(t)', 'FontSize', 12);
                xlabel('Time', 'FontSize', 11);
                ylabel('x', 'FontSize', 11);
                grid on;
                subplot(2, 2, 3);
                plot(result_lorenz.y(:, 1), result_lorenz.y(:, 3), 'Color', [253, 181, 21]/255, 'LineWidth', 1);
                title('x-z Projection', 'FontSize', 12);
                xlabel('x', 'FontSize', 11);
                ylabel('z', 'FontSize', 11);
                grid on;
                subplot(2, 2, 4);
                plot(result_lorenz.y(:, 2), result_lorenz.y(:, 3), 'Color', [59, 126, 161]/255, 'LineWidth', 1);
                title('y-z Projection', 'FontSize', 12);
                xlabel('y', 'FontSize', 11);
                ylabel('z', 'FontSize', 11);
                grid on;
                sgtitle('Lorenz Attractor', 'FontSize', 14);
            end
            fprintf('Demo completed. Check figures for results.\n');
        end
        function test_accuracy()
            %TEST_ACCURACY Test RK4 accuracy against analytical solutions
            fprintf('Testing RK4 Accuracy\n');
            fprintf('===================\n\n');
            % Test 1: Exponential decay
            fprintf('Test 1: Exponential decay dy/dt = -λy\n');
            lambda = 2;
            dydt = @(t, y) -lambda * y;
            analytical = @(t) exp(-lambda * t);
            solver = RungeKutta4();
            dt_values = [0.1, 0.05, 0.01, 0.005];
            t_final = 1;
            fprintf('dt\t\tError\t\tRate\n');
            fprintf('--------------------------------\n');
            prev_error = [];
            for i = 1:length(dt_values)
                dt = dt_values(i);
                result = solver.solve(dydt, 1, [0, t_final], dt);
                if result.success
                    y_exact = analytical(t_final);
                    error = abs(result.y(end) - y_exact);
                    if isempty(prev_error)
                        fprintf('%.4f\t\t%.2e\t\t--\n', dt, error);
                    else
                        rate = log(prev_error/error) / log(dt_values(i-1)/dt);
                        fprintf('%.4f\t\t%.2e\t\t%.2f\n', dt, error, rate);
                    end
                    prev_error = error;
                end
            end
            % Test 2: Harmonic oscillator (energy conservation)
            fprintf('\nTest 2: Harmonic oscillator energy conservation\n');
            omega = 2;
            harmonic = @(t, y) [y(2); -omega^2 * y(1)];
            result_harmonic = solver.solve(harmonic, [1; 0], [0, 4*pi], 0.01);
            if result_harmonic.success
                % Calculate energy at each time step
                kinetic = 0.5 * result_harmonic.y(:, 2).^2;
                potential = 0.5 * omega^2 * result_harmonic.y(:, 1).^2;
                total_energy = kinetic + potential;
                energy_error = std(total_energy) / mean(total_energy);
                fprintf('Relative energy error: %.2e\n', energy_error);
                figure;
                subplot(2, 1, 1);
                plot(result_harmonic.t, total_energy, 'Color', [0, 50, 98]/255, 'LineWidth', 2);
                title('Energy Conservation', 'FontSize', 12);
                xlabel('Time', 'FontSize', 11);
                ylabel('Total Energy', 'FontSize', 11);
                grid on;
                subplot(2, 1, 2);
                plot(result_harmonic.y(:, 1), result_harmonic.y(:, 2), ...
                    'Color', [253, 181, 21]/255, 'LineWidth', 2);
                title('Phase Portrait', 'FontSize', 12);
                xlabel('Position', 'FontSize', 11);
                ylabel('Velocity', 'FontSize', 11);
                grid on;
                axis equal;
                sgtitle('RK4: Harmonic Oscillator', 'FontSize', 14);
            end
            fprintf('\nExpected convergence rate for RK4: 4.0\n');
        end
        function comparison_with_euler()
            %COMPARISON_WITH_EULER Compare RK4 with Euler method
            fprintf('RK4 vs Explicit Euler Comparison\n');
            fprintf('================================\n\n');
            % Test problem: dy/dt = -y*sin(t), y(0) = 1
            % Analytical solution: y(t) = exp(cos(t) - 1)
            dydt = @(t, y) -y * sin(t);
            analytical = @(t) exp(cos(t) - 1);
            t_final = 2*pi;
            dt = 0.2;  % Relatively large step size to show differences
            % Solve with both methods
            rk4_solver = RungeKutta4();
            euler_solver = ExplicitEuler();
            result_rk4 = rk4_solver.solve(dydt, 1, [0, t_final], dt);
            result_euler = euler_solver.solve(dydt, 1, [0, t_final], dt);
            % Plot comparison
            figure;
            t_exact = linspace(0, t_final, 1000);
            y_exact = analytical(t_exact);
            hold on;
            plot(t_exact, y_exact, 'k-', 'LineWidth', 2, 'DisplayName', 'Analytical');
            if result_rk4.success
                plot(result_rk4.t, result_rk4.y, 'o-', 'Color', [0, 50, 98]/255, ...
                    'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'RK4');
                y_exact_rk4 = analytical(result_rk4.t);
                error_rk4 = max(abs(result_rk4.y - y_exact_rk4));
                fprintf('RK4 maximum error: %.6e\n', error_rk4);
            end
            if result_euler.success
                plot(result_euler.t, result_euler.y, 's-', 'Color', [253, 181, 21]/255, ...
                    'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Euler');
                y_exact_euler = analytical(result_euler.t);
                error_euler = max(abs(result_euler.y - y_exact_euler));
                fprintf('Euler maximum error: %.6e\n', error_euler);
                if result_rk4.success
                    fprintf('RK4 is %.1f times more accurate\n', error_euler/error_rk4);
                end
            end
            hold off;
            title(sprintf('RK4 vs Euler Comparison (dt = %.1f)', dt), 'FontSize', 14);
            xlabel('Time', 'FontSize', 12);
            ylabel('y(t)', 'FontSize', 12);
            legend('show', 'Location', 'best');
            grid on;
        end
    end
end