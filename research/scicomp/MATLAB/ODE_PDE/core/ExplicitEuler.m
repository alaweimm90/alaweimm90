classdef ExplicitEuler < ODESolver
    %EXPLICITEULER Explicit Euler ODE solver
    %   First-order explicit method: y_{n+1} = y_n + h*f(t_n, y_n)
    %
    %   Author: Berkeley SciComp Team
    %   Date: 2024
    methods
        function obj = ExplicitEuler(varargin)
            %EXPLICITEULER Constructor
            obj@ODESolver(varargin{:});
            obj.name = 'Explicit Euler';
            obj.order = 1;
        end
        function result = solve(obj, func, y0, t_span, dt, varargin)
            %SOLVE Solve ODE using explicit Euler method
            %   result = solve(obj, func, y0, t_span, dt)
            %
            %   Example:
            %       solver = ExplicitEuler();
            %       dydt = @(t, y) -2*y;  % Exponential decay
            %       result = solver.solve(dydt, 1, [0, 1], 0.01);
            %       plot(result.t, result.y);
            obj.reset();
            result = obj.solve_generic(func, y0, t_span, dt, varargin{:});
            if result.success
                result.method_info = struct(...
                    'name', 'Explicit Euler', ...
                    'order', 1, ...
                    'type', 'explicit', ...
                    'stability', 'conditionally stable', ...
                    'description', 'First-order forward Euler method'...
                );
            end
        end
        function y_new = step(obj, func, t, y, dt)
            %STEP Single step of explicit Euler method
            %   y_new = step(obj, func, t, y, dt)
            %
            %   Implements: y_{n+1} = y_n + h*f(t_n, y_n)
            y = y(:);  % Ensure column vector
            % Evaluate derivative
            dydt = func(t, y);
            dydt = dydt(:);  % Ensure column vector
            % Euler step
            y_new = y + dt * dydt;
            % Check for NaN or Inf
            if any(~isfinite(y_new))
                error('ExplicitEuler:NumericalError', ...
                    'Non-finite values detected in solution');
            end
        end
    end
    methods (Static)
        function demo()
            %DEMO Demonstration of explicit Euler solver
            fprintf('Explicit Euler Method Demo\n');
            fprintf('==========================\n\n');
            % Problem 1: Exponential decay
            fprintf('Problem 1: Exponential decay dy/dt = -2y, y(0) = 1\n');
            fprintf('Analytical solution: y(t) = exp(-2t)\n\n');
            solver = ExplicitEuler();
            dydt = @(t, y) -2*y;
            analytical = @(t) exp(-2*t);
            % Solve with different step sizes
            dt_values = [0.1, 0.05, 0.01];
            t_final = 1;
            figure;
            colors = [0, 50, 98; 253, 181, 21; 59, 126, 161]/255;
            hold on;
            % Plot analytical solution
            t_exact = linspace(0, t_final, 1000);
            plot(t_exact, analytical(t_exact), 'k--', 'LineWidth', 2, ...
                'DisplayName', 'Analytical');
            % Plot numerical solutions
            for i = 1:length(dt_values)
                dt = dt_values(i);
                result = solver.solve(dydt, 1, [0, t_final], dt);
                if result.success
                    plot(result.t, result.y, 'o-', 'Color', colors(i, :), ...
                        'LineWidth', 1.5, 'MarkerSize', 4, ...
                        'DisplayName', sprintf('dt = %.3f', dt));
                    % Calculate error
                    y_exact = analytical(result.t);
                    error = max(abs(result.y - y_exact));
                    fprintf('dt = %.3f: Max error = %.6f\n', dt, error);
                end
            end
            hold off;
            title('Explicit Euler: Exponential Decay', 'FontSize', 14);
            xlabel('Time', 'FontSize', 12);
            ylabel('y(t)', 'FontSize', 12);
            legend('show', 'Location', 'best');
            grid on;
            % Problem 2: Harmonic oscillator
            fprintf('\nProblem 2: Harmonic oscillator d²y/dt² + ω²y = 0\n');
            fprintf('Convert to system: dy1/dt = y2, dy2/dt = -ω²y1\n\n');
            omega = 2;
            system_ode = @(t, y) [y(2); -omega^2 * y(1)];
            y0_system = [1; 0];  % Initial position and velocity
            result_system = solver.solve(system_ode, y0_system, [0, 2*pi/omega], 0.01);
            if result_system.success
                figure;
                subplot(2, 1, 1);
                plot(result_system.t, result_system.y(:, 1), 'Color', colors(1, :), ...
                    'LineWidth', 2);
                title('Position vs Time', 'FontSize', 12);
                xlabel('Time', 'FontSize', 11);
                ylabel('Position', 'FontSize', 11);
                grid on;
                subplot(2, 1, 2);
                plot(result_system.y(:, 1), result_system.y(:, 2), 'Color', colors(2, :), ...
                    'LineWidth', 2);
                title('Phase Portrait', 'FontSize', 12);
                xlabel('Position', 'FontSize', 11);
                ylabel('Velocity', 'FontSize', 11);
                grid on;
                axis equal;
                sgtitle('Explicit Euler: Harmonic Oscillator', 'FontSize', 14);
            end
            fprintf('Demo completed. Check figures for results.\n');
        end
        function test_convergence()
            %TEST_CONVERGENCE Test convergence order
            fprintf('Testing Explicit Euler Convergence Order\n');
            fprintf('========================================\n\n');
            % Test problem: dy/dt = -y, y(0) = 1
            % Analytical solution: y(t) = exp(-t)
            dydt = @(t, y) -y;
            analytical = @(t) exp(-t);
            t_final = 1;
            y0 = 1;
            dt_values = [0.1, 0.05, 0.025, 0.0125];
            errors = zeros(size(dt_values));
            solver = ExplicitEuler();
            for i = 1:length(dt_values)
                dt = dt_values(i);
                result = solver.solve(dydt, y0, [0, t_final], dt);
                if result.success
                    y_exact = analytical(t_final);
                    errors(i) = abs(result.y(end) - y_exact);
                    fprintf('dt = %.4f: Error = %.6e\n', dt, errors(i));
                else
                    errors(i) = NaN;
                end
            end
            % Calculate convergence rates
            fprintf('\nConvergence rates:\n');
            for i = 2:length(errors)
                if ~isnan(errors(i)) && ~isnan(errors(i-1))
                    rate = log(errors(i-1)/errors(i)) / log(dt_values(i-1)/dt_values(i));
                    fprintf('Between dt=%.4f and dt=%.4f: Rate = %.2f\n', ...
                        dt_values(i-1), dt_values(i), rate);
                end
            end
            fprintf('\nExpected convergence rate for Explicit Euler: 1.0\n');
        end
        function stability_analysis()
            %STABILITY_ANALYSIS Analyze stability of explicit Euler
            fprintf('Explicit Euler Stability Analysis\n');
            fprintf('=================================\n\n');
            % Test problem: dy/dt = λy for different λ values
            lambda_values = [-1, -5, -10, -50];
            dt_stable = 2 ./ abs(lambda_values);  % Stability limit: |1 + λh| ≤ 1
            figure;
            for i = 1:length(lambda_values)
                lambda = lambda_values(i);
                dydt = @(t, y) lambda * y;
                subplot(2, 2, i);
                % Test stable and unstable time steps
                dt_test = [0.8 * dt_stable(i), 1.2 * dt_stable(i)];
                colors = [0, 50, 98; 255, 0, 0]/255;
                labels = {'Stable', 'Unstable'};
                hold on;
                for j = 1:2
                    dt = dt_test(j);
                    solver = ExplicitEuler();
                    try
                        result = solver.solve(dydt, 1, [0, 2], dt);
                        if result.success && all(isfinite(result.y))
                            plot(result.t, result.y, 'Color', colors(j, :), ...
                                'LineWidth', 2, 'DisplayName', ...
                                sprintf('%s (dt=%.3f)', labels{j}, dt));
                        else
                            fprintf('Solution failed for λ=%.1f, dt=%.3f\n', lambda, dt);
                        end
                    catch
                        fprintf('Numerical instability for λ=%.1f, dt=%.3f\n', lambda, dt);
                    end
                end
                hold off;
                title(sprintf('λ = %.1f', lambda), 'FontSize', 12);
                xlabel('Time', 'FontSize', 10);
                ylabel('y(t)', 'FontSize', 10);
                legend('show');
                grid on;
            end
            sgtitle('Explicit Euler Stability Analysis', 'FontSize', 14);
            fprintf('Stability condition for Explicit Euler: |1 + λh| ≤ 1\n');
            fprintf('For dy/dt = λy with λ < 0: h ≤ 2/|λ|\n\n');
            for i = 1:length(lambda_values)
                fprintf('λ = %3.1f: Stable dt ≤ %.4f\n', ...
                    lambda_values(i), dt_stable(i));
            end
        end
    end
end