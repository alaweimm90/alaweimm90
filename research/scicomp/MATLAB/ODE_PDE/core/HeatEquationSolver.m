classdef HeatEquationSolver < PDESolver
    %HEATEQUATIONSOLVER Solver for heat equation
    %   Solves ∂u/∂t = α∇²u + f(x,t) using finite differences
    %
    %   Author: Berkeley SciComp Team
    %   Date: 2024
    properties
        thermal_diffusivity     % Thermal diffusivity α
        scheme = 'implicit'     % Time stepping scheme
    end
    methods
        function obj = HeatEquationSolver(domain, boundary_conditions, thermal_diffusivity, varargin)
            %HEATEQUATIONSOLVER Constructor
            %   obj = HeatEquationSolver(domain, bc, alpha, ...)
            %
            %   Inputs:
            %       domain - Spatial domain structure with field 'x'
            %       boundary_conditions - BC structure
            %       thermal_diffusivity - Thermal diffusivity α
            obj@PDESolver(domain, boundary_conditions, varargin{:});
            p = inputParser;
            addRequired(p, 'thermal_diffusivity', @(x) isnumeric(x) && x > 0);
            addParameter(p, 'scheme', 'implicit', @(x) ismember(x, {'explicit', 'implicit', 'crank_nicolson'}));
            parse(p, thermal_diffusivity, varargin{:});
            obj.thermal_diffusivity = thermal_diffusivity;
            obj.scheme = p.Results.scheme;
            obj.name = 'Heat Equation Solver';
            obj.pde_type = 'parabolic';
        end
        function result = solve_steady(obj, source_term, varargin)
            %SOLVE_STEADY Solve steady heat equation
            %   -α∇²u = f(x)
            %
            %   result = solve_steady(obj, source_term)
            %
            %   Example:
            %       domain.x = linspace(0, 1, 51);
            %       bc.dirichlet = containers.Map({0, 50}, {0, 0});
            %       solver = HeatEquationSolver(domain, bc, 0.1);
            %       f = @(x) pi^2 * sin(pi*x);
            %       result = solver.solve_steady(f);
            try
                % Build finite difference matrix for -α∇²u = f
                [A, b] = obj.build_finite_difference_matrix(source_term);
                % Scale by thermal diffusivity
                A = A / obj.thermal_diffusivity;
                b = -b / obj.thermal_diffusivity;
                % Solve linear system
                u = A \ b;
                result = obj.create_result_structure(u, [], true, ...
                    'Steady heat equation solved successfully');
                % Add method information
                result.method_info = struct(...
                    'equation', 'Heat (steady)', ...
                    'thermal_diffusivity', obj.thermal_diffusivity, ...
                    'discretization', 'finite_difference', ...
                    'grid_points', length(obj.domain.x) ...
                );
            catch ME
                result = obj.create_result_structure([], [], false, ...
                    sprintf('Steady solve failed: %s', ME.message));
            end
        end
        function result = solve_transient(obj, initial_condition, varargin)
            %SOLVE_TRANSIENT Solve transient heat equation
            %   ∂u/∂t = α∇²u + f(x,t)
            %
            %   result = solve_transient(obj, u0, 'time_span', [0, 1], 'dt', 0.01)
            p = inputParser;
            addRequired(p, 'initial_condition');
            addParameter(p, 'time_span', [0, 1], @(x) length(x) == 2 && x(2) > x(1));
            addParameter(p, 'dt', 0.01, @(x) isnumeric(x) && x > 0);
            addParameter(p, 'source_term', [], @(x) isempty(x) || isa(x, 'function_handle'));
            parse(p, initial_condition, varargin{:});
            try
                % Setup
                x = obj.domain.x(:);
                n = length(x);
                dx = x(2) - x(1);
                t_span = p.Results.time_span;
                dt = p.Results.dt;
                t_vec = t_span(1):dt:t_span(2);
                n_steps = length(t_vec);
                % Check stability
                obj.check_cfl_condition(dt, [], obj.thermal_diffusivity);
                % Initialize solution
                if isa(initial_condition, 'function_handle')
                    u0 = initial_condition(x);
                else
                    u0 = initial_condition(:);
                end
                if length(u0) ~= n
                    error('Initial condition size mismatch');
                end
                % Storage for solution
                u_storage = zeros(n_steps, n);
                u_storage(1, :) = u0';
                u_current = u0;
                % Build spatial discretization matrix
                [L, ~] = obj.build_spatial_operator();
                % Time stepping
                switch lower(obj.scheme)
                    case 'explicit'
                        u_current = obj.explicit_time_step(u_current, L, t_vec, dt, p.Results.source_term, u_storage);
                    case 'implicit'
                        u_current = obj.implicit_time_step(u_current, L, t_vec, dt, p.Results.source_term, u_storage);
                    case 'crank_nicolson'
                        u_current = obj.crank_nicolson_time_step(u_current, L, t_vec, dt, p.Results.source_term, u_storage);
                end
                result = obj.create_result_structure(u_storage, t_vec', true, ...
                    sprintf('Transient heat equation solved (%s scheme)', obj.scheme));
                % Add method information
                result.method_info = struct(...
                    'equation', 'Heat (transient)', ...
                    'scheme', obj.scheme, ...
                    'thermal_diffusivity', obj.thermal_diffusivity, ...
                    'dt', dt, ...
                    'time_steps', n_steps, ...
                    'grid_points', n ...
                );
            catch ME
                result = obj.create_result_structure([], [], false, ...
                    sprintf('Transient solve failed: %s', ME.message));
            end
        end
        function [L, b_bc] = build_spatial_operator(obj)
            %BUILD_SPATIAL_OPERATOR Build spatial discretization operator
            x = obj.domain.x(:);
            n = length(x);
            dx = x(2) - x(1);
            % Second derivative operator: d²/dx²
            L = zeros(n, n);
            % Interior points
            for i = 2:n-1
                L(i, i-1) = 1;
                L(i, i) = -2;
                L(i, i+1) = 1;
            end
            L = L / dx^2;
            % Boundary condition modification vector
            b_bc = zeros(n, 1);
            % Apply boundary conditions
            if isfield(obj.boundary_conditions, 'dirichlet')
                dirichlet = obj.boundary_conditions.dirichlet;
                if isstruct(dirichlet)
                    if isfield(dirichlet, '0') || isfield(dirichlet, 'left')
                        L(1, :) = 0;
                        L(1, 1) = 1;
                        % b_bc will be set during time stepping
                    end
                    if isfield(dirichlet, num2str(n-1)) || isfield(dirichlet, 'right')
                        L(n, :) = 0;
                        L(n, n) = 1;
                        % b_bc will be set during time stepping
                    end
                elseif isa(dirichlet, 'containers.Map')
                    keys_list = cell(keys(dirichlet));
                    for i = 1:length(keys_list)
                        idx = keys_list{i} + 1;  % Convert to 1-based indexing
                        if idx >= 1 && idx <= n
                            L(idx, :) = 0;
                            L(idx, idx) = 1;
                        end
                    end
                end
            end
            % Apply thermal diffusivity
            L = obj.thermal_diffusivity * L;
        end
        function u_storage = explicit_time_step(obj, u_current, L, t_vec, dt, source_term, u_storage)
            %EXPLICIT_TIME_STEP Forward Euler time stepping
            n = length(u_current);
            for i = 2:length(t_vec)
                t = t_vec(i-1);
                % Source term
                if ~isempty(source_term)
                    f = source_term(obj.domain.x, t);
                    f = f(:);
                else
                    f = zeros(n, 1);
                end
                % Apply boundary conditions to source
                f = obj.apply_bc_to_source(f, t);
                % Explicit update: u^{n+1} = u^n + dt*(L*u^n + f^n)
                u_new = u_current + dt * (L * u_current + f);
                u_storage(i, :) = u_new';
                u_current = u_new;
                % Check for numerical instability
                if any(~isfinite(u_current))
                    error('Numerical instability detected');
                end
            end
        end
        function u_storage = implicit_time_step(obj, u_current, L, t_vec, dt, source_term, u_storage)
            %IMPLICIT_TIME_STEP Backward Euler time stepping
            n = length(u_current);
            I = eye(n);
            for i = 2:length(t_vec)
                t = t_vec(i);
                % Source term at new time level
                if ~isempty(source_term)
                    f = source_term(obj.domain.x, t);
                    f = f(:);
                else
                    f = zeros(n, 1);
                end
                % Apply boundary conditions to source
                f = obj.apply_bc_to_source(f, t);
                % Implicit update: (I - dt*L)*u^{n+1} = u^n + dt*f^{n+1}
                A = I - dt * L;
                b = u_current + dt * f;
                % Apply boundary conditions to system
                [A, b] = obj.apply_bc_to_system(A, b, t);
                u_new = A \ b;
                u_storage(i, :) = u_new';
                u_current = u_new;
            end
        end
        function u_storage = crank_nicolson_time_step(obj, u_current, L, t_vec, dt, source_term, u_storage)
            %CRANK_NICOLSON_TIME_STEP Crank-Nicolson time stepping
            n = length(u_current);
            I = eye(n);
            for i = 2:length(t_vec)
                t_old = t_vec(i-1);
                t_new = t_vec(i);
                % Source terms at old and new time levels
                if ~isempty(source_term)
                    f_old = source_term(obj.domain.x, t_old);
                    f_new = source_term(obj.domain.x, t_new);
                    f_old = f_old(:);
                    f_new = f_new(:);
                else
                    f_old = zeros(n, 1);
                    f_new = zeros(n, 1);
                end
                % Apply boundary conditions
                f_old = obj.apply_bc_to_source(f_old, t_old);
                f_new = obj.apply_bc_to_source(f_new, t_new);
                % Crank-Nicolson: (I - dt/2*L)*u^{n+1} = (I + dt/2*L)*u^n + dt/2*(f^n + f^{n+1})
                A = I - dt/2 * L;
                b = (I + dt/2 * L) * u_current + dt/2 * (f_old + f_new);
                % Apply boundary conditions to system
                [A, b] = obj.apply_bc_to_system(A, b, t_new);
                u_new = A \ b;
                u_storage(i, :) = u_new';
                u_current = u_new;
            end
        end
        function f = apply_bc_to_source(obj, f, t)
            %APPLY_BC_TO_SOURCE Apply boundary conditions to source term
            n = length(f);
            if isfield(obj.boundary_conditions, 'dirichlet')
                dirichlet = obj.boundary_conditions.dirichlet;
                if isa(dirichlet, 'containers.Map')
                    keys_list = cell(keys(dirichlet));
                    for i = 1:length(keys_list)
                        idx = keys_list{i} + 1;
                        if idx >= 1 && idx <= n
                            f(idx) = 0;  % Dirichlet nodes don't contribute to source
                        end
                    end
                end
            end
        end
        function [A, b] = apply_bc_to_system(obj, A, b, t)
            %APPLY_BC_TO_SYSTEM Apply boundary conditions to linear system
            n = size(A, 1);
            if isfield(obj.boundary_conditions, 'dirichlet')
                dirichlet = obj.boundary_conditions.dirichlet;
                if isa(dirichlet, 'containers.Map')
                    keys_list = cell(keys(dirichlet));
                    for i = 1:length(keys_list)
                        idx = keys_list{i} + 1;
                        if idx >= 1 && idx <= n
                            A(idx, :) = 0;
                            A(idx, idx) = 1;
                            % Evaluate boundary value (could be time-dependent)
                            bc_value = dirichlet(keys_list{i});
                            if isa(bc_value, 'function_handle')
                                b(idx) = bc_value(t);
                            else
                                b(idx) = bc_value;
                            end
                        end
                    end
                end
            end
        end
    end
    methods (Static)
        function demo()
            %DEMO Demonstration of heat equation solver
            fprintf('Heat Equation Solver Demo\n');
            fprintf('========================\n\n');
            % Problem 1: 1D heat diffusion with Dirichlet BC
            fprintf('Problem 1: 1D heat diffusion\n');
            fprintf('∂u/∂t = α∇²u, u(0,t) = 0, u(1,t) = 0\n');
            fprintf('Initial condition: u(x,0) = sin(πx)\n\n');
            % Setup domain and boundary conditions
            domain.x = linspace(0, 1, 51);
            bc.dirichlet = containers.Map({0, 50}, {0, 0});
            alpha = 0.1;
            solver = HeatEquationSolver(domain, bc, alpha);
            % Initial condition
            initial_condition = @(x) sin(pi * x);
            % Solve transient problem
            result = solver.solve_transient(initial_condition, ...
                'time_span', [0, 0.5], 'dt', 0.001);
            if result.success
                % Plot solution
                solver.plot_solution(result, 'title', 'Heat Diffusion', ...
                    'surface_plot', true);
                % Analytical solution for comparison
                figure;
                t_snapshots = [0, 0.1, 0.2, 0.3, 0.5];
                colors = [0, 50, 98; 253, 181, 21; 59, 126, 161; 196, 130, 14; 0, 176, 218]/255;
                hold on;
                for i = 1:length(t_snapshots)
                    t = t_snapshots(i);
                    % Find closest time index
                    [~, t_idx] = min(abs(result.t - t));
                    % Analytical solution: u(x,t) = sin(πx)exp(-π²αt)
                    u_analytical = sin(pi * domain.x) * exp(-pi^2 * alpha * t);
                    plot(domain.x, result.u(t_idx, :), 'o-', 'Color', colors(i, :), ...
                        'LineWidth', 2, 'MarkerSize', 4, ...
                        'DisplayName', sprintf('Numerical t=%.1f', t));
                    plot(domain.x, u_analytical, '--', 'Color', colors(i, :), ...
                        'LineWidth', 1, 'DisplayName', sprintf('Analytical t=%.1f', t));
                end
                hold off;
                title('Heat Equation: Numerical vs Analytical', 'FontSize', 14);
                xlabel('x', 'FontSize', 12);
                ylabel('Temperature', 'FontSize', 12);
                legend('show', 'Location', 'best');
                grid on;
                fprintf('Problem 1 completed successfully.\n');
            else
                fprintf('Problem 1 failed: %s\n', result.message);
            end
            % Problem 2: Steady heat equation with source
            fprintf('\nProblem 2: Steady heat equation with source\n');
            fprintf('-α∇²u = f(x), u(0) = 0, u(1) = 0\n');
            fprintf('Source: f(x) = π²sin(πx)\n\n');
            source_term = @(x) pi^2 * sin(pi * x);
            result_steady = solver.solve_steady(source_term);
            if result_steady.success
                figure;
                x = domain.x;
                u_analytical_steady = sin(pi * x) / alpha;  % Analytical solution
                plot(x, result_steady.u, 'o-', 'Color', [0, 50, 98]/255, ...
                    'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Numerical');
                hold on;
                plot(x, u_analytical_steady, '--', 'Color', [253, 181, 21]/255, ...
                    'LineWidth', 2, 'DisplayName', 'Analytical');
                hold off;
                title('Steady Heat Equation with Source', 'FontSize', 14);
                xlabel('x', 'FontSize', 12);
                ylabel('Temperature', 'FontSize', 12);
                legend('show', 'Location', 'best');
                grid on;
                % Calculate error
                error = max(abs(result_steady.u - u_analytical_steady'));
                fprintf('Maximum error: %.6e\n', error);
                fprintf('Problem 2 completed successfully.\n');
            else
                fprintf('Problem 2 failed: %s\n', result_steady.message);
            end
            fprintf('\nDemo completed. Check figures for results.\n');
        end
    end
end