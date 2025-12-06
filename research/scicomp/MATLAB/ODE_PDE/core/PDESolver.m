classdef PDESolver < handle
    %PDESOLVER Base class for PDE solvers
    %   Provides common interface and functionality for all PDE solving methods
    %
    %   Author: Berkeley SciComp Team
    %   Date: 2024
    properties (Access = protected)
        domain                  % Spatial domain
        boundary_conditions     % Boundary conditions
        tolerance = 1e-6       % Error tolerance
        max_iterations = 10000 % Maximum iterations
        berkeley_blue = [0, 50, 98]/255      % UC Berkeley Blue
        california_gold = [253, 181, 21]/255 % California Gold
    end
    properties (Access = public)
        name = 'Generic PDE Solver'  % Solver name
        dimension = 1                % Spatial dimension
        pde_type = 'generic'        % Type of PDE
    end
    methods (Abstract)
        result = solve_steady(obj, source_term, varargin)
        result = solve_transient(obj, initial_condition, varargin)
    end
    methods
        function obj = PDESolver(domain, boundary_conditions, varargin)
            %PDESOLVER Constructor
            %   obj = PDESolver(domain, boundary_conditions, ...)
            %
            %   Inputs:
            %       domain - Spatial domain structure
            %       boundary_conditions - Boundary condition structure
            % Parse input arguments
            p = inputParser;
            addRequired(p, 'domain', @isstruct);
            addRequired(p, 'boundary_conditions', @isstruct);
            addParameter(p, 'tolerance', 1e-6, @isnumeric);
            addParameter(p, 'max_iterations', 10000, @isnumeric);
            parse(p, domain, boundary_conditions, varargin{:});
            obj.domain = domain;
            obj.boundary_conditions = boundary_conditions;
            obj.tolerance = p.Results.tolerance;
            obj.max_iterations = p.Results.max_iterations;
            % Validate domain and boundary conditions
            obj.validate_domain();
            obj.validate_boundary_conditions();
        end
        function validate_domain(obj)
            %VALIDATE_DOMAIN Validate spatial domain
            if ~isfield(obj.domain, 'x')
                error('PDESolver:InvalidDomain', 'Domain must contain field ''x''');
            end
            if ~isnumeric(obj.domain.x) || ~isvector(obj.domain.x)
                error('PDESolver:InvalidDomain', 'Domain.x must be numeric vector');
            end
            % Ensure x is sorted
            if ~issorted(obj.domain.x)
                error('PDESolver:InvalidDomain', 'Domain.x must be sorted');
            end
            % Check for minimum grid size
            if length(obj.domain.x) < 3
                error('PDESolver:InvalidDomain', 'Domain must have at least 3 grid points');
            end
        end
        function validate_boundary_conditions(obj)
            %VALIDATE_BOUNDARY_CONDITIONS Validate boundary conditions
            if ~isstruct(obj.boundary_conditions)
                error('PDESolver:InvalidBC', 'Boundary conditions must be struct');
            end
            % Check for supported boundary condition types
            supported_types = {'dirichlet', 'neumann', 'robin', 'periodic'};
            bc_fields = fieldnames(obj.boundary_conditions);
            for i = 1:length(bc_fields)
                if ~ismember(bc_fields{i}, supported_types)
                    warning('PDESolver:UnknownBC', ...
                        'Unknown boundary condition type: %s', bc_fields{i});
                end
            end
            % Validate Dirichlet conditions
            if isfield(obj.boundary_conditions, 'dirichlet')
                dirichlet = obj.boundary_conditions.dirichlet;
                if ~isstruct(dirichlet) && ~isa(dirichlet, 'containers.Map')
                    error('PDESolver:InvalidBC', ...
                        'Dirichlet BC must be struct or containers.Map');
                end
            end
        end
        function [A, b] = build_finite_difference_matrix(obj, source_term)
            %BUILD_FINITE_DIFFERENCE_MATRIX Build FD matrix for Poisson equation
            %   -d²u/dx² = f(x) with appropriate boundary conditions
            x = obj.domain.x(:);
            n = length(x);
            dx = x(2) - x(1);  % Assume uniform grid
            % Initialize matrix and RHS
            A = zeros(n, n);
            b = zeros(n, 1);
            % Evaluate source term
            if isa(source_term, 'function_handle')
                f = source_term(x);
            else
                f = source_term * ones(n, 1);
            end
            % Interior points: -u_{i-1} + 2u_i - u_{i+1} = dx²*f_i
            for i = 2:n-1
                A(i, i-1) = -1;
                A(i, i) = 2;
                A(i, i+1) = -1;
                b(i) = dx^2 * f(i);
            end
            % Apply boundary conditions
            obj.apply_boundary_conditions_to_matrix(A, b);
        end
        function apply_boundary_conditions_to_matrix(obj, A, b)
            %APPLY_BOUNDARY_CONDITIONS_TO_MATRIX Apply BC to linear system
            n = size(A, 1);
            % Dirichlet boundary conditions
            if isfield(obj.boundary_conditions, 'dirichlet')
                dirichlet = obj.boundary_conditions.dirichlet;
                if isstruct(dirichlet)
                    indices = fieldnames(dirichlet);
                    for i = 1:length(indices)
                        idx = str2double(indices{i}) + 1;  % Convert to 1-based
                        if idx >= 1 && idx <= n
                            A(idx, :) = 0;
                            A(idx, idx) = 1;
                            b(idx) = dirichlet.(indices{i});
                        end
                    end
                elseif isa(dirichlet, 'containers.Map')
                    keys_list = keys(dirichlet);
                    for i = 1:length(keys_list)
                        idx = keys_list{i} + 1;  % Convert to 1-based
                        if idx >= 1 && idx <= n
                            A(idx, :) = 0;
                            A(idx, idx) = 1;
                            b(idx) = dirichlet(keys_list{i});
                        end
                    end
                end
            end
            % Neumann boundary conditions (simple implementation)
            if isfield(obj.boundary_conditions, 'neumann')
                neumann = obj.boundary_conditions.neumann;
                % Left boundary
                if isfield(neumann, 'left') || (isa(neumann, 'containers.Map') && isKey(neumann, 0))
                    dx = obj.domain.x(2) - obj.domain.x(1);
                    A(1, 1) = -1;
                    A(1, 2) = 1;
                    if isfield(neumann, 'left')
                        b(1) = dx * neumann.left;
                    else
                        b(1) = dx * neumann(0);
                    end
                end
                % Right boundary
                if isfield(neumann, 'right') || (isa(neumann, 'containers.Map') && isKey(neumann, n-1))
                    dx = obj.domain.x(2) - obj.domain.x(1);
                    A(n, n-1) = -1;
                    A(n, n) = 1;
                    if isfield(neumann, 'right')
                        b(n) = dx * neumann.right;
                    else
                        b(n) = dx * neumann(n-1);
                    end
                end
            end
        end
        function result = create_result_structure(obj, u, t, success, message)
            %CREATE_RESULT_STRUCTURE Create standardized result structure
            result.u = u;
            result.t = t;
            result.x = obj.domain.x;
            result.success = success;
            result.message = message;
            result.solver_name = obj.name;
            result.pde_type = obj.pde_type;
            result.dimension = obj.dimension;
            if success && ~isempty(u)
                result.solution_norm = norm(u(:));
                if size(u, 1) > 1  % Time-dependent solution
                    result.final_time = t(end);
                    result.time_steps = length(t);
                end
            end
        end
        function plot_solution(obj, result, varargin)
            %PLOT_SOLUTION Plot PDE solution with Berkeley styling
            p = inputParser;
            addParameter(p, 'title', 'PDE Solution', @ischar);
            addParameter(p, 'xlabel', 'x', @ischar);
            addParameter(p, 'ylabel', 'u(x)', @ischar);
            addParameter(p, 'linewidth', 2, @isnumeric);
            addParameter(p, 'surface_plot', false, @islogical);
            parse(p, varargin{:});
            if ~result.success
                warning('Cannot plot failed solution');
                return;
            end
            x = result.x;
            u = result.u;
            if size(u, 1) == 1 || isempty(result.t)
                % Steady-state solution
                figure;
                plot(x, u, 'Color', obj.berkeley_blue, 'LineWidth', p.Results.linewidth);
                obj.apply_berkeley_style();
                title(p.Results.title, 'FontSize', 14, 'FontWeight', 'bold');
                xlabel(p.Results.xlabel, 'FontSize', 12);
                ylabel(p.Results.ylabel, 'FontSize', 12);
                grid on;
            else
                % Time-dependent solution
                if p.Results.surface_plot
                    figure;
                    [X, T] = meshgrid(x, result.t);
                    surf(X, T, u, 'EdgeColor', 'none');
                    colormap(obj.get_berkeley_colormap());
                    obj.apply_berkeley_style();
                    title(p.Results.title, 'FontSize', 14, 'FontWeight', 'bold');
                    xlabel(p.Results.xlabel, 'FontSize', 12);
                    ylabel('Time', 'FontSize', 12);
                    zlabel(p.Results.ylabel, 'FontSize', 12);
                    view(45, 30);
                    colorbar;
                else
                    figure;
                    % Plot solution at different time snapshots
                    n_snapshots = min(5, size(u, 1));
                    time_indices = round(linspace(1, size(u, 1), n_snapshots));
                    colors = obj.get_berkeley_colors(n_snapshots);
                    hold on;
                    for i = 1:n_snapshots
                        idx = time_indices(i);
                        plot(x, u(idx, :), 'Color', colors(i, :), ...
                            'LineWidth', p.Results.linewidth, ...
                            'DisplayName', sprintf('t = %.3f', result.t(idx)));
                    end
                    hold off;
                    obj.apply_berkeley_style();
                    title(p.Results.title, 'FontSize', 14, 'FontWeight', 'bold');
                    xlabel(p.Results.xlabel, 'FontSize', 12);
                    ylabel(p.Results.ylabel, 'FontSize', 12);
                    legend('show', 'Location', 'best');
                    grid on;
                end
            end
        end
        function cmap = get_berkeley_colormap(obj)
            %GET_BERKELEY_COLORMAP Create Berkeley-themed colormap
            % Create colormap from white to Berkeley blue
            n_colors = 256;
            white = [1, 1, 1];
            cmap = zeros(n_colors, 3);
            for i = 1:n_colors
                alpha = (i-1) / (n_colors-1);
                cmap(i, :) = (1-alpha) * white + alpha * obj.berkeley_blue;
            end
        end
        function colors = get_berkeley_colors(obj, n_colors)
            %GET_BERKELEY_COLORS Get Berkeley color scheme
            base_colors = [
                obj.berkeley_blue;
                obj.california_gold;
                [59, 126, 161]/255;    % Berkeley Blue Light
                [196, 130, 14]/255;    % California Gold Dark
                [0, 176, 218]/255      % Berkeley Secondary Blue
            ];
            if n_colors <= size(base_colors, 1)
                colors = base_colors(1:n_colors, :);
            else
                % Generate additional colors by interpolation
                colors = zeros(n_colors, 3);
                colors(1:size(base_colors, 1), :) = base_colors;
                for i = size(base_colors, 1)+1:n_colors
                    idx = mod(i-1, size(base_colors, 1)) + 1;
                    colors(i, :) = base_colors(idx, :) * 0.7;
                end
            end
        end
        function apply_berkeley_style(obj)
            %APPLY_BERKELEY_STYLE Apply Berkeley visual styling
            set(gca, 'FontSize', 11);
            set(gca, 'LineWidth', 1.2);
            set(gca, 'GridAlpha', 0.3);
            set(gca, 'MinorGridAlpha', 0.1);
            % Set axis colors to Berkeley blue
            set(gca, 'XColor', obj.berkeley_blue);
            set(gca, 'YColor', obj.berkeley_blue);
        end
        function info = get_solver_info(obj)
            %GET_SOLVER_INFO Get solver information
            info.name = obj.name;
            info.pde_type = obj.pde_type;
            info.dimension = obj.dimension;
            info.tolerance = obj.tolerance;
            info.max_iterations = obj.max_iterations;
            info.domain_size = length(obj.domain.x);
            info.boundary_conditions = obj.boundary_conditions;
        end
        function check_cfl_condition(obj, dt, velocity, diffusivity)
            %CHECK_CFL_CONDITION Check CFL stability condition
            x = obj.domain.x;
            dx = x(2) - x(1);  % Assume uniform grid
            % Advection CFL condition
            if nargin >= 3 && ~isempty(velocity) && velocity ~= 0
                cfl_advection = abs(velocity) * dt / dx;
                if cfl_advection > 1
                    warning('PDESolver:CFLViolation', ...
                        'Advection CFL condition violated: %.3f > 1', cfl_advection);
                end
            end
            % Diffusion stability condition
            if nargin >= 4 && ~isempty(diffusivity) && diffusivity > 0
                r = diffusivity * dt / dx^2;
                if r > 0.5
                    warning('PDESolver:StabilityViolation', ...
                        'Diffusion stability condition violated: %.3f > 0.5', r);
                end
            end
        end
    end
end