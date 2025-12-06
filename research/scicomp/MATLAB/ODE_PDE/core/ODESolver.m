classdef ODESolver < handle
    %ODESOLVER Base class for ODE solvers
    %   Provides common interface and functionality for all ODE solving methods
    %
    %   Author: Berkeley SciComp Team
    %   Date: 2024
    properties (Access = protected)
        tolerance = 1e-8        % Error tolerance
        max_iterations = 10000  % Maximum iterations
        current_time = 0        % Current time
        current_solution        % Current solution vector
        step_count = 0          % Number of steps taken
        berkeley_blue = [0, 50, 98]/255      % UC Berkeley Blue
        california_gold = [253, 181, 21]/255 % California Gold
    end
    properties (Access = public)
        name = 'Generic ODE Solver'  % Solver name
        order = 1                    % Order of accuracy
    end
    methods (Abstract)
        result = solve(obj, func, y0, t_span, dt, varargin)
        y_new = step(obj, func, t, y, dt)
    end
    methods
        function obj = ODESolver(varargin)
            %ODESOLVER Constructor
            %   obj = ODESolver('tolerance', 1e-6, 'max_iterations', 5000)
            % Parse input arguments
            p = inputParser;
            addParameter(p, 'tolerance', 1e-8, @isnumeric);
            addParameter(p, 'max_iterations', 10000, @isnumeric);
            parse(p, varargin{:});
            obj.tolerance = p.Results.tolerance;
            obj.max_iterations = p.Results.max_iterations;
        end
        function result = solve_generic(obj, func, y0, t_span, dt, varargin)
            %SOLVE_GENERIC Generic solve method for simple ODE solvers
            %   result = solve_generic(obj, func, y0, t_span, dt)
            %
            %   Inputs:
            %       func - Function handle dy/dt = func(t, y)
            %       y0 - Initial condition
            %       t_span - [t_initial, t_final]
            %       dt - Time step
            %
            %   Outputs:
            %       result - Structure with fields:
            %           t - Time vector
            %           y - Solution matrix
            %           success - Boolean success flag
            %           message - Status message
            %           step_count - Number of steps taken
            try
                % Validate inputs
                obj.validate_inputs(func, y0, t_span, dt);
                % Setup
                t_start = t_span(1);
                t_end = t_span(2);
                y0 = y0(:);  % Ensure column vector
                n_vars = length(y0);
                % Determine number of steps
                if dt > 0
                    t_vec = t_start:dt:t_end;
                else
                    t_vec = t_start:dt:t_end;
                end
                n_steps = length(t_vec);
                % Initialize storage
                y_storage = zeros(n_steps, n_vars);
                y_storage(1, :) = y0';
                % Integration loop
                y_current = y0;
                for i = 2:n_steps
                    t_current = t_vec(i-1);
                    dt_current = t_vec(i) - t_current;
                    y_new = obj.step(func, t_current, y_current, dt_current);
                    y_storage(i, :) = y_new';
                    y_current = y_new;
                    obj.step_count = obj.step_count + 1;
                end
                % Return result
                result.t = t_vec';
                result.y = y_storage;
                result.success = true;
                result.message = sprintf('%s completed successfully', obj.name);
                result.step_count = obj.step_count;
                result.solver_name = obj.name;
                result.order = obj.order;
            catch ME
                result.t = [];
                result.y = [];
                result.success = false;
                result.message = sprintf('Error: %s', ME.message);
                result.step_count = obj.step_count;
                result.solver_name = obj.name;
                result.order = obj.order;
            end
        end
        function validate_inputs(~, func, y0, t_span, dt)
            %VALIDATE_INPUTS Validate solver inputs
            if ~isa(func, 'function_handle')
                error('ODESolver:InvalidInput', 'func must be a function handle');
            end
            if ~isnumeric(y0) || ~isreal(y0)
                error('ODESolver:InvalidInput', 'y0 must be real numeric');
            end
            if length(t_span) ~= 2 || t_span(2) <= t_span(1)
                error('ODESolver:InvalidInput', 't_span must be [t0, tf] with tf > t0');
            end
            if ~isnumeric(dt) || dt == 0
                error('ODESolver:InvalidInput', 'dt must be non-zero numeric');
            end
            % Test function evaluation
            try
                t_test = t_span(1);
                dydt_test = func(t_test, y0(:));
                if ~isnumeric(dydt_test) || length(dydt_test) ~= length(y0)
                    error('Function output dimension mismatch');
                end
            catch ME
                error('ODESolver:InvalidFunction', ...
                    'Function evaluation failed: %s', ME.message);
            end
        end
        function plot_solution(obj, result, varargin)
            %PLOT_SOLUTION Plot ODE solution with Berkeley styling
            %   plot_solution(obj, result)
            %   plot_solution(obj, result, 'title', 'My Solution')
            p = inputParser;
            addParameter(p, 'title', 'ODE Solution', @ischar);
            addParameter(p, 'xlabel', 'Time', @ischar);
            addParameter(p, 'ylabel', 'Solution', @ischar);
            addParameter(p, 'linewidth', 2, @isnumeric);
            parse(p, varargin{:});
            if ~result.success
                warning('Cannot plot failed solution');
                return;
            end
            figure;
            % Plot all solution components
            [n_points, n_vars] = size(result.y);
            colors = obj.get_berkeley_colors(n_vars);
            hold on;
            for i = 1:n_vars
                plot(result.t, result.y(:, i), 'Color', colors(i, :), ...
                    'LineWidth', p.Results.linewidth, ...
                    'DisplayName', sprintf('y_%d', i));
            end
            hold off;
            % Berkeley styling
            obj.apply_berkeley_style();
            title(p.Results.title, 'FontSize', 14, 'FontWeight', 'bold');
            xlabel(p.Results.xlabel, 'FontSize', 12);
            ylabel(p.Results.ylabel, 'FontSize', 12);
            if n_vars > 1
                legend('show', 'Location', 'best');
            end
            grid on;
            grid minor;
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
        function reset(obj)
            %RESET Reset solver state
            obj.current_time = 0;
            obj.current_solution = [];
            obj.step_count = 0;
        end
        function info = get_solver_info(obj)
            %GET_SOLVER_INFO Get solver information
            info.name = obj.name;
            info.order = obj.order;
            info.tolerance = obj.tolerance;
            info.max_iterations = obj.max_iterations;
            info.step_count = obj.step_count;
            info.description = sprintf('%s (Order %d)', obj.name, obj.order);
        end
    end
end