classdef GradientDescent < handle
    % GradientDescent - Gradient descent optimization algorithm
    %
    % BERKELEY SCICOMP - OPTIMIZATION TOOLBOX
    % =====================================
    %
    % This class implements the gradient descent algorithm for unconstrained
    % optimization with various line search methods and momentum options.
    %
    % Author: Berkeley SciComp Team
    % Date: 2024
    properties
        MaxIterations = 1000        % Maximum number of iterations
        Tolerance = 1e-6            % Convergence tolerance
        StepSize = 0.01            % Initial step size
        LineSearch = 'backtrack'    % Line search method
        Momentum = 0.0             % Momentum parameter
        Verbose = false            % Display progress
        % Berkeley colors
        BerkeleyBlue = [0 50 98]/255;
        CaliforniaGold = [253 181 21]/255;
    end
    properties (Access = private)
        History = struct()         % Optimization history
    end
    methods
        function obj = GradientDescent(varargin)
            % Constructor for GradientDescent
            %
            % Usage:
            %   gd = optimization.GradientDescent()
            %   gd = optimization.GradientDescent('MaxIterations', 2000, 'Tolerance', 1e-8)
            % Parse input arguments
            p = inputParser;
            addParameter(p, 'MaxIterations', 1000, @isnumeric);
            addParameter(p, 'Tolerance', 1e-6, @isnumeric);
            addParameter(p, 'StepSize', 0.01, @isnumeric);
            addParameter(p, 'LineSearch', 'backtrack', @ischar);
            addParameter(p, 'Momentum', 0.0, @isnumeric);
            addParameter(p, 'Verbose', false, @islogical);
            parse(p, varargin{:});
            obj.MaxIterations = p.Results.MaxIterations;
            obj.Tolerance = p.Results.Tolerance;
            obj.StepSize = p.Results.StepSize;
            obj.LineSearch = p.Results.LineSearch;
            obj.Momentum = p.Results.Momentum;
            obj.Verbose = p.Results.Verbose;
        end
        function result = minimize(obj, objective, x0, gradient)
            % Minimize objective function using gradient descent
            %
            % Inputs:
            %   objective - Function handle for objective function
            %   x0        - Initial guess (column vector)
            %   gradient  - Function handle for gradient (optional)
            %
            % Outputs:
            %   result - Structure with optimization results
            if nargin < 4
                gradient = @(x) obj.numericalGradient(objective, x);
            end
            x = x0(:);  % Ensure column vector
            n = length(x);
            % Initialize history
            obj.History.x = zeros(n, obj.MaxIterations + 1);
            obj.History.f = zeros(obj.MaxIterations + 1, 1);
            obj.History.grad_norm = zeros(obj.MaxIterations + 1, 1);
            % Initial evaluation
            f = objective(x);
            grad = gradient(x);
            obj.History.x(:, 1) = x;
            obj.History.f(1) = f;
            obj.History.grad_norm(1) = norm(grad);
            % Initialize momentum
            velocity = zeros(size(x));
            % Main optimization loop
            for iter = 1:obj.MaxIterations
                % Check convergence
                if norm(grad) < obj.Tolerance
                    if obj.Verbose
                        fprintf('Converged at iteration %d\n', iter);
                    end
                    break;
                end
                % Compute search direction
                direction = -grad;
                % Apply momentum
                if obj.Momentum > 0
                    velocity = obj.Momentum * velocity + (1 - obj.Momentum) * direction;
                    direction = velocity;
                end
                % Line search
                switch lower(obj.LineSearch)
                    case 'fixed'
                        step_size = obj.StepSize;
                    case 'backtrack'
                        step_size = obj.backtrackLineSearch(objective, x, direction, grad);
                    case 'wolfe'
                        step_size = obj.wolfeLineSearch(objective, gradient, x, direction, grad);
                    otherwise
                        step_size = obj.StepSize;
                end
                % Update variables
                x_new = x + step_size * direction;
                f_new = objective(x_new);
                grad_new = gradient(x_new);
                % Store history
                obj.History.x(:, iter + 1) = x_new;
                obj.History.f(iter + 1) = f_new;
                obj.History.grad_norm(iter + 1) = norm(grad_new);
                % Display progress
                if obj.Verbose && mod(iter, 100) == 0
                    fprintf('Iter %4d: f = %12.6e, ||grad|| = %12.6e\n', ...
                            iter, f_new, norm(grad_new));
                end
                % Update for next iteration
                x = x_new;
                f = f_new;
                grad = grad_new;
            end
            % Trim history
            actual_iters = min(iter, obj.MaxIterations);
            obj.History.x = obj.History.x(:, 1:actual_iters + 1);
            obj.History.f = obj.History.f(1:actual_iters + 1);
            obj.History.grad_norm = obj.History.grad_norm(1:actual_iters + 1);
            % Create result structure
            result = struct();
            result.x = x;
            result.fun = f;
            result.grad = grad;
            result.success = norm(grad) < obj.Tolerance;
            result.nit = actual_iters;
            result.nfev = actual_iters + 1;  % Approximate
            result.message = obj.getExitMessage(result.success, actual_iters);
            result.history = obj.History;
        end
        function step_size = backtrackLineSearch(obj, objective, x, direction, grad)
            % Backtracking line search with Armijo condition
            alpha = 1.0;
            c1 = 1e-4;  % Armijo parameter
            rho = 0.5;  % Backtracking parameter
            f0 = objective(x);
            grad_dot_dir = grad' * direction;
            while objective(x + alpha * direction) > f0 + c1 * alpha * grad_dot_dir
                alpha = rho * alpha;
                if alpha < 1e-10
                    break;
                end
            end
            step_size = alpha;
        end
        function step_size = wolfeLineSearch(obj, objective, gradient, x, direction, grad)
            % Wolfe line search (simplified implementation)
            alpha = 1.0;
            c1 = 1e-4;  % Armijo parameter
            c2 = 0.9;   % Curvature parameter
            f0 = objective(x);
            grad_dot_dir = grad' * direction;
            % Simple implementation - in practice, would use more sophisticated method
            for i = 1:10
                x_new = x + alpha * direction;
                f_new = objective(x_new);
                grad_new = gradient(x_new);
                % Check Armijo condition
                if f_new <= f0 + c1 * alpha * grad_dot_dir
                    % Check curvature condition
                    if grad_new' * direction >= c2 * grad_dot_dir
                        break;
                    end
                end
                alpha = 0.5 * alpha;
            end
            step_size = alpha;
        end
        function grad = numericalGradient(obj, objective, x)
            % Compute numerical gradient using finite differences
            h = 1e-8;
            n = length(x);
            grad = zeros(n, 1);
            for i = 1:n
                x_plus = x;
                x_minus = x;
                x_plus(i) = x_plus(i) + h;
                x_minus(i) = x_minus(i) - h;
                grad(i) = (objective(x_plus) - objective(x_minus)) / (2 * h);
            end
        end
        function message = getExitMessage(obj, success, iterations)
            % Get exit message based on optimization result
            if success
                message = 'Optimization terminated successfully';
            elseif iterations >= obj.MaxIterations
                message = 'Maximum number of iterations reached';
            else
                message = 'Optimization failed';
            end
        end
        function plotConvergence(obj)
            % Plot convergence history
            if isempty(obj.History)
                error('No optimization history available. Run minimize() first.');
            end
            figure;
            % Objective function
            subplot(2, 1, 1);
            semilogy(0:length(obj.History.f)-1, obj.History.f, 'b-', 'LineWidth', 2);
            xlabel('Iteration');
            ylabel('Objective Function');
            title('Gradient Descent Convergence', 'Color', obj.BerkeleyBlue);
            grid on;
            % Gradient norm
            subplot(2, 1, 2);
            semilogy(0:length(obj.History.grad_norm)-1, obj.History.grad_norm, 'r-', 'LineWidth', 2);
            xlabel('Iteration');
            ylabel('||Gradient||');
            title('Gradient Norm', 'Color', obj.BerkeleyBlue);
            grid on;
            % Set figure properties
            set(gcf, 'Position', [100, 100, 800, 600]);
        end
    end
end