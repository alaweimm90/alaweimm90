classdef BFGS < handle
    % BFGS - Broyden-Fletcher-Goldfarb-Shanno quasi-Newton method
    %
    % BERKELEY SCICOMP - OPTIMIZATION TOOLBOX
    % =====================================
    %
    % This class implements the BFGS algorithm for unconstrained optimization
    % with line search and various initialization strategies.
    %
    % Author: Berkeley SciComp Team
    % Date: 2024
    properties
        MaxIterations = 1000        % Maximum number of iterations
        Tolerance = 1e-6            % Convergence tolerance
        LineSearch = 'wolfe'        % Line search method
        InitialHessian = 'identity' % Initial Hessian approximation
        Verbose = false            % Display progress
        % Berkeley colors
        BerkeleyBlue = [0 50 98]/255;
        CaliforniaGold = [253 181 21]/255;
    end
    properties (Access = private)
        History = struct()         % Optimization history
    end
    methods
        function obj = BFGS(varargin)
            % Constructor for BFGS
            %
            % Usage:
            %   bfgs = optimization.BFGS()
            %   bfgs = optimization.BFGS('MaxIterations', 2000, 'Tolerance', 1e-8)
            % Parse input arguments
            p = inputParser;
            addParameter(p, 'MaxIterations', 1000, @isnumeric);
            addParameter(p, 'Tolerance', 1e-6, @isnumeric);
            addParameter(p, 'LineSearch', 'wolfe', @ischar);
            addParameter(p, 'InitialHessian', 'identity', @ischar);
            addParameter(p, 'Verbose', false, @islogical);
            parse(p, varargin{:});
            obj.MaxIterations = p.Results.MaxIterations;
            obj.Tolerance = p.Results.Tolerance;
            obj.LineSearch = p.Results.LineSearch;
            obj.InitialHessian = p.Results.InitialHessian;
            obj.Verbose = p.Results.Verbose;
        end
        function result = minimize(obj, objective, x0, gradient)
            % Minimize objective function using BFGS
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
            obj.History.step_size = zeros(obj.MaxIterations, 1);
            % Initial evaluation
            f = objective(x);
            grad = gradient(x);
            obj.History.x(:, 1) = x;
            obj.History.f(1) = f;
            obj.History.grad_norm(1) = norm(grad);
            % Initialize Hessian approximation
            H = obj.initializeHessian(n, objective, x, grad);
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
                direction = -H \ grad;
                % Line search
                switch lower(obj.LineSearch)
                    case 'backtrack'
                        step_size = obj.backtrackLineSearch(objective, x, direction, grad);
                    case 'wolfe'
                        step_size = obj.wolfeLineSearch(objective, gradient, x, direction, grad);
                    case 'strong_wolfe'
                        step_size = obj.strongWolfeLineSearch(objective, gradient, x, direction, grad);
                    otherwise
                        step_size = 1.0;
                end
                % Update variables
                x_new = x + step_size * direction;
                f_new = objective(x_new);
                grad_new = gradient(x_new);
                % Store step information for Hessian update
                s = x_new - x;  % Step vector
                y = grad_new - grad;  % Gradient difference
                % Update Hessian approximation using BFGS formula
                if s' * y > 1e-10  % Curvature condition
                    H = obj.updateHessian(H, s, y);
                else
                    if obj.Verbose
                        fprintf('Skipping BFGS update due to curvature condition\n');
                    end
                end
                % Store history
                obj.History.x(:, iter + 1) = x_new;
                obj.History.f(iter + 1) = f_new;
                obj.History.grad_norm(iter + 1) = norm(grad_new);
                obj.History.step_size(iter) = step_size;
                % Display progress
                if obj.Verbose && mod(iter, 10) == 0
                    fprintf('Iter %4d: f = %12.6e, ||grad|| = %12.6e, step = %12.6e\n', ...
                            iter, f_new, norm(grad_new), step_size);
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
            obj.History.step_size = obj.History.step_size(1:actual_iters);
            % Create result structure
            result = struct();
            result.x = x;
            result.fun = f;
            result.grad = grad;
            result.hess_inv = H;
            result.success = norm(grad) < obj.Tolerance;
            result.nit = actual_iters;
            result.nfev = actual_iters + 1;  % Approximate
            result.message = obj.getExitMessage(result.success, actual_iters);
            result.history = obj.History;
        end
        function H = initializeHessian(obj, n, objective, x, grad)
            % Initialize Hessian approximation
            switch lower(obj.InitialHessian)
                case 'identity'
                    H = eye(n);
                case 'scaled_identity'
                    % Scale by gradient norm
                    scale = max(1.0, norm(grad));
                    H = scale * eye(n);
                case 'diagonal'
                    % Diagonal approximation based on finite differences
                    h = 1e-6;
                    diag_elements = zeros(n, 1);
                    for i = 1:n
                        x_plus = x; x_plus(i) = x_plus(i) + h;
                        x_minus = x; x_minus(i) = x_minus(i) - h;
                        f_plus = objective(x_plus);
                        f_minus = objective(x_minus);
                        f_center = objective(x);
                        second_deriv = (f_plus - 2*f_center + f_minus) / h^2;
                        diag_elements(i) = max(1e-6, abs(second_deriv));
                    end
                    H = diag(1 ./ diag_elements);
                otherwise
                    H = eye(n);
            end
        end
        function H_new = updateHessian(obj, H, s, y)
            % Update Hessian approximation using BFGS formula
            %
            % H_new = H - (H*s*s'*H)/(s'*H*s) + (y*y')/(y'*s)
            rho = 1 / (y' * s);
            % BFGS update formula
            I = eye(size(H));
            H_new = (I - rho * s * y') * H * (I - rho * y * s') + rho * s * s';
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
            % Wolfe line search with Armijo and curvature conditions
            alpha = 1.0;
            c1 = 1e-4;  % Armijo parameter
            c2 = 0.9;   % Curvature parameter
            max_iters = 20;
            f0 = objective(x);
            grad_dot_dir = grad' * direction;
            for i = 1:max_iters
                x_new = x + alpha * direction;
                f_new = objective(x_new);
                % Check Armijo condition
                if f_new > f0 + c1 * alpha * grad_dot_dir
                    alpha = 0.5 * alpha;
                    continue;
                end
                % Check curvature condition
                grad_new = gradient(x_new);
                if grad_new' * direction >= c2 * grad_dot_dir
                    break;
                end
                alpha = min(2 * alpha, 1.0);
            end
            step_size = alpha;
        end
        function step_size = strongWolfeLineSearch(obj, objective, gradient, x, direction, grad)
            % Strong Wolfe line search (simplified implementation)
            alpha = 1.0;
            c1 = 1e-4;  % Armijo parameter
            c2 = 0.1;   % Curvature parameter (stronger than regular Wolfe)
            f0 = objective(x);
            grad_dot_dir = grad' * direction;
            % Simple implementation - in practice would use zoom procedure
            for i = 1:10
                x_new = x + alpha * direction;
                f_new = objective(x_new);
                grad_new = gradient(x_new);
                % Check Armijo condition
                armijo_satisfied = f_new <= f0 + c1 * alpha * grad_dot_dir;
                % Check strong curvature condition
                curvature_satisfied = abs(grad_new' * direction) <= c2 * abs(grad_dot_dir);
                if armijo_satisfied && curvature_satisfied
                    break;
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
            subplot(2, 2, 1);
            semilogy(0:length(obj.History.f)-1, obj.History.f, 'b-', 'LineWidth', 2);
            xlabel('Iteration');
            ylabel('Objective Function');
            title('BFGS Convergence', 'Color', obj.BerkeleyBlue);
            grid on;
            % Gradient norm
            subplot(2, 2, 2);
            semilogy(0:length(obj.History.grad_norm)-1, obj.History.grad_norm, 'r-', 'LineWidth', 2);
            xlabel('Iteration');
            ylabel('||Gradient||');
            title('Gradient Norm', 'Color', obj.BerkeleyBlue);
            grid on;
            % Step size
            subplot(2, 2, 3);
            semilogy(1:length(obj.History.step_size), obj.History.step_size, 'g-', 'LineWidth', 2);
            xlabel('Iteration');
            ylabel('Step Size');
            title('Step Size History', 'Color', obj.BerkeleyBlue);
            grid on;
            % Variable trajectory (for 2D problems)
            if size(obj.History.x, 1) == 2
                subplot(2, 2, 4);
                plot(obj.History.x(1, :), obj.History.x(2, :), 'o-', 'LineWidth', 2, 'MarkerSize', 4);
                xlabel('x_1');
                ylabel('x_2');
                title('Optimization Path', 'Color', obj.BerkeleyBlue);
                grid on;
            end
            % Set figure properties
            set(gcf, 'Position', [100, 100, 1000, 800]);
        end
    end
end