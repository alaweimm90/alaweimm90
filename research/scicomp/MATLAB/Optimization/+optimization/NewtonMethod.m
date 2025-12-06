classdef NewtonMethod < handle
    % NewtonMethod - Newton's method for unconstrained optimization
    %
    % BERKELEY SCICOMP - OPTIMIZATION TOOLBOX
    % =====================================
    %
    % This class implements Newton's method with line search and Hessian
    % modification strategies for unconstrained optimization.
    %
    % Author: Berkeley SciComp Team
    % Date: 2024
    properties
        MaxIterations = 1000        % Maximum number of iterations
        Tolerance = 1e-6            % Convergence tolerance
        LineSearch = 'backtrack'    % Line search method
        HessianModification = 'regularization'  % Hessian modification strategy
        RegularizationParam = 1e-6  % Regularization parameter
        Verbose = false            % Display progress
        % Berkeley colors
        BerkeleyBlue = [0 50 98]/255;
        CaliforniaGold = [253 181 21]/255;
    end
    properties (Access = private)
        History = struct()         % Optimization history
    end
    methods
        function obj = NewtonMethod(varargin)
            % Constructor for NewtonMethod
            %
            % Usage:
            %   nm = optimization.NewtonMethod()
            %   nm = optimization.NewtonMethod('MaxIterations', 2000, 'Tolerance', 1e-8)
            % Parse input arguments
            p = inputParser;
            addParameter(p, 'MaxIterations', 1000, @isnumeric);
            addParameter(p, 'Tolerance', 1e-6, @isnumeric);
            addParameter(p, 'LineSearch', 'backtrack', @ischar);
            addParameter(p, 'HessianModification', 'regularization', @ischar);
            addParameter(p, 'RegularizationParam', 1e-6, @isnumeric);
            addParameter(p, 'Verbose', false, @islogical);
            parse(p, varargin{:});
            obj.MaxIterations = p.Results.MaxIterations;
            obj.Tolerance = p.Results.Tolerance;
            obj.LineSearch = p.Results.LineSearch;
            obj.HessianModification = p.Results.HessianModification;
            obj.RegularizationParam = p.Results.RegularizationParam;
            obj.Verbose = p.Results.Verbose;
        end
        function result = minimize(obj, objective, x0, gradient, hessian)
            % Minimize objective function using Newton's method
            %
            % Inputs:
            %   objective - Function handle for objective function
            %   x0        - Initial guess (column vector)
            %   gradient  - Function handle for gradient (optional)
            %   hessian   - Function handle for Hessian (optional)
            %
            % Outputs:
            %   result - Structure with optimization results
            if nargin < 4
                gradient = @(x) obj.numericalGradient(objective, x);
            end
            if nargin < 5
                hessian = @(x) obj.numericalHessian(objective, x);
            end
            x = x0(:);  % Ensure column vector
            n = length(x);
            % Initialize history
            obj.History.x = zeros(n, obj.MaxIterations + 1);
            obj.History.f = zeros(obj.MaxIterations + 1, 1);
            obj.History.grad_norm = zeros(obj.MaxIterations + 1, 1);
            obj.History.condition_number = zeros(obj.MaxIterations + 1, 1);
            % Initial evaluation
            f = objective(x);
            grad = gradient(x);
            H = hessian(x);
            obj.History.x(:, 1) = x;
            obj.History.f(1) = f;
            obj.History.grad_norm(1) = norm(grad);
            obj.History.condition_number(1) = cond(H);
            % Main optimization loop
            for iter = 1:obj.MaxIterations
                % Check convergence
                if norm(grad) < obj.Tolerance
                    if obj.Verbose
                        fprintf('Converged at iteration %d\n', iter);
                    end
                    break;
                end
                % Modify Hessian if needed
                H_modified = obj.modifyHessian(H);
                % Compute Newton direction
                try
                    direction = -H_modified \ grad;
                catch
                    % Fallback to steepest descent if Hessian is singular
                    if obj.Verbose
                        warning('Hessian is singular, using steepest descent');
                    end
                    direction = -grad;
                end
                % Line search
                switch lower(obj.LineSearch)
                    case 'fixed'
                        step_size = 1.0;
                    case 'backtrack'
                        step_size = obj.backtrackLineSearch(objective, x, direction, grad);
                    case 'wolfe'
                        step_size = obj.wolfeLineSearch(objective, gradient, x, direction, grad);
                    otherwise
                        step_size = 1.0;
                end
                % Update variables
                x_new = x + step_size * direction;
                f_new = objective(x_new);
                grad_new = gradient(x_new);
                H_new = hessian(x_new);
                % Store history
                obj.History.x(:, iter + 1) = x_new;
                obj.History.f(iter + 1) = f_new;
                obj.History.grad_norm(iter + 1) = norm(grad_new);
                obj.History.condition_number(iter + 1) = cond(H_new);
                % Display progress
                if obj.Verbose && mod(iter, 10) == 0
                    fprintf('Iter %4d: f = %12.6e, ||grad|| = %12.6e, cond(H) = %12.6e\n', ...
                            iter, f_new, norm(grad_new), cond(H_new));
                end
                % Update for next iteration
                x = x_new;
                f = f_new;
                grad = grad_new;
                H = H_new;
            end
            % Trim history
            actual_iters = min(iter, obj.MaxIterations);
            obj.History.x = obj.History.x(:, 1:actual_iters + 1);
            obj.History.f = obj.History.f(1:actual_iters + 1);
            obj.History.grad_norm = obj.History.grad_norm(1:actual_iters + 1);
            obj.History.condition_number = obj.History.condition_number(1:actual_iters + 1);
            % Create result structure
            result = struct();
            result.x = x;
            result.fun = f;
            result.grad = grad;
            result.hess = H;
            result.success = norm(grad) < obj.Tolerance;
            result.nit = actual_iters;
            result.nfev = actual_iters + 1;  % Approximate
            result.message = obj.getExitMessage(result.success, actual_iters);
            result.history = obj.History;
        end
        function H_modified = modifyHessian(obj, H)
            % Modify Hessian to ensure positive definiteness
            switch lower(obj.HessianModification)
                case 'none'
                    H_modified = H;
                case 'regularization'
                    % Add regularization term
                    H_modified = H + obj.RegularizationParam * eye(size(H));
                case 'eigenvalue'
                    % Modify negative eigenvalues
                    [V, D] = eig(H);
                    D_modified = max(diag(D), obj.RegularizationParam);
                    H_modified = V * diag(D_modified) * V';
                case 'cholesky'
                    % Modified Cholesky factorization
                    try
                        chol(H);
                        H_modified = H;
                    catch
                        % Add diagonal term until positive definite
                        beta = obj.RegularizationParam;
                        while true
                            try
                                H_test = H + beta * eye(size(H));
                                chol(H_test);
                                H_modified = H_test;
                                break;
                            catch
                                beta = beta * 10;
                                if beta > 1e6
                                    H_modified = H + 1e6 * eye(size(H));
                                    break;
                                end
                            end
                        end
                    end
                otherwise
                    H_modified = H + obj.RegularizationParam * eye(size(H));
            end
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
            % Simple implementation
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
        function H = numericalHessian(obj, objective, x)
            % Compute numerical Hessian using finite differences
            h = 1e-6;
            n = length(x);
            H = zeros(n, n);
            % Diagonal elements
            for i = 1:n
                x_plus = x; x_plus(i) = x_plus(i) + h;
                x_minus = x; x_minus(i) = x_minus(i) - h;
                H(i, i) = (objective(x_plus) - 2*objective(x) + objective(x_minus)) / h^2;
            end
            % Off-diagonal elements
            for i = 1:n
                for j = i+1:n
                    x_pp = x; x_pp(i) = x_pp(i) + h; x_pp(j) = x_pp(j) + h;
                    x_pm = x; x_pm(i) = x_pm(i) + h; x_pm(j) = x_pm(j) - h;
                    x_mp = x; x_mp(i) = x_mp(i) - h; x_mp(j) = x_mp(j) + h;
                    x_mm = x; x_mm(i) = x_mm(i) - h; x_mm(j) = x_mm(j) - h;
                    H(i, j) = (objective(x_pp) - objective(x_pm) - objective(x_mp) + objective(x_mm)) / (4*h^2);
                    H(j, i) = H(i, j);
                end
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
            title('Newton Method Convergence', 'Color', obj.BerkeleyBlue);
            grid on;
            % Gradient norm
            subplot(2, 2, 2);
            semilogy(0:length(obj.History.grad_norm)-1, obj.History.grad_norm, 'r-', 'LineWidth', 2);
            xlabel('Iteration');
            ylabel('||Gradient||');
            title('Gradient Norm', 'Color', obj.BerkeleyBlue);
            grid on;
            % Condition number
            subplot(2, 2, 3);
            semilogy(0:length(obj.History.condition_number)-1, obj.History.condition_number, 'g-', 'LineWidth', 2);
            xlabel('Iteration');
            ylabel('Condition Number');
            title('Hessian Condition Number', 'Color', obj.BerkeleyBlue);
            grid on;
            % Variable trajectory (for 2D problems)
            if size(obj.History.x, 1) == 2
                subplot(2, 2, 4);
                plot(obj.History.x(1, :), obj.History.x(2, :), 'o-', 'LineWidth', 2, 'MarkerSize', 6);
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