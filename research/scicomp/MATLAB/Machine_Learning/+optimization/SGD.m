classdef SGD < handle
    % SGD - Stochastic Gradient Descent Optimizer
    %
    % This class implements various stochastic gradient descent algorithms
    % for optimizing machine learning models and scientific computing problems.
    %
    % Features:
    %   - Classic SGD with momentum
    %   - Adaptive learning rate schedules
    %   - Convergence monitoring
    %   - Berkeley-themed visualizations
    %   - Scientific computing integration
    %
    % Example:
    %   sgd = optimization.SGD(0.01, 'Momentum', 0.9, 'Schedule', 'exponential');
    %   [x_opt, f_opt, history] = sgd.minimize(@objective, x0, 'MaxIter', 1000);
    %
    % Author: Berkeley SciComp Team
    % Date: 2024
    properties (Access = private)
        velocity_
        history_
        isInitialized_ = false
    end
    properties
        LearningRate = 0.01     % Initial learning rate
        Momentum = 0.0          % Momentum parameter
        Schedule = 'constant'   % Learning rate schedule
        DecayRate = 0.95        % Decay rate for exponential schedule
        StepSize = 100          % Step size for step schedule
        Tolerance = 1e-6        % Convergence tolerance
        Verbose = false         % Print optimization progress
    end
    methods
        function obj = SGD(learningRate, varargin)
            % Constructor for SGD optimizer
            %
            % Parameters:
            %   learningRate - Initial learning rate
            %   'Momentum' - Momentum parameter (default: 0.0)
            %   'Schedule' - Learning rate schedule ('constant', 'exponential', 'step') (default: 'constant')
            %   'DecayRate' - Decay rate for exponential schedule (default: 0.95)
            %   'StepSize' - Step size for step schedule (default: 100)
            %   'Tolerance' - Convergence tolerance (default: 1e-6)
            %   'Verbose' - Print progress (default: false)
            % Parse input arguments
            p = inputParser;
            addRequired(p, 'learningRate', @(x) isnumeric(x) && x > 0);
            addParameter(p, 'Momentum', 0.0, @(x) isnumeric(x) && x >= 0 && x < 1);
            addParameter(p, 'Schedule', 'constant', @(x) ismember(x, {'constant', 'exponential', 'step'}));
            addParameter(p, 'DecayRate', 0.95, @(x) isnumeric(x) && x > 0 && x < 1);
            addParameter(p, 'StepSize', 100, @(x) isnumeric(x) && x > 0);
            addParameter(p, 'Tolerance', 1e-6, @(x) isnumeric(x) && x > 0);
            addParameter(p, 'Verbose', false, @islogical);
            parse(p, learningRate, varargin{:});
            obj.LearningRate = p.Results.learningRate;
            obj.Momentum = p.Results.Momentum;
            obj.Schedule = p.Results.Schedule;
            obj.DecayRate = p.Results.DecayRate;
            obj.StepSize = p.Results.StepSize;
            obj.Tolerance = p.Results.Tolerance;
            obj.Verbose = p.Results.Verbose;
        end
        function [x_opt, f_opt, history] = minimize(obj, objective, x0, varargin)
            % Minimize objective function using SGD
            %
            % Parameters:
            %   objective - Function handle for objective function
            %   x0 - Initial point
            %   'MaxIter' - Maximum iterations (default: 1000)
            %   'BatchSize' - Batch size for stochastic gradient (default: 32)
            %   'Data' - Data for stochastic gradients (default: [])
            %
            % Returns:
            %   x_opt - Optimal point
            %   f_opt - Optimal function value
            %   history - Optimization history
            % Parse arguments
            p = inputParser;
            addRequired(p, 'objective', @(x) isa(x, 'function_handle'));
            addRequired(p, 'x0', @isnumeric);
            addParameter(p, 'MaxIter', 1000, @(x) isnumeric(x) && x > 0);
            addParameter(p, 'BatchSize', 32, @(x) isnumeric(x) && x > 0);
            addParameter(p, 'Data', [], @isnumeric);
            parse(p, objective, x0, varargin{:});
            % Initialize
            x = x0(:);  % Ensure column vector
            n = length(x);
            if ~obj.isInitialized_
                obj.velocity_ = zeros(n, 1);
                obj.isInitialized_ = true;
            end
            % Initialize history
            obj.history_ = struct();
            obj.history_.x = zeros(n, p.Results.MaxIter + 1);
            obj.history_.f = zeros(p.Results.MaxIter + 1, 1);
            obj.history_.gradNorm = zeros(p.Results.MaxIter + 1, 1);
            obj.history_.learningRate = zeros(p.Results.MaxIter + 1, 1);
            % Initial evaluation
            [f, grad] = obj.evaluateObjective(objective, x, p.Results.Data, p.Results.BatchSize, 1);
            obj.history_.x(:, 1) = x;
            obj.history_.f(1) = f;
            obj.history_.gradNorm(1) = norm(grad);
            obj.history_.learningRate(1) = obj.LearningRate;
            if obj.Verbose
                fprintf('SGD Optimization\n');
                fprintf('================\n');
                fprintf('Iter\t\tF(x)\t\t||grad||\t\tLR\n');
                fprintf('%4d\t\t%.6e\t%.6e\t%.6e\n', 0, f, norm(grad), obj.LearningRate);
            end
            % Main optimization loop
            for iter = 1:p.Results.MaxIter
                % Compute current learning rate
                lr = obj.computeLearningRate(iter);
                % Compute gradient (stochastic if data provided)
                [f, grad] = obj.evaluateObjective(objective, x, p.Results.Data, p.Results.BatchSize, iter);
                % Update velocity with momentum
                obj.velocity_ = obj.Momentum * obj.velocity_ - lr * grad;
                % Update parameters
                x = x + obj.velocity_;
                % Store history
                obj.history_.x(:, iter + 1) = x;
                obj.history_.f(iter + 1) = f;
                obj.history_.gradNorm(iter + 1) = norm(grad);
                obj.history_.learningRate(iter + 1) = lr;
                % Check convergence
                if norm(grad) < obj.Tolerance
                    if obj.Verbose
                        fprintf('Converged at iteration %d\n', iter);
                    end
                    break;
                end
                % Print progress
                if obj.Verbose && mod(iter, 100) == 0
                    fprintf('%4d\t\t%.6e\t%.6e\t%.6e\n', iter, f, norm(grad), lr);
                end
            end
            % Trim history
            actualIter = min(iter + 1, p.Results.MaxIter + 1);
            obj.history_.x = obj.history_.x(:, 1:actualIter);
            obj.history_.f = obj.history_.f(1:actualIter);
            obj.history_.gradNorm = obj.history_.gradNorm(1:actualIter);
            obj.history_.learningRate = obj.history_.learningRate(1:actualIter);
            % Return results
            x_opt = x;
            f_opt = f;
            history = obj.history_;
        end
        function fig = plotHistory(obj, varargin)
            % Plot optimization history with Berkeley styling
            %
            % Parameters:
            %   'Title' - Plot title (default: 'SGD Optimization History')
            %
            % Returns:
            %   fig - Figure handle
            if isempty(obj.history_)
                error('No optimization history available');
            end
            p = inputParser;
            addParameter(p, 'Title', 'SGD Optimization History', @ischar);
            parse(p, varargin{:});
            % Berkeley colors
            berkeleyBlue = [0, 50, 98]/255;
            californiaGold = [253, 181, 21]/255;
            fig = figure('Position', [100, 100, 1200, 800]);
            iterations = 0:(length(obj.history_.f) - 1);
            % Objective function
            subplot(2, 2, 1);
            semilogy(iterations, obj.history_.f, 'Color', berkeleyBlue, 'LineWidth', 2);
            xlabel('Iteration');
            ylabel('Objective Value');
            title('Objective Function');
            grid on;
            grid minor;
            % Gradient norm
            subplot(2, 2, 2);
            semilogy(iterations, obj.history_.gradNorm, 'Color', californiaGold, 'LineWidth', 2);
            xlabel('Iteration');
            ylabel('Gradient Norm');
            title('Gradient Norm');
            grid on;
            grid minor;
            % Learning rate
            subplot(2, 2, 3);
            plot(iterations, obj.history_.learningRate, 'Color', 'red', 'LineWidth', 2);
            xlabel('Iteration');
            ylabel('Learning Rate');
            title('Learning Rate Schedule');
            grid on;
            grid minor;
            % Parameter trajectory (if 2D)
            subplot(2, 2, 4);
            if size(obj.history_.x, 1) == 2
                plot(obj.history_.x(1, :), obj.history_.x(2, :), 'Color', berkeleyBlue, 'LineWidth', 2);
                hold on;
                scatter(obj.history_.x(1, 1), obj.history_.x(2, 1), 100, 'green', 'filled', 'DisplayName', 'Start');
                scatter(obj.history_.x(1, end), obj.history_.x(2, end), 100, 'red', 'filled', 'DisplayName', 'End');
                xlabel('x1');
                ylabel('x2');
                title('Parameter Trajectory');
                legend();
            else
                % Plot first parameter if higher dimensional
                plot(iterations, obj.history_.x(1, :), 'Color', berkeleyBlue, 'LineWidth', 2);
                xlabel('Iteration');
                ylabel('Parameter 1');
                title('First Parameter Evolution');
            end
            grid on;
            grid minor;
            sgtitle(p.Results.Title, 'FontSize', 14, 'FontWeight', 'bold');
        end
    end
    methods (Access = private)
        function lr = computeLearningRate(obj, iter)
            % Compute learning rate based on schedule
            switch obj.Schedule
                case 'constant'
                    lr = obj.LearningRate;
                case 'exponential'
                    lr = obj.LearningRate * (obj.DecayRate ^ floor(iter / obj.StepSize));
                case 'step'
                    lr = obj.LearningRate / (1 + obj.DecayRate * iter);
            end
        end
        function [f, grad] = evaluateObjective(obj, objective, x, data, batchSize, iter)
            % Evaluate objective function and gradient
            if isempty(data)
                % Full gradient
                if nargout > 1
                    [f, grad] = objective(x);
                else
                    f = objective(x);
                    % Numerical gradient if not provided
                    grad = obj.numericalGradient(objective, x);
                end
            else
                % Stochastic gradient
                nSamples = size(data, 1);
                % Sample batch
                rng(iter); % For reproducibility
                batchIndices = randperm(nSamples, min(batchSize, nSamples));
                batchData = data(batchIndices, :);
                % Evaluate on batch
                if nargout > 1
                    [f, grad] = objective(x, batchData);
                else
                    f = objective(x, batchData);
                    grad = obj.numericalGradient(@(xi) objective(xi, batchData), x);
                end
            end
        end
        function grad = numericalGradient(obj, func, x)
            % Compute numerical gradient using finite differences
            h = 1e-8;
            n = length(x);
            grad = zeros(n, 1);
            for i = 1:n
                x_plus = x;
                x_minus = x;
                x_plus(i) = x_plus(i) + h;
                x_minus(i) = x_minus(i) - h;
                grad(i) = (func(x_plus) - func(x_minus)) / (2 * h);
            end
        end
    end
end