classdef Adam < handle
    % ADAM - Adaptive Moment Estimation Optimizer
    %
    % This class implements the Adam optimization algorithm with adaptive
    % learning rates and momentum estimation for efficient optimization
    % of machine learning models and scientific computing problems.
    %
    % Features:
    %   - Adaptive learning rates
    %   - First and second moment estimation
    %   - Bias correction
    %   - Convergence monitoring
    %   - Berkeley-themed visualizations
    %
    % Example:
    %   adam = optimization.Adam(0.001, 'Beta1', 0.9, 'Beta2', 0.999);
    %   [x_opt, f_opt, history] = adam.minimize(@objective, x0, 'MaxIter', 1000);
    %
    % Author: Berkeley SciComp Team
    % Date: 2024
    properties (Access = private)
        m_  % First moment estimate
        v_  % Second moment estimate
        t_  % Time step
        history_
        isInitialized_ = false
    end
    properties
        LearningRate = 0.001    % Learning rate
        Beta1 = 0.9             % Exponential decay rate for first moment
        Beta2 = 0.999           % Exponential decay rate for second moment
        Epsilon = 1e-8          % Small constant for numerical stability
        Tolerance = 1e-6        % Convergence tolerance
        Verbose = false         % Print optimization progress
    end
    methods
        function obj = Adam(learningRate, varargin)
            % Constructor for Adam optimizer
            %
            % Parameters:
            %   learningRate - Learning rate
            %   'Beta1' - Exponential decay rate for first moment (default: 0.9)
            %   'Beta2' - Exponential decay rate for second moment (default: 0.999)
            %   'Epsilon' - Small constant for numerical stability (default: 1e-8)
            %   'Tolerance' - Convergence tolerance (default: 1e-6)
            %   'Verbose' - Print progress (default: false)
            % Parse input arguments
            p = inputParser;
            addRequired(p, 'learningRate', @(x) isnumeric(x) && x > 0);
            addParameter(p, 'Beta1', 0.9, @(x) isnumeric(x) && x >= 0 && x < 1);
            addParameter(p, 'Beta2', 0.999, @(x) isnumeric(x) && x >= 0 && x < 1);
            addParameter(p, 'Epsilon', 1e-8, @(x) isnumeric(x) && x > 0);
            addParameter(p, 'Tolerance', 1e-6, @(x) isnumeric(x) && x > 0);
            addParameter(p, 'Verbose', false, @islogical);
            parse(p, learningRate, varargin{:});
            obj.LearningRate = p.Results.learningRate;
            obj.Beta1 = p.Results.Beta1;
            obj.Beta2 = p.Results.Beta2;
            obj.Epsilon = p.Results.Epsilon;
            obj.Tolerance = p.Results.Tolerance;
            obj.Verbose = p.Results.Verbose;
        end
        function [x_opt, f_opt, history] = minimize(obj, objective, x0, varargin)
            % Minimize objective function using Adam
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
                obj.m_ = zeros(n, 1);
                obj.v_ = zeros(n, 1);
                obj.t_ = 0;
                obj.isInitialized_ = true;
            end
            % Initialize history
            obj.history_ = struct();
            obj.history_.x = zeros(n, p.Results.MaxIter + 1);
            obj.history_.f = zeros(p.Results.MaxIter + 1, 1);
            obj.history_.gradNorm = zeros(p.Results.MaxIter + 1, 1);
            obj.history_.learningRate = zeros(p.Results.MaxIter + 1, 1);
            obj.history_.beta1Power = zeros(p.Results.MaxIter + 1, 1);
            obj.history_.beta2Power = zeros(p.Results.MaxIter + 1, 1);
            % Initial evaluation
            [f, grad] = obj.evaluateObjective(objective, x, p.Results.Data, p.Results.BatchSize, 1);
            obj.history_.x(:, 1) = x;
            obj.history_.f(1) = f;
            obj.history_.gradNorm(1) = norm(grad);
            obj.history_.learningRate(1) = obj.LearningRate;
            obj.history_.beta1Power(1) = obj.Beta1;
            obj.history_.beta2Power(1) = obj.Beta2;
            if obj.Verbose
                fprintf('Adam Optimization\n');
                fprintf('=================\n');
                fprintf('Iter\t\tF(x)\t\t||grad||\t\tLR_eff\n');
                fprintf('%4d\t\t%.6e\t%.6e\t%.6e\n', 0, f, norm(grad), obj.LearningRate);
            end
            % Main optimization loop
            for iter = 1:p.Results.MaxIter
                obj.t_ = obj.t_ + 1;
                % Compute gradient (stochastic if data provided)
                [f, grad] = obj.evaluateObjective(objective, x, p.Results.Data, p.Results.BatchSize, iter);
                % Update biased first moment estimate
                obj.m_ = obj.Beta1 * obj.m_ + (1 - obj.Beta1) * grad;
                % Update biased second raw moment estimate
                obj.v_ = obj.Beta2 * obj.v_ + (1 - obj.Beta2) * (grad .* grad);
                % Compute bias-corrected first moment estimate
                m_hat = obj.m_ / (1 - obj.Beta1^obj.t_);
                % Compute bias-corrected second raw moment estimate
                v_hat = obj.v_ / (1 - obj.Beta2^obj.t_);
                % Update parameters
                x = x - obj.LearningRate * m_hat ./ (sqrt(v_hat) + obj.Epsilon);
                % Compute effective learning rate
                lr_eff = obj.LearningRate / (1 - obj.Beta1^obj.t_) * sqrt(1 - obj.Beta2^obj.t_);
                % Store history
                obj.history_.x(:, iter + 1) = x;
                obj.history_.f(iter + 1) = f;
                obj.history_.gradNorm(iter + 1) = norm(grad);
                obj.history_.learningRate(iter + 1) = lr_eff;
                obj.history_.beta1Power(iter + 1) = obj.Beta1^obj.t_;
                obj.history_.beta2Power(iter + 1) = obj.Beta2^obj.t_;
                % Check convergence
                if norm(grad) < obj.Tolerance
                    if obj.Verbose
                        fprintf('Converged at iteration %d\n', iter);
                    end
                    break;
                end
                % Print progress
                if obj.Verbose && mod(iter, 100) == 0
                    fprintf('%4d\t\t%.6e\t%.6e\t%.6e\n', iter, f, norm(grad), lr_eff);
                end
            end
            % Trim history
            actualIter = min(iter + 1, p.Results.MaxIter + 1);
            obj.history_.x = obj.history_.x(:, 1:actualIter);
            obj.history_.f = obj.history_.f(1:actualIter);
            obj.history_.gradNorm = obj.history_.gradNorm(1:actualIter);
            obj.history_.learningRate = obj.history_.learningRate(1:actualIter);
            obj.history_.beta1Power = obj.history_.beta1Power(1:actualIter);
            obj.history_.beta2Power = obj.history_.beta2Power(1:actualIter);
            % Return results
            x_opt = x;
            f_opt = f;
            history = obj.history_;
        end
        function fig = plotHistory(obj, varargin)
            % Plot optimization history with Berkeley styling
            %
            % Parameters:
            %   'Title' - Plot title (default: 'Adam Optimization History')
            %
            % Returns:
            %   fig - Figure handle
            if isempty(obj.history_)
                error('No optimization history available');
            end
            p = inputParser;
            addParameter(p, 'Title', 'Adam Optimization History', @ischar);
            parse(p, varargin{:});
            % Berkeley colors
            berkeleyBlue = [0, 50, 98]/255;
            californiaGold = [253, 181, 21]/255;
            fig = figure('Position', [100, 100, 1400, 1000]);
            iterations = 0:(length(obj.history_.f) - 1);
            % Objective function
            subplot(2, 3, 1);
            semilogy(iterations, obj.history_.f, 'Color', berkeleyBlue, 'LineWidth', 2);
            xlabel('Iteration');
            ylabel('Objective Value');
            title('Objective Function');
            grid on;
            grid minor;
            % Gradient norm
            subplot(2, 3, 2);
            semilogy(iterations, obj.history_.gradNorm, 'Color', californiaGold, 'LineWidth', 2);
            xlabel('Iteration');
            ylabel('Gradient Norm');
            title('Gradient Norm');
            grid on;
            grid minor;
            % Effective learning rate
            subplot(2, 3, 3);
            plot(iterations, obj.history_.learningRate, 'Color', 'red', 'LineWidth', 2);
            xlabel('Iteration');
            ylabel('Effective Learning Rate');
            title('Effective Learning Rate');
            grid on;
            grid minor;
            % Beta powers
            subplot(2, 3, 4);
            semilogy(iterations, obj.history_.beta1Power, 'Color', berkeleyBlue, 'LineWidth', 2, 'DisplayName', '\beta_1^t');
            hold on;
            semilogy(iterations, obj.history_.beta2Power, 'Color', californiaGold, 'LineWidth', 2, 'DisplayName', '\beta_2^t');
            xlabel('Iteration');
            ylabel('Beta Powers');
            title('Exponential Decay Powers');
            legend();
            grid on;
            grid minor;
            % Parameter trajectory (if 2D)
            subplot(2, 3, 5);
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
            % Bias correction effect
            subplot(2, 3, 6);
            bias_correction_1 = 1 ./ (1 - obj.history_.beta1Power);
            bias_correction_2 = sqrt(1 - obj.history_.beta2Power);
            plot(iterations, bias_correction_1, 'Color', berkeleyBlue, 'LineWidth', 2, 'DisplayName', '1/(1-\beta_1^t)');
            hold on;
            plot(iterations, bias_correction_2, 'Color', californiaGold, 'LineWidth', 2, 'DisplayName', '\sqrt{1-\beta_2^t}');
            xlabel('Iteration');
            ylabel('Correction Factor');
            title('Bias Correction Factors');
            legend();
            grid on;
            grid minor;
            sgtitle(p.Results.Title, 'FontSize', 14, 'FontWeight', 'bold');
        end
    end
    methods (Access = private)
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