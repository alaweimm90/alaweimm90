classdef SimulatedAnnealing < handle
    % SimulatedAnnealing - Simulated annealing global optimization
    %
    % BERKELEY SCICOMP - OPTIMIZATION TOOLBOX
    % =====================================
    %
    % This class implements simulated annealing for global optimization
    % with various cooling schedules and acceptance criteria.
    %
    % Author: Berkeley SciComp Team
    % Date: 2024
    properties
        MaxIterations = 10000       % Maximum number of iterations
        InitialTemperature = 100    % Initial temperature
        FinalTemperature = 1e-8     % Final temperature
        CoolingSchedule = 'exponential'  % Cooling schedule
        StepSize = 1.0             % Step size for perturbations
        Verbose = false            % Display progress
        % Berkeley colors
        BerkeleyBlue = [0 50 98]/255;
        CaliforniaGold = [253 181 21]/255;
    end
    properties (Access = private)
        History = struct()         % Optimization history
    end
    methods
        function obj = SimulatedAnnealing(varargin)
            % Constructor for SimulatedAnnealing
            %
            % Usage:
            %   sa = optimization.SimulatedAnnealing()
            %   sa = optimization.SimulatedAnnealing('MaxIterations', 20000, 'InitialTemperature', 200)
            % Parse input arguments
            p = inputParser;
            addParameter(p, 'MaxIterations', 10000, @isnumeric);
            addParameter(p, 'InitialTemperature', 100, @isnumeric);
            addParameter(p, 'FinalTemperature', 1e-8, @isnumeric);
            addParameter(p, 'CoolingSchedule', 'exponential', @ischar);
            addParameter(p, 'StepSize', 1.0, @isnumeric);
            addParameter(p, 'Verbose', false, @islogical);
            parse(p, varargin{:});
            obj.MaxIterations = p.Results.MaxIterations;
            obj.InitialTemperature = p.Results.InitialTemperature;
            obj.FinalTemperature = p.Results.FinalTemperature;
            obj.CoolingSchedule = p.Results.CoolingSchedule;
            obj.StepSize = p.Results.StepSize;
            obj.Verbose = p.Results.Verbose;
        end
        function result = minimize(obj, objective, bounds, x0)
            % Minimize objective function using simulated annealing
            %
            % Inputs:
            %   objective - Function handle for objective function
            %   bounds    - Cell array of bounds {[lb1 ub1], [lb2 ub2], ...}
            %   x0        - Initial guess (optional)
            %
            % Outputs:
            %   result - Structure with optimization results
            n = length(bounds);
            % Initialize starting point
            if nargin < 4 || isempty(x0)
                x = zeros(n, 1);
                for i = 1:n
                    x(i) = bounds{i}(1) + rand() * (bounds{i}(2) - bounds{i}(1));
                end
            else
                x = x0(:);
            end
            % Initialize history
            obj.History.x = zeros(n, obj.MaxIterations + 1);
            obj.History.f = zeros(obj.MaxIterations + 1, 1);
            obj.History.temperature = zeros(obj.MaxIterations + 1, 1);
            obj.History.accepted = false(obj.MaxIterations, 1);
            obj.History.best_x = zeros(n, obj.MaxIterations + 1);
            obj.History.best_f = zeros(obj.MaxIterations + 1, 1);
            % Initial evaluation
            f = objective(x);
            best_x = x;
            best_f = f;
            % Store initial state
            obj.History.x(:, 1) = x;
            obj.History.f(1) = f;
            obj.History.temperature(1) = obj.InitialTemperature;
            obj.History.best_x(:, 1) = best_x;
            obj.History.best_f(1) = best_f;
            temperature = obj.InitialTemperature;
            % Main optimization loop
            for iter = 1:obj.MaxIterations
                % Generate new candidate solution
                x_new = obj.generateCandidate(x, bounds);
                f_new = objective(x_new);
                % Acceptance criterion
                if obj.acceptCandidate(f, f_new, temperature)
                    x = x_new;
                    f = f_new;
                    obj.History.accepted(iter) = true;
                    % Update best solution
                    if f_new < best_f
                        best_x = x_new;
                        best_f = f_new;
                    end
                end
                % Update temperature
                temperature = obj.updateTemperature(iter);
                % Store history
                obj.History.x(:, iter + 1) = x;
                obj.History.f(iter + 1) = f;
                obj.History.temperature(iter + 1) = temperature;
                obj.History.best_x(:, iter + 1) = best_x;
                obj.History.best_f(iter + 1) = best_f;
                % Display progress
                if obj.Verbose && mod(iter, 1000) == 0
                    acceptance_rate = sum(obj.History.accepted(max(1, iter-999):iter)) / 1000;
                    fprintf('Iter %5d: f = %12.6e, best_f = %12.6e, T = %12.6e, acc_rate = %.3f\n', ...
                            iter, f, best_f, temperature, acceptance_rate);
                end
                % Early termination if temperature is too low
                if temperature < obj.FinalTemperature
                    if obj.Verbose
                        fprintf('Temperature reached final value at iteration %d\n', iter);
                    end
                    break;
                end
            end
            % Trim history
            actual_iters = min(iter, obj.MaxIterations);
            obj.History.x = obj.History.x(:, 1:actual_iters + 1);
            obj.History.f = obj.History.f(1:actual_iters + 1);
            obj.History.temperature = obj.History.temperature(1:actual_iters + 1);
            obj.History.accepted = obj.History.accepted(1:actual_iters);
            obj.History.best_x = obj.History.best_x(:, 1:actual_iters + 1);
            obj.History.best_f = obj.History.best_f(1:actual_iters + 1);
            % Create result structure
            result = struct();
            result.x = best_x;
            result.fun = best_f;
            result.success = true;
            result.nit = actual_iters;
            result.nfev = actual_iters + 1;
            result.message = 'Simulated annealing completed';
            result.final_temperature = temperature;
            result.acceptance_rate = sum(obj.History.accepted) / length(obj.History.accepted);
            result.history = obj.History;
        end
        function x_new = generateCandidate(obj, x, bounds)
            % Generate new candidate solution
            n = length(x);
            x_new = x + obj.StepSize * randn(n, 1);
            % Apply bounds
            for i = 1:n
                if x_new(i) < bounds{i}(1)
                    x_new(i) = bounds{i}(1) + rand() * (bounds{i}(2) - bounds{i}(1));
                elseif x_new(i) > bounds{i}(2)
                    x_new(i) = bounds{i}(1) + rand() * (bounds{i}(2) - bounds{i}(1));
                end
            end
        end
        function accept = acceptCandidate(obj, f_current, f_new, temperature)
            % Determine whether to accept new candidate
            if f_new < f_current
                % Always accept better solutions
                accept = true;
            else
                % Accept worse solutions with probability exp(-ΔE/T)
                delta_f = f_new - f_current;
                probability = exp(-delta_f / temperature);
                accept = rand() < probability;
            end
        end
        function temperature = updateTemperature(obj, iteration)
            % Update temperature according to cooling schedule
            switch lower(obj.CoolingSchedule)
                case 'linear'
                    % Linear cooling
                    alpha = iteration / obj.MaxIterations;
                    temperature = obj.InitialTemperature * (1 - alpha);
                case 'exponential'
                    % Exponential cooling
                    alpha = log(obj.FinalTemperature / obj.InitialTemperature) / obj.MaxIterations;
                    temperature = obj.InitialTemperature * exp(alpha * iteration);
                case 'logarithmic'
                    % Logarithmic cooling
                    temperature = obj.InitialTemperature / log(1 + iteration);
                case 'power'
                    % Power law cooling
                    beta = 0.95;  % Cooling parameter
                    temperature = obj.InitialTemperature * (beta ^ iteration);
                case 'adaptive'
                    % Adaptive cooling based on acceptance rate
                    if iteration > 100
                        recent_acceptance = sum(obj.History.accepted(max(1, iteration-99):iteration-1)) / 100;
                        if recent_acceptance > 0.8
                            alpha = 0.99;  % Cool faster if accepting too much
                        elseif recent_acceptance < 0.2
                            alpha = 0.999; % Cool slower if accepting too little
                        else
                            alpha = 0.995; % Standard cooling
                        end
                        temperature = obj.History.temperature(iteration) * alpha;
                    else
                        temperature = obj.InitialTemperature;
                    end
                otherwise
                    % Default to exponential
                    alpha = log(obj.FinalTemperature / obj.InitialTemperature) / obj.MaxIterations;
                    temperature = obj.InitialTemperature * exp(alpha * iteration);
            end
            % Ensure temperature doesn't go below final temperature
            temperature = max(temperature, obj.FinalTemperature);
        end
        function plotConvergence(obj)
            % Plot convergence history
            if isempty(obj.History)
                error('No optimization history available. Run minimize() first.');
            end
            figure;
            % Best objective function
            subplot(2, 2, 1);
            semilogy(0:length(obj.History.best_f)-1, obj.History.best_f, 'b-', 'LineWidth', 2);
            hold on;
            plot(0:length(obj.History.f)-1, obj.History.f, 'g-', 'Alpha', 0.3);
            xlabel('Iteration');
            ylabel('Objective Function');
            title('Simulated Annealing Convergence', 'Color', obj.BerkeleyBlue);
            legend('Best', 'Current', 'Location', 'best');
            grid on;
            % Temperature
            subplot(2, 2, 2);
            loglog(0:length(obj.History.temperature)-1, obj.History.temperature, 'r-', 'LineWidth', 2);
            xlabel('Iteration');
            ylabel('Temperature');
            title('Cooling Schedule', 'Color', obj.BerkeleyBlue);
            grid on;
            % Acceptance rate (rolling average)
            subplot(2, 2, 3);
            window_size = min(1000, length(obj.History.accepted));
            acceptance_rate = zeros(length(obj.History.accepted) - window_size + 1, 1);
            for i = 1:length(acceptance_rate)
                acceptance_rate(i) = sum(obj.History.accepted(i:i+window_size-1)) / window_size;
            end
            plot(window_size:length(obj.History.accepted), acceptance_rate, 'g-', 'LineWidth', 2);
            xlabel('Iteration');
            ylabel('Acceptance Rate');
            title(sprintf('Acceptance Rate (window = %d)', window_size), 'Color', obj.BerkeleyBlue);
            grid on;
            ylim([0, 1]);
            % Variable trajectory (for 2D problems)
            if size(obj.History.x, 1) == 2
                subplot(2, 2, 4);
                plot(obj.History.x(1, :), obj.History.x(2, :), 'k-', 'Alpha', 0.3);
                hold on;
                plot(obj.History.best_x(1, :), obj.History.best_x(2, :), 'b-', 'LineWidth', 2);
                plot(obj.History.best_x(1, end), obj.History.best_x(2, end), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
                xlabel('x_1');
                ylabel('x_2');
                title('Optimization Path', 'Color', obj.BerkeleyBlue);
                legend('Current', 'Best', 'Final', 'Location', 'best');
                grid on;
            end
            % Set figure properties
            set(gcf, 'Position', [100, 100, 1000, 800]);
        end
    end
end