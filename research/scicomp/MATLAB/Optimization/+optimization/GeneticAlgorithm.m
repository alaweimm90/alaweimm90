classdef GeneticAlgorithm < handle
    % GeneticAlgorithm - Genetic algorithm for global optimization
    %
    % BERKELEY SCICOMP - OPTIMIZATION TOOLBOX
    % =====================================
    %
    % This class implements a genetic algorithm for global optimization
    % with various selection, crossover, and mutation operators.
    %
    % Author: Berkeley SciComp Team
    % Date: 2024
    properties
        PopulationSize = 50         % Population size
        MaxGenerations = 1000       % Maximum number of generations
        CrossoverRate = 0.8         % Crossover probability
        MutationRate = 0.1          % Mutation probability
        SelectionMethod = 'tournament'  % Selection method
        CrossoverMethod = 'uniform'     % Crossover method
        MutationMethod = 'gaussian'     % Mutation method
        ElitismRate = 0.1          % Fraction of elite individuals to preserve
        Verbose = false            % Display progress
        % Berkeley colors
        BerkeleyBlue = [0 50 98]/255;
        CaliforniaGold = [253 181 21]/255;
    end
    properties (Access = private)
        History = struct()         % Optimization history
    end
    methods
        function obj = GeneticAlgorithm(varargin)
            % Constructor for GeneticAlgorithm
            %
            % Usage:
            %   ga = optimization.GeneticAlgorithm()
            %   ga = optimization.GeneticAlgorithm('PopulationSize', 100, 'MaxGenerations', 2000)
            % Parse input arguments
            p = inputParser;
            addParameter(p, 'PopulationSize', 50, @isnumeric);
            addParameter(p, 'MaxGenerations', 1000, @isnumeric);
            addParameter(p, 'CrossoverRate', 0.8, @isnumeric);
            addParameter(p, 'MutationRate', 0.1, @isnumeric);
            addParameter(p, 'SelectionMethod', 'tournament', @ischar);
            addParameter(p, 'CrossoverMethod', 'uniform', @ischar);
            addParameter(p, 'MutationMethod', 'gaussian', @ischar);
            addParameter(p, 'ElitismRate', 0.1, @isnumeric);
            addParameter(p, 'Verbose', false, @islogical);
            parse(p, varargin{:});
            obj.PopulationSize = p.Results.PopulationSize;
            obj.MaxGenerations = p.Results.MaxGenerations;
            obj.CrossoverRate = p.Results.CrossoverRate;
            obj.MutationRate = p.Results.MutationRate;
            obj.SelectionMethod = p.Results.SelectionMethod;
            obj.CrossoverMethod = p.Results.CrossoverMethod;
            obj.MutationMethod = p.Results.MutationMethod;
            obj.ElitismRate = p.Results.ElitismRate;
            obj.Verbose = p.Results.Verbose;
        end
        function result = minimize(obj, objective, bounds)
            % Minimize objective function using genetic algorithm
            %
            % Inputs:
            %   objective - Function handle for objective function
            %   bounds    - Cell array of bounds {[lb1 ub1], [lb2 ub2], ...}
            %
            % Outputs:
            %   result - Structure with optimization results
            n = length(bounds);
            % Initialize history
            obj.History.best_fitness = zeros(obj.MaxGenerations + 1, 1);
            obj.History.mean_fitness = zeros(obj.MaxGenerations + 1, 1);
            obj.History.worst_fitness = zeros(obj.MaxGenerations + 1, 1);
            obj.History.diversity = zeros(obj.MaxGenerations + 1, 1);
            % Initialize population
            population = obj.initializePopulation(bounds);
            fitness = obj.evaluatePopulation(objective, population);
            % Store initial statistics
            [best_fitness, best_idx] = min(fitness);
            best_individual = population(best_idx, :);
            obj.History.best_fitness(1) = best_fitness;
            obj.History.mean_fitness(1) = mean(fitness);
            obj.History.worst_fitness(1) = max(fitness);
            obj.History.diversity(1) = obj.calculateDiversity(population);
            % Main evolution loop
            for generation = 1:obj.MaxGenerations
                % Selection
                parents = obj.selection(population, fitness);
                % Crossover and mutation
                offspring = obj.reproduction(parents, bounds);
                % Evaluate offspring
                offspring_fitness = obj.evaluatePopulation(objective, offspring);
                % Combine populations
                combined_population = [population; offspring];
                combined_fitness = [fitness; offspring_fitness];
                % Environmental selection (survival of the fittest)
                [population, fitness] = obj.environmentalSelection(combined_population, combined_fitness);
                % Update best solution
                [current_best_fitness, best_idx] = min(fitness);
                if current_best_fitness < best_fitness
                    best_fitness = current_best_fitness;
                    best_individual = population(best_idx, :);
                end
                % Store statistics
                obj.History.best_fitness(generation + 1) = best_fitness;
                obj.History.mean_fitness(generation + 1) = mean(fitness);
                obj.History.worst_fitness(generation + 1) = max(fitness);
                obj.History.diversity(generation + 1) = obj.calculateDiversity(population);
                % Display progress
                if obj.Verbose && mod(generation, 50) == 0
                    fprintf('Gen %4d: best = %12.6e, mean = %12.6e, diversity = %12.6e\n', ...
                            generation, best_fitness, mean(fitness), obj.History.diversity(generation + 1));
                end
                % Check convergence (optional)
                if generation > 10
                    improvement = obj.History.best_fitness(generation - 9) - obj.History.best_fitness(generation + 1);
                    if improvement < 1e-12
                        if obj.Verbose
                            fprintf('Converged at generation %d\n', generation);
                        end
                        break;
                    end
                end
            end
            % Trim history
            actual_generations = min(generation, obj.MaxGenerations);
            obj.History.best_fitness = obj.History.best_fitness(1:actual_generations + 1);
            obj.History.mean_fitness = obj.History.mean_fitness(1:actual_generations + 1);
            obj.History.worst_fitness = obj.History.worst_fitness(1:actual_generations + 1);
            obj.History.diversity = obj.History.diversity(1:actual_generations + 1);
            % Create result structure
            result = struct();
            result.x = best_individual';
            result.fun = best_fitness;
            result.success = true;
            result.nit = actual_generations;
            result.nfev = (actual_generations + 1) * obj.PopulationSize;
            result.message = 'Genetic algorithm completed';
            result.final_population = population;
            result.final_fitness = fitness;
            result.history = obj.History;
        end
        function population = initializePopulation(obj, bounds)
            % Initialize random population within bounds
            n = length(bounds);
            population = zeros(obj.PopulationSize, n);
            for i = 1:obj.PopulationSize
                for j = 1:n
                    population(i, j) = bounds{j}(1) + rand() * (bounds{j}(2) - bounds{j}(1));
                end
            end
        end
        function fitness = evaluatePopulation(obj, objective, population)
            % Evaluate fitness of entire population
            fitness = zeros(size(population, 1), 1);
            for i = 1:size(population, 1)
                fitness(i) = objective(population(i, :)');
            end
        end
        function parents = selection(obj, population, fitness)
            % Select parents for reproduction
            n_parents = obj.PopulationSize;
            parents = zeros(n_parents, size(population, 2));
            switch lower(obj.SelectionMethod)
                case 'tournament'
                    tournament_size = 3;
                    for i = 1:n_parents
                        candidates = randi(size(population, 1), tournament_size, 1);
                        [~, winner_idx] = min(fitness(candidates));
                        parents(i, :) = population(candidates(winner_idx), :);
                    end
                case 'roulette'
                    % Convert minimization to maximization for roulette wheel
                    if all(fitness >= 0)
                        weights = max(fitness) - fitness + 1e-10;
                    else
                        weights = fitness - min(fitness) + 1e-10;
                        weights = max(weights) - weights + 1e-10;
                    end
                    weights = weights / sum(weights);
                    for i = 1:n_parents
                        idx = obj.rouletteWheelSelection(weights);
                        parents(i, :) = population(idx, :);
                    end
                case 'rank'
                    [~, sorted_indices] = sort(fitness);
                    ranks = 1:length(fitness);
                    weights = ranks / sum(ranks);
                    for i = 1:n_parents
                        idx = obj.rouletteWheelSelection(weights);
                        parents(i, :) = population(sorted_indices(idx), :);
                    end
                otherwise
                    % Default to tournament selection
                    parents = obj.selection(population, fitness);
            end
        end
        function idx = rouletteWheelSelection(obj, weights)
            % Roulette wheel selection
            r = rand();
            cumulative = cumsum(weights);
            idx = find(cumulative >= r, 1, 'first');
            if isempty(idx)
                idx = length(weights);
            end
        end
        function offspring = reproduction(obj, parents, bounds)
            % Create offspring through crossover and mutation
            n_offspring = size(parents, 1);
            offspring = zeros(n_offspring, size(parents, 2));
            for i = 1:2:n_offspring-1
                % Select two parents
                parent1 = parents(i, :);
                parent2 = parents(min(i+1, n_offspring), :);
                % Crossover
                if rand() < obj.CrossoverRate
                    [child1, child2] = obj.crossover(parent1, parent2);
                else
                    child1 = parent1;
                    child2 = parent2;
                end
                % Mutation
                child1 = obj.mutation(child1, bounds);
                child2 = obj.mutation(child2, bounds);
                offspring(i, :) = child1;
                if i + 1 <= n_offspring
                    offspring(i + 1, :) = child2;
                end
            end
        end
        function [child1, child2] = crossover(obj, parent1, parent2)
            % Perform crossover between two parents
            switch lower(obj.CrossoverMethod)
                case 'uniform'
                    mask = rand(size(parent1)) < 0.5;
                    child1 = parent1;
                    child2 = parent2;
                    child1(mask) = parent2(mask);
                    child2(mask) = parent1(mask);
                case 'single_point'
                    crossover_point = randi(length(parent1) - 1);
                    child1 = [parent1(1:crossover_point), parent2(crossover_point+1:end)];
                    child2 = [parent2(1:crossover_point), parent1(crossover_point+1:end)];
                case 'two_point'
                    points = sort(randperm(length(parent1) - 1, 2));
                    child1 = parent1;
                    child2 = parent2;
                    child1(points(1)+1:points(2)) = parent2(points(1)+1:points(2));
                    child2(points(1)+1:points(2)) = parent1(points(1)+1:points(2));
                case 'arithmetic'
                    alpha = rand();
                    child1 = alpha * parent1 + (1 - alpha) * parent2;
                    child2 = (1 - alpha) * parent1 + alpha * parent2;
                otherwise
                    % Default to uniform crossover
                    [child1, child2] = obj.crossover(parent1, parent2);
            end
        end
        function individual = mutation(obj, individual, bounds)
            % Apply mutation to an individual
            for i = 1:length(individual)
                if rand() < obj.MutationRate
                    switch lower(obj.MutationMethod)
                        case 'gaussian'
                            sigma = 0.1 * (bounds{i}(2) - bounds{i}(1));
                            individual(i) = individual(i) + sigma * randn();
                        case 'uniform'
                            individual(i) = bounds{i}(1) + rand() * (bounds{i}(2) - bounds{i}(1));
                        case 'polynomial'
                            eta = 20;  % Distribution index
                            delta = bounds{i}(2) - bounds{i}(1);
                            u = rand();
                            if u < 0.5
                                delta_q = (2*u)^(1/(eta+1)) - 1;
                            else
                                delta_q = 1 - (2*(1-u))^(1/(eta+1));
                            end
                            individual(i) = individual(i) + delta_q * delta;
                        otherwise
                            % Default to gaussian mutation
                            individual = obj.mutation(individual, bounds);
                    end
                    % Apply bounds
                    individual(i) = max(bounds{i}(1), min(bounds{i}(2), individual(i)));
                end
            end
        end
        function [new_population, new_fitness] = environmentalSelection(obj, population, fitness)
            % Select survivors for next generation
            [sorted_fitness, sorted_indices] = sort(fitness);
            % Elitism: preserve best individuals
            n_elite = round(obj.ElitismRate * obj.PopulationSize);
            elite_indices = sorted_indices(1:n_elite);
            % Select remaining individuals
            remaining_indices = sorted_indices(n_elite+1:obj.PopulationSize);
            % Combine elite and selected individuals
            selected_indices = [elite_indices; remaining_indices];
            new_population = population(selected_indices, :);
            new_fitness = fitness(selected_indices);
        end
        function diversity = calculateDiversity(obj, population)
            % Calculate population diversity
            n = size(population, 1);
            distances = zeros(n, n);
            for i = 1:n
                for j = i+1:n
                    distances(i, j) = norm(population(i, :) - population(j, :));
                    distances(j, i) = distances(i, j);
                end
            end
            diversity = mean(distances(:));
        end
        function plotConvergence(obj)
            % Plot convergence history
            if isempty(obj.History)
                error('No optimization history available. Run minimize() first.');
            end
            figure;
            % Fitness evolution
            subplot(2, 2, 1);
            generations = 0:length(obj.History.best_fitness)-1;
            semilogy(generations, obj.History.best_fitness, 'b-', 'LineWidth', 2);
            hold on;
            semilogy(generations, obj.History.mean_fitness, 'g--', 'LineWidth', 1.5);
            semilogy(generations, obj.History.worst_fitness, 'r:', 'LineWidth', 1.5);
            xlabel('Generation');
            ylabel('Fitness');
            title('Genetic Algorithm Convergence', 'Color', obj.BerkeleyBlue);
            legend('Best', 'Mean', 'Worst', 'Location', 'best');
            grid on;
            % Diversity
            subplot(2, 2, 2);
            plot(generations, obj.History.diversity, 'k-', 'LineWidth', 2);
            xlabel('Generation');
            ylabel('Population Diversity');
            title('Population Diversity', 'Color', obj.BerkeleyBlue);
            grid on;
            % Fitness distribution (final generation)
            subplot(2, 2, 3);
            histogram(obj.History.final_fitness, 20, 'FaceColor', obj.CaliforniaGold, 'EdgeColor', 'black');
            xlabel('Fitness');
            ylabel('Frequency');
            title('Final Population Fitness Distribution', 'Color', obj.BerkeleyBlue);
            grid on;
            % Convergence rate
            subplot(2, 2, 4);
            improvement = -diff(obj.History.best_fitness);
            improvement(improvement <= 0) = 1e-15;  % Handle non-improving generations
            semilogy(1:length(improvement), improvement, 'b-', 'LineWidth', 2);
            xlabel('Generation');
            ylabel('Fitness Improvement');
            title('Convergence Rate', 'Color', obj.BerkeleyBlue);
            grid on;
            % Set figure properties
            set(gcf, 'Position', [100, 100, 1000, 800]);
        end
    end
end