function optimization_demo()
    % OPTIMIZATION_DEMO - Comprehensive demonstration of optimization algorithms
    %
    % BERKELEY SCICOMP - OPTIMIZATION TOOLBOX DEMO
    % ==========================================
    %
    % This script demonstrates various optimization algorithms implemented
    % in the Berkeley SciComp Optimization toolbox.
    %
    % Author: Berkeley SciComp Team
    % Date: 2024
    fprintf('BERKELEY SCICOMP - OPTIMIZATION TOOLBOX DEMO\n');
    fprintf('==========================================\n\n');
    % Berkeley colors
    berkeley_blue = [0 50 98]/255;
    california_gold = [253 181 21]/255;
    %% Test Problem 1: Rosenbrock Function (2D)
    fprintf('Test Problem 1: Rosenbrock Function (2D)\n');
    fprintf('========================================\n');
    fprintf('f(x,y) = 100*(y - x^2)^2 + (1 - x)^2\n');
    fprintf('Global minimum: f(1,1) = 0\n\n');
    % Define Rosenbrock function
    rosenbrock = @(x) 100*(x(2) - x(1)^2)^2 + (1 - x(1))^2;
    rosenbrock_grad = @(x) [-400*x(1)*(x(2) - x(1)^2) - 2*(1 - x(1)); 200*(x(2) - x(1)^2)];
    x0 = [-1.2; 1.0];  % Standard starting point
    % Test different algorithms
    algorithms = {'GradientDescent', 'NewtonMethod', 'BFGS'};
    results = cell(length(algorithms), 1);
    for i = 1:length(algorithms)
        fprintf('Testing %s...\n', algorithms{i});
        switch algorithms{i}
            case 'GradientDescent'
                optimizer = optimization.GradientDescent('MaxIterations', 5000, 'Tolerance', 1e-6, 'LineSearch', 'backtrack');
                results{i} = optimizer.minimize(rosenbrock, x0, rosenbrock_grad);
            case 'NewtonMethod'
                optimizer = optimization.NewtonMethod('MaxIterations', 100, 'Tolerance', 1e-6);
                results{i} = optimizer.minimize(rosenbrock, x0, rosenbrock_grad);
            case 'BFGS'
                optimizer = optimization.BFGS('MaxIterations', 1000, 'Tolerance', 1e-6);
                results{i} = optimizer.minimize(rosenbrock, x0, rosenbrock_grad);
        end
        fprintf('  Solution: x = [%.6f, %.6f]\n', results{i}.x(1), results{i}.x(2));
        fprintf('  Function value: f = %.8f\n', results{i}.fun);
        fprintf('  Iterations: %d\n', results{i}.nit);
        fprintf('  Success: %s\n\n', char(string(results{i}.success)));
    end
    %% Test Problem 2: Global Optimization (Ackley Function)
    fprintf('Test Problem 2: Ackley Function (Global Optimization)\n');
    fprintf('===================================================\n');
    fprintf('f(x,y) = -20*exp(-0.2*sqrt(0.5*(x^2 + y^2))) - exp(0.5*(cos(2*pi*x) + cos(2*pi*y))) + e + 20\n');
    fprintf('Global minimum: f(0,0) = 0\n\n');
    % Define Ackley function
    ackley = @(x) -20*exp(-0.2*sqrt(0.5*(x(1)^2 + x(2)^2))) - exp(0.5*(cos(2*pi*x(1)) + cos(2*pi*x(2)))) + exp(1) + 20;
    bounds = {[-5, 5], [-5, 5]};
    % Test Simulated Annealing
    fprintf('Testing Simulated Annealing...\n');
    sa = optimization.SimulatedAnnealing('MaxIterations', 10000, 'InitialTemperature', 100, 'Verbose', false);
    result_sa = sa.minimize(ackley, bounds);
    fprintf('  Solution: x = [%.6f, %.6f]\n', result_sa.x(1), result_sa.x(2));
    fprintf('  Function value: f = %.8f\n', result_sa.fun);
    fprintf('  Iterations: %d\n', result_sa.nit);
    fprintf('  Final temperature: %.2e\n\n', result_sa.final_temperature);
    % Test Genetic Algorithm
    fprintf('Testing Genetic Algorithm...\n');
    ga = optimization.GeneticAlgorithm('PopulationSize', 50, 'MaxGenerations', 200, 'Verbose', false);
    result_ga = ga.minimize(ackley, bounds);
    fprintf('  Solution: x = [%.6f, %.6f]\n', result_ga.x(1), result_ga.x(2));
    fprintf('  Function value: f = %.8f\n', result_ga.fun);
    fprintf('  Generations: %d\n\n', result_ga.nit);
    %% Test Problem 3: Linear Programming
    fprintf('Test Problem 3: Linear Programming\n');
    fprintf('=================================\n');
    fprintf('maximize    3*x + 2*y\n');
    fprintf('subject to  x + y <= 4\n');
    fprintf('            2*x + y <= 6\n');
    fprintf('            x, y >= 0\n');
    fprintf('Optimal solution: x = 2, y = 2, f = 10\n\n');
    % Define LP problem
    c = [3; 2];  % Objective coefficients
    A = [1, 1; 2, 1];  % Constraint matrix
    b = [4; 6];  % RHS vector
    % Solve using LinearProgramming class
    fprintf('Testing Linear Programming (Simplex Method)...\n');
    lp = optimization.LinearProgramming('Verbose', false);
    result_lp = lp.solve(c, A, b, 'max');
    fprintf('  Solution: x = [%.6f, %.6f]\n', result_lp.x(1), result_lp.x(2));
    fprintf('  Objective value: f = %.6f\n', result_lp.fun);
    fprintf('  Iterations: %d\n', result_lp.nit);
    fprintf('  Success: %s\n\n', char(string(result_lp.success)));
    %% Visualization
    fprintf('Creating visualization plots...\n');
    % Plot Rosenbrock function contours with optimization paths
    figure('Position', [100, 100, 1200, 800]);
    % Rosenbrock contour plot
    subplot(2, 3, 1);
    [X, Y] = meshgrid(-2:0.1:2, -1:0.1:3);
    Z = 100*(Y - X.^2).^2 + (1 - X).^2;
    contour(X, Y, log10(Z + 1), 20);
    hold on;
    % Plot optimization paths
    colors = {'r', 'g', 'b'};
    for i = 1:length(algorithms)
        if isfield(results{i}, 'history') && isfield(results{i}.history, 'x')
            path = results{i}.history.x;
            plot(path(1, :), path(2, :), [colors{i} 'o-'], 'LineWidth', 2, 'MarkerSize', 4);
        end
    end
    plot(1, 1, 'k*', 'MarkerSize', 15, 'LineWidth', 3);  % Global minimum
    xlabel('x_1');
    ylabel('x_2');
    title('Rosenbrock Function Optimization Paths', 'Color', berkeley_blue);
    legend([algorithms, 'Global Min'], 'Location', 'best');
    grid on;
    % Convergence plots for different algorithms
    for i = 1:length(algorithms)
        subplot(2, 3, i + 1);
        if isfield(results{i}, 'history') && isfield(results{i}.history, 'f')
            semilogy(0:length(results{i}.history.f)-1, results{i}.history.f, 'b-', 'LineWidth', 2);
            xlabel('Iteration');
            ylabel('Function Value');
            title([algorithms{i} ' Convergence'], 'Color', berkeley_blue);
            grid on;
        end
    end
    % Simulated Annealing visualization
    subplot(2, 3, 5);
    if isfield(result_sa, 'history')
        semilogy(0:length(result_sa.history.best_f)-1, result_sa.history.best_f, 'b-', 'LineWidth', 2);
        hold on;
        plot(0:length(result_sa.history.f)-1, result_sa.history.f, 'g-', 'Alpha', 0.3);
        xlabel('Iteration');
        ylabel('Function Value');
        title('Simulated Annealing Convergence', 'Color', berkeley_blue);
        legend('Best', 'Current', 'Location', 'best');
        grid on;
    end
    % Genetic Algorithm visualization
    subplot(2, 3, 6);
    if isfield(result_ga, 'history')
        generations = 0:length(result_ga.history.best_fitness)-1;
        semilogy(generations, result_ga.history.best_fitness, 'b-', 'LineWidth', 2);
        hold on;
        semilogy(generations, result_ga.history.mean_fitness, 'g--', 'LineWidth', 1.5);
        xlabel('Generation');
        ylabel('Fitness');
        title('Genetic Algorithm Convergence', 'Color', berkeley_blue);
        legend('Best', 'Mean', 'Location', 'best');
        grid on;
    end
    % Adjust overall figure properties
    sgtitle('Berkeley SciComp Optimization Toolbox Results', 'FontSize', 16, 'Color', berkeley_blue, 'FontWeight', 'bold');
    %% Performance Comparison
    fprintf('\nPerformance Comparison Summary\n');
    fprintf('============================\n');
    fprintf('Algorithm              Solution Error    Iterations    Success\n');
    fprintf('----------------------------------------------------------------\n');
    true_solution = [1; 1];
    for i = 1:length(algorithms)
        error = norm(results{i}.x - true_solution);
        fprintf('%-20s %12.6e %12d %12s\n', algorithms{i}, error, results{i}.nit, char(string(results{i}.success)));
    end
    fprintf('\nGlobal Optimization Results\n');
    fprintf('===========================\n');
    fprintf('Algorithm              Function Value    Solution Error\n');
    fprintf('-------------------------------------------------------\n');
    global_min = [0; 0];
    sa_error = norm(result_sa.x - global_min);
    ga_error = norm(result_ga.x - global_min);
    fprintf('%-20s %12.6f %12.6f\n', 'Simulated Annealing', result_sa.fun, sa_error);
    fprintf('%-20s %12.6f %12.6f\n', 'Genetic Algorithm', result_ga.fun, ga_error);
    fprintf('\nLinear Programming Result\n');
    fprintf('========================\n');
    fprintf('Optimal value: %.6f (expected: 10.0)\n', result_lp.fun);
    fprintf('Solution: [%.6f, %.6f] (expected: [2.0, 2.0])\n', result_lp.x(1), result_lp.x(2));
    fprintf('\n=== DEMO COMPLETED ===\n');
    fprintf('All optimization algorithms have been successfully demonstrated.\n');
    fprintf('The Berkeley SciComp Optimization Toolbox provides a comprehensive\n');
    fprintf('suite of algorithms for various optimization problems.\n\n');
end
% Helper function to create test problems
function problems = createTestProblems()
    % Collection of standard test problems for optimization
    problems = struct();
    % Rosenbrock function
    problems.rosenbrock.objective = @(x) 100*(x(2) - x(1)^2)^2 + (1 - x(1))^2;
    problems.rosenbrock.gradient = @(x) [-400*x(1)*(x(2) - x(1)^2) - 2*(1 - x(1)); 200*(x(2) - x(1)^2)];
    problems.rosenbrock.hessian = @(x) [-400*(x(2) - 3*x(1)^2) + 2, -400*x(1); -400*x(1), 200];
    problems.rosenbrock.x0 = [-1.2; 1.0];
    problems.rosenbrock.solution = [1; 1];
    problems.rosenbrock.optimum = 0;
    % Ackley function
    problems.ackley.objective = @(x) -20*exp(-0.2*sqrt(0.5*(x(1)^2 + x(2)^2))) - exp(0.5*(cos(2*pi*x(1)) + cos(2*pi*x(2)))) + exp(1) + 20;
    problems.ackley.bounds = {[-5, 5], [-5, 5]};
    problems.ackley.solution = [0; 0];
    problems.ackley.optimum = 0;
    % Sphere function
    problems.sphere.objective = @(x) sum(x.^2);
    problems.sphere.gradient = @(x) 2*x;
    problems.sphere.hessian = @(x) 2*eye(length(x));
    problems.sphere.bounds = {[-5, 5], [-5, 5]};
    problems.sphere.solution = [0; 0];
    problems.sphere.optimum = 0;
    % Himmelblau's function
    problems.himmelblau.objective = @(x) (x(1)^2 + x(2) - 11)^2 + (x(1) + x(2)^2 - 7)^2;
    problems.himmelblau.bounds = {[-5, 5], [-5, 5]};
    problems.himmelblau.solutions = [3, 2; -2.805118, 3.131312; -3.779310, -3.283186; 3.584428, -1.848126];
    problems.himmelblau.optimum = 0;
end