classdef LinearProgramming < handle
    % LinearProgramming - Simplex method for linear programming
    %
    % BERKELEY SCICOMP - OPTIMIZATION TOOLBOX
    % =====================================
    %
    % This class implements the simplex method for solving linear programming
    % problems with two-phase method for finding initial feasible solutions.
    %
    % Author: Berkeley SciComp Team
    % Date: 2024
    properties
        MaxIterations = 1000        % Maximum number of iterations
        Tolerance = 1e-8            % Numerical tolerance
        PivotingRule = 'bland'      % Pivoting rule ('bland', 'dantzig')
        Verbose = false            % Display progress
        % Berkeley colors
        BerkeleyBlue = [0 50 98]/255;
        CaliforniaGold = [253 181 21]/255;
    end
    properties (Access = private)
        History = struct()         % Optimization history
    end
    methods
        function obj = LinearProgramming(varargin)
            % Constructor for LinearProgramming
            %
            % Usage:
            %   lp = optimization.LinearProgramming()
            %   lp = optimization.LinearProgramming('MaxIterations', 2000, 'Tolerance', 1e-10)
            % Parse input arguments
            p = inputParser;
            addParameter(p, 'MaxIterations', 1000, @isnumeric);
            addParameter(p, 'Tolerance', 1e-8, @isnumeric);
            addParameter(p, 'PivotingRule', 'bland', @ischar);
            addParameter(p, 'Verbose', false, @islogical);
            parse(p, varargin{:});
            obj.MaxIterations = p.Results.MaxIterations;
            obj.Tolerance = p.Results.Tolerance;
            obj.PivotingRule = p.Results.PivotingRule;
            obj.Verbose = p.Results.Verbose;
        end
        function result = solve(obj, c, A, b, sense)
            % Solve linear programming problem
            %
            % Problem formulation:
            %   minimize/maximize  c'*x
            %   subject to         A*x <= b  (inequality constraints)
            %                      x >= 0   (non-negativity)
            %
            % Inputs:
            %   c     - Objective function coefficients (column vector)
            %   A     - Constraint matrix
            %   b     - Right-hand side vector
            %   sense - 'min' for minimization, 'max' for maximization
            %
            % Outputs:
            %   result - Structure with optimization results
            if nargin < 5
                sense = 'min';
            end
            % Convert to column vectors
            c = c(:);
            b = b(:);
            % Convert maximization to minimization
            if strcmpi(sense, 'max')
                c = -c;
            end
            [m, n] = size(A);
            % Convert to standard form by adding slack variables
            % A*x + s = b, x >= 0, s >= 0
            c_std = [c; zeros(m, 1)];
            A_std = [A, eye(m)];
            % Check if problem is in standard form (b >= 0)
            if any(b < 0)
                % Use two-phase method
                result = obj.twoPhaseMethod(c_std, A_std, b);
            else
                % Use single-phase method
                result = obj.singlePhaseMethod(c_std, A_std, b);
            end
            % Convert back to original variables
            if ~isempty(result.x)
                result.x = result.x(1:n);  % Remove slack variables
            end
            % Convert back to maximization if needed
            if strcmpi(sense, 'max')
                result.fun = -result.fun;
            end
            result.sense = sense;
        end
        function result = singlePhaseMethod(obj, c, A, b)
            % Single-phase simplex method
            [m, n] = size(A);
            % Create initial tableau
            % [A | I | b]
            % [c | 0 | 0]
            tableau = zeros(m + 1, n + 1);
            tableau(1:m, 1:n) = A;
            tableau(1:m, end) = b;
            tableau(end, 1:n) = c';
            % Basic variables are the last m variables (slack variables)
            basic_vars = (n-m+1):n;
            % Initialize history
            obj.History.tableau = cell(obj.MaxIterations + 1, 1);
            obj.History.basic_vars = cell(obj.MaxIterations + 1, 1);
            obj.History.objective = zeros(obj.MaxIterations + 1, 1);
            obj.History.tableau{1} = tableau;
            obj.History.basic_vars{1} = basic_vars;
            obj.History.objective(1) = tableau(end, end);
            % Main simplex loop
            iteration = 0;
            while iteration < obj.MaxIterations
                iteration = iteration + 1;
                % Check optimality conditions
                reduced_costs = tableau(end, 1:n);
                entering_var = obj.selectEnteringVariable(reduced_costs);
                if isempty(entering_var)
                    % Optimal solution found
                    x = obj.extractSolution(tableau, basic_vars, n);
                    result = struct();
                    result.x = x;
                    result.fun = tableau(end, end);
                    result.success = true;
                    result.nit = iteration - 1;
                    result.message = 'Optimal solution found';
                    result.tableau = tableau;
                    result.basic_vars = basic_vars;
                    result.history = obj.History;
                    return;
                end
                % Ratio test to find leaving variable
                leaving_var_idx = obj.ratioTest(tableau, entering_var);
                if isempty(leaving_var_idx)
                    % Problem is unbounded
                    result = struct();
                    result.x = [];
                    result.fun = -inf;
                    result.success = false;
                    result.nit = iteration - 1;
                    result.message = 'Problem is unbounded';
                    result.history = obj.History;
                    return;
                end
                % Pivot operation
                obj.pivot(tableau, leaving_var_idx, entering_var);
                basic_vars(leaving_var_idx) = entering_var;
                % Store history
                if iteration <= obj.MaxIterations
                    obj.History.tableau{iteration + 1} = tableau;
                    obj.History.basic_vars{iteration + 1} = basic_vars;
                    obj.History.objective(iteration + 1) = tableau(end, end);
                end
                % Display progress
                if obj.Verbose && mod(iteration, 10) == 0
                    fprintf('Iter %4d: Objective = %12.6e\n', iteration, tableau(end, end));
                end
            end
            % Maximum iterations reached
            x = obj.extractSolution(tableau, basic_vars, n);
            result = struct();
            result.x = x;
            result.fun = tableau(end, end);
            result.success = false;
            result.nit = iteration;
            result.message = 'Maximum iterations reached';
            result.history = obj.History;
        end
        function result = twoPhaseMethod(obj, c, A, b)
            % Two-phase simplex method for problems with b < 0
            [m, n] = size(A);
            % Phase I: Find basic feasible solution
            if obj.Verbose
                fprintf('Starting Phase I...\n');
            end
            % Make b non-negative by multiplying rows with negative b by -1
            negative_rows = b < 0;
            A(negative_rows, :) = -A(negative_rows, :);
            b(negative_rows) = -b(negative_rows);
            % Add artificial variables
            % minimize sum of artificial variables
            c_phase1 = [zeros(n, 1); ones(m, 1)];
            A_phase1 = [A, eye(m)];
            % Initial tableau for Phase I
            tableau_phase1 = zeros(m + 1, n + m + 1);
            tableau_phase1(1:m, 1:n+m) = A_phase1;
            tableau_phase1(1:m, end) = b;
            tableau_phase1(end, n+1:n+m) = ones(1, m);
            % Make tableau compatible with artificial variables in basis
            for i = 1:m
                tableau_phase1(end, :) = tableau_phase1(end, :) - tableau_phase1(i, :);
            end
            basic_vars_phase1 = (n+1):(n+m);
            % Solve Phase I
            result_phase1 = obj.solveTableau(tableau_phase1, basic_vars_phase1, n+m);
            if ~result_phase1.success || result_phase1.fun > obj.Tolerance
                % Problem is infeasible
                result = struct();
                result.x = [];
                result.fun = inf;
                result.success = false;
                result.nit = result_phase1.nit;
                result.message = 'Problem is infeasible';
                return;
            end
            if obj.Verbose
                fprintf('Phase I completed. Artificial variables sum: %e\n', result_phase1.fun);
            end
            % Phase II: Solve original problem
            if obj.Verbose
                fprintf('Starting Phase II...\n');
            end
            % Remove artificial variables from tableau
            tableau_phase2 = result_phase1.tableau(:, 1:n+1);
            % Set up original objective function
            tableau_phase2(end, 1:n) = c';
            tableau_phase2(end, end) = 0;
            % Find basic variables that are not artificial
            basic_vars_phase2 = result_phase1.basic_vars;
            valid_basic = basic_vars_phase2 <= n;
            if sum(valid_basic) < m
                % Some artificial variables are still basic - problem may be degenerate
                % For simplicity, we'll proceed with a warning
                if obj.Verbose
                    warning('Some artificial variables remain basic. Solution may be degenerate.');
                end
            end
            % Solve Phase II
            result = obj.solveTableau(tableau_phase2, basic_vars_phase2, n);
            result.nit = result.nit + result_phase1.nit;
        end
        function result = solveTableau(obj, tableau, basic_vars, n_vars)
            % Solve simplex tableau
            iteration = 0;
            while iteration < obj.MaxIterations
                iteration = iteration + 1;
                % Check optimality
                reduced_costs = tableau(end, 1:n_vars);
                entering_var = obj.selectEnteringVariable(reduced_costs);
                if isempty(entering_var)
                    % Optimal solution found
                    x = obj.extractSolution(tableau, basic_vars, n_vars);
                    result = struct();
                    result.x = x;
                    result.fun = tableau(end, end);
                    result.success = true;
                    result.nit = iteration - 1;
                    result.message = 'Optimal solution found';
                    result.tableau = tableau;
                    result.basic_vars = basic_vars;
                    return;
                end
                % Ratio test
                leaving_var_idx = obj.ratioTest(tableau, entering_var);
                if isempty(leaving_var_idx)
                    % Unbounded
                    result = struct();
                    result.x = [];
                    result.fun = -inf;
                    result.success = false;
                    result.nit = iteration - 1;
                    result.message = 'Problem is unbounded';
                    return;
                end
                % Pivot
                obj.pivot(tableau, leaving_var_idx, entering_var);
                basic_vars(leaving_var_idx) = entering_var;
                if obj.Verbose && mod(iteration, 20) == 0
                    fprintf('Iter %4d: Objective = %12.6e\n', iteration, tableau(end, end));
                end
            end
            % Max iterations
            x = obj.extractSolution(tableau, basic_vars, n_vars);
            result = struct();
            result.x = x;
            result.fun = tableau(end, end);
            result.success = false;
            result.nit = iteration;
            result.message = 'Maximum iterations reached';
            result.tableau = tableau;
            result.basic_vars = basic_vars;
        end
        function entering_var = selectEnteringVariable(obj, reduced_costs)
            % Select entering variable using specified pivoting rule
            switch lower(obj.PivotingRule)
                case 'dantzig'
                    % Most negative reduced cost
                    [min_cost, entering_var] = min(reduced_costs);
                    if min_cost >= -obj.Tolerance
                        entering_var = [];
                    end
                case 'bland'
                    % First negative reduced cost (Bland's rule)
                    entering_var = find(reduced_costs < -obj.Tolerance, 1, 'first');
                otherwise
                    % Default to Bland's rule
                    entering_var = find(reduced_costs < -obj.Tolerance, 1, 'first');
            end
        end
        function leaving_var_idx = ratioTest(obj, tableau, entering_var)
            % Perform ratio test to find leaving variable
            [m, ~] = size(tableau);
            m = m - 1;  % Exclude objective row
            pivot_column = tableau(1:m, entering_var);
            rhs = tableau(1:m, end);
            % Find positive pivot elements
            positive_pivots = pivot_column > obj.Tolerance;
            if ~any(positive_pivots)
                leaving_var_idx = [];  % Unbounded
                return;
            end
            % Calculate ratios
            ratios = inf(m, 1);
            ratios(positive_pivots) = rhs(positive_pivots) ./ pivot_column(positive_pivots);
            % Find minimum ratio
            [~, leaving_var_idx] = min(ratios);
            % Handle ties using Bland's rule (choose smallest index)
            min_ratio = ratios(leaving_var_idx);
            tied_indices = find(abs(ratios - min_ratio) < obj.Tolerance);
            leaving_var_idx = min(tied_indices);
        end
        function pivot(obj, tableau, pivot_row, pivot_col)
            % Perform pivot operation
            pivot_element = tableau(pivot_row, pivot_col);
            if abs(pivot_element) < obj.Tolerance
                error('Pivot element is too small');
            end
            % Normalize pivot row
            tableau(pivot_row, :) = tableau(pivot_row, :) / pivot_element;
            % Eliminate other elements in pivot column
            [m, n] = size(tableau);
            for i = 1:m
                if i ~= pivot_row && abs(tableau(i, pivot_col)) > obj.Tolerance
                    factor = tableau(i, pivot_col);
                    tableau(i, :) = tableau(i, :) - factor * tableau(pivot_row, :);
                end
            end
        end
        function x = extractSolution(obj, tableau, basic_vars, n)
            % Extract solution from tableau
            x = zeros(n, 1);
            for i = 1:length(basic_vars)
                var_idx = basic_vars(i);
                if var_idx <= n
                    x(var_idx) = tableau(i, end);
                end
            end
        end
        function plotConvergence(obj)
            % Plot convergence history
            if isempty(obj.History) || isempty(obj.History.objective)
                error('No optimization history available. Run solve() first.');
            end
            figure;
            % Objective function
            iterations = 0:length(obj.History.objective)-1;
            plot(iterations, obj.History.objective, 'b-', 'LineWidth', 2, 'Marker', 'o');
            xlabel('Iteration');
            ylabel('Objective Function');
            title('Linear Programming Convergence', 'Color', obj.BerkeleyBlue);
            grid on;
            % Set figure properties
            set(gcf, 'Position', [100, 100, 800, 400]);
        end
    end
end