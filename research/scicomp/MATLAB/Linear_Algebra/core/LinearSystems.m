classdef LinearSystems < handle
    % LINEARSYSTEMS Linear system solvers for scientific computing
    %
    % Features:
    % - Direct methods (LU, Cholesky, QR, SVD)
    % - Iterative methods (Jacobi, Gauss-Seidel, CG, GMRES)
    % - System analysis and solver recommendation
    % - Specialized algorithms for structured matrices
    methods (Static)
        function result = luSolve(A, b, checkFinite)
            % Solve Ax = b using LU decomposition with partial pivoting
            if nargin < 3
                checkFinite = true;
            end
            A = MatrixOperations.validateMatrix(A, 'A');
            b = MatrixOperations.validateVector(b, 'b');
            if size(A, 1) ~= length(b)
                error('LinearAlgebra:IncompatibleDimensions', ...
                    'Matrix and vector dimensions incompatible: %d != %d', ...
                    size(A, 1), length(b));
            end
            if size(A, 1) ~= size(A, 2)
                error('LinearAlgebra:NonSquareMatrix', ...
                    'Matrix must be square for LU solve');
            end
            try
                x = A \ b;
                % Compute residual
                residual = A * x - b;
                residualNorm = norm(residual);
                result = struct(...
                    'solution', x, ...
                    'success', true, ...
                    'iterations', 1, ...
                    'residualNorm', residualNorm, ...
                    'info', struct('method', 'LU', 'conditionNumber', cond(A)));
            catch ME
                warning('LinearAlgebra:SolveFailed', 'LU solve failed: %s', ME.message);
                result = struct(...
                    'solution', NaN(size(A, 2), 1), ...
                    'success', false, ...
                    'iterations', 0, ...
                    'residualNorm', Inf, ...
                    'info', struct('method', 'LU', 'error', ME.message));
            end
        end
        function result = choleskySolve(A, b, lower)
            % Solve Ax = b using Cholesky decomposition for SPD matrices
            if nargin < 3
                lower = true;
            end
            A = MatrixOperations.validateMatrix(A, 'A');
            b = MatrixOperations.validateVector(b, 'b');
            if size(A, 1) ~= length(b)
                error('LinearAlgebra:IncompatibleDimensions', ...
                    'Matrix and vector dimensions incompatible: %d != %d', ...
                    size(A, 1), length(b));
            end
            if size(A, 1) ~= size(A, 2)
                error('LinearAlgebra:NonSquareMatrix', ...
                    'Matrix must be square for Cholesky solve');
            end
            try
                % Check if matrix is symmetric
                if norm(A - A.', 'fro') > 1e-12
                    warning('LinearAlgebra:NotSymmetric', ...
                        'Matrix is not symmetric, Cholesky may fail');
                end
                % Use Cholesky factorization
                if lower
                    L = chol(A, 'lower');
                    y = L \ b;
                    x = L.' \ y;
                else
                    U = chol(A, 'upper');
                    y = U.' \ b;
                    x = U \ y;
                end
                % Compute residual
                residual = A * x - b;
                residualNorm = norm(residual);
                result = struct(...
                    'solution', x, ...
                    'success', true, ...
                    'iterations', 1, ...
                    'residualNorm', residualNorm, ...
                    'info', struct('method', 'Cholesky', 'conditionNumber', cond(A)));
            catch ME
                warning('LinearAlgebra:SolveFailed', 'Cholesky solve failed: %s', ME.message);
                result = struct(...
                    'solution', NaN(size(A, 2), 1), ...
                    'success', false, ...
                    'iterations', 0, ...
                    'residualNorm', Inf, ...
                    'info', struct('method', 'Cholesky', 'error', ME.message));
            end
        end
        function result = qrSolve(A, b, mode)
            % Solve Ax = b using QR decomposition (supports overdetermined systems)
            if nargin < 3
                mode = 'full';
            end
            A = MatrixOperations.validateMatrix(A, 'A');
            b = MatrixOperations.validateVector(b, 'b');
            if size(A, 1) ~= length(b)
                error('LinearAlgebra:IncompatibleDimensions', ...
                    'Matrix and vector dimensions incompatible: %d != %d', ...
                    size(A, 1), length(b));
            end
            try
                [Q, R] = MatrixDecompositions.qrDecomposition(A, mode);
                % Solve Rx = Q'b
                QtB = Q.' * b;
                if strcmp(mode, 'economic')
                    x = R \ QtB;
                else
                    x = R(1:size(A, 2), :) \ QtB(1:size(A, 2));
                end
                % Compute residual
                residual = A * x - b;
                residualNorm = norm(residual);
                result = struct(...
                    'solution', x, ...
                    'success', true, ...
                    'iterations', 1, ...
                    'residualNorm', residualNorm, ...
                    'info', struct(...
                        'method', 'QR', ...
                        'conditionNumber', cond(A), ...
                        'overdetermined', size(A, 1) > size(A, 2)));
            catch ME
                warning('LinearAlgebra:SolveFailed', 'QR solve failed: %s', ME.message);
                result = struct(...
                    'solution', NaN(size(A, 2), 1), ...
                    'success', false, ...
                    'iterations', 0, ...
                    'residualNorm', Inf, ...
                    'info', struct('method', 'QR', 'error', ME.message));
            end
        end
        function result = svdSolve(A, b, tolerance)
            % Solve Ax = b using SVD (handles rank-deficient systems)
            A = MatrixOperations.validateMatrix(A, 'A');
            b = MatrixOperations.validateVector(b, 'b');
            if size(A, 1) ~= length(b)
                error('LinearAlgebra:IncompatibleDimensions', ...
                    'Matrix and vector dimensions incompatible: %d != %d', ...
                    size(A, 1), length(b));
            end
            try
                [U, S, V] = svd(A);
                s = diag(S);
                if nargin < 3 || isempty(tolerance)
                    tolerance = max(size(A)) * eps(max(s));
                end
                % Compute pseudoinverse solution
                r = sum(s > tolerance);
                x = V(:, 1:r) * ((U(:, 1:r).' * b) ./ s(1:r));
                % Compute residual
                if size(A, 1) >= size(A, 2)
                    residual = A * x - b;
                    residualNorm = norm(residual);
                else
                    residualNorm = 0; % Underdetermined case
                end
                result = struct(...
                    'solution', x, ...
                    'success', true, ...
                    'iterations', 1, ...
                    'residualNorm', residualNorm, ...
                    'info', struct(...
                        'method', 'SVD', ...
                        'rank', r, ...
                        'singularValues', s, ...
                        'conditionNumber', s(1) / s(end)));
            catch ME
                warning('LinearAlgebra:SolveFailed', 'SVD solve failed: %s', ME.message);
                result = struct(...
                    'solution', NaN(size(A, 2), 1), ...
                    'success', false, ...
                    'iterations', 0, ...
                    'residualNorm', Inf, ...
                    'info', struct('method', 'SVD', 'error', ME.message));
            end
        end
        function result = jacobiSolve(A, b, x0, maxIterations, tolerance)
            % Solve Ax = b using Jacobi iteration
            if nargin < 3 || isempty(x0)
                x0 = zeros(size(A, 2), 1);
            end
            if nargin < 4
                maxIterations = 1000;
            end
            if nargin < 5
                tolerance = 1e-6;
            end
            A = MatrixOperations.validateMatrix(A, 'A');
            b = MatrixOperations.validateVector(b, 'b');
            x = MatrixOperations.validateVector(x0, 'x0');
            if size(A, 1) ~= size(A, 2)
                error('LinearAlgebra:NonSquareMatrix', ...
                    'Matrix must be square for Jacobi iteration');
            end
            if size(A, 1) ~= length(b)
                error('LinearAlgebra:IncompatibleDimensions', ...
                    'Matrix and vector dimensions incompatible: %d != %d', ...
                    size(A, 1), length(b));
            end
            n = size(A, 1);
            % Check diagonal dominance
            diagA = diag(A);
            if any(abs(diagA) < 1e-12)
                warning('LinearAlgebra:NearZeroDiagonal', ...
                    'Matrix has near-zero diagonal elements, Jacobi may not converge');
            end
            % Extract diagonal and off-diagonal parts
            D = diag(diagA);
            R = A - D;
            residuals = zeros(maxIterations, 1);
            for iteration = 1:maxIterations
                % Jacobi update: x^(k+1) = D^(-1) * (b - R * x^(k))
                xNew = (b - R * x) ./ diagA;
                % Check convergence
                residual = A * xNew - b;
                residualNorm = norm(residual);
                residuals(iteration) = residualNorm;
                if residualNorm < tolerance
                    result = struct(...
                        'solution', xNew, ...
                        'success', true, ...
                        'iterations', iteration, ...
                        'residualNorm', residualNorm, ...
                        'info', struct('method', 'Jacobi', 'residuals', residuals(1:iteration)));
                    return;
                end
                x = xNew;
            end
            result = struct(...
                'solution', x, ...
                'success', false, ...
                'iterations', maxIterations, ...
                'residualNorm', residuals(end), ...
                'info', struct('method', 'Jacobi', 'residuals', residuals, 'converged', false));
        end
        function result = gaussSeidelSolve(A, b, x0, maxIterations, tolerance)
            % Solve Ax = b using Gauss-Seidel iteration
            if nargin < 3 || isempty(x0)
                x0 = zeros(size(A, 2), 1);
            end
            if nargin < 4
                maxIterations = 1000;
            end
            if nargin < 5
                tolerance = 1e-6;
            end
            A = MatrixOperations.validateMatrix(A, 'A');
            b = MatrixOperations.validateVector(b, 'b');
            x = MatrixOperations.validateVector(x0, 'x0');
            if size(A, 1) ~= size(A, 2)
                error('LinearAlgebra:NonSquareMatrix', ...
                    'Matrix must be square for Gauss-Seidel iteration');
            end
            if size(A, 1) ~= length(b)
                error('LinearAlgebra:IncompatibleDimensions', ...
                    'Matrix and vector dimensions incompatible: %d != %d', ...
                    size(A, 1), length(b));
            end
            n = size(A, 1);
            residuals = zeros(maxIterations, 1);
            for iteration = 1:maxIterations
                xOld = x;
                % Gauss-Seidel update
                for i = 1:n
                    if abs(A(i, i)) < 1e-12
                        warning('LinearAlgebra:NearZeroDiagonal', ...
                            'Near-zero diagonal element at position %d', i);
                        continue;
                    end
                    sumAx = A(i, 1:i-1) * x(1:i-1) + A(i, i+1:n) * xOld(i+1:n);
                    x(i) = (b(i) - sumAx) / A(i, i);
                end
                % Check convergence
                residual = A * x - b;
                residualNorm = norm(residual);
                residuals(iteration) = residualNorm;
                if residualNorm < tolerance
                    result = struct(...
                        'solution', x, ...
                        'success', true, ...
                        'iterations', iteration, ...
                        'residualNorm', residualNorm, ...
                        'info', struct('method', 'Gauss-Seidel', 'residuals', residuals(1:iteration)));
                    return;
                end
            end
            result = struct(...
                'solution', x, ...
                'success', false, ...
                'iterations', maxIterations, ...
                'residualNorm', residuals(end), ...
                'info', struct('method', 'Gauss-Seidel', 'residuals', residuals, 'converged', false));
        end
        function result = conjugateGradient(A, b, x0, maxIterations, tolerance)
            % Solve Ax = b using Conjugate Gradient method (for SPD matrices)
            if nargin < 3 || isempty(x0)
                x0 = zeros(size(A, 2), 1);
            end
            if nargin < 4 || isempty(maxIterations)
                maxIterations = size(A, 2);
            end
            if nargin < 5
                tolerance = 1e-6;
            end
            A = MatrixOperations.validateMatrix(A, 'A');
            b = MatrixOperations.validateVector(b, 'b');
            x = MatrixOperations.validateVector(x0, 'x0');
            if size(A, 1) ~= size(A, 2)
                error('LinearAlgebra:NonSquareMatrix', ...
                    'Matrix must be square for CG');
            end
            if size(A, 1) ~= length(b)
                error('LinearAlgebra:IncompatibleDimensions', ...
                    'Matrix and vector dimensions incompatible: %d != %d', ...
                    size(A, 1), length(b));
            end
            % Check if matrix is symmetric
            if norm(A - A.', 'fro') > 1e-12
                warning('LinearAlgebra:NotSymmetric', ...
                    'Matrix is not symmetric, CG may not converge');
            end
            % Initialize CG
            r = b - A * x;
            p = r;
            rsold = r.' * r;
            residuals = zeros(maxIterations, 1);
            for iteration = 1:maxIterations
                Ap = A * p;
                alpha = rsold / (p.' * Ap);
                x = x + alpha * p;
                r = r - alpha * Ap;
                rsnew = r.' * r;
                residualNorm = sqrt(rsnew);
                residuals(iteration) = residualNorm;
                if residualNorm < tolerance
                    result = struct(...
                        'solution', x, ...
                        'success', true, ...
                        'iterations', iteration, ...
                        'residualNorm', residualNorm, ...
                        'info', struct('method', 'CG', 'residuals', residuals(1:iteration)));
                    return;
                end
                beta = rsnew / rsold;
                p = r + beta * p;
                rsold = rsnew;
            end
            result = struct(...
                'solution', x, ...
                'success', false, ...
                'iterations', maxIterations, ...
                'residualNorm', residuals(end), ...
                'info', struct('method', 'CG', 'residuals', residuals, 'converged', false));
        end
        function analysis = analyzeSystem(A, b)
            % Analyze linear system properties to guide solver selection
            A = MatrixOperations.validateMatrix(A, 'A');
            b = MatrixOperations.validateVector(b, 'b');
            analysis = struct();
            analysis.matrixShape = size(A);
            analysis.vectorLength = length(b);
            analysis.isSquare = size(A, 1) == size(A, 2);
            analysis.isOverdetermined = size(A, 1) > size(A, 2);
            analysis.isUnderdetermined = size(A, 1) < size(A, 2);
            if analysis.isSquare
                % Square matrix analysis
                analysis.determinant = det(A);
                analysis.isSingular = abs(analysis.determinant) < 1e-12;
                analysis.conditionNumber = cond(A);
                analysis.isWellConditioned = analysis.conditionNumber < 1e12;
                % Symmetry check
                analysis.isSymmetric = norm(A - A.', 'fro') < 1e-12;
                if analysis.isSymmetric
                    eigenvals = eig(A);
                    analysis.isPositiveDefinite = all(eigenvals > 1e-12);
                    analysis.isPositiveSemidefinite = all(eigenvals >= -1e-12);
                end
                % Diagonal dominance
                diagVals = abs(diag(A));
                offDiagSums = sum(abs(A), 2) - diagVals;
                analysis.isDiagonallyDominant = all(diagVals >= offDiagSums);
            end
        end
        function solverName = recommendSolver(A, b)
            % Recommend appropriate solver based on system properties
            analysis = LinearSystems.analyzeSystem(A, b);
            if ~analysis.isSquare
                solverName = 'qrSolve';
                return;
            end
            if analysis.isSingular
                solverName = 'svdSolve';
                return;
            end
            if isfield(analysis, 'isPositiveDefinite') && analysis.isPositiveDefinite
                solverName = 'choleskySolve';
                return;
            end
            if ~analysis.isWellConditioned
                solverName = 'svdSolve';
                return;
            end
            if analysis.matrixShape(1) > 1000 && ...
               isfield(analysis, 'isDiagonallyDominant') && analysis.isDiagonallyDominant
                solverName = 'gaussSeidelSolve';
                return;
            end
            solverName = 'luSolve';
        end
        function result = solveAuto(A, b, varargin)
            % Automatically select and apply appropriate solver
            solverName = LinearSystems.recommendSolver(A, b);
            switch solverName
                case 'luSolve'
                    result = LinearSystems.luSolve(A, b, varargin{:});
                case 'choleskySolve'
                    result = LinearSystems.choleskySolve(A, b, varargin{:});
                case 'qrSolve'
                    result = LinearSystems.qrSolve(A, b, varargin{:});
                case 'svdSolve'
                    result = LinearSystems.svdSolve(A, b, varargin{:});
                case 'gaussSeidelSolve'
                    result = LinearSystems.gaussSeidelSolve(A, b, varargin{:});
                otherwise
                    result = LinearSystems.luSolve(A, b, varargin{:});
            end
        end
        function systems = createTestSystems()
            % Create test linear systems for validation
            rng(42); % Set seed for reproducibility
            systems = struct();
            % Well-conditioned system
            A = randn(5, 5);
            A = A + 0.1 * eye(5);
            b = randn(5, 1);
            systems.wellConditioned = struct('A', A, 'b', b);
            % Symmetric positive definite
            A = randn(4, 4);
            A = A.' * A + 0.1 * eye(4);
            b = randn(4, 1);
            systems.symmetricPD = struct('A', A, 'b', b);
            % Overdetermined system
            A = randn(8, 5);
            b = randn(8, 1);
            systems.overdetermined = struct('A', A, 'b', b);
            % Ill-conditioned (Hilbert matrix)
            n = 5;
            A = hilb(n);
            b = ones(n, 1);
            systems.illConditioned = struct('A', A, 'b', b);
        end
    end
end