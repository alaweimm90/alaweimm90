classdef MatrixDecompositions < handle
    % MATRIXDECOMPOSITIONS Matrix decomposition algorithms
    %
    % Features:
    % - LU decomposition with partial pivoting
    % - QR decomposition (Householder and Givens)
    % - Cholesky decomposition
    % - Singular Value Decomposition (SVD)
    % - Eigenvalue decomposition
    % - Schur decomposition
    methods (Static)
        function [P, L, U] = luDecomposition(A)
            % LU decomposition with partial pivoting
            A = MatrixOperations.validateMatrix(A, 'A');
            if size(A, 1) ~= size(A, 2)
                error('LinearAlgebra:NonSquareMatrix', ...
                    'Matrix must be square for LU decomposition');
            end
            [L, U, P] = lu(A);
        end
        function [Q, R] = qrDecomposition(A, mode)
            % QR decomposition using Householder reflections
            A = MatrixOperations.validateMatrix(A, 'A');
            if nargin < 2
                mode = 'full';
            end
            if ~ismember(mode, {'full', 'economic'})
                error('LinearAlgebra:InvalidInput', ...
                    'Mode must be ''full'' or ''economic''');
            end
            if strcmp(mode, 'economic')
                [Q, R] = qr(A, 'econ');
            else
                [Q, R] = qr(A);
            end
        end
        function L = choleskyDecomposition(A, lower)
            % Cholesky decomposition for positive definite matrices
            A = MatrixOperations.validateMatrix(A, 'A');
            if nargin < 2
                lower = true;
            end
            if size(A, 1) ~= size(A, 2)
                error('LinearAlgebra:NonSquareMatrix', ...
                    'Matrix must be square for Cholesky decomposition');
            end
            % Check if matrix is symmetric (within tolerance)
            if norm(A - A.', 'fro') > 1e-12
                warning('LinearAlgebra:NotSymmetric', ...
                    'Matrix is not symmetric, results may be unreliable');
            end
            try
                if lower
                    L = chol(A, 'lower');
                else
                    L = chol(A, 'upper');
                end
            catch ME
                if contains(ME.message, 'positive definite')
                    error('LinearAlgebra:NotPositiveDefinite', ...
                        'Matrix is not positive definite');
                else
                    rethrow(ME);
                end
            end
        end
        function [U, S, V] = svd(A, fullMatrices, computeUV)
            % Singular Value Decomposition
            A = MatrixOperations.validateMatrix(A, 'A');
            if nargin < 2
                fullMatrices = true;
            end
            if nargin < 3
                computeUV = true;
            end
            if ~computeUV
                S = svd(A);
                U = [];
                V = [];
            else
                if fullMatrices
                    [U, S, V] = svd(A);
                else
                    [U, S, V] = svd(A, 'econ');
                end
                % Convert S to vector of singular values for consistency
                S = diag(S);
            end
        end
        function [eigenvals, eigenvecs] = eigendecomposition(A, rightVecs, leftVecs)
            % Eigenvalue decomposition for general matrices
            A = MatrixOperations.validateMatrix(A, 'A');
            if nargin < 2
                rightVecs = true;
            end
            if nargin < 3
                leftVecs = false;
            end
            if size(A, 1) ~= size(A, 2)
                error('LinearAlgebra:NonSquareMatrix', ...
                    'Matrix must be square for eigendecomposition');
            end
            if rightVecs && ~leftVecs
                [eigenvecs, eigenvals] = eig(A);
                eigenvals = diag(eigenvals);
                % Sort by eigenvalue magnitude (descending)
                [~, idx] = sort(abs(eigenvals), 'descend');
                eigenvals = eigenvals(idx);
                eigenvecs = eigenvecs(:, idx);
            elseif leftVecs
                % MATLAB doesn't have direct left eigenvector computation
                % Use transpose property: left eigenvectors of A are right eigenvectors of A'
                [rightEigenvecs, eigenvals] = eig(A);
                [leftEigenvecs, ~] = eig(A');
                eigenvals = diag(eigenvals);
                % Sort by eigenvalue magnitude (descending)
                [~, idx] = sort(abs(eigenvals), 'descend');
                eigenvals = eigenvals(idx);
                if rightVecs
                    eigenvecs = {rightEigenvecs(:, idx), leftEigenvecs(:, idx)};
                else
                    eigenvecs = leftEigenvecs(:, idx);
                end
            else
                eigenvals = eig(A);
                [~, idx] = sort(abs(eigenvals), 'descend');
                eigenvals = eigenvals(idx);
                eigenvecs = [];
            end
        end
        function [eigenvals, eigenvecs] = symmetricEigendecomposition(A)
            % Eigenvalue decomposition for symmetric/Hermitian matrices
            A = MatrixOperations.validateMatrix(A, 'A');
            if size(A, 1) ~= size(A, 2)
                error('LinearAlgebra:NonSquareMatrix', ...
                    'Matrix must be square');
            end
            % Check if matrix is symmetric/Hermitian
            if ~isreal(A)
                if norm(A - A', 'fro') > 1e-12
                    warning('LinearAlgebra:NotHermitian', ...
                        'Matrix is not Hermitian, results may be unreliable');
                end
            else
                if norm(A - A.', 'fro') > 1e-12
                    warning('LinearAlgebra:NotSymmetric', ...
                        'Matrix is not symmetric, results may be unreliable');
                end
            end
            [eigenvecs, eigenvals] = eig(A);
            eigenvals = diag(eigenvals);
            % Sort in ascending order (MATLAB default for symmetric matrices)
            [eigenvals, idx] = sort(eigenvals, 'ascend');
            eigenvecs = eigenvecs(:, idx);
        end
        function [T, Z] = schurDecomposition(A, outputType)
            % Schur decomposition
            A = MatrixOperations.validateMatrix(A, 'A');
            if nargin < 2
                outputType = 'real';
            end
            if size(A, 1) ~= size(A, 2)
                error('LinearAlgebra:NonSquareMatrix', ...
                    'Matrix must be square for Schur decomposition');
            end
            if strcmp(outputType, 'complex')
                [Z, T] = schur(A, 'complex');
            else
                [Z, T] = schur(A);
            end
        end
    end
end