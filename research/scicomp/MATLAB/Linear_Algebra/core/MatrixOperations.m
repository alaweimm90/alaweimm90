classdef MatrixOperations < handle
    % MATRIXOPERATIONS Core matrix operations for scientific computing
    %
    % Features:
    % - Basic matrix arithmetic with error checking
    % - Matrix decompositions (LU, QR, Cholesky, SVD)
    % - Eigenvalue and eigenvector computation
    % - Matrix norms and condition numbers
    % - Specialized algorithms for scientific computing
    methods (Static)
        function A = validateMatrix(A, name)
            % Validate matrix input and convert to numeric array
            if nargin < 2
                name = 'matrix';
            end
            if ~isnumeric(A)
                error('LinearAlgebra:InvalidInput', '%s must be numeric', name);
            end
            if ~ismatrix(A)
                error('LinearAlgebra:InvalidInput', '%s must be 2-dimensional', name);
            end
            if isempty(A)
                error('LinearAlgebra:InvalidInput', '%s cannot be empty', name);
            end
        end
        function v = validateVector(v, name)
            % Validate vector input and convert to column vector
            if nargin < 2
                name = 'vector';
            end
            if ~isnumeric(v)
                error('LinearAlgebra:InvalidInput', '%s must be numeric', name);
            end
            if ~isvector(v)
                error('LinearAlgebra:InvalidInput', '%s must be a vector', name);
            end
            % Convert to column vector
            v = v(:);
        end
        function C = matrixMultiply(A, B, checkCompatibility)
            % Matrix multiplication with dimension checking
            if nargin < 3
                checkCompatibility = true;
            end
            A = MatrixOperations.validateMatrix(A, 'A');
            if isvector(B)
                B = MatrixOperations.validateVector(B, 'B');
            else
                B = MatrixOperations.validateMatrix(B, 'B');
            end
            if checkCompatibility
                if size(A, 2) ~= size(B, 1)
                    error('LinearAlgebra:IncompatibleDimensions', ...
                        'Incompatible dimensions: A[%dx%d] * B[%dx%d]', ...
                        size(A, 1), size(A, 2), size(B, 1), size(B, 2));
                end
            end
            C = A * B;
        end
        function C = matrixAdd(A, B)
            % Matrix addition with dimension checking
            A = MatrixOperations.validateMatrix(A, 'A');
            B = MatrixOperations.validateMatrix(B, 'B');
            if ~isequal(size(A), size(B))
                error('LinearAlgebra:IncompatibleDimensions', ...
                    'Incompatible shapes: A[%dx%d] + B[%dx%d]', ...
                    size(A, 1), size(A, 2), size(B, 1), size(B, 2));
            end
            C = A + B;
        end
        function C = matrixSubtract(A, B)
            % Matrix subtraction with dimension checking
            A = MatrixOperations.validateMatrix(A, 'A');
            B = MatrixOperations.validateMatrix(B, 'B');
            if ~isequal(size(A), size(B))
                error('LinearAlgebra:IncompatibleDimensions', ...
                    'Incompatible shapes: A[%dx%d] - B[%dx%d]', ...
                    size(A, 1), size(A, 2), size(B, 1), size(B, 2));
            end
            C = A - B;
        end
        function C = matrixPower(A, n)
            % Matrix power A^n using repeated squaring
            A = MatrixOperations.validateMatrix(A, 'A');
            if size(A, 1) ~= size(A, 2)
                error('LinearAlgebra:NonSquareMatrix', ...
                    'Matrix must be square for matrix power');
            end
            if n < 0 || n ~= round(n)
                error('LinearAlgebra:InvalidInput', ...
                    'Power must be non-negative integer');
            end
            if n == 0
                C = eye(size(A, 1));
            else
                C = A^n;
            end
        end
        function At = transpose(A)
            % Matrix transpose
            A = MatrixOperations.validateMatrix(A, 'A');
            At = A.';
        end
        function Ah = conjugateTranspose(A)
            % Conjugate transpose (Hermitian transpose)
            A = MatrixOperations.validateMatrix(A, 'A');
            Ah = A';
        end
        function tr = trace(A)
            % Matrix trace (sum of diagonal elements)
            A = MatrixOperations.validateMatrix(A, 'A');
            if size(A, 1) ~= size(A, 2)
                error('LinearAlgebra:NonSquareMatrix', ...
                    'Matrix must be square to compute trace');
            end
            tr = trace(A);
        end
        function d = determinant(A)
            % Matrix determinant
            A = MatrixOperations.validateMatrix(A, 'A');
            if size(A, 1) ~= size(A, 2)
                error('LinearAlgebra:NonSquareMatrix', ...
                    'Matrix must be square to compute determinant');
            end
            d = det(A);
        end
        function r = rank(A, tolerance)
            % Matrix rank using SVD
            A = MatrixOperations.validateMatrix(A, 'A');
            if nargin < 2
                tolerance = max(size(A)) * eps(norm(A));
            end
            r = rank(A, tolerance);
        end
        function kappa = conditionNumber(A, p)
            % Matrix condition number
            A = MatrixOperations.validateMatrix(A, 'A');
            if nargin < 2
                kappa = cond(A);
            else
                kappa = cond(A, p);
            end
        end
        function normF = frobeniusNorm(A)
            % Frobenius norm of matrix
            A = MatrixOperations.validateMatrix(A, 'A');
            normF = norm(A, 'fro');
        end
        function norm2 = spectralNorm(A)
            % Spectral norm (largest singular value)
            A = MatrixOperations.validateMatrix(A, 'A');
            norm2 = norm(A, 2);
        end
        function normNuc = nuclearNorm(A)
            % Nuclear norm (sum of singular values)
            A = MatrixOperations.validateMatrix(A, 'A');
            s = svd(A);
            normNuc = sum(s);
        end
        function matrices = createTestMatrices()
            % Create test matrices for validation
            rng(42); % Set seed for reproducibility
            matrices = struct();
            % Random matrices
            matrices.random_3x3 = randn(3, 3);
            matrices.random_5x5 = randn(5, 5);
            % Symmetric matrix
            A = randn(4, 4);
            matrices.symmetric_4x4 = A + A.';
            % Positive definite matrix
            A = randn(3, 3);
            matrices.positive_definite_3x3 = A.' * A + 0.1 * eye(3);
            % Orthogonal matrix (from QR decomposition)
            [Q, ~] = qr(randn(4, 4));
            matrices.orthogonal_4x4 = Q;
            % Singular matrix
            matrices.singular_3x3 = [1, 2, 3; 2, 4, 6; 1, 2, 3];
            % Hilbert matrix (ill-conditioned)
            n = 5;
            matrices.hilbert_5x5 = hilb(n);
        end
    end
end