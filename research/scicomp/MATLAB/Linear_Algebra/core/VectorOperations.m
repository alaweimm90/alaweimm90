classdef VectorOperations < handle
    % VECTOROPERATIONS Core vector operations for scientific computing
    %
    % Features:
    % - Vector arithmetic with dimension checking
    % - Vector norms and metrics
    % - Inner and outer products
    % - Cross products and vector geometry
    % - Vector projections and orthogonalization
    methods (Static)
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
            if isempty(v)
                error('LinearAlgebra:InvalidInput', '%s cannot be empty', name);
            end
            % Convert to column vector
            v = v(:);
        end
        function w = add(u, v)
            % Vector addition with dimension checking
            u = VectorOperations.validateVector(u, 'u');
            v = VectorOperations.validateVector(v, 'v');
            if length(u) ~= length(v)
                error('LinearAlgebra:IncompatibleDimensions', ...
                    'Vector dimensions must match: %d != %d', length(u), length(v));
            end
            w = u + v;
        end
        function w = subtract(u, v)
            % Vector subtraction with dimension checking
            u = VectorOperations.validateVector(u, 'u');
            v = VectorOperations.validateVector(v, 'v');
            if length(u) ~= length(v)
                error('LinearAlgebra:IncompatibleDimensions', ...
                    'Vector dimensions must match: %d != %d', length(u), length(v));
            end
            w = u - v;
        end
        function w = scalarMultiply(scalar, v)
            % Scalar multiplication of vector
            v = VectorOperations.validateVector(v, 'v');
            if ~isscalar(scalar)
                error('LinearAlgebra:InvalidInput', 'First argument must be scalar');
            end
            w = scalar * v;
        end
        function result = dotProduct(u, v)
            % Dot product (inner product) of two vectors
            u = VectorOperations.validateVector(u, 'u');
            v = VectorOperations.validateVector(v, 'v');
            if length(u) ~= length(v)
                error('LinearAlgebra:IncompatibleDimensions', ...
                    'Vector dimensions must match: %d != %d', length(u), length(v));
            end
            result = u.' * v;
        end
        function result = outerProduct(u, v)
            % Outer product of two vectors
            u = VectorOperations.validateVector(u, 'u');
            v = VectorOperations.validateVector(v, 'v');
            result = u * v.';
        end
        function w = crossProduct(u, v)
            % Cross product of two 3D vectors
            u = VectorOperations.validateVector(u, 'u');
            v = VectorOperations.validateVector(v, 'v');
            if length(u) ~= 3 || length(v) ~= 3
                error('LinearAlgebra:InvalidInput', ...
                    'Cross product requires 3D vectors');
            end
            w = cross(u, v);
        end
        function result = tripleScalarProduct(u, v, w)
            % Scalar triple product: u · (v × w)
            u = VectorOperations.validateVector(u, 'u');
            v = VectorOperations.validateVector(v, 'v');
            w = VectorOperations.validateVector(w, 'w');
            if length(u) ~= 3 || length(v) ~= 3 || length(w) ~= 3
                error('LinearAlgebra:InvalidInput', ...
                    'Triple scalar product requires 3D vectors');
            end
            crossVW = VectorOperations.crossProduct(v, w);
            result = VectorOperations.dotProduct(u, crossVW);
        end
        function mag = magnitude(v)
            % Vector magnitude (Euclidean norm)
            v = VectorOperations.validateVector(v, 'v');
            mag = norm(v);
        end
        function vNorm = normalize(v, normType)
            % Normalize vector to unit length
            v = VectorOperations.validateVector(v, 'v');
            if nargin < 2
                normType = 2;
            end
            normV = norm(v, normType);
            if normV < 1e-12
                warning('LinearAlgebra:NearZeroNorm', ...
                    'Vector has near-zero norm, normalization may be unstable');
                vNorm = v;
            else
                vNorm = v / normV;
            end
        end
        function d = distance(u, v, normType)
            % Distance between two vectors
            if nargin < 3
                normType = 2;
            end
            diff = VectorOperations.subtract(u, v);
            d = norm(diff, normType);
        end
        function angle = angleBetween(u, v, degrees)
            % Angle between two vectors
            u = VectorOperations.validateVector(u, 'u');
            v = VectorOperations.validateVector(v, 'v');
            if nargin < 3
                degrees = false;
            end
            if length(u) ~= length(v)
                error('LinearAlgebra:IncompatibleDimensions', ...
                    'Vector dimensions must match: %d != %d', length(u), length(v));
            end
            % Normalize vectors
            uNorm = VectorOperations.normalize(u);
            vNorm = VectorOperations.normalize(v);
            % Compute angle using dot product
            cosAngle = max(-1, min(1, uNorm.' * vNorm)); % Clamp to [-1, 1]
            angle = acos(cosAngle);
            if degrees
                angle = rad2deg(angle);
            end
        end
        function proj = project(u, v)
            % Project vector u onto vector v
            u = VectorOperations.validateVector(u, 'u');
            v = VectorOperations.validateVector(v, 'v');
            if length(u) ~= length(v)
                error('LinearAlgebra:IncompatibleDimensions', ...
                    'Vector dimensions must match: %d != %d', length(u), length(v));
            end
            vDotV = v.' * v;
            if vDotV < 1e-12
                warning('LinearAlgebra:ZeroVector', ...
                    'Cannot project onto zero vector');
                proj = zeros(size(u));
            else
                proj = ((u.' * v) / vDotV) * v;
            end
        end
        function rej = reject(u, v)
            % Vector rejection: component of u orthogonal to v
            projection = VectorOperations.project(u, v);
            rej = VectorOperations.subtract(u, projection);
        end
        function isOrth = areOrthogonal(u, v, tolerance)
            % Check if two vectors are orthogonal
            if nargin < 3
                tolerance = 1e-10;
            end
            dotProd = VectorOperations.dotProduct(u, v);
            isOrth = abs(dotProd) < tolerance;
        end
        function isPar = areParallel(u, v, tolerance)
            % Check if two vectors are parallel
            u = VectorOperations.validateVector(u, 'u');
            v = VectorOperations.validateVector(v, 'v');
            if nargin < 3
                tolerance = 1e-10;
            end
            if length(u) ~= length(v)
                error('LinearAlgebra:IncompatibleDimensions', ...
                    'Vector dimensions must match: %d != %d', length(u), length(v));
            end
            % Check if cross product is zero (for 3D vectors)
            if length(u) == 3
                crossProd = VectorOperations.crossProduct(u, v);
                isPar = norm(crossProd) < tolerance;
            else
                % For general case, check if angle is 0 or π
                try
                    angle = VectorOperations.angleBetween(u, v);
                    isPar = abs(angle) < tolerance || abs(angle - pi) < tolerance;
                catch
                    isPar = false;
                end
            end
        end
        function orthoVectors = gramSchmidtOrthogonalization(vectors, normalize_flag)
            % Gram-Schmidt orthogonalization of vector list
            if nargin < 2
                normalize_flag = true;
            end
            if isempty(vectors)
                orthoVectors = {};
                return;
            end
            % Validate all vectors
            n = length(vectors);
            for i = 1:n
                vectors{i} = VectorOperations.validateVector(vectors{i}, sprintf('vector_%d', i));
            end
            % Check dimensions
            dim = length(vectors{1});
            for i = 2:n
                if length(vectors{i}) ~= dim
                    error('LinearAlgebra:IncompatibleDimensions', ...
                        'All vectors must have same dimension. Vector %d has dimension %d, expected %d', ...
                        i, length(vectors{i}), dim);
                end
            end
            orthoVectors = {};
            for i = 1:n
                % Start with the original vector
                orthoV = vectors{i};
                % Subtract projections onto all previous orthogonal vectors
                for j = 1:length(orthoVectors)
                    projection = VectorOperations.project(orthoV, orthoVectors{j});
                    orthoV = VectorOperations.subtract(orthoV, projection);
                end
                % Check if vector is linearly independent
                if norm(orthoV) < 1e-12
                    warning('LinearAlgebra:LinearlyDependent', ...
                        'Vector %d is linearly dependent and will be skipped', i);
                    continue;
                end
                % Normalize if requested
                if normalize_flag
                    orthoV = VectorOperations.normalize(orthoV);
                end
                orthoVectors{end+1} = orthoV; %#ok<AGROW>
            end
        end
        function vectors = createTestVectors()
            % Create test vectors for validation
            rng(42); % Set seed for reproducibility
            vectors = struct();
            % Standard vectors
            vectors.zero_3d = [0; 0; 0];
            vectors.unit_x = [1; 0; 0];
            vectors.unit_y = [0; 1; 0];
            vectors.unit_z = [0; 0; 1];
            % Random vectors
            vectors.random_3d = randn(3, 1);
            vectors.random_5d = randn(5, 1);
            vectors.random_10d = randn(10, 1);
            % Specific test cases
            vectors.orthogonal_pair_1 = [1; 0; 0];
            vectors.orthogonal_pair_2 = [0; 1; 0];
            vectors.parallel_pair_1 = [1; 2; 3];
            vectors.parallel_pair_2 = [2; 4; 6];
            % Vectors for Gram-Schmidt test
            vectors.gs_test_1 = [1; 1; 0];
            vectors.gs_test_2 = [1; 0; 1];
            vectors.gs_test_3 = [0; 1; 1];
        end
    end
end