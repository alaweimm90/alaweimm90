function varargout = matrix_utilities(operation, varargin)
% MATRIX_UTILITIES - Advanced matrix operations for spintronic simulations
%
% This function provides specialized matrix operations commonly used in
% spintronic device calculations including Pauli matrices, rotation matrices,
% tensor operations, and matrix analysis functions.
%
% Syntax:
%   result = matrix_utilities(operation, ...)
%   [result1, result2, ...] = matrix_utilities(operation, ...)
%
% Operations:
%   'pauli'           - Generate Pauli matrices
%   'rotation'        - Generate rotation matrices
%   'tensor_product'  - Compute tensor product of matrices
%   'commutator'      - Compute matrix commutator [A, B]
%   'anticommutator'  - Compute matrix anticommutator {A, B}
%   'matrix_exp'      - Matrix exponential with advanced algorithms
%   'matrix_log'      - Matrix logarithm
%   'spectral_radius' - Compute spectral radius
%   'condition_est'   - Condition number estimation
%   'block_diag'      - Create block diagonal matrix
%   'kron_sum'        - Kronecker sum A ⊕ B
%   'vectorize'       - Matrix vectorization operations
%   'spin_matrices'   - Generate spin-S matrices
%   'su2_generators'  - SU(2) group generators
%   'clifford'        - Clifford algebra matrices
%
% Examples:
%   % Get Pauli matrices
%   [sx, sy, sz] = matrix_utilities('pauli')
%   
%   % Rotation matrix around z-axis
%   Rz = matrix_utilities('rotation', 'z', pi/4)
%   
%   % Tensor product
%   result = matrix_utilities('tensor_product', A, B)
%   
%   % Commutator
%   comm = matrix_utilities('commutator', A, B)
%
% Author: Meshal Alawein <meshal@berkeley.edu>
% Copyright © 2025 Meshal Alawein — All rights reserved.
% License: MIT

    if nargin < 1
        error('Operation must be specified');
    end
    
    switch lower(operation)
        case 'pauli'
            [varargout{1:nargout}] = generate_pauli_matrices(varargin{:});
            
        case 'rotation'
            [varargout{1:nargout}] = generate_rotation_matrices(varargin{:});
            
        case 'tensor_product'
            [varargout{1:nargout}] = compute_tensor_product(varargin{:});
            
        case 'commutator'
            [varargout{1:nargout}] = compute_commutator(varargin{:});
            
        case 'anticommutator'
            [varargout{1:nargout}] = compute_anticommutator(varargin{:});
            
        case 'matrix_exp'
            [varargout{1:nargout}] = compute_matrix_exponential(varargin{:});
            
        case 'matrix_log'
            [varargout{1:nargout}] = compute_matrix_logarithm(varargin{:});
            
        case 'spectral_radius'
            [varargout{1:nargout}] = compute_spectral_radius(varargin{:});
            
        case 'condition_est'
            [varargout{1:nargout}] = estimate_condition_number(varargin{:});
            
        case 'block_diag'
            [varargout{1:nargout}] = create_block_diagonal(varargin{:});
            
        case 'kron_sum'
            [varargout{1:nargout}] = compute_kronecker_sum(varargin{:});
            
        case 'vectorize'
            [varargout{1:nargout}] = vectorize_operations(varargin{:});
            
        case 'spin_matrices'
            [varargout{1:nargout}] = generate_spin_matrices(varargin{:});
            
        case 'su2_generators'
            [varargout{1:nargout}] = generate_su2_generators(varargin{:});
            
        case 'clifford'
            [varargout{1:nargout}] = generate_clifford_matrices(varargin{:});
            
        otherwise
            error('Unknown operation: %s', operation);
    end
end

function varargout = generate_pauli_matrices(varargin)
    % Generate Pauli spin matrices
    
    % Pauli matrices
    sigma_x = [0, 1; 1, 0];
    sigma_y = [0, -1i; 1i, 0];
    sigma_z = [1, 0; 0, -1];
    sigma_0 = eye(2);  % Identity
    
    if nargout <= 1
        % Return as cell array or structure
        if nargin >= 1 && strcmpi(varargin{1}, 'struct')
            varargout{1} = struct('x', sigma_x, 'y', sigma_y, 'z', sigma_z, 'I', sigma_0);
        else
            varargout{1} = {sigma_x, sigma_y, sigma_z, sigma_0};
        end
    elseif nargout == 3
        varargout{1} = sigma_x;
        varargout{2} = sigma_y;
        varargout{3} = sigma_z;
    elseif nargout == 4
        varargout{1} = sigma_x;
        varargout{2} = sigma_y;
        varargout{3} = sigma_z;
        varargout{4} = sigma_0;
    else
        error('Invalid number of output arguments for Pauli matrices');
    end
end

function varargout = generate_rotation_matrices(axis, angle, varargin)
    % Generate rotation matrices
    
    if nargin < 2
        error('Axis and angle must be specified');
    end
    
    % Handle different input formats
    if ischar(axis)
        switch lower(axis)
            case 'x'
                axis_vector = [1, 0, 0];
            case 'y'
                axis_vector = [0, 1, 0];
            case 'z'
                axis_vector = [0, 0, 1];
            otherwise
                error('Unknown rotation axis: %s', axis);
        end
    elseif isnumeric(axis) && length(axis) == 3
        axis_vector = axis / norm(axis);  % Normalize
    else
        error('Invalid axis specification');
    end
    
    % Rodrigues' rotation formula
    K = [0, -axis_vector(3), axis_vector(2);
         axis_vector(3), 0, -axis_vector(1);
         -axis_vector(2), axis_vector(1), 0];
    
    R = eye(3) + sin(angle) * K + (1 - cos(angle)) * K^2;
    
    varargout{1} = R;
    
    % Also return quaternion representation if requested
    if nargout >= 2
        q = [cos(angle/2); sin(angle/2) * axis_vector(:)];
        varargout{2} = q;
    end
end

function result = compute_tensor_product(A, B, varargin)
    % Compute tensor (Kronecker) product
    
    if nargin < 2
        error('Two matrices required for tensor product');
    end
    
    result = kron(A, B);
end

function result = compute_commutator(A, B, varargin)
    % Compute commutator [A, B] = AB - BA
    
    if nargin < 2
        error('Two matrices required for commutator');
    end
    
    result = A * B - B * A;
end

function result = compute_anticommutator(A, B, varargin)
    % Compute anticommutator {A, B} = AB + BA
    
    if nargin < 2
        error('Two matrices required for anticommutator');
    end
    
    result = A * B + B * A;
end

function result = compute_matrix_exponential(A, varargin)
    % Advanced matrix exponential computation
    
    p = inputParser;
    addRequired(p, 'A', @(x) isnumeric(x) && ismatrix(x));
    addParameter(p, 'Method', 'pade', @(x) any(strcmpi(x, {'pade', 'series', 'eigen', 'scaling'})));
    addParameter(p, 'Tolerance', 1e-12, @(x) isscalar(x) && x > 0);
    addParameter(p, 'MaxTerms', 100, @(x) isscalar(x) && x > 0);
    
    parse(p, A, varargin{:});
    
    method = p.Results.Method;
    tol = p.Results.Tolerance;
    max_terms = p.Results.MaxTerms;
    
    switch lower(method)
        case 'pade'
            result = expm(A);  % MATLAB's built-in (uses Padé approximation)
            
        case 'series'
            % Taylor series expansion
            n = size(A, 1);
            result = eye(n);
            term = eye(n);
            
            for k = 1:max_terms
                term = term * A / k;
                old_result = result;
                result = result + term;
                
                if norm(result - old_result, 'fro') < tol
                    break;
                end
            end
            
        case 'eigen'
            % Eigenvalue decomposition method
            [V, D] = eig(A);
            result = V * diag(exp(diag(D))) / V;
            
        case 'scaling'
            % Scaling and squaring method
            norm_A = norm(A, 1);
            s = max(0, ceil(log2(norm_A)));
            A_scaled = A / 2^s;
            
            % Padé approximation of exp(A_scaled)
            result = pade_approximation(A_scaled, 6);
            
            % Square s times
            for i = 1:s
                result = result^2;
            end
    end
end

function result = pade_approximation(A, order)
    % Padé approximation for matrix exponential
    
    n = size(A, 1);
    I = eye(n);
    
    switch order
        case 6
            % (6,6) Padé approximant
            A2 = A^2;
            A4 = A2^2;
            A6 = A2 * A4;
            
            U = A * (64*I + 32*A2 + 4*A4 + A6/3);
            V = 64*I + 48*A2 + 12*A4 + A6;
            
            result = (V + U) / (V - U);
            
        otherwise
            % General Padé approximation
            result = expm(A);  % Fallback to built-in
    end
end

function result = compute_matrix_logarithm(A, varargin)
    % Matrix logarithm computation
    
    p = inputParser;
    addRequired(p, 'A', @(x) isnumeric(x) && ismatrix(x));
    addParameter(p, 'Method', 'schur', @(x) any(strcmpi(x, {'schur', 'series', 'eigen'})));
    
    parse(p, A, varargin{:});
    
    method = p.Results.Method;
    
    switch lower(method)
        case 'schur'
            result = logm(A);  % MATLAB's built-in (uses Schur decomposition)
            
        case 'eigen'
            [V, D] = eig(A);
            log_D = diag(log(diag(D)));
            result = V * log_D / V;
            
        case 'series'
            % Series expansion log(I + X) for X with small norm
            I = eye(size(A));
            X = A - I;
            
            if norm(X) >= 1
                error('Series method requires ||A - I|| < 1');
            end
            
            result = zeros(size(A));
            term = -X;
            
            for k = 1:100
                result = result - term / k;
                term = term * X;
                
                if norm(term) < 1e-12
                    break;
                end
            end
    end
end

function rho = compute_spectral_radius(A, varargin)
    % Compute spectral radius (largest eigenvalue magnitude)
    
    eigenvals = eig(A);
    rho = max(abs(eigenvals));
    
    % Handle edge case for empty matrix
    if isempty(eigenvals)
        rho = 0;
    end
end

function kappa = estimate_condition_number(A, varargin)
    % Estimate matrix condition number
    
    p = inputParser;
    addRequired(p, 'A');
    addParameter(p, 'Norm', '2', @(x) any(strcmpi(x, {'1', '2', 'inf', 'fro'})));
    
    parse(p, A, varargin{:});
    norm_type = p.Results.Norm;
    
    switch norm_type
        case '1'
            kappa = condest(A);
        case '2'
            kappa = cond(A, 2);
        case 'inf'
            kappa = cond(A, inf);
        case 'fro'
            kappa = cond(A, 'fro');
    end
end

function result = create_block_diagonal(varargin)
    % Create block diagonal matrix from input matrices
    
    if nargin == 0
        error('At least one matrix required');
    end
    
    % Handle cell array input
    if nargin == 1 && iscell(varargin{1})
        matrices = varargin{1};
    else
        matrices = varargin;
    end
    
    result = blkdiag(matrices{:});
end

function result = compute_kronecker_sum(A, B, varargin)
    % Compute Kronecker sum A ⊕ B = A ⊗ I + I ⊗ B
    
    if nargin < 2
        error('Two matrices required for Kronecker sum');
    end
    
    [m, n] = size(A);
    [p, q] = size(B);
    
    result = kron(A, eye(p)) + kron(eye(m), B);
end

function result = vectorize_operations(operation, A, varargin)
    % Matrix vectorization operations
    
    switch lower(operation)
        case 'vec'
            % Vectorize matrix (column-wise)
            result = A(:);
            
        case 'unvec'
            % Reshape vector back to matrix
            if nargin < 3
                error('Matrix dimensions required for unvec');
            end
            dims = varargin{1};
            result = reshape(A, dims);
            
        case 'vech'
            % Vectorize lower triangular part
            n = size(A, 1);
            idx = tril(true(n));
            result = A(idx);
            
        case 'svec'
            % Symmetric vectorization
            n = size(A, 1);
            result = zeros(n*(n+1)/2, 1);
            k = 1;
            for j = 1:n
                for i = j:n
                    if i == j
                        result(k) = A(i, j);
                    else
                        result(k) = sqrt(2) * A(i, j);
                    end
                    k = k + 1;
                end
            end
            
        otherwise
            error('Unknown vectorization operation: %s', operation);
    end
end

function varargout = generate_spin_matrices(S, varargin)
    % Generate spin-S matrices (Sx, Sy, Sz)
    
    if nargin < 1
        S = 1/2;  % Default to spin-1/2
    end
    
    % Dimension of spin-S space
    dim = 2*S + 1;
    
    % Generate basis states |S, m> with m = -S, -S+1, ..., S-1, S
    m_values = -S:S;
    
    % Spin raising and lowering operators
    Sp = zeros(dim, dim);  % S+
    Sm = zeros(dim, dim);  % S-
    Sz = zeros(dim, dim);  % Sz
    
    for i = 1:dim
        m = m_values(i);
        
        % Sz diagonal elements
        Sz(i, i) = m;
        
        % S+ off-diagonal elements
        if i > 1
            Sp(i-1, i) = sqrt(S*(S+1) - m*(m-1));
        end
        
        % S- off-diagonal elements  
        if i < dim
            Sm(i+1, i) = sqrt(S*(S+1) - m*(m+1));
        end
    end
    
    % Sx and Sy from raising and lowering operators
    Sx = (Sp + Sm) / 2;
    Sy = (Sp - Sm) / (2i);
    
    if nargout <= 1
        varargout{1} = struct('x', Sx, 'y', Sy, 'z', Sz, 'plus', Sp, 'minus', Sm);
    elseif nargout == 3
        varargout{1} = Sx;
        varargout{2} = Sy;
        varargout{3} = Sz;
    elseif nargout == 5
        varargout{1} = Sx;
        varargout{2} = Sy;
        varargout{3} = Sz;
        varargout{4} = Sp;
        varargout{5} = Sm;
    end
end

function varargout = generate_su2_generators(varargin)
    % Generate SU(2) group generators (Pauli matrices / 2)
    
    [sigma_x, sigma_y, sigma_z] = generate_pauli_matrices();
    
    % SU(2) generators are Pauli matrices divided by 2
    J1 = sigma_x / 2;
    J2 = sigma_y / 2;
    J3 = sigma_z / 2;
    
    if nargout <= 1
        varargout{1} = {J1, J2, J3};
    else
        varargout{1} = J1;
        varargout{2} = J2;
        varargout{3} = J3;
    end
end

function varargout = generate_clifford_matrices(dimension, varargin)
    % Generate Clifford algebra matrices (gamma matrices)
    
    if nargin < 1
        dimension = 4;  % Default to 4D (Dirac matrices)
    end
    
    switch dimension
        case 2
            % 2D Clifford algebra (equivalent to Pauli matrices)
            gamma1 = [0, 1; 1, 0];
            gamma2 = [0, -1i; 1i, 0];
            
            if nargout <= 1
                varargout{1} = {gamma1, gamma2};
            else
                varargout{1} = gamma1;
                varargout{2} = gamma2;
            end
            
        case 4
            % 4D Clifford algebra (Dirac gamma matrices)
            sigma = generate_pauli_matrices();
            I2 = eye(2);
            O2 = zeros(2);
            
            gamma0 = [I2, O2; O2, -I2];
            gamma1 = [O2, sigma{1}; -sigma{1}, O2];
            gamma2 = [O2, sigma{2}; -sigma{2}, O2];
            gamma3 = [O2, sigma{3}; -sigma{3}, O2];
            gamma5 = 1i * gamma0 * gamma1 * gamma2 * gamma3;
            
            if nargout <= 1
                varargout{1} = {gamma0, gamma1, gamma2, gamma3, gamma5};
            else
                varargout{1} = gamma0;
                varargout{2} = gamma1;
                varargout{3} = gamma2;
                varargout{4} = gamma3;
                if nargout >= 5
                    varargout{5} = gamma5;
                end
            end
            
        otherwise
            error('Clifford algebra not implemented for dimension %d', dimension);
    end
end