classdef LinearQuadraticRegulator < handle
    % LinearQuadraticRegulator - LQR optimal controller
    %
    % Solves the optimal control problem:
    %   min ∫(x'Qx + u'Ru + 2x'Nu) dt
    %
    % Subject to: dx/dt = Ax + Bu
    %
    % Example:
    %   sys = StateSpaceSystem(A, B, C, D);
    %   Q = eye(sys.n_states);
    %   R = eye(sys.n_inputs);
    %   lqr = LinearQuadraticRegulator(sys, Q, R);
    %   K = lqr.gainMatrix();
    properties (Access = private)
        system      % StateSpaceSystem object
        Q_matrix    % State weighting matrix
        R_matrix    % Input weighting matrix
        N_matrix    % Cross-term matrix
        P_matrix    % Solution to Riccati equation
    end
    properties (Dependent)
        Q           % State weighting matrix (read-only)
        R           % Input weighting matrix (read-only)
        N           % Cross-term matrix (read-only)
        P           % Riccati equation solution (read-only)
    end
    methods
        function obj = LinearQuadraticRegulator(system, Q, R, N)
            % Constructor
            %
            % Parameters:
            %   system - StateSpaceSystem object
            %   Q - State weighting matrix (n x n, positive semi-definite)
            %   R - Input weighting matrix (m x m, positive definite)
            %   N - Cross-term matrix (n x m, optional)
            validateattributes(system, {'StateSpaceSystem'}, {});
            validateattributes(Q, {'numeric'}, {'2d', 'finite'});
            validateattributes(R, {'numeric'}, {'2d', 'finite'});
            if nargin < 4 || isempty(N)
                N = zeros(system.n_states, system.n_inputs);
            else
                validateattributes(N, {'numeric'}, {'2d', 'finite'});
            end
            obj.system = system;
            obj.Q_matrix = Q;
            obj.R_matrix = R;
            obj.N_matrix = N;
            obj.validateWeights();
            obj.solveRiccati();
        end
        % Dependent property getters
        function Q = get.Q(obj)
            Q = obj.Q_matrix;
        end
        function R = get.R(obj)
            R = obj.R_matrix;
        end
        function N = get.N(obj)
            N = obj.N_matrix;
        end
        function P = get.P(obj)
            P = obj.P_matrix;
        end
        function K = gainMatrix(obj)
            % Compute optimal feedback gain matrix K
            K = obj.R_matrix \ (obj.system.B' * obj.P_matrix + obj.N_matrix');
        end
        function sys_cl = closedLoopSystem(obj)
            % Compute closed-loop system with LQR feedback
            K = obj.gainMatrix();
            A_cl = obj.system.A - obj.system.B * K;
            B_cl = obj.system.B;  % For reference input
            C_cl = obj.system.C;
            D_cl = obj.system.D;
            sys_cl = StateSpaceSystem(A_cl, B_cl, C_cl, D_cl);
        end
        function cost = costFunctionValue(obj, x0)
            % Compute optimal cost for given initial condition
            %
            % Parameters:
            %   x0 - Initial state
            %
            % Returns:
            %   cost - Optimal cost J* = x0' P x0
            validateattributes(x0, {'numeric'}, {'vector'});
            if length(x0) ~= obj.system.n_states
                error('x0 must have n_states elements');
            end
            cost = x0(:)' * obj.P_matrix * x0(:);
        end
        function margins = stabilityMargins(obj)
            % Compute stability margins for LQR controller
            %
            % Returns:
            %   margins - Structure with gain and phase margins
            K = obj.gainMatrix();
            % Loop transfer function: L(s) = K * (sI - A)^(-1) * B
            sys_cl = obj.closedLoopSystem();
            % Compute margins using MATLAB Control System Toolbox if available
            if exist('margin', 'file') == 2
                % Create transfer function for loop analysis
                [num, den] = ss2tf(obj.system.A - obj.system.B * K, ...
                                   obj.system.B, K, zeros(size(K, 1), size(obj.system.B, 2)));
                try
                    [gm, pm, wcg, wcp] = margin(tf(num, den));
                    margins = struct('GainMargin', gm, 'PhaseMargin', pm, ...
                                   'GMFrequency', wcg, 'PMFrequency', wcp);
                catch
                    warning('Could not compute stability margins');
                    margins = struct('GainMargin', NaN, 'PhaseMargin', NaN, ...
                                   'GMFrequency', NaN, 'PMFrequency', NaN);
                end
            else
                % Theoretical LQR margins (infinite gain margin, >=60° phase margin)
                margins = struct('GainMargin', Inf, 'PhaseMargin', 60, ...
                               'GMFrequency', NaN, 'PMFrequency', NaN);
                warning('Control System Toolbox not available. Using theoretical LQR margins.');
            end
        end
    end
    methods (Access = private)
        function validateWeights(obj)
            % Validate weighting matrices
            % Check dimensions
            if ~isequal(size(obj.Q_matrix), [obj.system.n_states, obj.system.n_states])
                error('Q matrix must be n x n');
            end
            if ~isequal(size(obj.R_matrix), [obj.system.n_inputs, obj.system.n_inputs])
                error('R matrix must be m x m');
            end
            if ~isequal(size(obj.N_matrix), [obj.system.n_states, obj.system.n_inputs])
                error('N matrix must be n x m');
            end
            % Check symmetry
            if ~isequal(obj.Q_matrix, obj.Q_matrix')
                warning('Q matrix is not symmetric');
            end
            if ~isequal(obj.R_matrix, obj.R_matrix')
                warning('R matrix is not symmetric');
            end
            % Check positive definiteness of R
            try
                chol(obj.R_matrix);
            catch
                error('R matrix must be positive definite');
            end
            % Check positive semi-definiteness of Q
            if min(eig(obj.Q_matrix)) < -1e-10
                warning('Q matrix may not be positive semi-definite');
            end
        end
        function solveRiccati(obj)
            % Solve algebraic Riccati equation
            try
                % Use MATLAB's built-in function if available
                if exist('care', 'file') == 2
                    obj.P_matrix = care(obj.system.A, obj.system.B, obj.Q_matrix, obj.R_matrix, obj.N_matrix);
                else
                    % Fallback: solve using Schur decomposition
                    obj.P_matrix = obj.solveRiccatiSchur();
                end
                % Verify solution
                if min(eig(obj.P_matrix)) < -1e-10
                    warning('Riccati solution may not be positive semi-definite');
                end
            catch ME
                error('Failed to solve Riccati equation: %s', ME.message);
            end
        end
        function P = solveRiccatiSchur(obj)
            % Solve Riccati equation using Schur decomposition
            % This is a simplified implementation
            A = obj.system.A;
            B = obj.system.B;
            Q = obj.Q_matrix;
            R = obj.R_matrix;
            N = obj.N_matrix;
            % Form Hamiltonian matrix
            R_inv = inv(R);
            H = [A - B * R_inv * N', -B * R_inv * B'; ...
                 -Q + N * R_inv * N', -(A - B * R_inv * N')'];
            % Schur decomposition
            [U, T] = schur(H, 'real');
            % Select stable eigenvalues
            n = obj.system.n_states;
            stable_idx = real(diag(T)) < 0;
            if sum(stable_idx) ~= n
                error('Cannot find n stable eigenvalues for Riccati solution');
            end
            % Reorder Schur form
            [U, T] = ordschur(U, T, stable_idx);
            % Extract solution
            U11 = U(1:n, 1:n);
            U21 = U(n+1:2*n, 1:n);
            if rcond(U11) < 1e-12
                error('Riccati solution is singular');
            end
            P = real(U21 / U11);
        end
    end
end