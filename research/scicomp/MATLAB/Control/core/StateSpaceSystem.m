classdef StateSpaceSystem < handle
    % StateSpaceSystem - Linear time-invariant state-space system
    %
    % Represents systems of the form:
    %   dx/dt = Ax + Bu
    %   y = Cx + Du
    %
    % Features:
    % - System analysis (poles, zeros, stability)
    % - Controllability and observability analysis
    % - Time and frequency response simulation
    % - System transformations
    %
    % Example:
    %   A = [-1, 1; 0, -2];
    %   B = [0; 1];
    %   C = [1, 0];
    %   D = 0;
    %   sys = StateSpaceSystem(A, B, C, D);
    %   fprintf('System is stable: %d\n', sys.isStable());
    properties (Access = private)
        A_matrix    % System matrix (n x n)
        B_matrix    % Input matrix (n x m)
        C_matrix    % Output matrix (p x n)
        D_matrix    % Feedthrough matrix (p x m)
    end
    properties (Dependent)
        A           % System matrix (read-only)
        B           % Input matrix (read-only)
        C           % Output matrix (read-only)
        D           % Feedthrough matrix (read-only)
        n_states    % Number of states
        n_inputs    % Number of inputs
        n_outputs   % Number of outputs
    end
    methods
        function obj = StateSpaceSystem(A, B, C, D)
            % Constructor
            %
            % Parameters:
            %   A - System matrix (n x n)
            %   B - Input matrix (n x m)
            %   C - Output matrix (p x n)
            %   D - Feedthrough matrix (p x m)
            % Validate inputs
            validateattributes(A, {'numeric'}, {'2d', 'finite'});
            validateattributes(B, {'numeric'}, {'2d', 'finite'});
            validateattributes(C, {'numeric'}, {'2d', 'finite'});
            validateattributes(D, {'numeric'}, {'2d', 'finite'});
            % Check dimensions
            [n, n_check] = size(A);
            if n ~= n_check
                error('A matrix must be square');
            end
            if size(B, 1) ~= n
                error('B matrix must have same number of rows as A');
            end
            if size(C, 2) ~= n
                error('C matrix must have same number of columns as A');
            end
            if ~isequal(size(D), [size(C, 1), size(B, 2)])
                error('D matrix dimensions must be (p x m)');
            end
            % Check for numerical issues
            if any(isnan([A(:); B(:); C(:); D(:)]))
                error('System matrices contain NaN values');
            end
            if any(isinf([A(:); B(:); C(:); D(:)]))
                error('System matrices contain infinite values');
            end
            obj.A_matrix = A;
            obj.B_matrix = B;
            obj.C_matrix = C;
            obj.D_matrix = D;
        end
        % Dependent property getters
        function A = get.A(obj)
            A = obj.A_matrix;
        end
        function B = get.B(obj)
            B = obj.B_matrix;
        end
        function C = get.C(obj)
            C = obj.C_matrix;
        end
        function D = get.D(obj)
            D = obj.D_matrix;
        end
        function n = get.n_states(obj)
            n = size(obj.A_matrix, 1);
        end
        function m = get.n_inputs(obj)
            m = size(obj.B_matrix, 2);
        end
        function p = get.n_outputs(obj)
            p = size(obj.C_matrix, 1);
        end
        function poles = poles(obj)
            % Compute system poles (eigenvalues of A)
            poles = eig(obj.A_matrix);
        end
        function stable = isStable(obj)
            % Check if system is asymptotically stable
            poles = obj.poles();
            stable = all(real(poles) < 0);
        end
        function Wc = controllabilityMatrix(obj)
            % Compute controllability matrix
            n = obj.n_states;
            Wc = obj.B_matrix;
            A_power = eye(n);
            for i = 2:n
                A_power = A_power * obj.A_matrix;
                Wc = [Wc, A_power * obj.B_matrix];
            end
        end
        function controllable = isControllable(obj)
            % Check if system is completely controllable
            Wc = obj.controllabilityMatrix();
            controllable = rank(Wc) == obj.n_states;
        end
        function Wo = observabilityMatrix(obj)
            % Compute observability matrix
            n = obj.n_states;
            Wo = obj.C_matrix;
            A_power = eye(n);
            for i = 2:n
                A_power = A_power * obj.A_matrix;
                Wo = [Wo; obj.C_matrix * A_power];
            end
        end
        function observable = isObservable(obj)
            % Check if system is completely observable
            Wo = obj.observabilityMatrix();
            observable = rank(Wo) == obj.n_states;
        end
        function [x_response, y_response] = simulate(obj, t, u, x0)
            % Simulate system response
            %
            % Parameters:
            %   t - Time vector
            %   u - Input signal (matrix: length(t) x n_inputs)
            %   x0 - Initial state (optional, default: zeros)
            %
            % Returns:
            %   x_response - State response (length(t) x n_states)
            %   y_response - Output response (length(t) x n_outputs)
            if nargin < 4 || isempty(x0)
                x0 = zeros(obj.n_states, 1);
            end
            validateattributes(t, {'numeric'}, {'vector', 'increasing'});
            validateattributes(u, {'numeric'}, {'2d'});
            validateattributes(x0, {'numeric'}, {'vector'});
            if size(u, 1) ~= length(t)
                error('Input u must have same number of rows as length(t)');
            end
            if size(u, 2) ~= obj.n_inputs
                error('Input u must have n_inputs columns');
            end
            if length(x0) ~= obj.n_states
                error('Initial state x0 must have n_states elements');
            end
            % Use ode45 for integration
            [~, x_response] = ode45(@(t_val, x) obj.systemDynamics(t_val, x, t, u), t, x0(:));
            % Compute output response
            y_response = zeros(length(t), obj.n_outputs);
            for i = 1:length(t)
                u_val = interp1(t, u, t(i), 'linear', 'extrap');
                y_response(i, :) = (obj.C_matrix * x_response(i, :)' + obj.D_matrix * u_val')';
            end
        end
        function [x_response, y_response] = stepResponse(obj, t, input_channel)
            % Compute step response
            %
            % Parameters:
            %   t - Time vector
            %   input_channel - Which input to apply step to (default: 1)
            %
            % Returns:
            %   x_response - State response
            %   y_response - Output response
            if nargin < 3
                input_channel = 1;
            end
            validateattributes(input_channel, {'numeric'}, {'scalar', 'integer', 'positive'});
            if input_channel > obj.n_inputs
                error('input_channel exceeds number of inputs');
            end
            u = zeros(length(t), obj.n_inputs);
            u(:, input_channel) = 1.0;
            [x_response, y_response] = obj.simulate(t, u);
        end
        function [x_response, y_response] = impulseResponse(obj, t, input_channel)
            % Compute impulse response
            %
            % Parameters:
            %   t - Time vector
            %   input_channel - Which input to apply impulse to (default: 1)
            %
            % Returns:
            %   x_response - State response
            %   y_response - Output response
            if nargin < 3
                input_channel = 1;
            end
            validateattributes(input_channel, {'numeric'}, {'scalar', 'integer', 'positive'});
            if input_channel > obj.n_inputs
                error('input_channel exceeds number of inputs');
            end
            dt = t(2) - t(1);  % Assume uniform sampling
            u = zeros(length(t), obj.n_inputs);
            u(1, input_channel) = 1.0 / dt;  % Approximation of delta function
            [x_response, y_response] = obj.simulate(t, u);
        end
    end
    methods (Access = private)
        function x_dot = systemDynamics(obj, t_val, x, t, u)
            % System dynamics for ODE integration
            u_val = interp1(t, u, t_val, 'linear', 'extrap');
            x_dot = obj.A_matrix * x + obj.B_matrix * u_val';
        end
    end
end