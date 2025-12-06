classdef PIDController < handle
    % PIDController - Professional PID controller implementation
    %
    % Features:
    % - Standard PID control with configurable gains
    % - Anti-windup protection
    % - Derivative filtering to reduce noise sensitivity
    % - Setpoint weighting for improved response
    % - Reset functionality
    %
    % Example:
    %   config = struct('kp', 2.0, 'ki', 0.5, 'kd', 0.1, 'dt', 0.01);
    %   controller = PIDController(config);
    %   output = controller.update(10.0, 8.5);
    properties (Access = private)
        config              % Configuration structure
        previous_error      % Previous error value
        integral            % Integral accumulator
        previous_measurement % Previous measurement
        previous_derivative % Previous derivative (for filtering)
    end
    methods
        function obj = PIDController(config)
            % Constructor
            %
            % Parameters:
            %   config - Configuration structure with fields:
            %            kp, ki, kd, dt, output_min, output_max,
            %            derivative_filter_tau
            validateattributes(config, {'struct'}, {'nonempty'});
            % Set default values
            defaults = struct(...
                'kp', 1.0, ...
                'ki', 0.0, ...
                'kd', 0.0, ...
                'dt', 0.01, ...
                'output_min', -Inf, ...
                'output_max', Inf, ...
                'derivative_filter_tau', 0.0 ...
            );
            % Merge with user config
            fields = fieldnames(defaults);
            for i = 1:length(fields)
                if ~isfield(config, fields{i})
                    config.(fields{i}) = defaults.(fields{i});
                end
            end
            obj.config = config;
            obj.reset();
        end
        function reset(obj)
            % Reset controller internal state
            obj.previous_error = 0.0;
            obj.integral = 0.0;
            obj.previous_measurement = 0.0;
            obj.previous_derivative = 0.0;
        end
        function output = update(obj, setpoint, measurement, dt)
            % Compute PID control output
            %
            % Parameters:
            %   setpoint - Desired value
            %   measurement - Current measured value
            %   dt - Time step (optional, uses config.dt if not provided)
            %
            % Returns:
            %   output - Control output
            if nargin < 4
                dt = obj.config.dt;
            end
            validateattributes(setpoint, {'numeric'}, {'scalar', 'finite'});
            validateattributes(measurement, {'numeric'}, {'scalar', 'finite'});
            validateattributes(dt, {'numeric'}, {'scalar', 'positive'});
            % Calculate error
            error = setpoint - measurement;
            % Proportional term
            proportional = obj.config.kp * error;
            % Integral term with anti-windup
            obj.integral = obj.integral + error * dt;
            % Anti-windup: clamp integral if output would saturate
            integral_term = obj.config.ki * obj.integral;
            provisional_output = proportional + integral_term;
            if provisional_output > obj.config.output_max
                obj.integral = (obj.config.output_max - proportional) / obj.config.ki;
            elseif provisional_output < obj.config.output_min
                obj.integral = (obj.config.output_min - proportional) / obj.config.ki;
            end
            integral = obj.config.ki * obj.integral;
            % Derivative term (on measurement to avoid derivative kick)
            derivative_raw = -(measurement - obj.previous_measurement) / dt;
            % Apply derivative filtering if specified
            if obj.config.derivative_filter_tau > 0
                alpha = dt / (obj.config.derivative_filter_tau + dt);
                derivative = alpha * derivative_raw + (1 - alpha) * obj.previous_derivative;
                obj.previous_derivative = derivative;
            else
                derivative = derivative_raw;
            end
            derivative_term = obj.config.kd * derivative;
            % Compute total output
            output = proportional + integral + derivative_term;
            % Apply output limits
            output = max(obj.config.output_min, min(obj.config.output_max, output));
            % Store values for next iteration
            obj.previous_error = error;
            obj.previous_measurement = measurement;
        end
        function [proportional, integral, derivative_term] = getComponents(obj, setpoint, measurement, dt)
            % Get individual PID components (for analysis/tuning)
            %
            % Parameters:
            %   setpoint - Desired value
            %   measurement - Current measured value
            %   dt - Time step (optional)
            %
            % Returns:
            %   proportional - Proportional component
            %   integral - Integral component
            %   derivative_term - Derivative component
            if nargin < 4
                dt = obj.config.dt;
            end
            error = setpoint - measurement;
            proportional = obj.config.kp * error;
            integral = obj.config.ki * obj.integral;
            derivative_raw = -(measurement - obj.previous_measurement) / dt;
            if obj.config.derivative_filter_tau > 0
                alpha = dt / (obj.config.derivative_filter_tau + dt);
                derivative = alpha * derivative_raw + (1 - alpha) * obj.previous_derivative;
            else
                derivative = derivative_raw;
            end
            derivative_term = obj.config.kd * derivative;
        end
        function new_config = tuneZieglerNichols(obj, ku, tu, method)
            % Auto-tune PID parameters using Ziegler-Nichols method
            %
            % Parameters:
            %   ku - Ultimate gain (gain at which system oscillates)
            %   tu - Ultimate period (period of oscillation)
            %   method - Tuning method ('classic', 'pessen', 'some_overshoot', 'no_overshoot')
            %
            % Returns:
            %   new_config - New configuration structure with tuned parameters
            if nargin < 4
                method = 'classic';
            end
            validateattributes(ku, {'numeric'}, {'scalar', 'positive'});
            validateattributes(tu, {'numeric'}, {'scalar', 'positive'});
            validatestring(method, {'classic', 'pessen', 'some_overshoot', 'no_overshoot'});
            switch method
                case 'classic'
                    kp = 0.6 * ku;
                    ki = 2.0 * kp / tu;
                    kd = kp * tu / 8.0;
                case 'pessen'
                    kp = 0.7 * ku;
                    ki = 2.5 * kp / tu;
                    kd = 0.15 * kp * tu;
                case 'some_overshoot'
                    kp = 0.33 * ku;
                    ki = 2.0 * kp / tu;
                    kd = kp * tu / 3.0;
                case 'no_overshoot'
                    kp = 0.2 * ku;
                    ki = 2.0 * kp / tu;
                    kd = kp * tu / 3.0;
            end
            new_config = obj.config;
            new_config.kp = kp;
            new_config.ki = ki;
            new_config.kd = kd;
        end
    end
end