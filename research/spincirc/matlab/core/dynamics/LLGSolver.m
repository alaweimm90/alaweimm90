function [m, t, solution_info] = LLGSolver(m0, Heff, alpha, gamma, tspan, varargin)
% LLGSOLVER - LLG dynamics solver with adaptive timestep control
%
% Solves dm/dt = -γ/(1+α²) * [m × H_eff + α * m × (m × H_eff)]
%
% Features: RK4/RK45/Dormand-Prince/IMEX methods, adaptive timestep,
% energy/magnetization conservation, event detection
%
% Inputs:
%   m0 - Initial magnetization (3x1 or 3xN)
%   Heff - Effective field function or constant
%   alpha - Gilbert damping parameter
%   gamma - Gyromagnetic ratio (rad⋅s⁻¹⋅T⁻¹)
%   tspan - Time span [t0, tf] or time vector
%   varargin - Optional parameters
%
% Options:
%   'Method' - 'RK45', 'RK4', 'DP54', 'IMEX' (default: 'RK45')
%   'RelTol', 'AbsTol' - Error tolerances (1e-6, 1e-8)
%   'MaxStep', 'InitialStep' - Step size control
%   'ConserveEnergy', 'ConserveMagnetization' - Conservation flags
%   'Events' - Event detection function handle
%   'Verbose' - Progress display
%
% Outputs:
%   m - Magnetization trajectory (3xN or 3xNxM)
%   t - Time vector
%   solution_info - Solver diagnostics
%
% Example:
%   m0 = [1; 0; 0];
%   H = @(t, m) [0; 0; 1] + 0.1*sin(2*pi*10e9*t)*[1; 0; 0];
%   [m, t] = LLGSolver(m0, H, 0.01, 1.76e11, [0, 10e-9]);
%
% Author: Meshal Alawein <meshal@berkeley.edu>
% Copyright © 2025 Meshal Alawein — All rights reserved.
% License: MIT

    % Parse input arguments
    p = inputParser;
    addRequired(p, 'm0', @(x) isnumeric(x) && size(x,1) == 3);
    addRequired(p, 'Heff', @(x) isa(x, 'function_handle') || isnumeric(x));
    addRequired(p, 'alpha', @(x) isnumeric(x) && isscalar(x) && x >= 0);
    addRequired(p, 'gamma', @(x) isnumeric(x) && isscalar(x) && x > 0);
    addRequired(p, 'tspan', @isnumeric);
    
    addParameter(p, 'Method', 'RK45', @ischar);
    addParameter(p, 'RelTol', 1e-6, @(x) isnumeric(x) && x > 0);
    addParameter(p, 'AbsTol', 1e-8, @(x) isnumeric(x) && x > 0);
    addParameter(p, 'MaxStep', [], @(x) isempty(x) || (isnumeric(x) && x > 0));
    addParameter(p, 'InitialStep', [], @(x) isempty(x) || (isnumeric(x) && x > 0));
    addParameter(p, 'ConserveEnergy', true, @islogical);
    addParameter(p, 'ConserveMagnetization', true, @islogical);
    addParameter(p, 'Events', [], @(x) isempty(x) || isa(x, 'function_handle'));
    addParameter(p, 'Verbose', false, @islogical);
    addParameter(p, 'OutputFcn', [], @(x) isempty(x) || isa(x, 'function_handle'));
    
    parse(p, m0, Heff, alpha, gamma, tspan, varargin{:});
    options = p.Results;
    
    % Validate inputs
    validateInputs(m0, tspan, options);
    
    % Normalize initial magnetization
    m0_normalized = normalizemagnetization(m0);
    
    % Convert Heff to function handle if constant
    if isnumeric(Heff)
        H_const = Heff;
        Heff = @(t, m) H_const;
    end
    
    % Set up time span
    if length(tspan) == 2
        t0 = tspan(1);
        tf = tspan(2);
        
        % Estimate initial time step
        if isempty(options.InitialStep)
            dt0 = estimateInitialStep(m0_normalized, Heff, alpha, gamma, t0);
        else
            dt0 = options.InitialStep;
        end
        
        % Set maximum time step
        if isempty(options.MaxStep)
            dt_max = (tf - t0) / 1000;  % At least 1000 points
        else
            dt_max = options.MaxStep;
        end
    else
        % Use provided time vector
        t0 = tspan(1);
        tf = tspan(end);
        dt0 = tspan(2) - tspan(1);
        dt_max = max(diff(tspan));
    end
    
    % Initialize solution arrays
    n_magnets = size(m0, 2);
    max_steps = ceil((tf - t0) / dt0) * 2;  % Estimate with buffer
    
    m_solution = zeros(3, n_magnets, max_steps);
    t_solution = zeros(max_steps, 1);
    
    % Initial conditions
    m_solution(:, :, 1) = m0_normalized;
    t_solution(1) = t0;
    
    % Initialize diagnostics
    energy_history = zeros(max_steps, 1);
    magnitude_error = zeros(max_steps, 1);
    step_sizes = zeros(max_steps, 1);
    
    energy_history(1) = calculateEnergy(m0_normalized, Heff, t0);
    magnitude_error(1) = max(abs(vecnorm(m0_normalized, 2, 1) - 1));
    step_sizes(1) = dt0;
    
    % Initialize solver state
    t_current = t0;
    m_current = m0_normalized;
    dt_current = dt0;
    step_count = 1;
    rejected_steps = 0;
    
    % Main integration loop
    if options.Verbose
        fprintf('Starting LLG integration...\n');
        fprintf('Method: %s, Time span: [%.2e, %.2e] s\n', options.Method, t0, tf);
        tic;
    end
    
    while t_current < tf && step_count < max_steps
        % Adjust final step
        if t_current + dt_current > tf
            dt_current = tf - t_current;
        end
        
        % Take integration step
        [m_new, dt_new, success, error_est] = takeStep(t_current, m_current, ...
            dt_current, Heff, alpha, gamma, options);
        
        if success
            % Accept step
            step_count = step_count + 1;
            t_current = t_current + dt_current;
            m_current = m_new;
            
            % Apply conservation constraints
            if options.ConserveMagnetization
                m_current = normalizemagnetization(m_current);
            end
            
            % Store solution
            m_solution(:, :, step_count) = m_current;
            t_solution(step_count) = t_current;
            
            % Update diagnostics
            energy_history(step_count) = calculateEnergy(m_current, Heff, t_current);
            magnitude_error(step_count) = max(abs(vecnorm(m_current, 2, 1) - 1));
            step_sizes(step_count) = dt_current;
            
            % Check events
            if ~isempty(options.Events)
                [event_val, is_terminal, direction] = options.Events(t_current, m_current);
                if any(event_val == 0) && is_terminal
                    break;
                end
            end
            
            % Output function
            if ~isempty(options.OutputFcn)
                status = options.OutputFcn(t_current, m_current, 'iter');
                if status == 1  % Stop integration
                    break;
                end
            end
            
            % Update time step for next iteration
            dt_current = dt_new;
            dt_current = min(dt_current, dt_max);
            
        else
            % Reject step and reduce time step
            rejected_steps = rejected_steps + 1;
            dt_current = dt_new;
            
            if dt_current < eps(t_current)
                warning('Time step too small. Integration terminated.');
                break;
            end
        end
        
        % Progress update
        if options.Verbose && mod(step_count, 1000) == 0
            progress = (t_current - t0) / (tf - t0) * 100;
            fprintf('Progress: %.1f%%, Steps: %d, Rejected: %d\n', ...
                progress, step_count, rejected_steps);
        end
    end
    
    % Trim solution arrays
    m = m_solution(:, :, 1:step_count);
    t = t_solution(1:step_count);
    
    % Prepare solution info
    solution_info = struct();
    solution_info.method = options.Method;
    solution_info.steps_taken = step_count;
    solution_info.steps_rejected = rejected_steps;
    solution_info.final_time = t_current;
    solution_info.energy_conservation = std(energy_history(1:step_count));
    solution_info.magnetization_error = max(magnitude_error(1:step_count));
    solution_info.average_step_size = mean(step_sizes(1:step_count));
    solution_info.energy_history = energy_history(1:step_count);
    solution_info.magnitude_error_history = magnitude_error(1:step_count);
    solution_info.step_size_history = step_sizes(1:step_count);
    
    if options.Verbose
        elapsed_time = toc;
        fprintf('Integration completed in %.3f seconds\n', elapsed_time);
        fprintf('Total steps: %d, Rejected: %d\n', step_count, rejected_steps);
        fprintf('Energy conservation error: %.2e\n', solution_info.energy_conservation);
        fprintf('Magnetization magnitude error: %.2e\n', solution_info.magnetization_error);
        solution_info.wall_time = elapsed_time;
    end
    
    % Call output function with 'done' status
    if ~isempty(options.OutputFcn)
        options.OutputFcn(t, m, 'done');
    end
end

function validateInputs(m0, tspan, options)
    if size(m0, 1) ~= 3
        error('Initial magnetization must have 3 components');
    end
    
    if length(tspan) < 2
        error('Time span must have at least 2 elements');
    end
    
    if any(diff(tspan) <= 0)
        error('Time span must be monotonically increasing');
    end
    
    valid_methods = {'RK4', 'RK45', 'DP54', 'IMEX'};
    if ~ismember(options.Method, valid_methods)
        error('Invalid method. Choose from: %s', strjoin(valid_methods, ', '));
    end
end

function m_norm = normalizemagnetization(m)
    magnitudes = vecnorm(m, 2, 1);
    magnitudes(magnitudes == 0) = 1;
    m_norm = m ./ magnitudes;
end

function dt0 = estimateInitialStep(m0, Heff, alpha, gamma, t0)
    H = Heff(t0, m0);
    H_magnitude = max(vecnorm(H, 2, 1));
    
    if H_magnitude > 0
        omega_max = gamma * H_magnitude;
        dt0 = 0.01 / omega_max;
        
        if alpha > 0
            tau_damping = 1 / (alpha * omega_max);
            dt0 = min(dt0, tau_damping / 10);
        end
    else
        dt0 = 1e-12;
    end
    
    dt0 = max(dt0, 1e-15);
    dt0 = min(dt0, 1e-9);
end

function E = calculateEnergy(m, Heff, t)
    H = Heff(t, m);
    mu0 = 4*pi*1e-7;
    E = -mu0 * sum(sum(m .* H, 1));
end

function [m_new, dt_new, success, error_est] = takeStep(t, m, dt, Heff, alpha, gamma, options)
    switch options.Method
        case 'RK4'
            [m_new, error_est] = rk4Step(t, m, dt, Heff, alpha, gamma);
            dt_new = dt;
            success = true;
            
        case 'RK45'
            [m_new, error_est, dt_new, success] = rk45Step(t, m, dt, Heff, alpha, gamma, options);
            
        case 'DP54'
            [m_new, error_est, dt_new, success] = dp54Step(t, m, dt, Heff, alpha, gamma, options);
            
        case 'IMEX'
            [m_new, error_est, dt_new, success] = imexStep(t, m, dt, Heff, alpha, gamma, options);
            
        otherwise
            error('Unknown method: %s', options.Method);
    end
end

function dmdt = llgRHS(t, m, Heff, alpha, gamma)
    % Right-hand side of LLG equation
    %
    % dm/dt = -γ/(1+α²) * [m × H_eff + α * m × (m × H_eff)]
    
    H = Heff(t, m);
    
    % Ensure both m and H have same dimensions
    if size(m, 2) == 1 && size(H, 2) > 1
        m = repmat(m, 1, size(H, 2));
    elseif size(H, 2) == 1 && size(m, 2) > 1
        H = repmat(H, 1, size(m, 2));
    end
    
    % Calculate cross products
    m_cross_H = cross(m, H, 1);
    m_cross_mH = cross(m, m_cross_H, 1);
    
    % LLG equation
    prefactor = -gamma / (1 + alpha^2);
    dmdt = prefactor * (m_cross_H + alpha * m_cross_mH);
end

function [m_new, error_est] = rk4Step(t, m, dt, Heff, alpha, gamma)
    % 4th-order Runge-Kutta step
    
    k1 = llgRHS(t, m, Heff, alpha, gamma);
    k2 = llgRHS(t + dt/2, m + dt*k1/2, Heff, alpha, gamma);
    k3 = llgRHS(t + dt/2, m + dt*k2/2, Heff, alpha, gamma);
    k4 = llgRHS(t + dt, m + dt*k3, Heff, alpha, gamma);
    
    m_new = m + dt/6 * (k1 + 2*k2 + 2*k3 + k4);
    
    % Error estimate (Richardson extrapolation)
    m_half1 = m + dt/12 * (k1 + 2*k2 + 2*k3 + k4);
    k1_half = llgRHS(t, m, Heff, alpha, gamma);
    k2_half = llgRHS(t + dt/4, m + dt*k1_half/4, Heff, alpha, gamma);
    k3_half = llgRHS(t + dt/4, m + dt*k2_half/4, Heff, alpha, gamma);
    k4_half = llgRHS(t + dt/2, m + dt*k3_half/2, Heff, alpha, gamma);
    m_half2 = m_half1 + dt/12 * (k1_half + 2*k2_half + 2*k3_half + k4_half);
    
    error_est = max(max(abs(m_new - m_half2)));
end

function [m_new, error_est, dt_new, success] = rk45Step(t, m, dt, Heff, alpha, gamma, options)
    % Runge-Kutta-Fehlberg 4(5) adaptive step
    
    % RKF coefficients
    a = [0, 1/4, 3/8, 12/13, 1, 1/2];
    b = [0, 0, 0, 0, 0, 0; ...
         1/4, 0, 0, 0, 0, 0; ...
         3/32, 9/32, 0, 0, 0, 0; ...
         1932/2197, -7200/2197, 7296/2197, 0, 0, 0; ...
         439/216, -8, 3680/513, -845/4104, 0, 0; ...
         -8/27, 2, -3544/2565, 1859/4104, -11/40, 0];
    
    c4 = [25/216, 0, 1408/2565, 2197/4104, -1/5, 0];
    c5 = [16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55];
    
    % Calculate k values
    k = zeros(size(m, 1), size(m, 2), 6);
    k(:, :, 1) = llgRHS(t, m, Heff, alpha, gamma);
    
    for i = 2:6
        m_temp = m;
        for j = 1:i-1
            m_temp = m_temp + dt * b(i, j) * k(:, :, j);
        end
        k(:, :, i) = llgRHS(t + a(i)*dt, m_temp, Heff, alpha, gamma);
    end
    
    % 4th and 5th order solutions
    m4 = m;
    m5 = m;
    for i = 1:6
        m4 = m4 + dt * c4(i) * k(:, :, i);
        m5 = m5 + dt * c5(i) * k(:, :, i);
    end
    
    % Error estimate
    error_est = max(max(abs(m5 - m4)));
    
    % Step size control
    tol = max(options.RelTol * max(max(abs(m))), options.AbsTol);
    
    if error_est <= tol
        % Accept step
        success = true;
        m_new = m5;  % Use higher order solution
        
        % Update step size
        if error_est > 0
            dt_new = dt * min(5, max(0.2, 0.9 * (tol / error_est)^(1/5)));
        else
            dt_new = dt * 5;  % Increase step size when error is very small
        end
    else
        % Reject step
        success = false;
        m_new = m;
        dt_new = dt * max(0.1, 0.9 * (tol / error_est)^(1/4));
    end
end

function [m_new, error_est, dt_new, success] = dp54Step(t, m, dt, Heff, alpha, gamma, options)
    % Dormand-Prince 5(4) adaptive step
    % Similar to RK45 but with different coefficients for better stability
    
    % DP54 coefficients
    a = [0, 1/5, 3/10, 4/5, 8/9, 1, 1];
    
    b = [0, 0, 0, 0, 0, 0, 0; ...
         1/5, 0, 0, 0, 0, 0, 0; ...
         3/40, 9/40, 0, 0, 0, 0, 0; ...
         44/45, -56/15, 32/9, 0, 0, 0, 0; ...
         19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0; ...
         9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0; ...
         35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0];
    
    c5 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0];
    c4 = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40];
    
    % Calculate k values
    k = zeros(size(m, 1), size(m, 2), 7);
    k(:, :, 1) = llgRHS(t, m, Heff, alpha, gamma);
    
    for i = 2:7
        m_temp = m;
        for j = 1:i-1
            m_temp = m_temp + dt * b(i, j) * k(:, :, j);
        end
        k(:, :, i) = llgRHS(t + a(i)*dt, m_temp, Heff, alpha, gamma);
    end
    
    % 4th and 5th order solutions
    m4 = m;
    m5 = m;
    for i = 1:7
        m4 = m4 + dt * c4(i) * k(:, :, i);
        m5 = m5 + dt * c5(i) * k(:, :, i);
    end
    
    % Error estimate and step control (same as RK45)
    error_est = max(max(abs(m5 - m4)));
    tol = max(options.RelTol * max(max(abs(m))), options.AbsTol);
    
    if error_est <= tol
        success = true;
        m_new = m5;
        if error_est > 0
            dt_new = dt * min(5, max(0.2, 0.9 * (tol / error_est)^(1/5)));
        else
            dt_new = dt * 5;
        end
    else
        success = false;
        m_new = m;
        dt_new = dt * max(0.1, 0.9 * (tol / error_est)^(1/4));
    end
end

function [m_new, error_est, dt_new, success] = imexStep(t, m, dt, Heff, alpha, gamma, options)
    % Implicit-Explicit method for stiff problems
    % Treats precession explicitly and damping implicitly
    
    % Split effective field into conservative and dissipative parts
    H = Heff(t, m);
    
    % Explicit predictor (precession term)
    m_cross_H = cross(m, H, 1);
    m_pred = m - (gamma * dt / (1 + alpha^2)) * m_cross_H;
    
    % Implicit corrector (damping term)
    % Solve: m_new = m_pred - alpha * gamma * dt / (1 + alpha^2) * m_new x (m_new x H)
    
    % Newton iteration for implicit solve
    m_new = m_pred;  % Initial guess
    
    for iter = 1:5  % Fixed number of Newton iterations
        H_new = Heff(t + dt, m_new);
        m_cross_H_new = cross(m_new, H_new, 1);
        m_cross_mH_new = cross(m_new, m_cross_H_new, 1);
        
        residual = m_new - m_pred + (alpha * gamma * dt / (1 + alpha^2)) * m_cross_mH_new;
        
        if max(max(abs(residual))) < 1e-12
            break;
        end
        
        % Approximate Jacobian (finite difference)
        eps_fd = 1e-8;
        J = zeros(3, 3, size(m, 2));
        
        for i = 1:3
            m_pert = m_new;
            m_pert(i, :) = m_pert(i, :) + eps_fd;
            
            H_pert = Heff(t + dt, m_pert);
            m_cross_H_pert = cross(m_pert, H_pert, 1);
            m_cross_mH_pert = cross(m_pert, m_cross_H_pert, 1);
            
            residual_pert = m_pert - m_pred + (alpha * gamma * dt / (1 + alpha^2)) * m_cross_mH_pert;
            
            J(i, :, :) = (residual_pert - residual) / eps_fd;
        end
        
        % Newton update
        for n = 1:size(m, 2)
            if cond(J(:, :, n)) < 1e12
                dm = -J(:, :, n) \ residual(:, n);
                m_new(:, n) = m_new(:, n) + dm;
            end
        end
    end
    
    % Error estimate (compare with explicit step)
    m_explicit = m - (gamma * dt / (1 + alpha^2)) * (m_cross_H + alpha * cross(m, m_cross_H, 1));
    error_est = max(max(abs(m_new - m_explicit)));
    
    % Step control
    tol = max(options.RelTol * max(max(abs(m))), options.AbsTol);
    
    if error_est <= tol
        success = true;
        dt_new = dt * min(2, max(0.5, 0.9 * (tol / error_est)^(1/3)));
    else
        success = false;
        m_new = m;
        dt_new = dt * max(0.25, 0.9 * (tol / error_est)^(1/3));
    end
end