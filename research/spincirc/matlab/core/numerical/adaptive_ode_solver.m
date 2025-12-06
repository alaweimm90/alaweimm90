function [t, y, info] = adaptive_ode_solver(ode_func, tspan, y0, varargin)
% ADAPTIVE_ODE_SOLVER - Advanced adaptive ODE solver with multiple methods
%
% This function provides multiple adaptive ODE integration methods optimized
% for spintronic device simulations, including stiff and non-stiff systems.
%
% Syntax:
%   [t, y, info] = adaptive_ode_solver(ode_func, tspan, y0)
%   [t, y, info] = adaptive_ode_solver(ode_func, tspan, y0, 'param', value, ...)
%
% Inputs:
%   ode_func - Function handle for ODE system: dy/dt = f(t, y)
%   tspan    - Time span [t_initial, t_final] or time vector
%   y0       - Initial conditions vector
%
% Optional Parameters:
%   'Method'       - Integration method ('DP54', 'RK4', 'IMEX', 'BDF') 
%   'RelTol'       - Relative tolerance (default: 1e-6)
%   'AbsTol'       - Absolute tolerance (default: 1e-9)
%   'InitialStep'  - Initial step size (auto if not specified)
%   'MaxStep'      - Maximum step size (default: inf)
%   'MinStep'      - Minimum step size (default: eps)
%   'MaxSteps'     - Maximum number of steps (default: 100000)
%   'Verbose'      - Display progress (default: false)
%   'EventFcn'     - Event function for root finding
%   'Jacobian'     - Jacobian matrix function (for stiff methods)
%
% Outputs:
%   t    - Time vector
%   y    - Solution matrix (each column is solution at corresponding time)
%   info - Information structure with solver statistics
%
% Author: Meshal Alawein <meshal@berkeley.edu>
% Copyright © 2025 Meshal Alawein — All rights reserved.
% License: MIT

    % Parse input arguments
    p = inputParser;
    addRequired(p, 'ode_func', @(x) isa(x, 'function_handle'));
    addRequired(p, 'tspan', @(x) isnumeric(x) && length(x) >= 2);
    addRequired(p, 'y0', @(x) isnumeric(x) && isvector(x));
    
    addParameter(p, 'Method', 'DP54', @(x) ischar(x) || isstring(x));
    addParameter(p, 'RelTol', 1e-6, @(x) isscalar(x) && x > 0);
    addParameter(p, 'AbsTol', 1e-9, @(x) isscalar(x) && x > 0);
    addParameter(p, 'InitialStep', [], @(x) isempty(x) || (isscalar(x) && x > 0));
    addParameter(p, 'MaxStep', inf, @(x) isscalar(x) && x > 0);
    addParameter(p, 'MinStep', eps, @(x) isscalar(x) && x > 0);
    addParameter(p, 'MaxSteps', 100000, @(x) isscalar(x) && x > 0);
    addParameter(p, 'Verbose', false, @islogical);
    addParameter(p, 'EventFcn', [], @(x) isempty(x) || isa(x, 'function_handle'));
    addParameter(p, 'Jacobian', [], @(x) isempty(x) || isa(x, 'function_handle'));
    
    parse(p, ode_func, tspan, y0, varargin{:});
    
    % Extract parameters
    method = p.Results.Method;
    rel_tol = p.Results.RelTol;
    abs_tol = p.Results.AbsTol;
    h_init = p.Results.InitialStep;
    h_max = p.Results.MaxStep;
    h_min = p.Results.MinStep;
    max_steps = p.Results.MaxSteps;
    verbose = p.Results.Verbose;
    event_fcn = p.Results.EventFcn;
    jacobian_fcn = p.Results.Jacobian;
    
    % Initialize
    y0 = y0(:);  % Ensure column vector
    n = length(y0);
    
    if length(tspan) == 2
        t_start = tspan(1);
        t_end = tspan(2);
        output_times = [];
    else
        t_start = tspan(1);
        t_end = tspan(end);
        output_times = tspan(:);
    end
    
    % Initialize solution storage
    t = t_start;
    y = y0;
    
    % Initialize step size
    if isempty(h_init)
        h_init = estimate_initial_step(ode_func, t_start, y0, rel_tol);
    end
    h = min(h_init, h_max);
    
    % Initialize info structure
    info = struct();
    info.method = method;
    info.rel_tol = rel_tol;
    info.abs_tol = abs_tol;
    info.steps_taken = 0;
    info.steps_rejected = 0;
    info.function_evaluations = 0;
    info.jacobian_evaluations = 0;
    info.last_step_size = h;
    info.success = false;
    
    if verbose
        fprintf('Starting adaptive ODE integration with method: %s\n', method);
        fprintf('Time span: [%.3e, %.3e], Initial step: %.3e\n', t_start, t_end, h);
    end
    
    % Main integration loop
    t_current = t_start;
    y_current = y0;
    step_count = 0;
    
    while t_current < t_end && step_count < max_steps
        
        % Adjust step size to not overshoot
        h = min(h, t_end - t_current);
        
        % Take integration step
        switch upper(method)
            case 'DP54'
                [t_new, y_new, h_new, accepted, stats] = dormand_prince_step(...
                    ode_func, t_current, y_current, h, rel_tol, abs_tol, h_min, h_max);
            case 'RK4'
                [t_new, y_new, h_new, accepted, stats] = runge_kutta_4_step(...
                    ode_func, t_current, y_current, h, rel_tol, abs_tol, h_min, h_max);
            case 'IMEX'
                [t_new, y_new, h_new, accepted, stats] = imex_step(...
                    ode_func, jacobian_fcn, t_current, y_current, h, rel_tol, abs_tol, h_min, h_max);
            case 'BDF'
                [t_new, y_new, h_new, accepted, stats] = bdf_step(...
                    ode_func, jacobian_fcn, t_current, y_current, h, rel_tol, abs_tol, h_min, h_max);
            otherwise
                error('Unknown integration method: %s', method);
        end
        
        % Update statistics
        info.function_evaluations = info.function_evaluations + stats.f_evals;
        if isfield(stats, 'j_evals')
            info.jacobian_evaluations = info.jacobian_evaluations + stats.j_evals;
        end
        
        if accepted
            % Step accepted
            t_current = t_new;
            y_current = y_new;
            
            % Store solution
            t = [t; t_current];
            y = [y, y_current];
            
            info.steps_taken = info.steps_taken + 1;
            
            % Check for events
            if ~isempty(event_fcn)
                [event_occurred, t_event, y_event] = check_events(event_fcn, ...
                    t(end-1), y(:,end-1), t_current, y_current);
                
                if event_occurred
                    t = [t(1:end-1); t_event];
                    y = [y(:,1:end-1), y_event];
                    info.event_time = t_event;
                    info.event_occurred = true;
                    break;
                end
            end
            
            % Progress reporting
            if verbose && mod(info.steps_taken, 1000) == 0
                fprintf('Step %d: t = %.3e, h = %.3e\n', info.steps_taken, t_current, h);
            end
            
        else
            % Step rejected
            info.steps_rejected = info.steps_rejected + 1;
        end
        
        % Update step size for next step
        h = h_new;
        step_count = step_count + 1;
        
        % Check for minimum step size
        if h < h_min
            warning('Step size fell below minimum. Integration may be inaccurate.');
            break;
        end
    end
    
    % Finalize info
    info.last_step_size = h;
    info.success = (t_current >= t_end - 1e-12);
    info.final_time = t_current;
    
    if step_count >= max_steps
        warning('Maximum number of steps exceeded.');
    end
    
    % Interpolate to requested output times if specified
    if ~isempty(output_times) && length(output_times) > 2
        y_interp = interp_solution(t, y, output_times, method);
        t = output_times;
        y = y_interp;
    end
    
    if verbose
        fprintf('Integration completed. Steps taken: %d, rejected: %d\n', ...
            info.steps_taken, info.steps_rejected);
        fprintf('Function evaluations: %d\n', info.function_evaluations);
    end
end

function h0 = estimate_initial_step(ode_func, t0, y0, rtol)
    % Estimate initial step size using the algorithm from Hairer & Wanner
    
    d0 = norm(y0);
    d1 = norm(ode_func(t0, y0));
    
    if d0 < 1e-5 || d1 < 1e-5
        h0 = 1e-6;
    else
        h0 = 0.01 * d0 / d1;
    end
    
    % Perform one explicit Euler step
    y1 = y0 + h0 * ode_func(t0, y0);
    d2 = norm(ode_func(t0 + h0, y1) - ode_func(t0, y0)) / h0;
    
    if max(d1, d2) <= 1e-15
        h1 = max(1e-6, h0 * 1e-3);
    else
        h1 = (0.01 / max(d1, d2))^(1/5);
    end
    
    h0 = min(100 * h0, h1, rtol^0.5);
end

function [t_new, y_new, h_new, accepted, stats] = dormand_prince_step(ode_func, t, y, h, rtol, atol, h_min, h_max)
    % Dormand-Prince 5(4) adaptive step
    
    % Butcher tableau coefficients
    a21 = 1/5;
    a31 = 3/40; a32 = 9/40;
    a41 = 44/45; a42 = -56/15; a43 = 32/9;
    a51 = 19372/6561; a52 = -25360/2187; a53 = 64448/6561; a54 = -212/729;
    a61 = 9017/3168; a62 = -355/33; a63 = 46732/5247; a64 = 49/176; a65 = -5103/18656;
    a71 = 35/384; a72 = 0; a73 = 500/1113; a74 = 125/192; a75 = -2187/6784; a76 = 11/84;
    
    % Alternative b coefficients for error estimation
    b1 = 5179/57600; b2 = 0; b3 = 7571/16695; b4 = 393/640; b5 = -92097/339200; b6 = 187/2100; b7 = 1/40;
    
    % Stage calculations
    k1 = ode_func(t, y);
    k2 = ode_func(t + h/5, y + h * a21*k1);
    k3 = ode_func(t + 3*h/10, y + h * (a31*k1 + a32*k2));
    k4 = ode_func(t + 4*h/5, y + h * (a41*k1 + a42*k2 + a43*k3));
    k5 = ode_func(t + 8*h/9, y + h * (a51*k1 + a52*k2 + a53*k3 + a54*k4));
    k6 = ode_func(t + h, y + h * (a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5));
    k7 = ode_func(t + h, y + h * (a71*k1 + a72*k2 + a73*k3 + a74*k4 + a75*k5 + a76*k6));
    
    % 5th order solution
    y5 = y + h * (a71*k1 + a73*k3 + a74*k4 + a75*k5 + a76*k6);
    
    % 4th order solution for error estimation
    y4 = y + h * (b1*k1 + b3*k3 + b4*k4 + b5*k5 + b6*k6 + b7*k7);
    
    % Error estimation
    error_est = norm(y5 - y4) / (atol + rtol * max(norm(y), norm(y5)));
    
    % Step size control
    safety = 0.9;
    if error_est <= 1
        % Accept step
        accepted = true;
        t_new = t + h;
        y_new = y5;
        
        % Update step size
        if error_est == 0
            h_new = h_max;
        else
            h_new = min(h_max, max(h_min, safety * h * (1/error_est)^(1/5)));
        end
    else
        % Reject step
        accepted = false;
        t_new = t;
        y_new = y;
        h_new = max(h_min, safety * h * (1/error_est)^(1/5));
    end
    
    stats.f_evals = 7;
end

function [t_new, y_new, h_new, accepted, stats] = runge_kutta_4_step(ode_func, t, y, h, rtol, atol, h_min, h_max)
    % Classical 4th order Runge-Kutta with adaptive step control
    
    % RK4 step with step size h
    k1 = ode_func(t, y);
    k2 = ode_func(t + h/2, y + h/2 * k1);
    k3 = ode_func(t + h/2, y + h/2 * k2);
    k4 = ode_func(t + h, y + h * k3);
    
    y_h = y + h/6 * (k1 + 2*k2 + 2*k3 + k4);
    
    % RK4 step with step size h/2 (twice)
    k1_half = k1;
    k2_half = ode_func(t + h/4, y + h/4 * k1_half);
    k3_half = ode_func(t + h/4, y + h/4 * k2_half);
    k4_half = ode_func(t + h/2, y + h/2 * k3_half);
    
    y_half = y + h/12 * (k1_half + 2*k2_half + 2*k3_half + k4_half);
    
    k1_half2 = ode_func(t + h/2, y_half);
    k2_half2 = ode_func(t + 3*h/4, y_half + h/4 * k1_half2);
    k3_half2 = ode_func(t + 3*h/4, y_half + h/4 * k2_half2);
    k4_half2 = ode_func(t + h, y_half + h/2 * k3_half2);
    
    y_h2 = y_half + h/12 * (k1_half2 + 2*k2_half2 + 2*k3_half2 + k4_half2);
    
    % Error estimation using Richardson extrapolation
    error_est = norm(y_h2 - y_h) / (15 * (atol + rtol * max(norm(y), norm(y_h2))));
    
    % Step control
    if error_est <= 1
        accepted = true;
        t_new = t + h;
        y_new = y_h2;  % Use more accurate solution
        
        if error_est == 0
            h_new = h_max;
        else
            h_new = min(h_max, max(h_min, 0.9 * h * (1/error_est)^0.25));
        end
    else
        accepted = false;
        t_new = t;
        y_new = y;
        h_new = max(h_min, 0.9 * h * (1/error_est)^0.25);
    end
    
    stats.f_evals = 8;
end

function [t_new, y_new, h_new, accepted, stats] = imex_step(ode_func, jac_func, t, y, h, rtol, atol, h_min, h_max)
    % IMEX (Implicit-Explicit) step for stiff systems
    % TODO: Implement higher-order IMEX schemes (e.g., IMEX-RK)
    % This is a simplified implementation - more sophisticated methods exist
    
    if isempty(jac_func)
        error('Jacobian function required for IMEX method');
    end
    
    % Explicit predictor (forward Euler)
    f_n = ode_func(t, y);
    y_pred = y + h * f_n;
    
    % Implicit corrector (backward Euler)
    J = jac_func(t + h, y_pred);
    
    % Newton iteration for implicit part
    max_newton_iter = 10;
    newton_tol = 1e-12;
    
    y_new = y_pred;
    for iter = 1:max_newton_iter
        f_new = ode_func(t + h, y_new);
        residual = y_new - y - h * f_new;
        
        if norm(residual) < newton_tol
            break;
        end
        
        % Solve linear system (I - h*J) * delta_y = -residual
        A = eye(length(y)) - h * J;
        delta_y = A \ (-residual);
        y_new = y_new + delta_y;
    end
    
    % Simple error estimation (can be improved)
    error_est = norm(y_new - y_pred) / (atol + rtol * max(norm(y), norm(y_new)));
    
    if error_est <= 1 || iter <= max_newton_iter
        accepted = true;
        t_new = t + h;
        h_new = min(h_max, max(h_min, 0.9 * h * min(2, (1/max(error_est, 1e-12))^0.5)));
    else
        accepted = false;
        t_new = t;
        y_new = y;
        h_new = max(h_min, 0.5 * h);
    end
    
    stats.f_evals = 2 + iter;
    stats.j_evals = 1;
end

function [t_new, y_new, h_new, accepted, stats] = bdf_step(ode_func, jac_func, t, y, h, rtol, atol, h_min, h_max)
    % Backward Differentiation Formula (simplified BDF2)
    % TODO: Implement variable-coefficient BDF methods
    
    if isempty(jac_func)
        error('Jacobian function required for BDF method');
    end
    
    % For first step, use backward Euler
    persistent y_prev t_prev h_prev
    
    if isempty(y_prev) || isempty(t_prev)
        % First step - use backward Euler
        y_pred = y + h * ode_func(t, y);
        
        % Newton iteration
        max_iter = 10;
        tol = 1e-12;
        y_new = y_pred;
        
        for iter = 1:max_iter
            f = ode_func(t + h, y_new);
            residual = y_new - y - h * f;
            
            if norm(residual) < tol
                break;
            end
            
            J = jac_func(t + h, y_new);
            A = eye(length(y)) - h * J;
            delta_y = A \ (-residual);
            y_new = y_new + delta_y;
        end
        
        accepted = true;
        t_new = t + h;
        
        % Store for next step
        y_prev = y;
        t_prev = t;
        h_prev = h;
        
    else
        % BDF2 step
        if abs(h - h_prev) > 1e-12 * h
            % Variable step size - use interpolation
            y_new = y + h * ode_func(t, y);  % Fall back to BE
        else
            % Constant step size BDF2
            y_pred = (4*y - y_prev) / 3 + 2*h/3 * ode_func(t + h, y);
            
            % Newton iteration
            max_iter = 10;
            tol = 1e-12;
            y_new = y_pred;
            
            for iter = 1:max_iter
                f = ode_func(t + h, y_new);
                residual = y_new - (4*y - y_prev)/3 - 2*h/3 * f;
                
                if norm(residual) < tol
                    break;
                end
                
                J = jac_func(t + h, y_new);
                A = eye(length(y)) - 2*h/3 * J;
                delta_y = A \ (-residual);
                y_new = y_new + delta_y;
            end
        end
        
        accepted = true;
        t_new = t + h;
        
        % Update history
        y_prev = y;
        t_prev = t;
        h_prev = h;
    end
    
    % Simple step size control
    h_new = h;  % Keep step size constant for BDF
    
    stats.f_evals = 2;
    stats.j_evals = 1;
end

function [occurred, t_event, y_event] = check_events(event_fcn, t1, y1, t2, y2)
    % Simple event detection using sign change
    
    [value1, isterminal1, direction1] = event_fcn(t1, y1);
    [value2, isterminal2, direction2] = event_fcn(t2, y2);
    
    occurred = false;
    t_event = t2;
    y_event = y2;
    
    % Check for sign changes
    for i = 1:length(value1)
        if isterminal1(i) && sign(value1(i)) ~= sign(value2(i))
            if (direction1(i) == 0) || (direction1(i) * (value2(i) - value1(i)) > 0)
                occurred = true;
                % Linear interpolation to find event time
                alpha = -value1(i) / (value2(i) - value1(i));
                t_event = t1 + alpha * (t2 - t1);
                y_event = y1 + alpha * (y2 - y1);
                break;
            end
        end
    end
end

function y_interp = interp_solution(t_sol, y_sol, t_out, method)
    % Interpolate solution to desired output times
    
    switch upper(method)
        case 'DP54'
            % Use cubic Hermite interpolation for high-order methods
            y_interp = zeros(size(y_sol, 1), length(t_out));
            for i = 1:size(y_sol, 1)
                y_interp(i, :) = pchip(t_sol, y_sol(i, :), t_out);
            end
        otherwise
            % Use linear interpolation for lower-order methods
            y_interp = zeros(size(y_sol, 1), length(t_out));
            for i = 1:size(y_sol, 1)
                y_interp(i, :) = interp1(t_sol, y_sol(i, :), t_out, 'linear');
            end
    end
end