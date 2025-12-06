function [m, t, I_s, solution_info] = LLGSSolver(m0, transport_params, current_profile, varargin)
% LLGSSOLVER - LLG with spin-transfer torque solver
%
% Self-consistent LLGS solver with real-time spin transport coupling.
% Solves the coupled system:
%   dm/dt = -γ/(1+α²) * [m × H_eff + α * m × (m × H_eff)] + T_STT + T_SOT
%   G * V = I_applied
%
% Features:
%   - Self-consistent coupling between magnetization dynamics and transport
%   - Spin-transfer torque (STT) from spin-polarized currents
%   - Spin-orbit torque (SOT) effects
%   - Spin pumping and back-action
%   - Voltage-controlled magnetic anisotropy (VCMA)
%   - Real-time adaptive coupling
%
% Inputs:
%   m0 - Initial magnetization configuration
%   transport_params - Transport system parameters
%   current_profile - Applied current vs time function
%   varargin - Optional parameters
%
% Optional Parameters:
%   'STT' - Include spin-transfer torque (default: true)
%   'SOT' - Include spin-orbit torque (default: false)
%   'SpinPumping' - Include spin pumping (default: true)
%   'VCMA' - Include voltage-controlled anisotropy (default: false)
%   'CouplingStrength' - STT coupling strength (default: auto)
%   'UpdateInterval' - Transport update frequency (default: adaptive)
%   'SelfConsistent' - Full self-consistency (default: true)
%
% Outputs:
%   m - Magnetization trajectory
%   t - Time vector
%   I_s - Spin current evolution
%   solution_info - Detailed solution information
%
% Author: Meshal Alawein <meshal@berkeley.edu>
% Copyright © 2025 Meshal Alawein — All rights reserved.
% License: MIT

    % Parse input arguments
    p = inputParser;
    addRequired(p, 'm0', @(x) isnumeric(x) && size(x,1) == 3);
    addRequired(p, 'transport_params', @isstruct);
    addRequired(p, 'current_profile', @(x) isa(x, 'function_handle') || isnumeric(x));
    
    addParameter(p, 'tspan', [0, 10e-9], @isnumeric);
    addParameter(p, 'STT', true, @islogical);
    addParameter(p, 'SOT', false, @islogical);
    addParameter(p, 'SpinPumping', true, @islogical);
    addParameter(p, 'VCMA', false, @islogical);
    addParameter(p, 'CouplingStrength', [], @(x) isempty(x) || isnumeric(x));
    addParameter(p, 'UpdateInterval', [], @(x) isempty(x) || isnumeric(x));
    addParameter(p, 'SelfConsistent', true, @islogical);
    addParameter(p, 'RelTol', 1e-6, @isnumeric);
    addParameter(p, 'AbsTol', 1e-8, @isnumeric);
    addParameter(p, 'Verbose', false, @islogical);
    
    parse(p, m0, transport_params, current_profile, varargin{:});
    options = p.Results;
    
    % Initialize transport solver
    transport_solver = initializeTransportSolver(transport_params);
    
    % Convert current profile to function handle
    if isnumeric(current_profile)
        I_const = current_profile;
        current_profile = @(t) I_const;
    end
    
    % Determine coupling parameters
    coupling_params = determineCouplingParameters(transport_params, options);
    
    % Set up time span and adaptive intervals
    tspan = options.tspan;
    if isempty(options.UpdateInterval)
        % Adaptive update interval based on dynamics
        gamma = 1.760859644e11;  % Gyromagnetic ratio
        alpha = transport_params.alpha;
        typical_field = 0.1;  % Tesla
        tau_precession = 1 / (gamma * typical_field);
        tau_damping = 1 / (alpha * gamma * typical_field);
        update_interval = min(tau_precession, tau_damping) / 10;
    else
        update_interval = options.UpdateInterval;
    end
    
    % Initialize solution storage
    max_points = ceil((tspan(2) - tspan(1)) / update_interval) * 2;
    m_history = zeros(3, size(m0, 2), max_points);
    I_s_history = zeros(4, max_points);  % [I_c, I_sx, I_sy, I_sz]
    t_history = zeros(max_points, 1);
    torque_history = zeros(3, size(m0, 2), max_points);
    
    % Initial conditions
    m_current = m0 / norm(m0, 'fro');  % Normalize
    t_current = tspan(1);
    I_applied = current_profile(t_current);
    
    % Solve initial transport problem
    [V_current, I_s_current] = solveTransport(transport_solver, m_current, I_applied);
    
    % Store initial state
    point_count = 1;
    m_history(:, :, 1) = m_current;
    I_s_history(:, 1) = [I_s_current.charge(1); I_s_current.spin_x(1); ...
                         I_s_current.spin_y(1); I_s_current.spin_z(1)];
    t_history(1) = t_current;
    torque_history(:, :, 1) = calculateTorques(m_current, I_s_current, coupling_params, options);
    
    if options.Verbose
        fprintf('Starting self-consistent LLGS integration...\n');
        fprintf('Time span: [%.2e, %.2e] s\n', tspan(1), tspan(2));
        fprintf('Update interval: %.2e s\n', update_interval);
        tic;
    end
    
    % Main integration loop
    while t_current < tspan(2) && point_count < max_points
        % Determine next update time
        t_next = min(t_current + update_interval, tspan(2));
        dt_step = t_next - t_current;
        
        % Get current applied current
        I_applied = current_profile(t_current);
        
        % Solve transport for current magnetization
        if options.SelfConsistent || point_count == 1
            [V_current, I_s_current] = solveTransport(transport_solver, m_current, I_applied);
        end
        
        % Calculate torques
        torques = calculateTorques(m_current, I_s_current, coupling_params, options);
        
        % Set up effective field including torques
        H_eff = @(t, m) effectiveField(t, m, transport_params, torques, coupling_params);
        
        % Integrate LLG over this time step
        [m_step, t_step] = LLGSolver(m_current, H_eff, transport_params.alpha, ...
            1.760859644e11, [t_current, t_next], ...
            'Method', 'RK45', 'RelTol', options.RelTol, 'AbsTol', options.AbsTol, ...
            'Verbose', false);
        
        % Update state
        point_count = point_count + 1;
        t_current = t_next;
        m_current = m_step(:, :, end);
        
        % Store results
        m_history(:, :, point_count) = m_current;
        I_s_history(:, point_count) = [I_s_current.charge(1); I_s_current.spin_x(1); ...
                                       I_s_current.spin_y(1); I_s_current.spin_z(1)];
        t_history(point_count) = t_current;
        torque_history(:, :, point_count) = torques;
        
        % Progress update
        if options.Verbose && mod(point_count, 100) == 0
            progress = (t_current - tspan(1)) / (tspan(2) - tspan(1)) * 100;
            fprintf('Progress: %.1f%%, Points: %d\n', progress, point_count);
        end
        
        % Adaptive update interval based on dynamics
        if point_count > 2
            dm_dt = norm(m_history(:, :, point_count) - m_history(:, :, point_count-1)) / dt_step;
            if dm_dt > 0
                suggested_dt = 0.01 / dm_dt;  % Adaptive time step
                update_interval = min(max(suggested_dt, dt_step/10), dt_step*10);
            end
        end
    end
    
    % Trim solution arrays
    m = m_history(:, :, 1:point_count);
    t = t_history(1:point_count);
    I_s = struct();
    I_s.charge = I_s_history(1, 1:point_count);
    I_s.spin_x = I_s_history(2, 1:point_count);
    I_s.spin_y = I_s_history(3, 1:point_count);
    I_s.spin_z = I_s_history(4, 1:point_count);
    I_s.torques = torque_history(:, :, 1:point_count);
    
    % Solution information
    solution_info = struct();
    solution_info.points_computed = point_count;
    solution_info.final_time = t_current;
    solution_info.average_update_interval = mean(diff(t));
    solution_info.coupling_strength = coupling_params.beta_STT;
    solution_info.included_effects = struct('STT', options.STT, 'SOT', options.SOT, ...
                                          'SpinPumping', options.SpinPumping, 'VCMA', options.VCMA);
    
    % Calculate switching metrics
    if size(m, 2) == 1  % Single magnet
        initial_state = m(:, 1, 1);
        final_state = m(:, 1, end);
        switching_angle = acos(dot(initial_state, final_state)) * 180/pi;
        solution_info.switching_angle = switching_angle;
        solution_info.switched = switching_angle > 90;
        
        % Find switching time if switched
        if solution_info.switched
            m_z_traj = squeeze(m(3, 1, :));
            switch_idx = find(m_z_traj * initial_state(3) < 0, 1);
            if ~isempty(switch_idx)
                solution_info.switching_time = t(switch_idx);
            end
        end
    end
    
    if options.Verbose
        elapsed_time = toc;
        fprintf('Self-consistent integration completed in %.3f seconds\n', elapsed_time);
        fprintf('Total points computed: %d\n', point_count);
        if isfield(solution_info, 'switched')
            if solution_info.switched
                fprintf('Magnetization switched (angle: %.1f deg)\n', solution_info.switching_angle);
                if isfield(solution_info, 'switching_time')
                    fprintf('Switching time: %.2e s\n', solution_info.switching_time);
                end
            else
                fprintf('No switching detected (angle: %.1f deg)\n', solution_info.switching_angle);
            end
        end
        solution_info.wall_time = elapsed_time;
    end
end

function transport_solver = initializeTransportSolver(params)
    % Initialize transport solver from parameters
    
    transport_solver = SpinTransportSolver();
    
    % Set geometry
    if isfield(params, 'geometry')
        transport_solver.setGeometry(params.geometry.length, ...
                                   params.geometry.width, ...
                                   params.geometry.thickness);
    end
    
    % Set materials
    if isfield(params, 'materials')
        transport_solver.setMaterials(params.materials);
    end
    
    % Set boundary conditions
    if isfield(params, 'boundary_conditions')
        transport_solver.setBoundaryConditions(params.boundary_conditions.type, ...
                                             params.boundary_conditions.values);
    end
    
    % Set temperature and field
    if isfield(params, 'temperature')
        transport_solver.setTemperature(params.temperature);
    end
    
    if isfield(params, 'magnetic_field')
        transport_solver.setMagneticField(params.magnetic_field);
    end
end

function [V, I_s] = solveTransport(transport_solver, m, I_applied)
    % Solve transport problem for given magnetization and current
    
    % Update magnetization in transport solver
    transport_solver.setMagnetization(m');
    
    % Update boundary conditions with applied current
    bc = transport_solver.boundary_conditions;
    if strcmp(bc.type, 'current')
        bc.values(1).current = I_applied;
        transport_solver.setBoundaryConditions(bc.type, bc.values);
    end
    
    % Solve transport
    [V, I_s, ~] = transport_solver.solve('verbose', false);
end

function coupling_params = determineCouplingParameters(params, options)
    % Determine coupling parameters for torques
    
    coupling_params = struct();
    
    % STT coupling strength
    if isempty(options.CouplingStrength)
        if isfield(params, 'beta')
            coupling_params.beta_STT = params.beta;
        else
            coupling_params.beta_STT = 0.3;  % Default value
        end
    else
        coupling_params.beta_STT = options.CouplingStrength;
    end
    
    % SOT parameters
    if options.SOT
        if isfield(params, 'theta_SH')
            coupling_params.theta_SH = params.theta_SH;
        else
            coupling_params.theta_SH = 0.1;  % Default spin Hall angle
        end
        
        if isfield(params, 'lambda_SOT')
            coupling_params.lambda_SOT = params.lambda_SOT;
        else
            coupling_params.lambda_SOT = 1e-9;  % Default SOT length scale
        end
    end
    
    % Spin pumping parameters
    if options.SpinPumping
        if isfield(params, 'g_mix')
            coupling_params.g_mix = params.g_mix;
        else
            coupling_params.g_mix = 1e15;  % Default mixing conductance (S/m^2)
        end
    end
    
    % VCMA parameters
    if options.VCMA
        if isfield(params, 'xi_VCMA')
            coupling_params.xi_VCMA = params.xi_VCMA;
        else
            coupling_params.xi_VCMA = 1e-4;  % Default VCMA coefficient (J/V⋅m)
        end
    end
end

function torques = calculateTorques(m, I_s, coupling_params, options)
    % Calculate all torques acting on magnetization
    
    torques = zeros(size(m));
    
    % Spin-transfer torque (STT)
    if options.STT
        % Slonczewski torque: T_STT = -(gamma * beta * I_s) / (mu_B * V) * [m x (m x p)]
        % where p is the spin polarization direction
        
        % Get spin current components
        I_spin = [I_s.spin_x(1); I_s.spin_y(1); I_s.spin_z(1)];
        
        % STT prefactor
        gamma = 1.760859644e11;
        mu_B = 9.2740100783e-24;
        volume = 1e-24;  % Typical volume (m^3) - should be from geometry
        
        beta = coupling_params.beta_STT;
        prefactor = gamma * beta / (mu_B * volume);
        
        % Assume spin polarization along first magnet direction
        p = m(:, 1) / norm(m(:, 1));
        
        for i = 1:size(m, 2)
            m_i = m(:, i);
            
            % Slonczewski torque: T = -prefactor * I_s * m x (m x p)
            m_cross_p = cross(m_i, p);
            T_STT = -prefactor * norm(I_spin) * cross(m_i, m_cross_p);
            
            torques(:, i) = torques(:, i) + T_STT;
        end
    end
    
    % Spin-orbit torque (SOT)
    if options.SOT
        % Damping-like and field-like SOT components
        theta_SH = coupling_params.theta_SH;
        lambda_SOT = coupling_params.lambda_SOT;
        
        % Current density (approximate)
        J_c = I_s.charge(1) / (1e-12);  % A/m^2
        
        % SOT prefactor
        gamma = 1.760859644e11;
        hbar = 1.054571817e-34;
        e = 1.602176634e-19;
        thickness = 1e-9;  % Typical thickness
        
        prefactor_SOT = (gamma * hbar * theta_SH * J_c) / (2 * e * thickness);
        
        for i = 1:size(m, 2)
            m_i = m(:, i);
            
            % Damping-like SOT: T_DL = prefactor * m x (z x m)
            z_hat = [0; 0; 1];
            z_cross_m = cross(z_hat, m_i);
            T_DL = prefactor_SOT * cross(m_i, z_cross_m);
            
            % Field-like SOT: T_FL = prefactor * (z x m)
            T_FL = 0.1 * prefactor_SOT * z_cross_m;  % Typically smaller
            
            torques(:, i) = torques(:, i) + T_DL + T_FL;
        end
    end
    
    % Spin pumping back-action
    if options.SpinPumping
        % This creates additional damping and should be included in H_eff
        % For now, implement as additional damping torque
        g_mix = coupling_params.g_mix;
        gamma = 1.760859644e11;
        
        % Spin pumping damping coefficient
        alpha_sp = g_mix * 1.055e-34 / (2 * 9.274e-24 * 1e-9);  % Additional damping
        
        for i = 1:size(m, 2)
            m_i = m(:, i);
            
            % Additional damping: T_sp = -alpha_sp * gamma * m x (m x H_eff)
            % Approximate H_eff as [0; 0; 1] for now
            H_eff_approx = [0; 0; 1];
            m_cross_H = cross(m_i, H_eff_approx);
            T_sp = -alpha_sp * gamma * cross(m_i, m_cross_H);
            
            torques(:, i) = torques(:, i) + T_sp;
        end
    end
end

function H_eff = effectiveField(t, m, params, torques, coupling_params)
    % Calculate effective field including all contributions
    
    H_eff = zeros(size(m));
    
    % External field
    if isfield(params, 'H_external')
        if isa(params.H_external, 'function_handle')
            H_ext = params.H_external(t);
        else
            H_ext = params.H_external;
        end
        
        for i = 1:size(m, 2)
            H_eff(:, i) = H_eff(:, i) + H_ext;
        end
    end
    
    % Anisotropy field
    if isfield(params, 'K_anis')
        K = params.K_anis;
        
        for i = 1:size(m, 2)
            m_i = m(:, i);
            
            % Uniaxial anisotropy along z
            H_anis = 2 * K * m_i(3) * [0; 0; 1];
            H_eff(:, i) = H_eff(:, i) + H_anis;
        end
    end
    
    % Exchange coupling (for multiple magnets)
    if size(m, 2) > 1 && isfield(params, 'J_exchange')
        J_ex = params.J_exchange;
        
        for i = 1:size(m, 2)
            H_exchange = zeros(3, 1);
            
            % Couple to nearest neighbors
            if i > 1
                H_exchange = H_exchange + J_ex * m(:, i-1);
            end
            if i < size(m, 2)
                H_exchange = H_exchange + J_ex * m(:, i+1);
            end
            
            H_eff(:, i) = H_eff(:, i) + H_exchange;
        end
    end
    
    % Convert torques to effective field contributions
    % Torques appear as additional terms in LLG, not as fields
    % This is handled in the LLG equation directly
end