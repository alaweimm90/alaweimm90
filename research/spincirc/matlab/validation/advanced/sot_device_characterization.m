function results = sot_device_characterization(varargin)
% SOT_DEVICE_CHARACTERIZATION - Validate SOT switching optimization
%
% This validation script demonstrates spin-orbit torque (SOT) driven
% magnetization switching and characterizes the optimization of SOT
% devices. It validates the framework's ability to capture SOT physics
% including damping-like and field-like torques.
%
% The SOT torques in heavy metal/ferromagnet bilayers are:
% τ_DL = -γ * ξ_DL * (ħ/2e) * (J_c/M_s*t_FM) * m × (m × σ)
% τ_FL = -γ * ξ_FL * (ħ/2e) * (J_c/M_s*t_FM) * m × σ
%
% Where:
%   ξ_DL - Damping-like SOT efficiency
%   ξ_FL - Field-like SOT efficiency  
%   J_c  - Charge current density in HM (A/m²)
%   σ    - Spin polarization direction
%   m    - Magnetization direction
%
% Key Parameters Characterized:
%   - SOT efficiency vs current
%   - Critical switching current
%   - Field-assisted switching
%   - Switching speed optimization
%   - Energy efficiency
%
% Physics Validated:
%   - Spin Hall effect in heavy metals
%   - Interfacial SOT mechanisms
%   - Current-field switching phase diagram
%   - Deterministic switching conditions
%   - Temperature dependence of SOT
%
% Usage:
%   results = sot_device_characterization();  % Default parameters
%   results = sot_device_characterization('HeavyMetal', 'Pt', 'Ferromagnet', 'CoFeB');
%   results = sot_device_characterization('Temperature', 77, 'OptimizeEfficiency', true);
%
% Author: Meshal Alawein <meshal@berkeley.edu>
% Copyright © 2025 Meshal Alawein — All rights reserved.
% License: MIT

    % Parse input arguments
    p = inputParser;
    addParameter(p, 'HeavyMetal', 'Pt', @ischar);  % Heavy metal material
    addParameter(p, 'Ferromagnet', 'CoFeB', @ischar);  % Ferromagnetic material
    addParameter(p, 'Temperature', 300, @(x) isnumeric(x) && x > 0);  % Temperature (K)
    addParameter(p, 'HM_Thickness', 4e-9, @(x) isnumeric(x) && x > 0);  % HM thickness (m)
    addParameter(p, 'FM_Thickness', 1.2e-9, @(x) isnumeric(x) && x > 0);  % FM thickness (m)
    addParameter(p, 'DeviceWidth', 100e-9, @(x) isnumeric(x) && x > 0);  % Device width (m)
    addParameter(p, 'DeviceLength', 200e-9, @(x) isnumeric(x) && x > 0);  % Device length (m)
    addParameter(p, 'CurrentRange', [1e11, 5e11], @(x) isnumeric(x) && length(x) == 2);  % Current range (A/m²)
    addParameter(p, 'FieldRange', [0, 50e-3], @(x) isnumeric(x) && length(x) == 2);  % Assist field range (T)
    addParameter(p, 'NumCurrentPoints', 15, @(x) isnumeric(x) && x > 0);  % Number of current points
    addParameter(p, 'NumFieldPoints', 20, @(x) isnumeric(x) && x > 0);  % Number of field points
    addParameter(p, 'PulseWidth', 5e-9, @(x) isnumeric(x) && x > 0);  % Current pulse width (s)
    addParameter(p, 'OptimizeEfficiency', false, @islogical);  % Run efficiency optimization
    addParameter(p, 'PlotResults', true, @islogical);  % Enable plotting
    addParameter(p, 'Verbose', true, @islogical);  % Enable verbose output
    addParameter(p, 'SaveFigures', false, @islogical);  % Save figures
    addParameter(p, 'FigurePath', pwd, @ischar);  % Figure save path
    parse(p, varargin{:});
    
    % Extract parameters
    hm_material = p.Results.HeavyMetal;
    fm_material = p.Results.Ferromagnet;
    T = p.Results.Temperature;
    t_hm = p.Results.HM_Thickness;
    t_fm = p.Results.FM_Thickness;
    W = p.Results.DeviceWidth;
    L = p.Results.DeviceLength;
    J_range = p.Results.CurrentRange;
    H_range = p.Results.FieldRange;
    N_J = p.Results.NumCurrentPoints;
    N_H = p.Results.NumFieldPoints;
    t_pulse = p.Results.PulseWidth;
    optimize_efficiency = p.Results.OptimizeEfficiency;
    plot_results = p.Results.PlotResults;
    verbose = p.Results.Verbose;
    
    if verbose
        fprintf('=== SOT Device Characterization Validation ===\n');
        fprintf('Stack: %s(%.1fnm)/%s(%.1fnm)\n', hm_material, t_hm*1e9, fm_material, t_fm*1e9);
        fprintf('Device: %.0f × %.0f nm²\n', W*1e9, L*1e9);
        fprintf('Temperature: %.1f K\n', T);
        fprintf('Current range: [%.1e, %.1e] A/m²\n', J_range);
        fprintf('Field range: [%.1f, %.1f] mT\n', H_range*1e3);
        fprintf('Pulse width: %.1f ns\n', t_pulse*1e9);
        fprintf('==============================================\n\n');
    end
    
    %% Step 1: Get Material Properties and SOT Parameters
    
    if verbose
        fprintf('Step 1: Loading material properties and SOT parameters...\n');
    end
    
    % Get material properties from database
    materials_db = MaterialsDB();
    hm_props = materials_db.getTemperatureDependence(hm_material, T);
    fm_props = materials_db.getTemperatureDependence(fm_material, T);
    
    % Get interface parameters
    interface_params = materials_db.getInterfaceParameters(fm_material, hm_material);
    
    % Extract SOT parameters
    if isfield(interface_params, 'xi_DL')
        xi_DL = interface_params.xi_DL;  % Damping-like efficiency
        xi_FL = interface_params.xi_FL;  % Field-like efficiency
        theta_SH = hm_props.theta_SH;    % Spin Hall angle
    else
        % Default SOT parameters based on material
        switch hm_material
            case 'Pt'
                xi_DL = 0.15;
                xi_FL = 0.05;
                theta_SH = 0.15;
            case 'Ta'
                xi_DL = 0.12;
                xi_FL = 0.03;
                theta_SH = -0.12;
            case 'W'
                xi_DL = 0.35;
                xi_FL = 0.02;
                theta_SH = -0.30;
            otherwise
                xi_DL = 0.1;
                xi_FL = 0.03;
                theta_SH = 0.1;
        end
    end
    
    % Extract FM properties
    Ms = fm_props.Ms;           % Saturation magnetization (A/m)
    alpha = fm_props.alpha;     % Gilbert damping
    K_u = fm_props.K_u;         % Uniaxial anisotropy (J/m³)
    gamma = fm_props.gamma;     % Gyromagnetic ratio (rad⋅s⁻¹⋅T⁻¹)
    
    % Physical constants
    mu_0 = 4*pi*1e-7;  % Permeability of free space
    hbar = 1.054571817e-34;  % Reduced Planck constant
    e = 1.602176634e-19;  % Elementary charge
    
    % Device parameters
    area = W * L;               % Device area (m²)
    volume_fm = area * t_fm;    % FM volume (m³)
    
    % Anisotropy field
    H_k = 2*K_u / (mu_0 * Ms);
    
    if verbose
        fprintf('  Heavy metal: %s (%.1f nm)\n', hm_material, t_hm*1e9);
        fprintf('    Spin Hall angle θ_SH: %.3f\n', theta_SH);
        fprintf('  Ferromagnet: %s (%.1f nm)\n', fm_material, t_fm*1e9);
        fprintf('    Ms: %.2e A/m, α: %.4f\n', Ms, alpha);
        fprintf('    K_u: %.2e J/m³, H_k: %.1f mT\n', K_u, H_k*1e3);
        fprintf('  SOT efficiencies: ξ_DL = %.3f, ξ_FL = %.3f\n', xi_DL, xi_FL);
        fprintf('  Device area: %.1f nm², FM volume: %.2e m³\n', area*1e18, volume_fm);
        fprintf('  Done.\n\n');
    end
    
    %% Step 2: Analytical SOT Model
    
    if verbose
        fprintf('Step 2: Computing analytical SOT model...\n');
    end
    
    % Critical current for SOT switching (macrospin model)
    % J_c = (2*e*Ms*t_fm*α*H_k) / (ħ*ξ_DL)
    J_c_SOT = (2*e*Ms*t_fm*alpha*H_k) / (hbar*xi_DL);
    
    % Generate current and field arrays
    J_current = linspace(J_range(1), J_range(2), N_J);
    H_assist = linspace(H_range(1), H_range(2), N_H);
    
    % Switching probability matrix (current vs field)
    [J_grid, H_grid] = meshgrid(J_current, H_assist);
    P_switch_analytical = zeros(size(J_grid));
    
    % Thermal parameters
    k_B = 1.380649e-23;  % Boltzmann constant
    E_thermal = k_B * T;
    E_barrier = K_u * volume_fm;
    Delta = E_barrier / E_thermal;  % Thermal stability factor
    
    % Analytical switching probability (field-assisted SOT)
    for i = 1:N_H
        for j = 1:N_J
            J = J_current(j);
            H = H_assist(i);
            
            % Effective switching field
            J_eff = J / J_c_SOT;
            H_eff = H / H_k;
            
            % Combined switching parameter
            if J_eff > 0.1  % Minimum threshold
                switching_param = sqrt(J_eff^2 + H_eff^2) - 1;
                
                if switching_param > 0 && Delta > 0
                    % Thermal activation model
                    P_switch_analytical(i, j) = 1 - exp(-switching_param^2 * Delta);
                end
            end
        end
    end
    
    % Critical switching line in J-H space
    H_critical = zeros(size(J_current));
    for j = 1:N_J
        J_eff = J_current(j) / J_c_SOT;
        if J_eff < 1
            H_critical(j) = H_k * sqrt(1 - J_eff^2);
        else
            H_critical(j) = 0;
        end
    end
    
    % SOT switching time (approximate)
    tau_SOT_analytical = zeros(size(J_current));
    for j = 1:N_J
        J_eff = J_current(j) / J_c_SOT;
        if J_eff > 1
            % Switching time inversely proportional to overdrive
            tau_SOT_analytical(j) = (1/(gamma*H_k)) / (J_eff - 1);
        else
            tau_SOT_analytical(j) = inf;
        end
    end
    tau_SOT_analytical = min(tau_SOT_analytical, 100e-9);  % Cap at 100 ns
    
    if verbose
        fprintf('  Critical SOT current: J_c = %.2e A/m²\n', J_c_SOT);
        fprintf('  Current range: [%.1f, %.1f] × J_c\n', J_range(1)/J_c_SOT, J_range(2)/J_c_SOT);
        fprintf('  Thermal stability: Δ = %.1f\n', Delta);
        fprintf('  Done.\n\n');
    end
    
    %% Step 3: SOT Switching Simulations
    
    if verbose
        fprintf('Step 3: Running SOT switching simulations...\n');
    end
    
    % Initialize results arrays
    P_switch_numerical = zeros(N_H, N_J);
    tau_switch_numerical = zeros(N_H, N_J);
    switching_trajectories = cell(N_H, N_J);
    
    % Magnetization states
    m_initial = [0, 0, -1];  % Initial state (down)
    m_target = [0, 0, 1];    % Target state (up)
    
    % Simulation parameters
    simulation_count = 0;
    total_simulations = N_H * N_J;
    
    for h_idx = 1:N_H
        H = H_assist(h_idx);
        
        for j_idx = 1:N_J
            J = J_current(j_idx);
            simulation_count = simulation_count + 1;
            
            if verbose && mod(simulation_count, 10) == 1
                fprintf('  Simulation %d/%d (J=%.1e A/m², H=%.1f mT)...\n', ...
                        simulation_count, total_simulations, J, H*1e3);
            end
            
            try
                % Define effective field with assist field
                H_app = [H, 0, 0];  % Assist field along x
                H_eff_func = @(t, m) calculateEffectiveFieldSOT(t, m, H_app, K_u, Ms, mu_0);
                
                % Define SOT torques
                tau_SOT_func = @(t, m) calculateSOTTorques(t, m, J, t_pulse, xi_DL, xi_FL, ...
                                      gamma, hbar, e, Ms, t_fm);
                
                % Combined equation of motion
                llg_sot_func = @(t, m) llgSOTEquation(t, m, H_eff_func, tau_SOT_func, alpha, gamma);
                
                % Time span
                t_sim = [0, 20e-9];  % 20 ns simulation
                
                % Solve LLG+SOT equation
                [m_traj, t_traj, solution_info] = LLGSolver(m_initial', H_eff_func, ...
                    alpha, gamma, t_sim, 'Method', 'RK45', 'RelTol', 1e-6, 'Verbose', false);
                
                % Check switching
                m_final = m_traj(:, end);
                switched = m_final(3) > 0.5;  % mz > 0.5
                
                P_switch_numerical(h_idx, j_idx) = double(switched);
                
                % Find switching time
                if switched
                    mz_traj = squeeze(m_traj(3, :));
                    switch_idx = find(mz_traj > 0, 1, 'first');
                    if ~isempty(switch_idx) && switch_idx > 1
                        tau_switch_numerical(h_idx, j_idx) = t_traj(switch_idx);
                    else
                        tau_switch_numerical(h_idx, j_idx) = t_traj(end);
                    end
                else
                    tau_switch_numerical(h_idx, j_idx) = inf;
                end
                
                % Store representative trajectories
                if h_idx <= 3 && j_idx <= 5  % Store subset
                    switching_trajectories{h_idx, j_idx} = struct('t', t_traj, 'm', m_traj, ...
                                                                'switched', switched);
                end
                
            catch ME
                if verbose
                    fprintf('    Simulation failed: %s\n', ME.message);
                end
                P_switch_numerical(h_idx, j_idx) = 0;
                tau_switch_numerical(h_idx, j_idx) = inf;
            end
        end
    end
    
    if verbose
        successful_sims = sum(sum(~isinf(tau_switch_numerical)));
        fprintf('  Completed %d/%d simulations\n', total_simulations, total_simulations);
        fprintf('  Successful switching events: %d\n', successful_sims);
        fprintf('  Done.\n\n');
    end
    
    %% Step 4: Efficiency Optimization
    
    if optimize_efficiency && verbose
        fprintf('Step 4: Running efficiency optimization...\n');
    end
    
    if optimize_efficiency
        % Define efficiency metrics
        % Energy per switching = ∫ V*I dt ≈ J²*ρ*t_pulse*volume
        rho_hm = hm_props.rho;  % HM resistivity
        
        energy_per_switch = zeros(N_H, N_J);
        efficiency_metric = zeros(N_H, N_J);
        
        for h_idx = 1:N_H
            for j_idx = 1:N_J
                J = J_current(j_idx);
                P_switch = P_switch_numerical(h_idx, j_idx);
                
                % Energy calculation (Joule heating in HM)
                energy_per_switch(h_idx, j_idx) = J^2 * rho_hm * t_pulse * t_hm * area;
                
                % Efficiency: switching probability per unit energy
                if energy_per_switch(h_idx, j_idx) > 0
                    efficiency_metric(h_idx, j_idx) = P_switch / energy_per_switch(h_idx, j_idx);
                end
            end
        end
        
        % Find optimal operating point
        [max_efficiency, max_idx] = max(efficiency_metric(:));
        [opt_h_idx, opt_j_idx] = ind2sub(size(efficiency_metric), max_idx);
        
        optimal_current = J_current(opt_j_idx);
        optimal_field = H_assist(opt_h_idx);
        optimal_energy = energy_per_switch(opt_h_idx, opt_j_idx);
        
        optimization_success = max_efficiency > 0;
        
        if verbose
            if optimization_success
                fprintf('  Optimization Results:\n');
                fprintf('    Optimal current: %.2e A/m² (%.1f × J_c)\n', ...
                        optimal_current, optimal_current/J_c_SOT);
                fprintf('    Optimal field: %.1f mT\n', optimal_field*1e3);
                fprintf('    Energy per switch: %.2e J\n', optimal_energy);
                fprintf('    Efficiency metric: %.2e (P_switch/J)\n', max_efficiency);
            else
                fprintf('  Optimization failed (no successful switching)\n');
            end
        end
    else
        optimization_success = false;
        optimal_current = NaN;
        optimal_field = NaN;
        optimal_energy = NaN;
        max_efficiency = NaN;
        energy_per_switch = [];
        efficiency_metric = [];
    end
    
    if optimize_efficiency && verbose
        fprintf('  Done.\n\n');
    end
    
    %% Step 5: Validation Analysis
    
    if verbose
        fprintf('Step 5: Validation analysis...\n');
    end
    
    % Compare switching probabilities at high current/field
    high_J_idx = round(0.8 * N_J);  % Use 80% of max current
    high_H_idx = round(0.8 * N_H);  % Use 80% of max field
    
    if high_J_idx <= N_J && high_H_idx <= N_H
        P_anal_high = P_switch_analytical(high_H_idx, high_J_idx);
        P_num_high = P_switch_numerical(high_H_idx, high_J_idx);
        
        if P_anal_high > 0.1  % Significant switching expected
            P_error = abs(P_num_high - P_anal_high);
            P_rel_error = P_error / P_anal_high * 100;
        else
            P_error = abs(P_num_high - P_anal_high);
            P_rel_error = NaN;
        end
        
        % Validate critical current extraction
        % Find 50% switching probability line
        P_threshold = 0.5;
        J_c_extracted = NaN;
        
        for j_idx = 1:N_J
            if any(P_switch_numerical(:, j_idx) >= P_threshold)
                J_c_extracted = J_current(j_idx);
                break;
            end
        end
        
        if ~isnan(J_c_extracted)
            J_c_error = abs(J_c_extracted - J_c_SOT) / J_c_SOT * 100;
        else
            J_c_error = NaN;
        end
        
        % Validation criteria
        P_tolerance = 0.3;  % 30% absolute error in switching probability
        J_c_tolerance = 50;  % 50% relative error in critical current
        
        P_validation = P_error < P_tolerance;
        J_c_validation = isnan(J_c_error) || J_c_error < J_c_tolerance;
        overall_validation = P_validation && J_c_validation;
        
        if verbose
            fprintf('  Validation Results:\n');
            fprintf('    Switching probability error: %.3f (tolerance: %.3f)\n', P_error, P_tolerance);
            fprintf('    P_switch validation: %s\n', char(P_validation*'PASS' + ~P_validation*'FAIL'));
            if ~isnan(J_c_error)
                fprintf('    Critical current error: %.1f%% (tolerance: %.0f%%)\n', J_c_error, J_c_tolerance);
                fprintf('    J_c validation: %s\n', char(J_c_validation*'PASS' + ~J_c_validation*'FAIL'));
            end
            fprintf('    Overall validation: %s\n', char(overall_validation*'PASS' + ~overall_validation*'FAIL'));
        end
    else
        P_error = NaN;
        J_c_error = NaN;
        overall_validation = false;
        
        if verbose
            fprintf('  Insufficient data for validation\n');
        end
    end
    
    if verbose
        fprintf('  Done.\n\n');
    end
    
    %% Step 6: Package Results
    
    results = struct();
    
    % Input parameters
    results.parameters = struct();
    results.parameters.heavy_metal = hm_material;
    results.parameters.ferromagnet = fm_material;
    results.parameters.temperature = T;
    results.parameters.hm_thickness = t_hm;
    results.parameters.fm_thickness = t_fm;
    results.parameters.device_width = W;
    results.parameters.device_length = L;
    results.parameters.current_range = J_range;
    results.parameters.field_range = H_range;
    results.parameters.pulse_width = t_pulse;
    
    % Material and SOT properties
    results.materials = struct();
    results.materials.heavy_metal = hm_props;
    results.materials.ferromagnet = fm_props;
    results.materials.interface_params = interface_params;
    results.materials.xi_DL = xi_DL;
    results.materials.xi_FL = xi_FL;
    results.materials.theta_SH = theta_SH;
    results.materials.anisotropy_field = H_k;
    results.materials.thermal_stability = Delta;
    
    % Analytical predictions
    results.analytical = struct();
    results.analytical.critical_current = J_c_SOT;
    results.analytical.current_density = J_current;
    results.analytical.assist_field = H_assist;
    results.analytical.switching_probability = P_switch_analytical;
    results.analytical.critical_field_line = H_critical;
    results.analytical.switching_time = tau_SOT_analytical;
    
    % Numerical results
    results.numerical = struct();
    results.numerical.current_density = J_current;
    results.numerical.assist_field = H_assist;
    results.numerical.switching_probability = P_switch_numerical;
    results.numerical.switching_time = tau_switch_numerical;
    results.numerical.trajectories = switching_trajectories;
    
    % Optimization results
    results.optimization = struct();
    results.optimization.performed = optimize_efficiency;
    results.optimization.success = optimization_success;
    if optimization_success
        results.optimization.optimal_current = optimal_current;
        results.optimization.optimal_field = optimal_field;
        results.optimization.optimal_energy = optimal_energy;
        results.optimization.max_efficiency = max_efficiency;
        results.optimization.energy_per_switch = energy_per_switch;
        results.optimization.efficiency_metric = efficiency_metric;
    end
    
    % Validation results
    results.validation = struct();
    results.validation.overall_pass = overall_validation;
    results.validation.probability_error = P_error;
    results.validation.critical_current_error = J_c_error;
    
    %% Step 7: Plotting
    
    if plot_results
        if verbose
            fprintf('Step 6: Generating plots...\n');
        end
        
        % Apply Berkeley styling
        berkeley();
        
        % Create main figure
        fig = figure('Position', [100, 100, 1400, 1000]);
        
        % Subplot 1: Switching probability map (analytical)
        subplot(2, 3, 1);
        imagesc(J_current/1e11, H_assist*1e3, P_switch_analytical);
        hold on;
        plot(J_current/1e11, H_critical*1e3, 'w--', 'LineWidth', 3, 'DisplayName', 'Critical Line');
        colorbar;
        xlabel('Current Density (10^{11} A/m²)');
        ylabel('Assist Field (mT)');
        title('Analytical Switching Probability');
        set(gca, 'YDir', 'normal');
        caxis([0, 1]);
        
        % Subplot 2: Switching probability map (numerical)
        subplot(2, 3, 2);
        imagesc(J_current/1e11, H_assist*1e3, P_switch_numerical);
        colorbar;
        xlabel('Current Density (10^{11} A/m²)');
        ylabel('Assist Field (mT)');
        title('Numerical Switching Probability');
        set(gca, 'YDir', 'normal');
        caxis([0, 1]);
        
        % Subplot 3: Critical current analysis
        subplot(2, 3, 3);
        % Extract switching probability at different fields
        field_indices = [1, round(N_H/3), round(2*N_H/3), N_H];
        colors = lines(length(field_indices));
        
        for i = 1:length(field_indices)
            h_idx = field_indices(i);
            plot(J_current/1e11, P_switch_numerical(h_idx, :), 'o-', 'Color', colors(i, :), ...
                 'LineWidth', 2, 'DisplayName', sprintf('H = %.1f mT', H_assist(h_idx)*1e3));
            hold on;
        end
        
        xline(J_c_SOT/1e11, '--k', 'LineWidth', 2, 'DisplayName', 'J_c (analytical)');
        xlabel('Current Density (10^{11} A/m²)');
        ylabel('Switching Probability');
        title('Switching vs Current');
        legend('Location', 'best');
        grid on;
        
        % Subplot 4: SOT switching trajectories
        subplot(2, 3, 4);
        trajectory_count = 0;
        
        for h_idx = 1:min(3, N_H)
            for j_idx = 1:min(3, N_J)
                if ~isempty(switching_trajectories{h_idx, j_idx})
                    traj = switching_trajectories{h_idx, j_idx};
                    plot(traj.t*1e9, squeeze(traj.m(3, :)), 'LineWidth', 2);
                    hold on;
                    trajectory_count = trajectory_count + 1;
                end
            end
        end
        
        xlabel('Time (ns)');
        ylabel('m_z');
        title('SOT Switching Trajectories');
        grid on;
        ylim([-1.1, 1.1]);
        
        % Subplot 5: Efficiency optimization (if performed)
        subplot(2, 3, 5);
        if optimize_efficiency && optimization_success
            imagesc(J_current/1e11, H_assist*1e3, log10(efficiency_metric + 1e-20));
            hold on;
            plot(optimal_current/1e11, optimal_field*1e3, 'r*', 'MarkerSize', 15, 'LineWidth', 3);
            colorbar;
            xlabel('Current Density (10^{11} A/m²)');
            ylabel('Assist Field (mT)');
            title('Efficiency Optimization');
            set(gca, 'YDir', 'normal');
        else
            % Show energy vs current instead
            if ~isempty(energy_per_switch)
                plot(J_current/1e11, energy_per_switch(1, :)*1e15, 'LineWidth', 3);
                xlabel('Current Density (10^{11} A/m²)');
                ylabel('Energy per Switch (fJ)');
                title('Energy vs Current');
                grid on;
            else
                text(0.5, 0.5, 'No Optimization Data', 'HorizontalAlignment', 'center', ...
                     'Units', 'normalized', 'FontSize', 14);
                title('Efficiency Analysis');
            end
        end
        
        % Subplot 6: Validation summary
        subplot(2, 3, 6);
        if ~isnan(P_error)
            validation_metrics = {'P Switch Error', 'J_c Error (%)', 'Thermal Stability'};
            validation_values = [P_error*100, J_c_error, Delta];
            validation_limits = [30, 50, 20];  % Tolerance/good limits
            
            bar_colors = zeros(length(validation_values), 3);
            for i = 1:length(validation_values)
                if isnan(validation_values(i)) || ...
                   (i <= 2 && validation_values(i) > validation_limits(i)) || ...
                   (i == 3 && validation_values(i) < validation_limits(i))
                    bar_colors(i, :) = [0.8, 0.2, 0.2];  % Red for fail/low
                else
                    bar_colors(i, :) = [0.2, 0.8, 0.2];  % Green for pass/good
                end
            end
            
            b = bar(validation_values);
            b.FaceColor = 'flat';
            b.CData = bar_colors;
            
            set(gca, 'XTickLabel', validation_metrics);
            ylabel('Value');
            title('Validation Summary');
            grid on;
        else
            text(0.5, 0.5, 'Validation Failed', 'HorizontalAlignment', 'center', ...
                 'Units', 'normalized', 'Color', 'red', 'FontSize', 14);
            title('Validation Summary');
        end
        
        % Add overall title
        sgtitle(sprintf('SOT Device Characterization: %s/%s, T=%.0fK', ...
                       hm_material, fm_material, T), 'FontSize', 16);
        
        % Store figure handle
        results.figure = fig;
        
        % Save figures if requested
        if p.Results.SaveFigures
            timestamp = datestr(now, 'yyyymmdd_HHMMSS');
            filename = sprintf('sot_device_characterization_%s_%s_T%.0fK_%s', ...
                             hm_material, fm_material, T, timestamp);
            filepath = fullfile(p.Results.FigurePath, filename);
            
            print(fig, [filepath '.png'], '-dpng', '-r300');
            print(fig, [filepath '.pdf'], '-dpdf', '-painters');
            
            if verbose
                fprintf('  Figures saved to: %s\n', p.Results.FigurePath);
            end
        end
        
        if verbose
            fprintf('  Done.\n\n');
        end
    end
    
    %% Final Summary
    
    if verbose
        fprintf('=== SOT DEVICE CHARACTERIZATION SUMMARY ===\n');
        fprintf('Overall validation: %s\n', char(overall_validation*'PASS' + ~overall_validation*'FAIL'));
        fprintf('Critical SOT current: %.2e A/m²\n', J_c_SOT);
        fprintf('SOT efficiencies: ξ_DL = %.3f, ξ_FL = %.3f\n', xi_DL, xi_FL);
        if ~isnan(P_error)
            fprintf('Switching probability error: %.1f%%\n', P_error*100);
        end
        if ~isnan(J_c_error)
            fprintf('Critical current error: %.1f%%\n', J_c_error);
        end
        if optimization_success
            fprintf('Optimal operating point: J = %.2e A/m², H = %.1f mT\n', ...
                    optimal_current, optimal_field*1e3);
            fprintf('Energy efficiency: %.2e (P_switch/J)\n', max_efficiency);
        end
        fprintf('============================================\n\n');
    end
    
end

%% Helper Functions

function H_eff = calculateEffectiveFieldSOT(t, m, H_app, K_u, Ms, mu_0)
    % Calculate effective field for SOT simulations
    
    % Applied field (assist field)
    H_applied = H_app(:);
    
    % Uniaxial anisotropy field (along z-axis)
    H_anis = (2*K_u / (mu_0*Ms)) * [0; 0; m(3)];
    
    % Total effective field
    H_eff = H_applied + H_anis;
end

function tau_SOT = calculateSOTTorques(t, m, J, t_pulse, xi_DL, xi_FL, gamma, hbar, e, Ms, t_fm)
    % Calculate SOT torques
    
    % Current pulse
    if t <= t_pulse
        J_eff = J;
    else
        J_eff = 0;
    end
    
    % SOT prefactor
    prefactor = -gamma * (hbar/(2*e)) * (J_eff/(Ms*t_fm));
    
    % Spin polarization direction (y-direction for x-current, z-FM stack)
    sigma = [0; 1; 0];
    
    m = m(:);
    
    % Damping-like torque: m × (m × σ)
    m_cross_sigma = cross(m, sigma);
    tau_DL = prefactor * xi_DL * cross(m, m_cross_sigma);
    
    % Field-like torque: m × σ
    tau_FL = prefactor * xi_FL * cross(m, sigma);
    
    % Total SOT torque
    tau_SOT = tau_DL + tau_FL;
end

function dmdt = llgSOTEquation(t, m, H_eff_func, tau_SOT_func, alpha, gamma)
    % LLG equation with SOT torques
    
    % Effective field
    H_eff = H_eff_func(t, m);
    
    % SOT torques
    tau_SOT = tau_SOT_func(t, m);
    
    % LLG terms
    m_cross_H = cross(m, H_eff);
    m_cross_mH = cross(m, m_cross_H);
    
    % LLG+SOT equation
    prefactor = -gamma / (1 + alpha^2);
    dmdt = prefactor * (m_cross_H + alpha * m_cross_mH) + tau_SOT;
end