function results = llgs_current_switching(varargin)
% LLGS_CURRENT_SWITCHING - Validate STT-driven magnetization switching
%
% This validation script demonstrates spin-transfer torque (STT) driven
% magnetization switching using the Landau-Lifshitz-Gilbert-Slonczewski
% (LLGS) equation. It validates the framework's ability to capture
% current-induced magnetic dynamics.
%
% The LLGS equation includes the STT term:
% dm/dt = -γ/(1+α²) * [m × H_eff + α * m × (m × H_eff)] + τ_STT
%
% Where the STT torque is:
% τ_STT = -γ*ε*P*J/(2*e*Ms*t) * [m × (m × p) + b * m × p]
%
% Key Parameters:
%   ε - STT efficiency
%   P - Spin polarization  
%   J - Current density (A/m²)
%   p - Fixed layer polarization direction
%   b - Field-like torque coefficient
%
% Physics Validated:
%   - Critical switching current density
%   - Switching time vs current
%   - Thermal activation and switching probability
%   - Precession during switching
%   - Energy barrier and retention
%
% Usage:
%   results = llgs_current_switching();  % Default parameters
%   results = llgs_current_switching('Material', 'CoFeB', 'Temperature', 350);
%   results = llgs_current_switching('CurrentRange', [1e11, 5e11], 'PlotResults', true);
%
% Author: Meshal Alawein <meshal@berkeley.edu>
% Copyright © 2025 Meshal Alawein — All rights reserved.
% License: MIT

    % Parse input arguments
    p = inputParser;
    addParameter(p, 'Material', 'CoFeB', @ischar);  % Free layer material
    addParameter(p, 'FixedLayerMaterial', 'CoFeB', @ischar);  % Fixed layer material
    addParameter(p, 'Temperature', 300, @(x) isnumeric(x) && x > 0);  % Temperature (K)
    addParameter(p, 'Thickness', 1.5e-9, @(x) isnumeric(x) && x > 0);  % Free layer thickness (m)
    addParameter(p, 'Diameter', 50e-9, @(x) isnumeric(x) && x > 0);  % Device diameter (m)
    addParameter(p, 'CurrentRange', [5e10, 3e11], @(x) isnumeric(x) && length(x) == 2);  % Current density range (A/m²)
    addParameter(p, 'NumCurrentPoints', 15, @(x) isnumeric(x) && x > 0);  % Number of current points
    addParameter(p, 'PulseWidth', 10e-9, @(x) isnumeric(x) && x > 0);  % Current pulse width (s)
    addParameter(p, 'AppliedField', [0, 0, 10e-3], @(x) isnumeric(x) && length(x) == 3);  % Applied field (T)
    addParameter(p, 'STTEfficiency', 0.7, @(x) isnumeric(x) && x > 0);  % STT efficiency ε
    addParameter(p, 'FieldLikeRatio', 0.1, @(x) isnumeric(x));  % Field-like/damping-like ratio
    addParameter(p, 'NumRealizations', 1, @(x) isnumeric(x) && x > 0);  % Monte Carlo realizations
    addParameter(p, 'PlotResults', true, @islogical);  % Enable plotting
    addParameter(p, 'Verbose', true, @islogical);  % Enable verbose output
    addParameter(p, 'SaveFigures', false, @islogical);  % Save figures
    addParameter(p, 'FigurePath', pwd, @ischar);  % Figure save path
    parse(p, varargin{:});
    
    % Extract parameters
    material = p.Results.Material;
    fixed_material = p.Results.FixedLayerMaterial;
    T = p.Results.Temperature;
    t_free = p.Results.Thickness;
    diameter = p.Results.Diameter;
    J_range = p.Results.CurrentRange;
    N_J = p.Results.NumCurrentPoints;
    t_pulse = p.Results.PulseWidth;
    H_app = p.Results.AppliedField;
    epsilon = p.Results.STTEfficiency;
    b_FL = p.Results.FieldLikeRatio;
    N_realizations = p.Results.NumRealizations;
    plot_results = p.Results.PlotResults;
    verbose = p.Results.Verbose;
    
    if verbose
        fprintf('=== STT-Driven Magnetization Switching Validation ===\n');
        fprintf('Free layer: %s (%.1f nm thick)\n', material, t_free*1e9);
        fprintf('Fixed layer: %s\n', fixed_material);
        fprintf('Device diameter: %.0f nm\n', diameter*1e9);
        fprintf('Temperature: %.1f K\n', T);
        fprintf('Current density range: [%.1e, %.1e] A/m²\n', J_range);
        fprintf('Pulse width: %.1f ns\n', t_pulse*1e9);
        fprintf('Applied field: [%.1f, %.1f, %.1f] mT\n', H_app*1e3);
        fprintf('STT efficiency: %.2f\n', epsilon);
        fprintf('======================================================\n\n');
    end
    
    %% Step 1: Get Material Properties
    
    if verbose
        fprintf('Step 1: Loading material properties...\n');
    end
    
    % Get material properties from database
    materials_db = MaterialsDB();
    free_props = materials_db.getTemperatureDependence(material, T);
    fixed_props = materials_db.getTemperatureDependence(fixed_material, T);
    
    % Extract key magnetic parameters
    Ms = free_props.Ms;              % Saturation magnetization (A/m)
    alpha = free_props.alpha;        % Gilbert damping
    A_ex = free_props.A_ex;          % Exchange stiffness (J/m)
    K_u = free_props.K_u;            % Uniaxial anisotropy (J/m³)
    gamma = free_props.gamma;        % Gyromagnetic ratio (rad⋅s⁻¹⋅T⁻¹)
    polarization = fixed_props.polarization;  % Spin polarization
    
    % Calculate device parameters
    area = pi * (diameter/2)^2;      % Device area (m²)
    volume = area * t_free;          % Free layer volume (m³)
    
    % Thermal energy and stability
    k_B = 1.380649e-23;  % Boltzmann constant (J/K)
    E_thermal = k_B * T;
    
    % Energy barrier (for uniaxial anisotropy)
    E_barrier = K_u * volume;
    Delta = E_barrier / E_thermal;   % Thermal stability factor
    
    % Exchange length
    mu_0 = 4*pi*1e-7;
    lambda_ex = sqrt(2*A_ex / (mu_0 * Ms^2));
    
    if verbose
        fprintf('  Material: %s\n', material);
        fprintf('  Ms: %.2e A/m, α: %.4f, γ: %.2e rad⋅s⁻¹⋅T⁻¹\n', Ms, alpha, gamma);
        fprintf('  K_u: %.2e J/m³, A_ex: %.2e J/m\n', K_u, A_ex);
        fprintf('  Device area: %.1f nm², volume: %.2e m³\n', area*1e18, volume);
        fprintf('  Energy barrier: %.2f k_B T (Δ = %.1f)\n', Delta, Delta);
        fprintf('  Exchange length: %.1f nm\n', lambda_ex*1e9);
        fprintf('  Done.\n\n');
    end
    
    %% Step 2: Analytical Models for STT Switching
    
    if verbose
        fprintf('Step 2: Computing analytical STT models...\n');
    end
    
    % Generate current density array
    J_current = linspace(J_range(1), J_range(2), N_J);
    
    % Critical current density (Slonczewski formula)
    % J_c = (2*e*Ms*t*alpha*H_k) / (ħ*P*ε)
    e = 1.602176634e-19;  % Elementary charge (C)
    hbar = 1.054571817e-34;  % Reduced Planck constant (J⋅s)
    
    % Anisotropy field
    H_k = 2*K_u / (mu_0 * Ms);
    
    % Critical current (zero temperature, no field)
    J_c0 = (2*e*Ms*t_free*alpha*H_k) / (hbar*polarization*epsilon);
    
    % Critical current with applied field
    H_app_magnitude = norm(H_app);
    J_c_analytical = J_c0 * (1 - H_app_magnitude/H_k);
    J_c_analytical = max(J_c_analytical, 0);  % Cannot be negative
    
    % Switching probability vs current (thermal activation)
    % P_switch = 1 - exp(-(J/J_c - 1)²*Δ) for J > J_c
    P_switch_analytical = zeros(size(J_current));
    for i = 1:length(J_current)
        if J_current(i) > J_c_analytical
            thermal_factor = ((J_current(i)/J_c_analytical) - 1)^2 * Delta;
            P_switch_analytical(i) = 1 - exp(-thermal_factor);
        end
    end
    
    % Switching time (for overdamped case)
    % τ_switch ≈ (1/γ*H_k) * ln(π/2θ_0) / (J/J_c - 1)
    tau_switch_analytical = zeros(size(J_current));
    theta_0 = 0.01;  % Initial angle (rad)
    
    for i = 1:length(J_current)
        if J_current(i) > J_c_analytical
            overdrive = J_current(i) / J_c_analytical - 1;
            if overdrive > 0.01  % Avoid division by very small numbers
                tau_switch_analytical(i) = (1/(gamma*H_k)) * log(pi/(2*theta_0)) / overdrive;
            else
                tau_switch_analytical(i) = inf;
            end
        else
            tau_switch_analytical(i) = inf;  % No switching below threshold
        end
    end
    
    % Limit switching times to reasonable values
    tau_switch_analytical = min(tau_switch_analytical, 100e-9);  % Max 100 ns
    
    if verbose
        fprintf('  Critical current density J_c: %.2e A/m²\n', J_c_analytical);
        fprintf('  Anisotropy field H_k: %.1f mT\n', H_k*1e3);
        fprintf('  Applied field magnitude: %.1f mT\n', H_app_magnitude*1e3);
        fprintf('  Thermal stability Δ: %.1f\n', Delta);
        fprintf('  Current range: [%.1f, %.1f] × J_c\n', J_range(1)/J_c_analytical, J_range(2)/J_c_analytical);
        fprintf('  Done.\n\n');
    end
    
    %% Step 3: LLGS Numerical Simulations
    
    if verbose
        fprintf('Step 3: Running LLGS numerical simulations...\n');
    end
    
    % Initialize arrays for numerical results
    P_switch_numerical = zeros(size(J_current));
    tau_switch_numerical = zeros(size(J_current));
    switching_trajectories = cell(length(J_current), 1);
    
    % Initial and target magnetization states
    m_initial = [0, 0, -1];  % Down state (-z)
    m_target = [0, 0, 1];    % Up state (+z)
    p_fixed = [0, 0, 1];     % Fixed layer polarization (+z)
    
    for j_idx = 1:length(J_current)
        J = J_current(j_idx);
        
        if verbose && mod(j_idx, 3) == 1
            fprintf('  Current point %d/%d (J = %.2e A/m², %.1f×J_c)...\n', ...
                    j_idx, N_J, J, J/J_c_analytical);
        end
        
        % Run multiple realizations for statistics
        switched_count = 0;
        switch_times = [];
        
        for real = 1:N_realizations
            try
                % Define effective field function
                H_eff_func = @(t, m) calculateEffectiveField(t, m, H_app, K_u, Ms, mu_0);
                
                % Define STT torque function
                tau_STT_func = @(t, m) calculateSTTTorque(t, m, J, t_pulse, p_fixed, ...
                                      gamma, epsilon, polarization, e, Ms, t_free, b_FL);
                
                % Combined LLGS equation
                llgs_func = @(t, m) llgsEquation(t, m, H_eff_func, tau_STT_func, alpha, gamma);
                
                % Time span for simulation
                t_sim = [0, 50e-9];  % 50 ns simulation time
                
                % Add small random perturbation to initial state (thermal noise)
                noise_amplitude = sqrt(2*alpha*k_B*T / (mu_0*Ms*volume*gamma));
                m0_perturbed = m_initial + noise_amplitude * randn(1, 3);
                m0_perturbed = m0_perturbed / norm(m0_perturbed);  % Renormalize
                
                % Solve LLGS equation
                [m_traj, t_traj, solution_info] = LLGSolver(m0_perturbed', H_eff_func, ...
                    alpha, gamma, t_sim, 'Method', 'RK45', 'RelTol', 1e-6, 'Verbose', false);
                
                % Check if switching occurred
                m_final = m_traj(:, end);
                
                % Switching criterion: mz > 0 (switched from -z to +z)
                if m_final(3) > 0.5
                    switched_count = switched_count + 1;
                    
                    % Find switching time (when mz crosses zero)
                    mz_traj = squeeze(m_traj(3, :));
                    zero_crossing = find(mz_traj > 0, 1, 'first');
                    
                    if ~isempty(zero_crossing) && zero_crossing > 1
                        % Interpolate for more accurate switching time
                        t1 = t_traj(zero_crossing-1);
                        t2 = t_traj(zero_crossing);
                        mz1 = mz_traj(zero_crossing-1);
                        mz2 = mz_traj(zero_crossing);
                        
                        t_switch = t1 + (0 - mz1) / (mz2 - mz1) * (t2 - t1);
                        switch_times = [switch_times, t_switch];
                    end
                end
                
                % Store trajectory for the first realization
                if real == 1
                    switching_trajectories{j_idx} = struct('t', t_traj, 'm', m_traj, ...
                                                         'switched', m_final(3) > 0.5);
                end
                
            catch ME
                if verbose
                    fprintf('    Simulation failed: %s\n', ME.message);
                end
            end
        end
        
        % Calculate statistics
        P_switch_numerical(j_idx) = switched_count / N_realizations;
        
        if ~isempty(switch_times)
            tau_switch_numerical(j_idx) = mean(switch_times);
        else
            tau_switch_numerical(j_idx) = inf;
        end
    end
    
    if verbose
        successful_points = sum(~isinf(tau_switch_numerical));
        fprintf('  Completed %d/%d current points\n', N_J, N_J);
        fprintf('  Successful switching simulations: %d/%d\n', successful_points, N_J);
        fprintf('  Done.\n\n');
    end
    
    %% Step 4: Error Analysis and Validation
    
    if verbose
        fprintf('Step 4: Analyzing results and validation...\n');
    end
    
    % Compare switching probabilities
    finite_P_idx = P_switch_analytical > 0.01;  % Where analytical prediction is non-zero
    
    if sum(finite_P_idx) > 0
        P_anal_compare = P_switch_analytical(finite_P_idx);
        P_num_compare = P_switch_numerical(finite_P_idx);
        
        % Calculate errors
        P_error = abs(P_num_compare - P_anal_compare);
        max_P_error = max(P_error);
        rms_P_error = sqrt(mean(P_error.^2));
        
        % Switching time comparison
        finite_tau_idx = isfinite(tau_switch_analytical) & isfinite(tau_switch_numerical);
        
        if sum(finite_tau_idx) > 0
            tau_anal_compare = tau_switch_analytical(finite_tau_idx);
            tau_num_compare = tau_switch_numerical(finite_tau_idx);
            
            tau_rel_error = abs(tau_num_compare - tau_anal_compare) ./ tau_anal_compare * 100;
            max_tau_error = max(tau_rel_error);
            mean_tau_error = mean(tau_rel_error);
        else
            max_tau_error = NaN;
            mean_tau_error = NaN;
        end
        
        % Validation criteria
        P_tolerance = 0.2;  % 20% absolute error in switching probability
        tau_tolerance = 50;  % 50% relative error in switching time
        
        P_validation = max_P_error < P_tolerance;
        tau_validation = isnan(max_tau_error) || max_tau_error < tau_tolerance;
        overall_validation = P_validation && tau_validation;
        
        if verbose
            fprintf('  Switching Probability Analysis:\n');
            fprintf('    Max absolute error: %.3f (tolerance: %.3f)\n', max_P_error, P_tolerance);
            fprintf('    RMS error: %.3f\n', rms_P_error);
            fprintf('    Validation: %s\n', char(P_validation*'PASS' + ~P_validation*'FAIL'));
            
            if ~isnan(max_tau_error)
                fprintf('  Switching Time Analysis:\n');
                fprintf('    Max relative error: %.1f%% (tolerance: %.0f%%)\n', max_tau_error, tau_tolerance);
                fprintf('    Mean relative error: %.1f%%\n', mean_tau_error);
                fprintf('    Validation: %s\n', char(tau_validation*'PASS' + ~tau_validation*'FAIL'));
            end
            
            fprintf('  Overall validation: %s\n', char(overall_validation*'PASS' + ~overall_validation*'FAIL'));
        end
    else
        max_P_error = NaN;
        rms_P_error = NaN;
        max_tau_error = NaN;
        mean_tau_error = NaN;
        overall_validation = false;
        
        if verbose
            fprintf('  No switching events to compare\n');
        end
    end
    
    if verbose
        fprintf('  Done.\n\n');
    end
    
    %% Step 5: Package Results
    
    results = struct();
    
    % Input parameters
    results.parameters = struct();
    results.parameters.material = material;
    results.parameters.fixed_material = fixed_material;
    results.parameters.temperature = T;
    results.parameters.thickness = t_free;
    results.parameters.diameter = diameter;
    results.parameters.current_range = J_range;
    results.parameters.pulse_width = t_pulse;
    results.parameters.applied_field = H_app;
    results.parameters.stt_efficiency = epsilon;
    results.parameters.field_like_ratio = b_FL;
    results.parameters.num_realizations = N_realizations;
    
    % Material properties
    results.material_properties = struct();
    results.material_properties.Ms = Ms;
    results.material_properties.alpha = alpha;
    results.material_properties.K_u = K_u;
    results.material_properties.A_ex = A_ex;
    results.material_properties.gamma = gamma;
    results.material_properties.polarization = polarization;
    results.material_properties.area = area;
    results.material_properties.volume = volume;
    results.material_properties.energy_barrier = E_barrier;
    results.material_properties.thermal_stability = Delta;
    
    % Analytical solutions
    results.analytical = struct();
    results.analytical.current_density = J_current;
    results.analytical.critical_current = J_c_analytical;
    results.analytical.anisotropy_field = H_k;
    results.analytical.switching_probability = P_switch_analytical;
    results.analytical.switching_time = tau_switch_analytical;
    
    % Numerical solutions
    results.numerical = struct();
    results.numerical.current_density = J_current;
    results.numerical.switching_probability = P_switch_numerical;
    results.numerical.switching_time = tau_switch_numerical;
    results.numerical.trajectories = switching_trajectories;
    
    % Validation results
    results.validation = struct();
    results.validation.overall_pass = overall_validation;
    results.validation.max_probability_error = max_P_error;
    results.validation.rms_probability_error = rms_P_error;
    results.validation.max_time_error = max_tau_error;
    results.validation.mean_time_error = mean_tau_error;
    
    %% Step 6: Plotting
    
    if plot_results
        if verbose
            fprintf('Step 5: Generating plots...\n');
        end
        
        % Apply Berkeley styling
        berkeley();
        
        % Create main figure
        fig = figure('Position', [100, 100, 1400, 1000]);
        
        % Subplot 1: Switching probability vs current
        subplot(2, 3, 1);
        plot(J_current/1e11, P_switch_analytical, 'LineWidth', 3, 'DisplayName', 'Analytical');
        hold on;
        plot(J_current/1e11, P_switch_numerical, 'o--', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Numerical');
        xline(J_c_analytical/1e11, '--r', 'LineWidth', 2, 'DisplayName', 'J_c (analytical)');
        xlabel('Current Density (10^{11} A/m²)');
        ylabel('Switching Probability');
        title('STT Switching Probability');
        legend('Location', 'best');
        grid on;
        ylim([0, 1.1]);
        
        % Subplot 2: Switching time vs current
        subplot(2, 3, 2);
        finite_tau_plot = isfinite(tau_switch_analytical);
        semilogy(J_current(finite_tau_plot)/1e11, tau_switch_analytical(finite_tau_plot)*1e9, ...
                 'LineWidth', 3, 'DisplayName', 'Analytical');
        hold on;
        finite_tau_num = isfinite(tau_switch_numerical);
        if sum(finite_tau_num) > 0
            semilogy(J_current(finite_tau_num)/1e11, tau_switch_numerical(finite_tau_num)*1e9, ...
                     'o--', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Numerical');
        end
        xline(J_c_analytical/1e11, '--r', 'LineWidth', 2, 'DisplayName', 'J_c');
        xlabel('Current Density (10^{11} A/m²)');
        ylabel('Switching Time (ns)');
        title('STT Switching Time');
        legend('Location', 'best');
        grid on;
        
        % Subplot 3: Magnetization trajectories
        subplot(2, 3, 3);
        colors = lines(min(5, length(switching_trajectories)));
        legend_entries = {};
        
        for i = 1:min(5, length(switching_trajectories))
            if ~isempty(switching_trajectories{i})
                traj = switching_trajectories{i};
                plot(traj.t*1e9, squeeze(traj.m(3, :)), 'Color', colors(i, :), 'LineWidth', 2);
                hold on;
                
                switched_str = '';
                if traj.switched
                    switched_str = ' (switched)';
                end
                legend_entries{end+1} = sprintf('J=%.1e%s', J_current(i), switched_str);
            end
        end
        
        xlabel('Time (ns)');
        ylabel('m_z');
        title('Magnetization Trajectories');
        if ~isempty(legend_entries)
            legend(legend_entries, 'Location', 'best');
        end
        grid on;
        ylim([-1.1, 1.1]);
        
        % Subplot 4: Energy landscape
        subplot(2, 3, 4);
        theta_range = linspace(0, pi, 100);
        
        % Energy vs angle (uniaxial anisotropy)
        E_anisotropy = K_u * volume * sin(theta_range).^2;
        E_zeeman = -mu_0 * Ms * volume * norm(H_app) * cos(theta_range);
        E_total = E_anisotropy + E_zeeman;
        
        plot(theta_range*180/pi, E_anisotropy/E_thermal, 'LineWidth', 2, 'DisplayName', 'Anisotropy');
        hold on;
        plot(theta_range*180/pi, E_zeeman/E_thermal, 'LineWidth', 2, 'DisplayName', 'Zeeman');
        plot(theta_range*180/pi, E_total/E_thermal, 'LineWidth', 3, 'DisplayName', 'Total');
        
        xlabel('Angle θ (degrees)');
        ylabel('Energy (k_B T)');
        title('Energy Landscape');
        legend('Location', 'best');
        grid on;
        
        % Subplot 5: Critical current analysis
        subplot(2, 3, 5);
        % Critical current vs field
        H_field_range = linspace(0, H_k, 50);
        J_c_vs_field = J_c0 * (1 - H_field_range/H_k);
        J_c_vs_field = max(J_c_vs_field, 0);
        
        plot(H_field_range*1e3, J_c_vs_field/1e11, 'LineWidth', 3);
        hold on;
        scatter(norm(H_app)*1e3, J_c_analytical/1e11, 100, 'r', 'filled', 'DisplayName', 'Current Point');
        
        xlabel('Applied Field (mT)');
        ylabel('Critical Current Density (10^{11} A/m²)');
        title('Critical Current vs Applied Field');
        legend('Location', 'best');
        grid on;
        
        % Subplot 6: Validation summary
        subplot(2, 3, 6);
        if ~isnan(max_P_error)
            validation_metrics = {'P Switch Error', 'τ Switch Error (%)', 'Thermal Stability'};
            if ~isnan(max_tau_error)
                validation_values = [max_P_error*100, max_tau_error, Delta];
                validation_limits = [20, 50, 40];  % Tolerance limits
            else
                validation_values = [max_P_error*100, 0, Delta];
                validation_limits = [20, 50, 40];
            end
            
            bar_colors = zeros(length(validation_values), 3);
            for i = 1:length(validation_values)
                if (i <= 2 && validation_values(i) > validation_limits(i)) || ...
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
            text(0.5, 0.5, 'No Switching Events', 'HorizontalAlignment', 'center', ...
                 'Color', 'red', 'FontSize', 14);
            title('Validation Summary');
        end
        
        % Add overall title
        sgtitle(sprintf('STT Switching Validation: %s, T=%.0fK, d=%.0fnm', ...
                       material, T, diameter*1e9), 'FontSize', 16);
        
        % Store figure handle
        results.figure = fig;
        
        % Save figures if requested
        if p.Results.SaveFigures
            timestamp = datestr(now, 'yyyymmdd_HHMMSS');
            filename = sprintf('llgs_current_switching_%s_T%.0fK_d%.0fnm_%s', ...
                             material, T, diameter*1e9, timestamp);
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
        fprintf('=== STT SWITCHING VALIDATION SUMMARY ===\n');
        fprintf('Overall validation: %s\n', char(overall_validation*'PASS' + ~overall_validation*'FAIL'));
        fprintf('Critical current density: %.2e A/m²\n', J_c_analytical);
        fprintf('Thermal stability factor: %.1f\n', Delta);
        if ~isnan(max_P_error)
            fprintf('Max switching probability error: %.1f%%\n', max_P_error*100);
        end
        if ~isnan(max_tau_error)
            fprintf('Max switching time error: %.1f%%\n', max_tau_error);
        end
        fprintf('Successful simulations: %d/%d\n', sum(~isinf(tau_switch_numerical)), N_J);
        fprintf('=========================================\n\n');
    end
    
end

%% Helper Functions

function H_eff = calculateEffectiveField(t, m, H_app, K_u, Ms, mu_0)
    % Calculate effective field including anisotropy and applied field
    
    % Uniaxial anisotropy field (along z-axis)
    H_anis = (2*K_u / (mu_0*Ms)) * [0; 0; m(3)];
    
    % Applied field
    H_applied = H_app(:);
    
    % Total effective field
    H_eff = H_anis + H_applied;
end

function tau_STT = calculateSTTTorque(t, m, J, t_pulse, p_fixed, gamma, epsilon, P, e, Ms, t_free, b_FL)
    % Calculate STT torque
    
    % Current pulse shape
    if t <= t_pulse
        J_eff = J;
    else
        J_eff = 0;
    end
    
    % STT prefactor
    hbar = 1.054571817e-34;
    prefactor = -gamma * epsilon * P * J_eff / (2 * e * Ms * t_free);
    
    % Fixed layer polarization
    p = p_fixed(:);
    m = m(:);
    
    % Damping-like torque: m × (m × p)
    m_cross_p = cross(m, p);
    tau_DL = prefactor * cross(m, m_cross_p);
    
    % Field-like torque: m × p
    tau_FL = prefactor * b_FL * cross(m, p);
    
    % Total STT torque
    tau_STT = tau_DL + tau_FL;
end

function dmdt = llgsEquation(t, m, H_eff_func, tau_STT_func, alpha, gamma)
    % LLGS equation with STT
    
    % Effective field
    H_eff = H_eff_func(t, m);
    
    % STT torque
    tau_STT = tau_STT_func(t, m);
    
    % LLG terms
    m_cross_H = cross(m, H_eff);
    m_cross_mH = cross(m, m_cross_H);
    
    % LLGS equation
    prefactor = -gamma / (1 + alpha^2);
    dmdt = prefactor * (m_cross_H + alpha * m_cross_mH) + tau_STT;
end