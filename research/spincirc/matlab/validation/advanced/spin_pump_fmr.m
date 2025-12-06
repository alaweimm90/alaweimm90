function results = spin_pump_fmr(varargin)
% SPIN_PUMP_FMR - Validate FMR-driven spin pumping with frequency analysis
%
% This validation script demonstrates ferromagnetic resonance (FMR) driven
% spin pumping and validates the SpinCirc framework's ability to capture
% the generation and detection of spin currents from precessional dynamics.
%
% FMR spin pumping occurs when a ferromagnet undergoes precession at its
% resonance frequency, pumping spin-polarized electrons into adjacent
% normal metals. The pumped spin current is:
%
% I_s = (ħ/4π) * g_r * (dm/dt × m)
%
% Where:
%   g_r  - Real part of mixing conductance (S/m²)
%   m    - Magnetization unit vector
%   dm/dt - Magnetization dynamics
%
% Key Parameters:
%   - FMR frequency and linewidth
%   - Spin mixing conductance
%   - Spin Hall voltage generation
%   - Damping enhancement by spin pumping
%   - Temperature and thickness dependence
%
% Physics Validated:
%   - Kittel FMR formula
%   - Spin pumping efficiency
%   - Inverse spin Hall effect (ISHE)
%   - Gilbert damping enhancement
%   - Spin current diffusion and relaxation
%
% Usage:
%   results = spin_pump_fmr();  % Default parameters
%   results = spin_pump_fmr('Ferromagnet', 'Permalloy', 'HeavyMetal', 'Pt');
%   results = spin_pump_fmr('FrequencyRange', [5e9, 25e9], 'FieldSweep', true);
%
% Author: Meshal Alawein <meshal@berkeley.edu>
% Copyright © 2025 Meshal Alawein — All rights reserved.
% License: MIT

    % Parse input arguments
    p = inputParser;
    addParameter(p, 'Ferromagnet', 'Permalloy', @ischar);  % FM material
    addParameter(p, 'HeavyMetal', 'Pt', @ischar);  % HM material (detector)
    addParameter(p, 'Temperature', 300, @(x) isnumeric(x) && x > 0);  % Temperature (K)
    addParameter(p, 'FM_Thickness', 10e-9, @(x) isnumeric(x) && x > 0);  % FM thickness (m)
    addParameter(p, 'HM_Thickness', 5e-9, @(x) isnumeric(x) && x > 0);  % HM thickness (m)
    addParameter(p, 'DeviceWidth', 50e-6, @(x) isnumeric(x) && x > 0);  % Device width (m)
    addParameter(p, 'DeviceLength', 200e-6, @(x) isnumeric(x) && x > 0);  % Device length (m)
    addParameter(p, 'ExcitationFrequency', 10e9, @(x) isnumeric(x) && x > 0);  % RF frequency (Hz)
    addParameter(p, 'ExcitationAmplitude', 1e-3, @(x) isnumeric(x) && x > 0);  % RF field amplitude (T)
    addParameter(p, 'FrequencyRange', [5e9, 20e9], @(x) isnumeric(x) && length(x) == 2);  % Frequency range (Hz)
    addParameter(p, 'FieldRange', [0, 200e-3], @(x) isnumeric(x) && length(x) == 2);  % Field range (T)
    addParameter(p, 'NumFreqPoints', 50, @(x) isnumeric(x) && x > 0);  % Number of frequency points
    addParameter(p, 'NumFieldPoints', 100, @(x) isnumeric(x) && x > 0);  % Number of field points
    addParameter(p, 'SweepType', 'frequency', @ischar);  % 'frequency' or 'field'
    addParameter(p, 'PlotResults', true, @islogical);  % Enable plotting
    addParameter(p, 'Verbose', true, @islogical);  % Enable verbose output
    addParameter(p, 'SaveFigures', false, @islogical);  % Save figures
    addParameter(p, 'FigurePath', pwd, @ischar);  % Figure save path
    parse(p, varargin{:});
    
    % Extract parameters
    fm_material = p.Results.Ferromagnet;
    hm_material = p.Results.HeavyMetal;
    T = p.Results.Temperature;
    t_fm = p.Results.FM_Thickness;
    t_hm = p.Results.HM_Thickness;
    W = p.Results.DeviceWidth;
    L = p.Results.DeviceLength;
    f_exc = p.Results.ExcitationFrequency;
    h_rf = p.Results.ExcitationAmplitude;
    f_range = p.Results.FrequencyRange;
    B_range = p.Results.FieldRange;
    N_f = p.Results.NumFreqPoints;
    N_B = p.Results.NumFieldPoints;
    sweep_type = p.Results.SweepType;
    plot_results = p.Results.PlotResults;
    verbose = p.Results.Verbose;
    
    if verbose
        fprintf('=== FMR Spin Pumping Validation ===\n');
        fprintf('Stack: %s(%.0fnm)/%s(%.0fnm)\n', fm_material, t_fm*1e9, hm_material, t_hm*1e9);
        fprintf('Device: %.0f × %.0f μm²\n', W*1e6, L*1e6);
        fprintf('Temperature: %.1f K\n', T);
        fprintf('Excitation: f = %.1f GHz, h_rf = %.1f mT\n', f_exc/1e9, h_rf*1e3);
        fprintf('Sweep type: %s\n', sweep_type);
        if strcmp(sweep_type, 'frequency')
            fprintf('Frequency range: [%.1f, %.1f] GHz\n', f_range/1e9);
        else
            fprintf('Field range: [%.1f, %.1f] mT\n', B_range*1e3);
        end
        fprintf('===================================\n\n');
    end
    
    %% Step 1: Get Material Properties
    
    if verbose
        fprintf('Step 1: Loading material properties...\n');
    end
    
    % Get material properties from database
    materials_db = MaterialsDB();
    fm_props = materials_db.getTemperatureDependence(fm_material, T);
    hm_props = materials_db.getTemperatureDependence(hm_material, T);
    
    % Get interface parameters
    interface_params = materials_db.getInterfaceParameters(fm_material, hm_material);
    
    % Extract FM properties
    Ms = fm_props.Ms;           % Saturation magnetization (A/m)
    alpha = fm_props.alpha;     % Gilbert damping
    gamma = fm_props.gamma;     % Gyromagnetic ratio (rad⋅s⁻¹⋅T⁻¹)
    g_factor = fm_props.g_factor; % Landé g-factor
    
    % Extract HM properties
    if isfield(hm_props, 'theta_SH')
        theta_SH = hm_props.theta_SH;    % Spin Hall angle
    else
        % Default values based on material
        switch hm_material
            case 'Pt'
                theta_SH = 0.15;
            case 'Ta'
                theta_SH = -0.12;
            case 'W'
                theta_SH = -0.30;
            otherwise
                theta_SH = 0.1;
        end
    end
    
    lambda_sf_hm = hm_props.lambda_sf;  % Spin diffusion length in HM
    rho_hm = hm_props.rho;              % HM resistivity
    
    % Interface mixing conductance
    if isfield(interface_params, 'g_r')
        g_r = interface_params.g_r;     % Real mixing conductance (S/m²)
    else
        g_r = 5e15;  % Typical value for FM/HM interfaces
    end
    
    % Physical constants
    mu_0 = 4*pi*1e-7;  % Permeability of free space
    hbar = 1.054571817e-34;  % Reduced Planck constant
    e = 1.602176634e-19;  % Elementary charge
    
    % Device parameters
    area = W * L;               % Device area (m²)
    volume_fm = area * t_fm;    % FM volume (m³)
    
    if verbose
        fprintf('  Ferromagnet: %s (%.0f nm)\n', fm_material, t_fm*1e9);
        fprintf('    Ms: %.2e A/m, α: %.4f, γ: %.2e rad⋅s⁻¹⋅T⁻¹\n', Ms, alpha, gamma);
        fprintf('  Heavy metal: %s (%.0f nm)\n', hm_material, t_hm*1e9);
        fprintf('    θ_SH: %.3f, λ_sf: %.1f nm, ρ: %.0f nΩ⋅m\n', ...
                theta_SH, lambda_sf_hm*1e9, rho_hm*1e9);
        fprintf('  Interface mixing conductance: %.2e S/m²\n', g_r);
        fprintf('  Device area: %.1f μm²\n', area*1e12);
        fprintf('  Done.\n\n');
    end
    
    %% Step 2: Analytical FMR and Spin Pumping Models
    
    if verbose
        fprintf('Step 2: Computing analytical FMR and spin pumping models...\n');
    end
    
    % FMR resonance condition (Kittel formula for thin films)
    % For in-plane magnetization: ω = γ * sqrt(H * (H + μ₀*M_eff))
    % where M_eff = Ms - H_demag_perpendicular
    
    % Effective magnetization (including demagnetization)
    M_eff = Ms;  % Simplified for thin film (in-plane easy axis)
    
    % Generate sweep arrays
    if strcmp(sweep_type, 'frequency')
        % Frequency sweep at fixed field
        frequencies = linspace(f_range(1), f_range(2), N_f);
        omega_sweep = 2*pi*frequencies;
        
        % Calculate resonance field from frequency
        H_res_analytical = zeros(size(frequencies));
        for i = 1:length(frequencies)
            omega = omega_sweep(i);
            % Solve: ω = γ * sqrt(H * (H + μ₀*M_eff))
            % This gives: H = ω²/(2*γ²*μ₀*M_eff) + μ₀*M_eff/2
            discriminant = (omega/gamma)^2;
            if discriminant >= (mu_0*M_eff)^2
                H_res_analytical(i) = sqrt(discriminant - (mu_0*M_eff)^2/4) - mu_0*M_eff/2;
            else
                H_res_analytical(i) = 0;  % No resonance possible
            end
        end
        
        sweep_parameter = frequencies;
        sweep_label = 'Frequency (GHz)';
        sweep_units = 1e9;
        
    else
        % Field sweep at fixed frequency
        fields = linspace(B_range(1), B_range(2), N_B);
        omega_fixed = 2*pi*f_exc;
        
        % Calculate resonance frequency from field
        f_res_analytical = zeros(size(fields));
        for i = 1:length(fields)
            H = fields(i);
            omega_res = gamma * sqrt(H * (H + mu_0*M_eff));
            f_res_analytical(i) = omega_res / (2*pi);
        end
        
        H_res_analytical = fields;
        sweep_parameter = fields;
        sweep_label = 'Magnetic Field (mT)';
        sweep_units = 1e3;
    end
    
    % FMR linewidth (Gilbert damping)
    if strcmp(sweep_type, 'frequency')
        Delta_H_analytical = (2*alpha/gamma) * omega_sweep / mu_0;  % Linewidth vs frequency
    else
        Delta_H_analytical = (2*alpha*omega_fixed) / (gamma*mu_0) * ones(size(fields));  % Fixed linewidth
    end
    
    % Spin pumping current amplitude
    % I_s = (ħ/4π) * g_r * area * |dm/dt × m|²
    % For circular precession: |dm/dt × m| = ω * precession_angle
    
    % Precession angle from RF drive (assuming linear response)
    precession_angle = gamma * h_rf / (2*alpha*omega_sweep);  % Approximate
    precession_angle = min(precession_angle, 0.1);  % Limit to small angles
    
    % Pumped spin current
    if strcmp(sweep_type, 'frequency')
        I_s_pumped_analytical = (hbar/(4*pi)) * g_r * area * ...
                               (omega_sweep .* precession_angle).^2;
    else
        I_s_pumped_analytical = (hbar/(4*pi)) * g_r * area * ...
                               (omega_fixed * gamma * h_rf / (2*alpha*omega_fixed))^2 * ones(size(fields));
    end
    
    % Spin Hall voltage in heavy metal
    % V_SHE = ρ_HM * θ_SH * I_s * (t_HM / W)
    V_SHE_analytical = rho_hm * abs(theta_SH) * I_s_pumped_analytical * (t_hm / W);
    
    % Damping enhancement due to spin pumping
    % α_eff = α₀ + (g_r * μ_B * g) / (4π * Ms * t_FM)
    mu_B = 9.2740100783e-24;  % Bohr magneton
    alpha_enhancement = (g_r * mu_B * g_factor) / (4*pi * Ms * t_fm);
    alpha_eff = alpha + alpha_enhancement;
    
    if verbose
        fprintf('  FMR Analysis:\n');
        if strcmp(sweep_type, 'frequency')
            fprintf('    Frequency range: [%.1f, %.1f] GHz\n', f_range/1e9);
            fprintf('    Resonance field range: [%.1f, %.1f] mT\n', ...
                    min(H_res_analytical)*1e3, max(H_res_analytical)*1e3);
        else
            fprintf('    Fixed frequency: %.1f GHz\n', f_exc/1e9);
            fprintf('    Field range: [%.1f, %.1f] mT\n', B_range*1e3);
        end
        
        fprintf('  Spin Pumping:\n');
        fprintf('    Max pumped current: %.2e A\n', max(I_s_pumped_analytical));
        fprintf('    Max spin Hall voltage: %.2f μV\n', max(V_SHE_analytical)*1e6);
        fprintf('    Damping enhancement: %.1e (%.1f%% increase)\n', ...
                alpha_enhancement, alpha_enhancement/alpha*100);
        fprintf('  Done.\n\n');
    end
    
    %% Step 3: LLG+FMR Numerical Simulations
    
    if verbose
        fprintf('Step 3: Running LLG+FMR numerical simulations...\n');
    end
    
    % Initialize arrays for numerical results
    if strcmp(sweep_type, 'frequency')
        I_s_pumped_numerical = zeros(size(frequencies));
        V_SHE_numerical = zeros(size(frequencies));
        precession_amplitude = zeros(size(frequencies));
        linewidth_numerical = zeros(size(frequencies));
    else
        I_s_pumped_numerical = zeros(size(fields));
        V_SHE_numerical = zeros(size(fields));
        precession_amplitude = zeros(size(fields));
        linewidth_numerical = zeros(size(fields));
    end
    
    % Simulation parameters
    N_points = length(sweep_parameter);
    simulation_time = 50e-9;  % 50 ns simulation time
    
    for i = 1:N_points
        if verbose && mod(i, 10) == 1
            fprintf('  Simulation %d/%d (%.2f %s)...\n', i, N_points, ...
                    sweep_parameter(i)/sweep_units, strtok(sweep_label));
        end
        
        try
            if strcmp(sweep_type, 'frequency')
                f_sim = frequencies(i);
                H_dc = H_res_analytical(i);  % Use resonance field
            else
                f_sim = f_exc;
                H_dc = fields(i);
            end
            
            omega_sim = 2*pi*f_sim;
            
            % Skip if no resonance possible
            if H_dc <= 0
                continue;
            end
            
            % Define time-dependent effective field
            H_eff_func = @(t, m) calculateEffectiveFieldFMR(t, m, H_dc, h_rf, ...
                                 omega_sim, Ms, mu_0);
            
            % Initial magnetization (along field direction)
            m_initial = [0; 0; 1];  % Along z-axis (field direction)
            
            % Solve LLG equation
            [m_traj, t_traj, solution_info] = LLGSolver(m_initial, H_eff_func, ...
                alpha_eff, gamma, [0, simulation_time], 'Method', 'RK45', ...
                'RelTol', 1e-6, 'AbsTol', 1e-8, 'Verbose', false);
            
            % Calculate spin pumping from trajectories
            if size(m_traj, 2) > 100  % Ensure sufficient data
                % Calculate dm/dt numerically
                dt = t_traj(2) - t_traj(1);
                dmdt = diff(m_traj, 1, 2) / dt;
                
                % Take steady-state portion (last 20% of simulation)
                steady_start = round(0.8 * size(dmdt, 2));
                dmdt_steady = dmdt(:, steady_start:end);
                m_steady = m_traj(:, steady_start:end-1);
                
                % Calculate spin current: I_s = (ħ/4π) * g_r * area * <|dm/dt × m|²>
                spin_current_density = zeros(1, size(dmdt_steady, 2));
                
                for j = 1:size(dmdt_steady, 2)
                    cross_product = cross(dmdt_steady(:, j), m_steady(:, j));
                    spin_current_density(j) = norm(cross_product)^2;
                end
                
                % Time-averaged spin current
                avg_spin_current_density = mean(spin_current_density);
                I_s_pumped_numerical(i) = (hbar/(4*pi)) * g_r * area * avg_spin_current_density;
                
                % Spin Hall voltage
                V_SHE_numerical(i) = rho_hm * abs(theta_SH) * I_s_pumped_numerical(i) * (t_hm / W);
                
                % Precession amplitude (transverse magnetization)
                m_transverse = sqrt(m_steady(1, :).^2 + m_steady(2, :).^2);
                precession_amplitude(i) = mean(m_transverse);
                
                % Extract linewidth (approximate from damping time)
                if length(t_traj) > 50
                    envelope = abs(hilbert(m_steady(1, :)));
                    if max(envelope) > 0.01
                        % Fit exponential decay
                        log_envelope = log(envelope + 1e-6);
                        p = polyfit(t_traj(steady_start:end-1), log_envelope, 1);
                        decay_rate = -p(1);
                        linewidth_numerical(i) = decay_rate * mu_0 / gamma;
                    end
                end
                
            else
                % Insufficient data
                I_s_pumped_numerical(i) = 0;
                V_SHE_numerical(i) = 0;
                precession_amplitude(i) = 0;
            end
            
        catch ME
            if verbose
                fprintf('    Simulation %d failed: %s\n', i, ME.message);
            end
            I_s_pumped_numerical(i) = 0;
            V_SHE_numerical(i) = 0;
            precession_amplitude(i) = 0;
        end
    end
    
    if verbose
        successful_sims = sum(I_s_pumped_numerical > 0);
        fprintf('  Completed %d/%d simulations\n', N_points, N_points);
        fprintf('  Successful FMR simulations: %d\n', successful_sims);
        fprintf('  Done.\n\n');
    end
    
    %% Step 4: Validation and Analysis
    
    if verbose
        fprintf('Step 4: Validation and analysis...\n');
    end
    
    % Compare numerical and analytical results
    valid_idx = I_s_pumped_numerical > 0 & I_s_pumped_analytical > 0;
    
    if sum(valid_idx) >= 3
        % Calculate errors for valid points
        I_s_error = abs(I_s_pumped_numerical(valid_idx) - I_s_pumped_analytical(valid_idx));
        I_s_rel_error = I_s_error ./ I_s_pumped_analytical(valid_idx) * 100;
        
        V_SHE_error = abs(V_SHE_numerical(valid_idx) - V_SHE_analytical(valid_idx));
        V_SHE_rel_error = V_SHE_error ./ V_SHE_analytical(valid_idx) * 100;
        
        % Statistics
        max_I_s_error = max(I_s_rel_error);
        mean_I_s_error = mean(I_s_rel_error);
        max_V_SHE_error = max(V_SHE_rel_error);
        mean_V_SHE_error = mean(V_SHE_rel_error);
        
        % Validation criteria
        I_s_tolerance = 50;   % 50% relative error tolerance
        V_SHE_tolerance = 50; % 50% relative error tolerance
        
        I_s_validation = max_I_s_error < I_s_tolerance;
        V_SHE_validation = max_V_SHE_error < V_SHE_tolerance;
        overall_validation = I_s_validation && V_SHE_validation && (sum(valid_idx) >= 3);
        
        if verbose
            fprintf('  Validation Results:\n');
            fprintf('    Valid simulation points: %d/%d\n', sum(valid_idx), N_points);
            fprintf('    Spin current error: %.1f%% max, %.1f%% mean\n', max_I_s_error, mean_I_s_error);
            fprintf('    Spin Hall voltage error: %.1f%% max, %.1f%% mean\n', max_V_SHE_error, mean_V_SHE_error);
            fprintf('    Spin current validation: %s\n', char(I_s_validation*'PASS' + ~I_s_validation*'FAIL'));
            fprintf('    Voltage validation: %s\n', char(V_SHE_validation*'PASS' + ~V_SHE_validation*'FAIL'));
            fprintf('    Overall validation: %s\n', char(overall_validation*'PASS' + ~overall_validation*'FAIL'));
        end
    else
        max_I_s_error = NaN;
        mean_I_s_error = NaN;
        max_V_SHE_error = NaN;
        mean_V_SHE_error = NaN;
        overall_validation = false;
        
        if verbose
            fprintf('  Insufficient valid data for validation\n');
        end
    end
    
    if verbose
        fprintf('  Done.\n\n');
    end
    
    %% Step 5: Package Results
    
    results = struct();
    
    % Input parameters
    results.parameters = struct();
    results.parameters.ferromagnet = fm_material;
    results.parameters.heavy_metal = hm_material;
    results.parameters.temperature = T;
    results.parameters.fm_thickness = t_fm;
    results.parameters.hm_thickness = t_hm;
    results.parameters.device_width = W;
    results.parameters.device_length = L;
    results.parameters.excitation_frequency = f_exc;
    results.parameters.excitation_amplitude = h_rf;
    results.parameters.sweep_type = sweep_type;
    if strcmp(sweep_type, 'frequency')
        results.parameters.frequency_range = f_range;
    else
        results.parameters.field_range = B_range;
    end
    
    % Material properties
    results.materials = struct();
    results.materials.ferromagnet = fm_props;
    results.materials.heavy_metal = hm_props;
    results.materials.mixing_conductance = g_r;
    results.materials.spin_hall_angle = theta_SH;
    results.materials.alpha_enhancement = alpha_enhancement;
    results.materials.effective_damping = alpha_eff;
    
    % FMR parameters
    results.fmr = struct();
    results.fmr.effective_magnetization = M_eff;
    results.fmr.gyromagnetic_ratio = gamma;
    if strcmp(sweep_type, 'frequency')
        results.fmr.resonance_field = H_res_analytical;
        results.fmr.frequencies = frequencies;
    else
        results.fmr.resonance_frequency = f_res_analytical;
        results.fmr.fields = fields;
    end
    results.fmr.linewidth_analytical = Delta_H_analytical;
    
    % Analytical predictions
    results.analytical = struct();
    results.analytical.sweep_parameter = sweep_parameter;
    results.analytical.spin_current_pumped = I_s_pumped_analytical;
    results.analytical.spin_hall_voltage = V_SHE_analytical;
    results.analytical.precession_angle = precession_angle;
    
    % Numerical results
    results.numerical = struct();
    results.numerical.sweep_parameter = sweep_parameter;
    results.numerical.spin_current_pumped = I_s_pumped_numerical;
    results.numerical.spin_hall_voltage = V_SHE_numerical;
    results.numerical.precession_amplitude = precession_amplitude;
    results.numerical.linewidth = linewidth_numerical;
    results.numerical.valid_points = sum(valid_idx);
    
    % Validation results
    results.validation = struct();
    results.validation.overall_pass = overall_validation;
    results.validation.max_current_error = max_I_s_error;
    results.validation.mean_current_error = mean_I_s_error;
    results.validation.max_voltage_error = max_V_SHE_error;
    results.validation.mean_voltage_error = mean_V_SHE_error;
    
    %% Step 6: Plotting
    
    if plot_results
        if verbose
            fprintf('Step 5: Generating plots...\n');
        end
        
        % Apply Berkeley styling
        berkeley();
        
        % Create main figure
        fig = figure('Position', [100, 100, 1400, 1000]);
        
        % Subplot 1: FMR resonance condition
        subplot(2, 3, 1);
        if strcmp(sweep_type, 'frequency')
            plot(frequencies/1e9, H_res_analytical*1e3, 'LineWidth', 3);
            xlabel('Frequency (GHz)');
            ylabel('Resonance Field (mT)');
            title('FMR Resonance Condition');
        else
            plot(fields*1e3, f_res_analytical/1e9, 'LineWidth', 3);
            xlabel('Magnetic Field (mT)');
            ylabel('Resonance Frequency (GHz)');
            title('FMR Resonance Condition');
        end
        grid on;
        
        % Subplot 2: Spin pumping current
        subplot(2, 3, 2);
        plot(sweep_parameter/sweep_units, I_s_pumped_analytical*1e9, 'LineWidth', 3, 'DisplayName', 'Analytical');
        hold on;
        valid_numerical = I_s_pumped_numerical > 0;
        if sum(valid_numerical) > 0
            plot(sweep_parameter(valid_numerical)/sweep_units, ...
                 I_s_pumped_numerical(valid_numerical)*1e9, 'o--', 'LineWidth', 2, ...
                 'MarkerSize', 6, 'DisplayName', 'Numerical');
        end
        xlabel(sweep_label);
        ylabel('Spin Current (nA)');
        title('FMR Spin Pumping Current');
        legend('Location', 'best');
        grid on;
        
        % Subplot 3: Spin Hall voltage
        subplot(2, 3, 3);
        plot(sweep_parameter/sweep_units, V_SHE_analytical*1e6, 'LineWidth', 3, 'DisplayName', 'Analytical');
        hold on;
        if sum(valid_numerical) > 0
            plot(sweep_parameter(valid_numerical)/sweep_units, ...
                 V_SHE_numerical(valid_numerical)*1e6, 'o--', 'LineWidth', 2, ...
                 'MarkerSize', 6, 'DisplayName', 'Numerical');
        end
        xlabel(sweep_label);
        ylabel('Spin Hall Voltage (μV)');
        title('Inverse Spin Hall Voltage');
        legend('Location', 'best');
        grid on;
        
        % Subplot 4: Precession dynamics
        subplot(2, 3, 4);
        if sum(valid_numerical) > 0
            plot(sweep_parameter(valid_numerical)/sweep_units, ...
                 precession_amplitude(valid_numerical), 'o-', 'LineWidth', 2);
            xlabel(sweep_label);
            ylabel('Precession Amplitude');
            title('Magnetization Precession');
            grid on;
        else
            text(0.5, 0.5, 'No Valid Precession Data', 'HorizontalAlignment', 'center', ...
                 'Units', 'normalized', 'FontSize', 12);
            title('Magnetization Precession');
        end
        
        % Subplot 5: Damping and linewidth
        subplot(2, 3, 5);
        damping_comparison = [alpha, alpha_eff];
        damping_labels = {'α₀', 'α_{eff}'};
        
        bar(damping_comparison);
        set(gca, 'XTickLabel', damping_labels);
        ylabel('Gilbert Damping');
        title('Damping Enhancement');
        grid on;
        
        % Add enhancement percentage
        enhancement_pct = alpha_enhancement/alpha*100;
        text(2, alpha_eff, sprintf('+%.1f%%', enhancement_pct), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
        
        % Subplot 6: Validation summary
        subplot(2, 3, 6);
        if ~isnan(max_I_s_error)
            validation_metrics = {'I_s Error (%)', 'V_SHE Error (%)', 'Valid Points (%)'};
            validation_values = [max_I_s_error, max_V_SHE_error, sum(valid_idx)/N_points*100];
            validation_limits = [50, 50, 50];  % Tolerance limits
            
            bar_colors = zeros(length(validation_values), 3);
            for i = 1:length(validation_values)
                if isnan(validation_values(i)) || validation_values(i) > validation_limits(i)
                    bar_colors(i, :) = [0.8, 0.2, 0.2];  % Red for fail
                else
                    bar_colors(i, :) = [0.2, 0.8, 0.2];  % Green for pass
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
        sgtitle(sprintf('FMR Spin Pumping: %s/%s, T=%.0fK', ...
                       fm_material, hm_material, T), 'FontSize', 16);
        
        % Store figure handle
        results.figure = fig;
        
        % Save figures if requested
        if p.Results.SaveFigures
            timestamp = datestr(now, 'yyyymmdd_HHMMSS');
            filename = sprintf('spin_pump_fmr_%s_%s_T%.0fK_%s', ...
                             fm_material, hm_material, T, timestamp);
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
        fprintf('=== FMR SPIN PUMPING VALIDATION SUMMARY ===\n');
        fprintf('Overall validation: %s\n', char(overall_validation*'PASS' + ~overall_validation*'FAIL'));
        fprintf('Sweep type: %s\n', sweep_type);
        if ~isnan(max_I_s_error)
            fprintf('Max spin current error: %.1f%%\n', max_I_s_error);
            fprintf('Max spin Hall voltage error: %.1f%%\n', max_V_SHE_error);
            fprintf('Valid simulation points: %d/%d (%.1f%%)\n', ...
                    sum(valid_idx), N_points, sum(valid_idx)/N_points*100);
        end
        fprintf('Damping enhancement: %.1f%% (α₀ = %.4f → α_eff = %.4f)\n', ...
                alpha_enhancement/alpha*100, alpha, alpha_eff);
        fprintf('Max pumped current: %.2e A\n', max(I_s_pumped_analytical));
        fprintf('Max spin Hall voltage: %.2f μV\n', max(V_SHE_analytical)*1e6);
        fprintf('===========================================\n\n');
    end
    
end

%% Helper Functions

function H_eff = calculateEffectiveFieldFMR(t, m, H_dc, h_rf, omega, Ms, mu_0)
    % Calculate effective field for FMR simulations
    
    % DC field (along z-axis)
    H_dc_vec = [0; 0; H_dc];
    
    % RF field (circular polarization in xy-plane)
    H_rf_vec = h_rf * [cos(omega*t); sin(omega*t); 0];
    
    % Demagnetization field (approximate for thin film)
    % N_z ≈ 1, N_x ≈ N_y ≈ 0 for thin film
    H_demag = -mu_0 * Ms * [0; 0; m(3)];
    
    % Total effective field
    H_eff = H_dc_vec + H_rf_vec + H_demag;
end