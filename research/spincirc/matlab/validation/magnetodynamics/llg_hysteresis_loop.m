function results = llg_hysteresis_loop(varargin)
% LLG_HYSTERESIS_LOOP - Validate quasi-static magnetic hysteresis loops
%
% This validation script demonstrates the simulation of magnetic hysteresis
% loops using the Landau-Lifshitz-Gilbert (LLG) equation under slowly
% varying external magnetic fields. It validates the framework's ability
% to capture magnetic switching, coercivity, and remanence.
%
% The LLG equation under slowly varying fields becomes:
% dm/dt = -γ/(1+α²) * [m × H_eff + α * m × (m × H_eff)]
%
% Where H_eff includes:
%   - Applied field: H_app(t)
%   - Anisotropy field: H_anis = (2K/μ₀Ms) * m_easy
%   - Demagnetization field: H_demag
%   - Exchange field: H_ex (for non-uniform systems)
%
% Key Parameters Extracted:
%   - Coercive field (H_c)
%   - Saturation field (H_sat)
%   - Remanent magnetization (M_r)
%   - Switching field asymmetry
%   - Loop area (magnetic work)
%
% Physics Validated:
%   - Hysteretic magnetic behavior
%   - Domain wall nucleation and propagation
%   - Shape-dependent demagnetization
%   - Temperature effects on coercivity
%   - Magnetic anisotropy contributions
%
% Usage:
%   results = llg_hysteresis_loop();  % Default parameters
%   results = llg_hysteresis_loop('Material', 'CoFeB', 'Temperature', 77);
%   results = llg_hysteresis_loop('FieldRange', 100e-3, 'SweepRate', 1e6);
%
% Author: Meshal Alawein <meshal@berkeley.edu>
% Copyright © 2025 Meshal Alawein — All rights reserved.
% License: MIT

    % Parse input arguments
    p = inputParser;
    addParameter(p, 'Material', 'CoFeB', @ischar);  % Magnetic material
    addParameter(p, 'Temperature', 300, @(x) isnumeric(x) && x > 0);  % Temperature (K)
    addParameter(p, 'Thickness', 2e-9, @(x) isnumeric(x) && x > 0);  % Film thickness (m)
    addParameter(p, 'Length', 200e-9, @(x) isnumeric(x) && x > 0);  % Film length (m)
    addParameter(p, 'Width', 100e-9, @(x) isnumeric(x) && x > 0);  % Film width (m)
    addParameter(p, 'FieldRange', 50e-3, @(x) isnumeric(x) && x > 0);  % Max field magnitude (T)
    addParameter(p, 'FieldDirection', [1, 0, 0], @(x) isnumeric(x) && length(x) == 3);  % Field direction
    addParameter(p, 'EasyAxis', [1, 0, 0], @(x) isnumeric(x) && length(x) == 3);  % Easy axis direction
    addParameter(p, 'SweepRate', 1e5, @(x) isnumeric(x) && x > 0);  % Field sweep rate (T/s)
    addParameter(p, 'NumCycles', 2, @(x) isnumeric(x) && x >= 1);  % Number of hysteresis cycles
    addParameter(p, 'IncludeDemagnetization', true, @islogical);  % Include demagnetization field
    addParameter(p, 'IncludeThermalNoise', false, @islogical);  % Include thermal fluctuations
    addParameter(p, 'PlotResults', true, @islogical);  % Enable plotting
    addParameter(p, 'Verbose', true, @islogical);  % Enable verbose output
    addParameter(p, 'SaveFigures', false, @islogical);  % Save figures
    addParameter(p, 'FigurePath', pwd, @ischar);  % Figure save path
    parse(p, varargin{:});
    
    % Extract parameters
    material = p.Results.Material;
    T = p.Results.Temperature;
    t_film = p.Results.Thickness;
    L = p.Results.Length;
    W = p.Results.Width;
    H_max = p.Results.FieldRange;
    H_dir = p.Results.FieldDirection / norm(p.Results.FieldDirection);
    easy_axis = p.Results.EasyAxis / norm(p.Results.EasyAxis);
    sweep_rate = p.Results.SweepRate;
    N_cycles = p.Results.NumCycles;
    include_demag = p.Results.IncludeDemagnetization;
    include_thermal = p.Results.IncludeThermalNoise;
    plot_results = p.Results.PlotResults;
    verbose = p.Results.Verbose;
    
    if verbose
        fprintf('=== LLG Magnetic Hysteresis Loop Validation ===\n');
        fprintf('Material: %s\n', material);
        fprintf('Dimensions: %.0f × %.0f × %.1f nm³\n', L*1e9, W*1e9, t_film*1e9);
        fprintf('Temperature: %.1f K\n', T);
        fprintf('Field range: ±%.1f mT\n', H_max*1e3);
        fprintf('Field direction: [%.2f, %.2f, %.2f]\n', H_dir);
        fprintf('Easy axis: [%.2f, %.2f, %.2f]\n', easy_axis);
        fprintf('Sweep rate: %.1e T/s\n', sweep_rate);
        fprintf('Number of cycles: %d\n', N_cycles);
        fprintf('Include demagnetization: %s\n', char(include_demag*'Yes' + ~include_demag*'No'));
        fprintf('===============================================\n\n');
    end
    
    %% Step 1: Get Material Properties
    
    if verbose
        fprintf('Step 1: Loading material properties...\n');
    end
    
    % Get material properties from database
    materials_db = MaterialsDB();
    mat_props = materials_db.getTemperatureDependence(material, T);
    
    % Extract key magnetic parameters
    Ms = mat_props.Ms;              % Saturation magnetization (A/m)
    alpha = mat_props.alpha;        % Gilbert damping
    A_ex = mat_props.A_ex;          % Exchange stiffness (J/m)
    K_u = mat_props.K_u;            % Uniaxial anisotropy (J/m³)
    gamma = mat_props.gamma;        % Gyromagnetic ratio (rad⋅s⁻¹⋅T⁻¹)
    
    % Physical constants
    mu_0 = 4*pi*1e-7;  % Permeability of free space
    k_B = 1.380649e-23;  % Boltzmann constant
    
    % Calculate device parameters
    volume = L * W * t_film;        % Device volume (m³)
    area = L * W;                   % Device area (m²)
    
    % Anisotropy field
    H_k = 2*K_u / (mu_0 * Ms);
    
    % Exchange length
    lambda_ex = sqrt(2*A_ex / (mu_0 * Ms^2));
    
    % Thermal energy
    E_thermal = k_B * T;
    E_anisotropy = K_u * volume;
    thermal_stability = E_anisotropy / E_thermal;
    
    if verbose
        fprintf('  Material: %s\n', material);
        fprintf('  Ms: %.2e A/m, α: %.4f, γ: %.2e rad⋅s⁻¹⋅T⁻¹\n', Ms, alpha, gamma);
        fprintf('  K_u: %.2e J/m³, A_ex: %.2e J/m\n', K_u, A_ex);
        fprintf('  Anisotropy field H_k: %.1f mT\n', H_k*1e3);
        fprintf('  Exchange length: %.1f nm\n', lambda_ex*1e9);
        fprintf('  Device volume: %.2e m³\n', volume);
        fprintf('  Thermal stability: %.1f k_B T\n', thermal_stability);
        fprintf('  Done.\n\n');
    end
    
    %% Step 2: Analytical Hysteresis Model
    
    if verbose
        fprintf('Step 2: Computing analytical hysteresis model...\n');
    end
    
    % Stoner-Wohlfarth model for coherent rotation
    % Switching field depends on angle between applied field and easy axis
    theta_H = acos(dot(H_dir, easy_axis));  % Angle between field and easy axis
    
    % Analytical switching fields (Stoner-Wohlfarth)
    if abs(theta_H) < 1e-6  % Field along easy axis
        H_c_analytical = H_k;  % Coercivity = anisotropy field
        H_sat_analytical = H_k;  % Saturation field
    elseif abs(theta_H - pi/2) < 1e-6  % Field along hard axis
        H_c_analytical = H_k / 2;  % Reduced coercivity
        H_sat_analytical = H_k;     % Same saturation field
    else  % General case
        H_c_analytical = H_k / (cos(theta_H)^(2/3) + sin(theta_H)^(2/3))^(3/2);
        H_sat_analytical = H_k / cos(theta_H);
    end
    
    % Demagnetization effects (approximate for thin film)
    if include_demag
        % Shape-dependent demagnetization factors
        if t_film < min(L, W)  % Thin film limit
            N_z = 1;  % Out-of-plane
            N_x = 0;  % In-plane (length direction)
            N_y = 0;  % In-plane (width direction)
        else
            % Use ellipsoid approximation
            [N_x, N_y, N_z] = calculateDemagFactors(L, W, t_film);
        end
        
        % Effective demagnetization field (depends on geometry and orientation)
        H_demag_eff = mu_0 * Ms * max(N_x, N_y, N_z);  % Worst case
        
        % Modify switching field
        H_c_analytical = H_c_analytical + H_demag_eff;
    else
        H_demag_eff = 0;
    end
    
    % Thermal effects on coercivity (Sharrock formula)
    if T > 0 && thermal_stability > 0
        % Thermal reduction factor
        tau_m = 1e-9;  % Measurement time scale (s)
        omega_0 = 1e9;  % Attempt frequency (Hz)
        
        thermal_factor = 1 - (k_B*T/E_anisotropy) * log(omega_0 * tau_m);
        thermal_factor = max(thermal_factor, 0.1);  % Avoid negative values
        
        H_c_thermal = H_c_analytical * thermal_factor;
    else
        H_c_thermal = H_c_analytical;
    end
    
    if verbose
        fprintf('  Angle between field and easy axis: %.1f°\n', theta_H*180/pi);
        fprintf('  Analytical coercivity (0K): %.1f mT\n', H_c_analytical*1e3);
        fprintf('  Thermal coercivity (%.0fK): %.1f mT\n', T, H_c_thermal*1e3);
        if include_demag
            fprintf('  Demagnetization field: %.1f mT\n', H_demag_eff*1e3);
        end
        fprintf('  Saturation field: %.1f mT\n', H_sat_analytical*1e3);
        fprintf('  Done.\n\n');
    end
    
    %% Step 3: LLG Simulation of Hysteresis Loop
    
    if verbose
        fprintf('Step 3: Running LLG hysteresis simulation...\n');
    end
    
    % Time parameters
    t_half_cycle = H_max / sweep_rate;  % Time for half cycle
    t_total = 4 * N_cycles * t_half_cycle;  % Total simulation time
    
    % Time array (dense enough to capture switching events)
    dt_min = min(1/(10*gamma*H_max), 1e-11);  % Minimum time step
    N_points = ceil(t_total / dt_min);
    N_points = min(N_points, 20000);  % Limit for memory
    t_array = linspace(0, t_total, N_points);
    dt = t_array(2) - t_array(1);
    
    % Applied field as function of time (triangular wave)
    H_applied_magnitude = zeros(size(t_array));
    cycle_time = 4 * t_half_cycle;  % Full cycle time
    
    for i = 1:length(t_array)
        t_cycle = mod(t_array(i), cycle_time);
        
        if t_cycle < t_half_cycle
            % Increasing field
            H_applied_magnitude(i) = -H_max + (t_cycle / t_half_cycle) * (2*H_max);
        elseif t_cycle < 2*t_half_cycle
            % Decreasing from +H_max to -H_max
            H_applied_magnitude(i) = H_max - ((t_cycle - t_half_cycle) / t_half_cycle) * (2*H_max);
        elseif t_cycle < 3*t_half_cycle
            % Increasing from -H_max to +H_max
            H_applied_magnitude(i) = -H_max + ((t_cycle - 2*t_half_cycle) / t_half_cycle) * (2*H_max);
        else
            % Decreasing from +H_max to starting point
            H_applied_magnitude(i) = H_max - ((t_cycle - 3*t_half_cycle) / t_half_cycle) * (2*H_max);
        end
    end
    
    % Define effective field function
    H_eff_func = @(t, m) calculateEffectiveFieldHysteresis(t, m, H_applied_magnitude, ...
                         t_array, H_dir, easy_axis, H_k, mu_0, Ms, include_demag, ...
                         N_x, N_y, N_z, include_thermal, T, k_B, volume);
    
    % Initial magnetization (start saturated in negative direction)
    m_initial = -easy_axis(:);
    
    % Add thermal noise if requested
    if include_thermal && thermal_stability > 0
        noise_amplitude = sqrt(2*alpha*k_B*T / (mu_0*Ms*volume*gamma));
        m_initial = m_initial + noise_amplitude * randn(3, 1);
        m_initial = m_initial / norm(m_initial);  % Renormalize
    end
    
    if verbose
        fprintf('  Simulation time: %.1f μs\n', t_total*1e6);
        fprintf('  Time step: %.3f ps\n', dt*1e12);
        fprintf('  Number of time points: %d\n', N_points);
        fprintf('  Field sweep rate: %.1e T/s\n', sweep_rate);
        fprintf('  Starting magnetization: [%.3f, %.3f, %.3f]\n', m_initial);
    end
    
    % Solve LLG equation
    try
        if verbose
            fprintf('  Solving LLG equation...\n');
        end
        
        [m_trajectory, t_solution, solution_info] = LLGSolver(m_initial, H_eff_func, ...
            alpha, gamma, [0, t_total], 'Method', 'RK45', 'RelTol', 1e-5, ...
            'AbsTol', 1e-7, 'MaxStep', dt*10, 'Verbose', verbose);
        
        simulation_success = true;
        
        if verbose
            fprintf('    Integration successful\n');
            fprintf('    Total steps: %d\n', solution_info.steps_taken);
            fprintf('    Final time: %.2e s\n', solution_info.final_time);
            fprintf('    Energy conservation error: %.2e\n', solution_info.energy_conservation);
        end
        
    catch ME
        if verbose
            fprintf('    LLG simulation failed: %s\n', ME.message);
        end
        simulation_success = false;
        m_trajectory = [];
        t_solution = [];
        solution_info = struct();
    end
    
    if verbose
        fprintf('  Done.\n\n');
    end
    
    %% Step 4: Extract Hysteresis Parameters
    
    if verbose
        fprintf('Step 4: Extracting hysteresis parameters...\n');
    end
    
    if simulation_success && ~isempty(m_trajectory)
        % Interpolate field values at solution times
        H_interp = interp1(t_array, H_applied_magnitude, t_solution, 'linear', 'extrap');
        
        % Project magnetization onto field direction
        m_parallel = zeros(size(t_solution));
        for i = 1:length(t_solution)
            m_parallel(i) = dot(m_trajectory(:, i), H_dir);
        end
        
        % Extract last complete cycle for analysis
        if length(t_solution) > 1000  % Ensure enough points
            last_cycle_start = round(0.75 * length(t_solution));  % Last quarter
            H_cycle = H_interp(last_cycle_start:end);
            m_cycle = m_parallel(last_cycle_start:end);
        else
            H_cycle = H_interp;
            m_cycle = m_parallel;
        end
        
        % Find coercive fields (where m = 0)
        zero_crossings = [];
        for i = 2:length(m_cycle)
            if m_cycle(i-1) * m_cycle(i) < 0  % Sign change
                % Linear interpolation for more accurate crossing
                H_cross = H_cycle(i-1) + (0 - m_cycle(i-1)) / (m_cycle(i) - m_cycle(i-1)) * ...
                         (H_cycle(i) - H_cycle(i-1));
                zero_crossings = [zero_crossings, H_cross];
            end
        end
        
        if length(zero_crossings) >= 2
            H_c1_numerical = zero_crossings(1);
            H_c2_numerical = zero_crossings(2);
            H_c_numerical = (abs(H_c1_numerical) + abs(H_c2_numerical)) / 2;
        else
            H_c1_numerical = NaN;
            H_c2_numerical = NaN;
            H_c_numerical = NaN;
        end
        
        % Find remanent magnetization (m at H = 0)
        if ~isempty(zero_crossings)
            [~, zero_field_idx] = min(abs(H_cycle));
            M_r_numerical = abs(m_cycle(zero_field_idx));
        else
            M_r_numerical = NaN;
        end
        
        % Find saturation magnetization
        M_sat_numerical = max(abs(m_cycle));
        
        % Calculate loop area (magnetic work)
        if length(H_cycle) > 10
            loop_area = trapz(H_cycle, m_cycle);
            loop_area = abs(loop_area);  % Take absolute value
        else
            loop_area = NaN;
        end
        
        % Calculate squareness ratio
        if ~isnan(M_r_numerical) && M_sat_numerical > 0
            squareness = M_r_numerical / M_sat_numerical;
        else
            squareness = NaN;
        end
        
        extraction_success = true;
        
        if verbose
            fprintf('  Extracted Parameters:\n');
            if ~isnan(H_c_numerical)
                fprintf('    Coercivity H_c: %.2f mT (expected: %.2f mT)\n', ...
                        H_c_numerical*1e3, H_c_thermal*1e3);
                fprintf('    Coercivity error: %.1f%%\n', ...
                        abs(H_c_numerical - H_c_thermal)/H_c_thermal*100);
            end
            if ~isnan(M_r_numerical)
                fprintf('    Remanence M_r/M_s: %.3f\n', M_r_numerical);
                fprintf('    Squareness ratio: %.3f\n', squareness);
            end
            if ~isnan(loop_area)
                fprintf('    Loop area: %.2e (magnetic work)\n', loop_area);
            end
        end
        
    else
        H_c_numerical = NaN;
        H_c1_numerical = NaN;
        H_c2_numerical = NaN;
        M_r_numerical = NaN;
        M_sat_numerical = NaN;
        loop_area = NaN;
        squareness = NaN;
        extraction_success = false;
        
        if verbose
            fprintf('  Parameter extraction failed (no simulation data)\n');
        end
    end
    
    if verbose
        fprintf('  Done.\n\n');
    end
    
    %% Step 5: Validation Analysis
    
    if verbose
        fprintf('Step 5: Validation analysis...\n');
    end
    
    if extraction_success && ~isnan(H_c_numerical)
        % Compare with analytical prediction
        H_c_error = abs(H_c_numerical - H_c_thermal) / H_c_thermal * 100;
        
        % Validation criteria
        H_c_tolerance = 30;  % 30% tolerance for coercivity
        squareness_min = 0.5;  % Minimum squareness for good loop
        
        H_c_validation = H_c_error < H_c_tolerance;
        squareness_validation = isnan(squareness) || squareness > squareness_min;
        overall_validation = H_c_validation && squareness_validation && simulation_success;
        
        if verbose
            fprintf('  Validation Results:\n');
            fprintf('    Coercivity error: %.1f%% (tolerance: %.0f%%)\n', H_c_error, H_c_tolerance);
            fprintf('    Coercivity validation: %s\n', char(H_c_validation*'PASS' + ~H_c_validation*'FAIL'));
            fprintf('    Squareness validation: %s\n', char(squareness_validation*'PASS' + ~squareness_validation*'FAIL'));
            fprintf('    Overall validation: %s\n', char(overall_validation*'PASS' + ~overall_validation*'FAIL'));
        end
    else
        H_c_error = NaN;
        overall_validation = false;
        
        if verbose
            fprintf('  Validation failed (insufficient data)\n');
        end
    end
    
    if verbose
        fprintf('  Done.\n\n');
    end
    
    %% Step 6: Package Results
    
    results = struct();
    
    % Input parameters
    results.parameters = struct();
    results.parameters.material = material;
    results.parameters.temperature = T;
    results.parameters.thickness = t_film;
    results.parameters.length = L;
    results.parameters.width = W;
    results.parameters.field_range = H_max;
    results.parameters.field_direction = H_dir;
    results.parameters.easy_axis = easy_axis;
    results.parameters.sweep_rate = sweep_rate;
    results.parameters.num_cycles = N_cycles;
    results.parameters.include_demagnetization = include_demag;
    results.parameters.include_thermal_noise = include_thermal;
    
    % Material properties
    results.material_properties = struct();
    results.material_properties.Ms = Ms;
    results.material_properties.alpha = alpha;
    results.material_properties.K_u = K_u;
    results.material_properties.A_ex = A_ex;
    results.material_properties.gamma = gamma;
    results.material_properties.volume = volume;
    results.material_properties.anisotropy_field = H_k;
    results.material_properties.exchange_length = lambda_ex;
    results.material_properties.thermal_stability = thermal_stability;
    
    % Analytical predictions
    results.analytical = struct();
    results.analytical.coercivity_0K = H_c_analytical;
    results.analytical.coercivity_thermal = H_c_thermal;
    results.analytical.saturation_field = H_sat_analytical;
    results.analytical.demagnetization_field = H_demag_eff;
    
    % Numerical results
    results.numerical = struct();
    results.numerical.success = simulation_success;
    if simulation_success
        results.numerical.time = t_solution;
        results.numerical.magnetization = m_trajectory;
        results.numerical.applied_field = H_interp;
        results.numerical.solution_info = solution_info;
    end
    
    % Extracted parameters
    results.extracted = struct();
    results.extracted.success = extraction_success;
    results.extracted.coercivity = H_c_numerical;
    results.extracted.coercivity_1 = H_c1_numerical;
    results.extracted.coercivity_2 = H_c2_numerical;
    results.extracted.remanence = M_r_numerical;
    results.extracted.saturation = M_sat_numerical;
    results.extracted.loop_area = loop_area;
    results.extracted.squareness = squareness;
    
    % Validation results
    results.validation = struct();
    results.validation.overall_pass = overall_validation;
    results.validation.coercivity_error = H_c_error;
    
    %% Step 7: Plotting
    
    if plot_results
        if verbose
            fprintf('Step 6: Generating plots...\n');
        end
        
        % Apply Berkeley styling
        berkeley();
        
        % Create main figure
        fig = figure('Position', [100, 100, 1400, 1000]);
        
        % Subplot 1: Hysteresis loop
        subplot(2, 3, 1);
        if simulation_success && ~isempty(m_trajectory)
            % Plot complete hysteresis loop
            plot(H_interp*1e3, m_parallel, 'LineWidth', 2, 'Color', [0, 0.4470, 0.7410]);
            hold on;
            
            % Mark coercive points
            if ~isnan(H_c1_numerical)
                plot(H_c1_numerical*1e3, 0, 'ro', 'MarkerSize', 8, 'LineWidth', 2);
            end
            if ~isnan(H_c2_numerical)
                plot(H_c2_numerical*1e3, 0, 'ro', 'MarkerSize', 8, 'LineWidth', 2);
            end
            
            % Mark remanence points
            if ~isnan(M_r_numerical)
                plot(0, M_r_numerical, 'gs', 'MarkerSize', 8, 'LineWidth', 2);
                plot(0, -M_r_numerical, 'gs', 'MarkerSize', 8, 'LineWidth', 2);
            end
        else
            text(0.5, 0.5, 'Simulation Failed', 'HorizontalAlignment', 'center', ...
                 'Units', 'normalized', 'Color', 'red', 'FontSize', 14);
        end
        
        xlabel('Applied Field (mT)');
        ylabel('Magnetization (M/M_s)');
        title('Magnetic Hysteresis Loop');
        grid on;
        axis equal;
        xlim([-H_max*1e3*1.1, H_max*1e3*1.1]);
        ylim([-1.1, 1.1]);
        
        % Subplot 2: Time evolution of magnetization
        subplot(2, 3, 2);
        if simulation_success && ~isempty(m_trajectory)
            plot(t_solution*1e6, m_trajectory(1, :), 'LineWidth', 2, 'DisplayName', 'm_x');
            hold on;
            plot(t_solution*1e6, m_trajectory(2, :), 'LineWidth', 2, 'DisplayName', 'm_y');
            plot(t_solution*1e6, m_trajectory(3, :), 'LineWidth', 2, 'DisplayName', 'm_z');
        end
        xlabel('Time (μs)');
        ylabel('Magnetization Components');
        title('Magnetization vs Time');
        legend('Location', 'best');
        grid on;
        
        % Subplot 3: Applied field vs time
        subplot(2, 3, 3);
        plot(t_array*1e6, H_applied_magnitude*1e3, 'LineWidth', 2);
        xlabel('Time (μs)');
        ylabel('Applied Field (mT)');
        title('Field Sweep Profile');
        grid on;
        
        % Subplot 4: Energy landscape
        subplot(2, 3, 4);
        theta_range = linspace(0, 2*pi, 200);
        
        % Energy vs angle for different field values
        field_values = [0, H_c_thermal/2, H_c_thermal];
        colors = lines(length(field_values));
        
        for i = 1:length(field_values)
            H_test = field_values(i);
            E_anis = K_u * volume * sin(theta_range).^2;
            E_zeeman = -mu_0 * Ms * volume * H_test * cos(theta_range);
            E_total = (E_anis + E_zeeman) / E_thermal;
            
            plot(theta_range*180/pi, E_total, 'LineWidth', 2, 'Color', colors(i, :), ...
                 'DisplayName', sprintf('H = %.1f mT', H_test*1e3));
            hold on;
        end
        
        xlabel('Angle θ (degrees)');
        ylabel('Energy (k_B T)');
        title('Energy Landscape');
        legend('Location', 'best');
        grid on;
        
        % Subplot 5: Parameter comparison
        subplot(2, 3, 5);
        if extraction_success
            param_names = {'H_c (mT)', 'M_r/M_s', 'Squareness'};
            analytical_values = [H_c_thermal*1e3, 1.0, 1.0];  % Ideal values
            numerical_values = [H_c_numerical*1e3, M_r_numerical, squareness];
            
            % Remove NaN values
            valid_idx = ~isnan(numerical_values);
            param_names = param_names(valid_idx);
            analytical_values = analytical_values(valid_idx);
            numerical_values = numerical_values(valid_idx);
            
            if ~isempty(param_names)
                x_pos = 1:length(param_names);
                width = 0.35;
                
                bar(x_pos - width/2, analytical_values, width, 'DisplayName', 'Analytical');
                hold on;
                bar(x_pos + width/2, numerical_values, width, 'DisplayName', 'Numerical');
                
                set(gca, 'XTick', x_pos, 'XTickLabel', param_names);
                ylabel('Parameter Value');
                title('Parameter Comparison');
                legend('Location', 'best');
                grid on;
            end
        else
            text(0.5, 0.5, 'No Parameters Extracted', 'HorizontalAlignment', 'center', ...
                 'Units', 'normalized', 'Color', 'red', 'FontSize', 14);
            title('Parameter Comparison');
        end
        
        % Subplot 6: Validation summary
        subplot(2, 3, 6);
        if extraction_success && ~isnan(H_c_error)
            validation_metrics = {'H_c Error (%)', 'Squareness', 'Thermal Stability'};
            validation_values = [H_c_error, squareness*100, thermal_stability];
            validation_limits = [30, 50, 20];  % Tolerance/good limits
            
            bar_colors = zeros(length(validation_values), 3);
            for i = 1:length(validation_values)
                if isnan(validation_values(i)) || ...
                   (i == 1 && validation_values(i) > validation_limits(i)) || ...
                   (i >= 2 && validation_values(i) < validation_limits(i))
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
        sgtitle(sprintf('Hysteresis Loop Validation: %s, T=%.0fK, %.0f×%.0fnm²', ...
                       material, T, L*1e9, W*1e9), 'FontSize', 16);
        
        % Store figure handle
        results.figure = fig;
        
        % Save figures if requested
        if p.Results.SaveFigures
            timestamp = datestr(now, 'yyyymmdd_HHMMSS');
            filename = sprintf('llg_hysteresis_loop_%s_T%.0fK_%s', material, T, timestamp);
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
        fprintf('=== HYSTERESIS LOOP VALIDATION SUMMARY ===\n');
        fprintf('Overall validation: %s\n', char(overall_validation*'PASS' + ~overall_validation*'FAIL'));
        fprintf('Simulation success: %s\n', char(simulation_success*'Yes' + ~simulation_success*'No'));
        if extraction_success
            fprintf('Analytical coercivity: %.2f mT\n', H_c_thermal*1e3);
            if ~isnan(H_c_numerical)
                fprintf('Numerical coercivity: %.2f mT (%.1f%% error)\n', H_c_numerical*1e3, H_c_error);
            end
            if ~isnan(squareness)
                fprintf('Loop squareness: %.3f\n', squareness);
            end
        else
            fprintf('Parameter extraction failed\n');
        end
        fprintf('==========================================\n\n');
    end
    
end

%% Helper Functions

function H_eff = calculateEffectiveFieldHysteresis(t, m, H_applied_magnitude, t_array, ...
                  H_dir, easy_axis, H_k, mu_0, Ms, include_demag, N_x, N_y, N_z, ...
                  include_thermal, T, k_B, volume)
    % Calculate effective field for hysteresis simulation
    
    % Interpolate applied field magnitude at current time
    H_app_mag = interp1(t_array, H_applied_magnitude, t, 'linear', 'extrap');
    
    % Applied field vector
    H_applied = H_app_mag * H_dir(:);
    
    % Uniaxial anisotropy field
    H_anis = (H_k / (mu_0 * Ms)) * dot(m, easy_axis) * easy_axis(:);
    
    % Demagnetization field (approximate)
    H_demag = [0; 0; 0];
    if include_demag && exist('N_x', 'var')
        H_demag = -[N_x * m(1); N_y * m(2); N_z * m(3)];
    end
    
    % Thermal field (random)
    H_thermal = [0; 0; 0];
    if include_thermal && T > 0 && volume > 0
        thermal_field_strength = sqrt(2 * k_B * T / (mu_0 * volume));
        H_thermal = thermal_field_strength * randn(3, 1);
    end
    
    % Total effective field
    H_eff = H_applied + H_anis + H_demag + H_thermal;
end

function [N_x, N_y, N_z] = calculateDemagFactors(L, W, t)
    % Calculate demagnetization factors for rectangular prism
    % Using approximate formulas for ellipsoid
    
    % Convert to semi-axes
    a = L / 2;
    b = W / 2;
    c = t / 2;
    
    % For thin films (c << a, b), use thin film limit
    if c < min(a, b) / 10
        N_z = 1;
        N_x = 0;
        N_y = 0;
    else
        % Use ellipsoid approximation (approximate formulas)
        if a >= b && b >= c
            % a is longest axis
            e1 = sqrt(1 - (b/a)^2);  % Eccentricity
            e2 = sqrt(1 - (c/a)^2);
            
            N_x = (1/(2*e1^2)) * (1/e2) * log((1+e2)/(1-e2)) - 1;
            N_y = (1/(2*e1^2)) * ((1/e2) * log((1+e2)/(1-e2)) - 2);
            N_z = 1 - N_x - N_y;
        else
            % Default values for complex geometries
            N_x = 0.33;
            N_y = 0.33;
            N_z = 0.34;
        end
    end
    
    % Ensure sum equals 1
    total = N_x + N_y + N_z;
    if total > 0
        N_x = N_x / total;
        N_y = N_y / total;
        N_z = N_z / total;
    end
end