function results = transport_hanle_precession(varargin)
% TRANSPORT_HANLE_PRECESSION - Validate Hanle precession in spin transport
%
% This validation script demonstrates the Hanle effect - the dephasing of
% spin accumulation under transverse magnetic fields. It validates the
% SpinCirc framework's ability to capture spin precession in transport.
%
% The Hanle effect occurs when injected spin-polarized electrons precess
% around an applied magnetic field during their diffusion. This leads to
% a characteristic Lorentzian line shape in the nonlocal resistance:
%
%   R_nl(B) = R_nl(0) / (1 + (ω_L * τ_sf)²)
%
% Where:
%   ω_L = γ * B  - Larmor frequency (rad/s)
%   τ_sf         - Spin relaxation time (s)
%   γ            - Gyromagnetic ratio (rad⋅s⁻¹⋅T⁻¹)
%
% Physics Validated:
%   - Spin precession under magnetic fields
%   - Larmor frequency dependence
%   - Spin dephasing and relaxation
%   - Hanle line shape extraction
%   - Temperature dependence of spin coherence
%
% Usage:
%   results = transport_hanle_precession();  % Default parameters
%   results = transport_hanle_precession('Material', 'Cu', 'Temperature', 77);
%   results = transport_hanle_precession('FieldRange', [-100e-3, 100e-3], 'PlotResults', true);
%
% Author: Meshal Alawein <meshal@berkeley.edu>
% Copyright © 2025 Meshal Alawein — All rights reserved.
% License: MIT

    % Parse input arguments
    p = inputParser;
    addParameter(p, 'Material', 'Cu', @ischar);  % Transport channel material
    addParameter(p, 'Length', 2e-6, @(x) isnumeric(x) && x > 0);  % Channel length (m)
    addParameter(p, 'Width', 200e-9, @(x) isnumeric(x) && x > 0);  % Channel width (m)
    addParameter(p, 'Thickness', 50e-9, @(x) isnumeric(x) && x > 0);  % Channel thickness (m)
    addParameter(p, 'InjectorMaterial', 'Permalloy', @ischar);  % Spin injector material
    addParameter(p, 'DetectorMaterial', 'Permalloy', @ischar);  % Spin detector material
    addParameter(p, 'Temperature', 300, @(x) isnumeric(x) && x > 0);  % Temperature (K)
    addParameter(p, 'Current', 100e-6, @(x) isnumeric(x) && x > 0);  % Injection current (A)
    addParameter(p, 'FieldRange', [-50e-3, 50e-3], @(x) isnumeric(x) && length(x) == 2);  % B-field range (T)
    addParameter(p, 'NumFieldPoints', 51, @(x) isnumeric(x) && x > 0);  % Number of field points
    addParameter(p, 'FieldDirection', [0, 0, 1], @(x) isnumeric(x) && length(x) == 3);  % Field direction
    addParameter(p, 'InjectorDistance', 0.5e-6, @(x) isnumeric(x) && x > 0);  % Injector-detector separation
    addParameter(p, 'PlotResults', true, @islogical);  % Enable plotting
    addParameter(p, 'Verbose', true, @islogical);  % Enable verbose output
    addParameter(p, 'SaveFigures', false, @islogical);  % Save figures
    addParameter(p, 'FigurePath', pwd, @ischar);  % Figure save path
    parse(p, varargin{:});
    
    % Extract parameters
    material = p.Results.Material;
    L = p.Results.Length;
    W = p.Results.Width;
    t = p.Results.Thickness;
    inj_material = p.Results.InjectorMaterial;
    det_material = p.Results.DetectorMaterial;
    T = p.Results.Temperature;
    I_inj = p.Results.Current;
    B_range = p.Results.FieldRange;
    N_B = p.Results.NumFieldPoints;
    B_dir = p.Results.FieldDirection / norm(p.Results.FieldDirection);
    d_sep = p.Results.InjectorDistance;
    plot_results = p.Results.PlotResults;
    verbose = p.Results.Verbose;
    
    if verbose
        fprintf('=== Hanle Precession Transport Validation ===\n');
        fprintf('Channel: %s, %.1f μm × %.0f nm × %.0f nm\n', material, L*1e6, W*1e9, t*1e9);
        fprintf('Injector/Detector: %s/%s\n', inj_material, det_material);
        fprintf('Temperature: %.1f K\n', T);
        fprintf('Injection current: %.0f μA\n', I_inj*1e6);
        fprintf('Field range: [%.1f, %.1f] mT\n', B_range*1e3);
        fprintf('Field direction: [%.2f, %.2f, %.2f]\n', B_dir);
        fprintf('===============================================\n\n');
    end
    
    %% Step 1: Get Material Properties
    
    if verbose
        fprintf('Step 1: Loading material properties...\n');
    end
    
    % Get material properties from database
    materials_db = MaterialsDB();
    channel_props = materials_db.getTemperatureDependence(material, T);
    injector_props = materials_db.getTemperatureDependence(inj_material, T);
    detector_props = materials_db.getTemperatureDependence(det_material, T);
    
    % Extract key transport parameters
    lambda_sf = channel_props.lambda_sf;
    rho = channel_props.rho;
    if isfield(channel_props, 'tau_sf')
        tau_sf = channel_props.tau_sf;
    else
        % Calculate from diffusion relation: τ_sf = λ_sf² / D
        D = 0.1;  % Typical diffusion constant (m²/s)
        tau_sf = lambda_sf^2 / D;
    end
    
    % Gyromagnetic ratio (use electron value)
    gamma = 1.76e11;  % rad⋅s⁻¹⋅T⁻¹
    
    if verbose
        fprintf('  Channel material: %s\n', material);
        fprintf('  Spin diffusion length: %.0f nm\n', lambda_sf*1e9);
        fprintf('  Spin relaxation time: %.2f ps\n', tau_sf*1e12);
        fprintf('  Resistivity: %.2e Ω⋅m\n', rho);
        fprintf('  Gyromagnetic ratio: %.2e rad⋅s⁻¹⋅T⁻¹\n', gamma);
        fprintf('  Done.\n\n');
    end
    
    %% Step 2: Analytical Solution for Hanle Effect
    
    if verbose
        fprintf('Step 2: Computing analytical Hanle solution...\n');
    end
    
    % Generate magnetic field array
    B_field = linspace(B_range(1), B_range(2), N_B);
    
    % Calculate Larmor frequency for each field
    omega_L = gamma * abs(B_field);
    
    % Analytical Hanle line shape (Lorentzian)
    % For a nonlocal spin valve configuration with separation d:
    % R_nl(B) = R_nl(0) * exp(-d/λ_sf) / (1 + (ω_L * τ_sf)²)
    
    % Calculate zero-field nonlocal resistance
    % Using 1D spin diffusion model
    R_area = rho * lambda_sf / (W * t);  % Area resistance
    polarization = 0.1;  % Effective spin polarization
    R_nl_0 = R_area * polarization^2 * exp(-d_sep / lambda_sf);
    
    % Hanle line shape
    hanle_factor = 1 ./ (1 + (omega_L * tau_sf).^2);
    R_nl_analytical = R_nl_0 * hanle_factor;
    
    % Also calculate for different field orientations
    % Transverse field (Hanle effect)
    hanle_transverse = hanle_factor;
    
    % Longitudinal field (no dephasing)
    hanle_longitudinal = ones(size(B_field));  % No precession
    
    if verbose
        fprintf('  Zero-field R_nl: %.2e Ω\n', R_nl_0);
        fprintf('  Characteristic field B_1/2: %.2f mT\n', (1/(gamma*tau_sf))*1e3);
        fprintf('  Field range: [%.1f, %.1f] mT\n', B_range*1e3);
        fprintf('  Done.\n\n');
    end
    
    %% Step 3: Numerical Simulation using SpinTransport Solver
    
    if verbose
        fprintf('Step 3: Running numerical simulations...\n');
    end
    
    % Initialize arrays for numerical results
    R_nl_numerical = zeros(size(B_field));
    solve_times = zeros(size(B_field));
    
    for i = 1:length(B_field)
        if verbose && mod(i, 10) == 1
            fprintf('  Field point %d/%d (B = %.2f mT)...\n', i, N_B, B_field(i)*1e3);
        end
        
        try
            % Set up SpinTransport solver
            solver = SpinTransportSolver();
            
            % Configure geometry for nonlocal spin valve
            solver.setGeometry(L, W, t);
            
            % Set temperature
            solver.setTemperature(T);
            
            % Set magnetic field
            B_vector = B_field(i) * B_dir;
            solver.setMagneticField(B_vector);
            
            % Set up materials array (single channel)
            materials_array = channel_props;
            materials_array.type = 'N';  % Nonmagnetic channel
            solver.setMaterials(materials_array);
            
            % Generate mesh
            nx = 101;  % Fine mesh for accurate field effects
            solver.generateMesh('nx', nx, 'ny', 5, 'nz', 5);
            
            % Set boundary conditions
            % Inject spin current at one end, detect voltage at separation distance
            inj_node = round(0.1 * nx);  % Injector position
            det_node = round((0.1 + d_sep/L) * nx);  % Detector position
            
            bc_values = [];
            bc_values(1).node = inj_node;
            bc_values(1).current = I_inj;  % Spin current injection
            bc_values(2).node = det_node;
            bc_values(2).voltage = 0;      % Voltage detection (ground)
            bc_values(3).node = 1;
            bc_values(3).voltage = 0;      % Ground contact
            bc_values(4).node = nx;
            bc_values(4).voltage = 0;      % Ground contact
            
            solver.setBoundaryConditions('mixed', bc_values);
            
            % Solve transport equation
            tic;
            [V, I_s, solve_info] = solver.solve('verbose', false);
            solve_times(i) = toc;
            
            % Extract nonlocal resistance
            % R_nl = (V_det - V_ref) / I_inj
            n_nodes = length(V) / 4;
            V_spin = V(n_nodes+1:2*n_nodes);  % Spin voltage
            
            if det_node <= length(V_spin) && inj_node <= length(V_spin)
                V_nl = V_spin(det_node) - V_spin(inj_node);
                R_nl_numerical(i) = V_nl / I_inj;
            else
                R_nl_numerical(i) = 0;
            end
            
        catch ME
            if verbose
                fprintf('    Simulation failed: %s\n', ME.message);
            end
            R_nl_numerical(i) = NaN;
            solve_times(i) = NaN;
        end
    end
    
    if verbose
        valid_points = sum(~isnan(R_nl_numerical));
        avg_solve_time = nanmean(solve_times);
        fprintf('  Completed %d/%d field points\n', valid_points, N_B);
        fprintf('  Average solve time: %.3f seconds\n', avg_solve_time);
        fprintf('  Done.\n\n');
    end
    
    %% Step 4: Fit Hanle Line Shape and Extract Parameters
    
    if verbose
        fprintf('Step 4: Fitting Hanle line shape...\n');
    end
    
    % Remove NaN values for fitting
    valid_idx = ~isnan(R_nl_numerical);
    B_fit = B_field(valid_idx);
    R_fit = R_nl_numerical(valid_idx);
    
    if sum(valid_idx) >= 5  % Need minimum points for fitting
        % Fit to Lorentzian: R(B) = R0 / (1 + (B/B_half)²)
        % Where B_half = 1/(γ*τ_sf) is the half-width field
        
        % Initial guess
        R0_guess = max(R_fit);
        B_half_guess = B_fit(length(B_fit)//2);  % Rough estimate
        
        % Define fitting function
        hanle_fit_func = @(params) params(1) ./ (1 + (gamma * abs(B_fit) * params(2)).^2);
        
        % Objective function (least squares)
        objective = @(params) sum((hanle_fit_func(params) - R_fit).^2);
        
        % Fit parameters
        try
            options = optimset('Display', 'off');
            fitted_params = fminsearch(objective, [R0_guess, tau_sf], options);
            
            R0_fitted = fitted_params(1);
            tau_sf_fitted = fitted_params(2);
            B_half_fitted = 1 / (gamma * tau_sf_fitted);
            
            % Calculate goodness of fit
            R_fit_curve = hanle_fit_func(fitted_params);
            R_squared = 1 - sum((R_fit - R_fit_curve).^2) / sum((R_fit - mean(R_fit)).^2);
            
            fitting_success = true;
            
            if verbose
                fprintf('  Fitted parameters:\n');
                fprintf('    R_nl(0): %.2e Ω (expected: %.2e Ω)\n', R0_fitted, R_nl_0);
                fprintf('    τ_sf: %.2f ps (expected: %.2f ps)\n', tau_sf_fitted*1e12, tau_sf*1e12);
                fprintf('    B_1/2: %.2f mT (expected: %.2f mT)\n', B_half_fitted*1e3, (1/(gamma*tau_sf))*1e3);
                fprintf('    R²: %.4f\n', R_squared);
            end
            
        catch
            if verbose
                fprintf('  Fitting failed\n');
            end
            fitting_success = false;
            R0_fitted = NaN;
            tau_sf_fitted = NaN;
            B_half_fitted = NaN;
            R_squared = NaN;
        end
    else
        if verbose
            fprintf('  Insufficient data points for fitting\n');
        end
        fitting_success = false;
        R0_fitted = NaN;
        tau_sf_fitted = NaN;
        B_half_fitted = NaN;
        R_squared = NaN;
    end
    
    if verbose
        fprintf('  Done.\n\n');
    end
    
    %% Step 5: Error Analysis and Validation
    
    if verbose
        fprintf('Step 5: Analyzing results...\n');
    end
    
    % Calculate errors between analytical and numerical
    if sum(valid_idx) > 0
        R_analytical_interp = interp1(B_field, R_nl_analytical, B_fit, 'linear', 'extrap');
        
        % Calculate errors
        abs_error = abs(R_fit - R_analytical_interp);
        rel_error = abs_error ./ abs(R_analytical_interp) * 100;
        
        max_abs_error = max(abs_error);
        max_rel_error = max(rel_error);
        rms_error = sqrt(mean(abs_error.^2));
        mean_rel_error = mean(rel_error);
        
        % Validation criteria
        tolerance_rel = 20;  % 20% relative error tolerance
        validation_pass = max_rel_error < tolerance_rel && fitting_success;
        
        if verbose
            fprintf('  Error Analysis:\n');
            fprintf('    Max absolute error: %.2e Ω\n', max_abs_error);
            fprintf('    Max relative error: %.1f%%\n', max_rel_error);
            fprintf('    RMS error: %.2e Ω\n', rms_error);
            fprintf('    Mean relative error: %.1f%%\n', mean_rel_error);
            fprintf('    Validation: %s (< %.0f%% required)\n', ...
                    char(validation_pass*'PASS' + ~validation_pass*'FAIL'), tolerance_rel);
        end
    else
        max_abs_error = NaN;
        max_rel_error = NaN;
        rms_error = NaN;
        mean_rel_error = NaN;
        validation_pass = false;
        
        if verbose
            fprintf('  No valid numerical data for error analysis\n');
        end
    end
    
    if verbose
        fprintf('  Done.\n\n');
    end
    
    %% Step 6: Package Results
    
    % Create results structure
    results = struct();
    
    % Input parameters
    results.parameters = struct();
    results.parameters.material = material;
    results.parameters.length = L;
    results.parameters.width = W;
    results.parameters.thickness = t;
    results.parameters.injector_material = inj_material;
    results.parameters.detector_material = det_material;
    results.parameters.temperature = T;
    results.parameters.injection_current = I_inj;
    results.parameters.field_range = B_range;
    results.parameters.field_direction = B_dir;
    results.parameters.injector_distance = d_sep;
    
    % Material properties
    results.material_properties = struct();
    results.material_properties.lambda_sf = lambda_sf;
    results.material_properties.tau_sf = tau_sf;
    results.material_properties.resistivity = rho;
    results.material_properties.gamma = gamma;
    
    % Analytical solution
    results.analytical = struct();
    results.analytical.magnetic_field = B_field;
    results.analytical.resistance_nl = R_nl_analytical;
    results.analytical.larmor_frequency = omega_L;
    results.analytical.hanle_factor = hanle_factor;
    results.analytical.R_nl_zero_field = R_nl_0;
    results.analytical.characteristic_field = 1/(gamma*tau_sf);
    
    % Numerical solution
    results.numerical = struct();
    results.numerical.magnetic_field = B_field;
    results.numerical.resistance_nl = R_nl_numerical;
    results.numerical.solve_times = solve_times;
    results.numerical.valid_points = sum(valid_idx);
    
    % Fitting results
    results.fitting = struct();
    results.fitting.success = fitting_success;
    results.fitting.R0_fitted = R0_fitted;
    results.fitting.tau_sf_fitted = tau_sf_fitted;
    results.fitting.B_half_fitted = B_half_fitted;
    results.fitting.r_squared = R_squared;
    
    % Validation results
    results.validation = struct();
    results.validation.overall_pass = validation_pass;
    results.validation.max_abs_error = max_abs_error;
    results.validation.max_rel_error = max_rel_error;
    results.validation.rms_error = rms_error;
    results.validation.mean_rel_error = mean_rel_error;
    
    %% Step 7: Plotting
    
    if plot_results
        if verbose
            fprintf('Step 6: Generating plots...\n');
        end
        
        % Apply Berkeley styling
        berkeley();
        
        % Create main figure
        fig = figure('Position', [100, 100, 1400, 1000]);
        
        % Subplot 1: Hanle curve comparison
        subplot(2, 3, 1);
        plot(B_field*1e3, R_nl_analytical*1e6, 'LineWidth', 3, 'DisplayName', 'Analytical');
        hold on;
        if sum(valid_idx) > 0
            plot(B_field*1e3, R_nl_numerical*1e6, 'o--', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Numerical');
        end
        if fitting_success
            B_fine = linspace(B_range(1), B_range(2), 200);
            R_fit_fine = R0_fitted ./ (1 + (gamma * abs(B_fine) * tau_sf_fitted).^2);
            plot(B_fine*1e3, R_fit_fine*1e6, ':', 'LineWidth', 2, 'DisplayName', 'Fitted');
        end
        xlabel('Magnetic Field (mT)');
        ylabel('Nonlocal Resistance (μΩ)');
        title('Hanle Effect in Spin Transport');
        legend('Location', 'best');
        grid on;
        
        % Subplot 2: Normalized Hanle curves
        subplot(2, 3, 2);
        plot(B_field*1e3, R_nl_analytical/R_nl_0, 'LineWidth', 3, 'DisplayName', 'Analytical');
        hold on;
        if sum(valid_idx) > 0 && ~isnan(R0_fitted)
            plot(B_field(valid_idx)*1e3, R_nl_numerical(valid_idx)/R0_fitted, 'o--', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Numerical');
        end
        xlabel('Magnetic Field (mT)');
        ylabel('Normalized R_{nl}/R_{nl}(0)');
        title('Normalized Hanle Line Shape');
        legend('Location', 'best');
        grid on;
        ylim([0, 1.1]);
        
        % Subplot 3: Larmor frequency dependence
        subplot(2, 3, 3);
        omega_L_plot = omega_L / 1e9;  % Convert to GHz
        plot(omega_L_plot, hanle_factor, 'LineWidth', 3);
        xlabel('Larmor Frequency (GHz)');
        ylabel('Hanle Factor');
        title('Frequency Domain Representation');
        grid on;
        
        % Add characteristic frequency line
        omega_char = 1/tau_sf / 1e9;  % GHz
        xline(omega_char, '--r', 'LineWidth', 2, 'DisplayName', sprintf('1/τ_{sf} = %.1f GHz', omega_char));
        legend('Location', 'best');
        
        % Subplot 4: Error analysis
        subplot(2, 3, 4);
        if sum(valid_idx) > 0
            yyaxis left;
            plot(B_fit*1e3, abs_error*1e6, 'o-', 'LineWidth', 2, 'Color', [0, 0.4470, 0.7410]);
            ylabel('Absolute Error (μΩ)');
            yyaxis right;
            plot(B_fit*1e3, rel_error, 's--', 'LineWidth', 2, 'Color', [0.8500, 0.3250, 0.0980]);
            ylabel('Relative Error (%)');
            xlabel('Magnetic Field (mT)');
            title('Numerical vs Analytical Error');
            grid on;
        else
            text(0.5, 0.5, 'No Valid Numerical Data', 'HorizontalAlignment', 'center');
            title('Error Analysis');
        end
        
        % Subplot 5: Parameter extraction validation
        subplot(2, 3, 5);
        if fitting_success
            param_names = {'R_{nl}(0) (μΩ)', 'τ_{sf} (ps)', 'B_{1/2} (mT)'};
            expected = [R_nl_0*1e6, tau_sf*1e12, (1/(gamma*tau_sf))*1e3];
            extracted = [R0_fitted*1e6, tau_sf_fitted*1e12, B_half_fitted*1e3];
            
            x_pos = 1:length(param_names);
            width = 0.35;
            
            bar(x_pos - width/2, expected, width, 'DisplayName', 'Expected');
            hold on;
            bar(x_pos + width/2, extracted, width, 'DisplayName', 'Extracted');
            
            set(gca, 'XTick', x_pos, 'XTickLabel', param_names);
            ylabel('Parameter Value');
            title('Parameter Extraction Validation');
            legend('Location', 'best');
            grid on;
        else
            text(0.5, 0.5, 'Parameter Extraction Failed', 'HorizontalAlignment', 'center');
            title('Parameter Extraction');
        end
        
        % Subplot 6: Validation summary
        subplot(2, 3, 6);
        if sum(valid_idx) > 0
            validation_metrics = {'Max Rel Error (%)', 'RMS Error (μΩ)', 'Fit R²'};
            if fitting_success
                validation_values = [max_rel_error, rms_error*1e6, R_squared*100];
                validation_limits = [20, inf, 90];  % Tolerance limits
            else
                validation_values = [max_rel_error, rms_error*1e6, 0];
                validation_limits = [20, inf, 90];
            end
            
            bar_colors = zeros(length(validation_values), 3);
            for i = 1:length(validation_values)
                if isnan(validation_values(i)) || ...
                   (i <= 2 && validation_values(i) > validation_limits(i)) || ...
                   (i == 3 && validation_values(i) < validation_limits(i))
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
            text(0.5, 0.5, 'Validation Failed\n(No Numerical Data)', ...
                 'HorizontalAlignment', 'center', 'Color', 'red', 'FontSize', 14);
            title('Validation Summary');
        end
        
        % Add overall title
        sgtitle(sprintf('Hanle Precession Validation: %s, T=%.0fK, d=%.1fμm', ...
                       material, T, d_sep*1e6), 'FontSize', 16);
        
        % Store figure handle
        results.figure = fig;
        
        % Save figures if requested
        if p.Results.SaveFigures
            timestamp = datestr(now, 'yyyymmdd_HHMMSS');
            filename = sprintf('transport_hanle_precession_%s_T%.0fK_%s', material, T, timestamp);
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
        fprintf('=== HANLE PRECESSION VALIDATION SUMMARY ===\n');
        fprintf('Overall validation: %s\n', char(validation_pass*'PASS' + ~validation_pass*'FAIL'));
        if sum(valid_idx) > 0
            fprintf('Maximum relative error: %.1f%%\n', max_rel_error);
            fprintf('RMS error: %.2e Ω\n', rms_error);
        end
        if fitting_success
            fprintf('Parameter extraction R²: %.4f\n', R_squared);
            fprintf('Extracted τ_sf: %.2f ps (expected: %.2f ps)\n', tau_sf_fitted*1e12, tau_sf*1e12);
        else
            fprintf('Parameter extraction failed\n');
        end
        fprintf('===========================================\n\n');
    end
    
end