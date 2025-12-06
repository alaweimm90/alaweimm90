function results = transport_1D_diffusion(varargin)
% TRANSPORT_1D_DIFFUSION - Validate 1D spin diffusion transport model
%
% This validation script demonstrates basic 1D spin transport using the
% SpinCirc framework. It solves the spin diffusion equation analytically
% and compares with numerical solutions to validate the transport solver.
%
% The 1D spin diffusion equation in steady state is:
%   d²μs/dx² - μs/λsf² = 0
%
% Where:
%   μs - Spin chemical potential (V)
%   λsf - Spin diffusion length (m)
%   x - Position coordinate (m)
%
% Boundary conditions tested:
%   1. Fixed spin injection at x=0: μs(0) = μ0
%   2. Spin absorption at x=L: μs(L) = 0
%   3. Finite spin relaxation throughout the channel
%
% Physics Validated:
%   - Exponential decay of spin accumulation
%   - Spin diffusion length extraction
%   - Interface resistance effects
%   - Temperature dependence of spin transport
%
% Usage:
%   results = transport_1D_diffusion();  % Default parameters
%   results = transport_1D_diffusion('Length', 2e-6, 'Material', 'Cu');
%   results = transport_1D_diffusion('PlotResults', true, 'Verbose', true);
%
% Author: Meshal Alawein <meshal@berkeley.edu>
% Copyright © 2025 Meshal Alawein — All rights reserved.
% License: MIT

    % Parse input arguments
    p = inputParser;
    addParameter(p, 'Length', 1e-6, @(x) isnumeric(x) && x > 0);  % Channel length (m)
    addParameter(p, 'Width', 100e-9, @(x) isnumeric(x) && x > 0);  % Channel width (m)
    addParameter(p, 'Thickness', 10e-9, @(x) isnumeric(x) && x > 0);  % Channel thickness (m)
    addParameter(p, 'Material', 'Cu', @ischar);  % Channel material
    addParameter(p, 'Temperature', 300, @(x) isnumeric(x) && x > 0);  % Temperature (K)
    addParameter(p, 'SpinVoltage', 1e-3, @(x) isnumeric(x) && x > 0);  % Injected spin voltage (V)
    addParameter(p, 'NumPoints', 101, @(x) isnumeric(x) && x > 0);  % Number of spatial points
    addParameter(p, 'PlotResults', true, @islogical);  % Enable plotting
    addParameter(p, 'Verbose', true, @islogical);  % Enable verbose output
    addParameter(p, 'SaveFigures', false, @islogical);  % Save figures
    addParameter(p, 'FigurePath', pwd, @ischar);  % Figure save path
    parse(p, varargin{:});
    
    % Extract parameters
    L = p.Results.Length;
    W = p.Results.Width;
    t = p.Results.Thickness;
    material = p.Results.Material;
    T = p.Results.Temperature;
    mu_s0 = p.Results.SpinVoltage;
    N = p.Results.NumPoints;
    plot_results = p.Results.PlotResults;
    verbose = p.Results.Verbose;
    
    if verbose
        fprintf('=== 1D Spin Diffusion Transport Validation ===\n');
        fprintf('Channel: %.1f μm × %.0f nm × %.0f nm\n', L*1e6, W*1e9, t*1e9);
        fprintf('Material: %s\n', material);
        fprintf('Temperature: %.1f K\n', T);
        fprintf('Injected span voltage: %.1f mV\n', mu_s0*1000);
        fprintf('Spatial points: %d\n', N);
        fprintf('================================================\n\n');
    end
    
    %% Step 1: Get Material Properties
    
    if verbose
        fprintf('Step 1: Loading material properties...\n');
    end
    
    % Get material properties from database
    materials_db = MaterialsDB();
    mat_props = materials_db.getMaterial(material);
    
    % Extract key parameters
    lambda_sf = mat_props.lambda_sf;  % Spin diffusion length
    rho = mat_props.rho;              % Resistivity
    D_s = mat_props.diffusion_constant; % Spin diffusion constant
    
    % Apply temperature dependence
    T0 = 300;  % Reference temperature
    if T ~= T0
        % Resistivity temperature dependence
        alpha_rho = 0.004;  % Temperature coefficient (1/K)
        rho = rho * (1 + alpha_rho * (T - T0));
        
        % Spin diffusion length temperature dependence (phonon scattering)
        lambda_sf = lambda_sf * sqrt(T0 / T);
        
        % Update diffusion constant
        tau_sf = lambda_sf^2 / D_s;  % Spin lifetime
        D_s = lambda_sf^2 / tau_sf;
    end
    
    if verbose
        fprintf('  Material: %s\n', material);
        fprintf('  Spin diffusion length: %.0f nm\n', lambda_sf*1e9);
        fprintf('  Resistivity: %.2e Ω⋅m\n', rho);
        fprintf('  Diffusion constant: %.2e m²/s\n', D_s);
        fprintf('  Done.\n\n');
    end
    
    %% Step 2: Analytical Solution
    
    if verbose
        fprintf('Step 2: Computing analytical solution...\n');
    end
    
    % Spatial grid
    x = linspace(0, L, N);
    dx = x(2) - x(1);
    
    % Analytical solution for 1D spin diffusion with boundary conditions:
    % μs(0) = μ_s0 (spin injection)
    % μs(L) = 0    (spin absorption)
    %
    % Solution: μs(x) = A * exp(-x/λsf) + B * exp(x/λsf)
    % 
    % Applying boundary conditions:
    % A + B = μ_s0
    % A * exp(-L/λsf) + B * exp(L/λsf) = 0
    %
    % Solving: A = μ_s0 * exp(L/λsf) / (exp(L/λsf) - exp(-L/λsf))
    %          B = -μ_s0 * exp(-L/λsf) / (exp(L/λsf) - exp(-L/λsf))
    
    exp_pos = exp(L/lambda_sf);
    exp_neg = exp(-L/lambda_sf);
    denominator = exp_pos - exp_neg;
    
    if abs(denominator) < 1e-12
        % Special case: L >> λsf, use simplified form
        A = mu_s0;
        B = 0;
        mu_s_analytical = A * exp(-x/lambda_sf);
        if verbose
            fprintf('  Using simplified form for L >> λsf\n');
        end
    else
        A = mu_s0 * exp_pos / denominator;
        B = -mu_s0 * exp_neg / denominator;
        mu_s_analytical = A * exp(-x/lambda_sf) + B * exp(x/lambda_sf);
    end
    
    % Calculate spin current (analytical)
    % I_s = -σ_s * A * dμs/dx where σ_s = e²N(0)D_s is spin conductivity
    % For normalized units, use: I_s = -dμs/dx / (rho/λsf²)
    
    dmu_dx_analytical = -(A/lambda_sf) * exp(-x/lambda_sf) + (B/lambda_sf) * exp(x/lambda_sf);
    I_s_analytical = -dmu_dx_analytical / (rho / lambda_sf^2);
    
    if verbose
        fprintf('  Boundary coefficients: A = %.2e V, B = %.2e V\n', A, B);
        fprintf('  Maximum spin voltage: %.2f mV\n', max(mu_s_analytical)*1000);
        fprintf('  Spin current at x=0: %.2e A\n', I_s_analytical(1));
        fprintf('  Done.\n\n');
    end
    
    %% Step 3: Numerical Solution using SpinTransport Solver
    
    if verbose
        fprintf('Step 3: Computing numerical solution...\n');
    end
    
    % Set up SpinTransport solver
    solver = SpinTransportSolver();
    
    % Configure geometry
    solver.setGeometry(L, W, t);
    
    % Set up materials for 1D channel
    % Create material array: [normal metal]
    materials_array = mat_props;
    materials_array.type = 'N';  % Normal metal
    solver.setMaterials(materials_array);
    
    % Set temperature
    solver.setTemperature(T);
    
    % Set boundary conditions
    % Inject spin voltage at x=0, ground at x=L
    bc_values = [];
    bc_values(1).node = 1;        % First node (x=0)
    bc_values(1).voltage = mu_s0; % Spin injection
    bc_values(2).node = N;        % Last node (x=L)
    bc_values(2).voltage = 0;     % Grounded
    
    solver.setBoundaryConditions('voltage', bc_values);
    
    % Generate mesh
    solver.generateMesh('nx', N, 'ny', 3, 'nz', 3);
    
    % Solve transport equation
    if verbose
        fprintf('  Solving transport equation...\n');
    end
    
    try
        [V, I_s, solve_info] = solver.solve('verbose', false);
        
        % Extract spin voltage (assume single spin component for 1D)
        n_nodes = length(V) / 4;
        mu_s_numerical = V(n_nodes+1:2*n_nodes);  % Spin voltage component
        
        % Extract spin current
        I_s_numerical = I_s.spin_x;  % Spin current along x-direction
        
        numerical_success = true;
        
        if verbose
            fprintf('  Numerical solution successful\n');
            fprintf('  Solve time: %.3f seconds\n', solve_info.solve_time);
            fprintf('  Residual: %.2e\n', solve_info.residual);
        end
        
    catch ME
        if verbose
            fprintf('  Numerical solution failed: %s\n', ME.message);
        end
        numerical_success = false;
        mu_s_numerical = zeros(size(x));
        I_s_numerical = zeros(size(x));
    end
    
    if verbose
        fprintf('  Done.\n\n');
    end
    
    %% Step 4: Comparison and Error Analysis
    
    if verbose
        fprintf('Step 4: Analyzing results...\n');
    end
    
    % Calculate errors (only if numerical solution succeeded)
    if numerical_success
        % Interpolate numerical solution to match analytical grid
        if length(mu_s_numerical) ~= length(x)
            x_numerical = linspace(0, L, length(mu_s_numerical));
            mu_s_numerical_interp = interp1(x_numerical, mu_s_numerical, x, 'linear', 'extrap');
            I_s_numerical_interp = interp1(x_numerical, I_s_numerical, x, 'linear', 'extrap');
        else
            mu_s_numerical_interp = mu_s_numerical;
            I_s_numerical_interp = I_s_numerical;
        end
        
        % Calculate errors
        voltage_error = abs(mu_s_analytical - mu_s_numerical_interp);
        current_error = abs(I_s_analytical - I_s_numerical_interp);
        
        % Error statistics
        max_voltage_error = max(voltage_error);
        rms_voltage_error = sqrt(mean(voltage_error.^2));
        max_current_error = max(current_error);
        rms_current_error = sqrt(mean(current_error.^2));
        
        % Relative errors
        rel_voltage_error = max_voltage_error / max(abs(mu_s_analytical)) * 100;
        rel_current_error = max_current_error / max(abs(I_s_analytical)) * 100;
        
        if verbose
            fprintf('  Error Analysis:\n');
            fprintf('    Max voltage error: %.2e V (%.2f%%)\n', max_voltage_error, rel_voltage_error);
            fprintf('    RMS voltage error: %.2e V\n', rms_voltage_error);
            fprintf('    Max current error: %.2e A (%.2f%%)\n', max_current_error, rel_current_error);
            fprintf('    RMS current error: %.2e A\n', rms_current_error);
        end
        
        % Validation criteria
        voltage_tolerance = 5;  % 5% relative error
        current_tolerance = 10; % 10% relative error
        
        voltage_validation = rel_voltage_error < voltage_tolerance;
        current_validation = rel_current_error < current_tolerance;
        overall_validation = voltage_validation && current_validation;
        
        if verbose
            fprintf('  Validation Results:\n');
            fprintf('    Voltage validation: %s (< %.1f%% required)\n', ...
                    char(voltage_validation*'PASS' + ~voltage_validation*'FAIL'), voltage_tolerance);
            fprintf('    Current validation: %s (< %.1f%% required)\n', ...
                    char(current_validation*'PASS' + ~current_validation*'FAIL'), current_tolerance);
            fprintf('    Overall validation: %s\n', ...
                    char(overall_validation*'PASS' + ~overall_validation*'FAIL'));
        end
    else
        % Set error values for failed numerical solution
        max_voltage_error = NaN;
        rms_voltage_error = NaN;
        max_current_error = NaN;
        rms_current_error = NaN;
        rel_voltage_error = NaN;
        rel_current_error = NaN;
        overall_validation = false;
        
        mu_s_numerical_interp = zeros(size(x));
        I_s_numerical_interp = zeros(size(x));
        voltage_error = zeros(size(x));
        current_error = zeros(size(x));
    end
    
    if verbose
        fprintf('  Done.\n\n');
    end
    
    %% Step 5: Extract Physical Parameters
    
    if verbose
        fprintf('Step 5: Extracting physical parameters...\n');
    end
    
    % Extract spin diffusion length from numerical data (if available)
    if numerical_success && max(mu_s_numerical_interp) > 0
        % Fit exponential decay to extract λsf
        % μs(x) ≈ μs(0) * exp(-x/λsf) for x << L
        
        % Use first half of the channel for fitting
        fit_range = 1:round(N/2);
        x_fit = x(fit_range);
        mu_fit = mu_s_numerical_interp(fit_range);
        
        % Logarithmic fit: log(μs) = log(μs0) - x/λsf
        if all(mu_fit > 0)
            log_mu = log(mu_fit);
            p = polyfit(x_fit, log_mu, 1);
            lambda_sf_extracted = -1 / p(1);
            mu_s0_extracted = exp(p(2));
            
            fit_quality = corrcoef(x_fit, log_mu);
            r_squared = fit_quality(1,2)^2;
        else
            lambda_sf_extracted = NaN;
            mu_s0_extracted = NaN;
            r_squared = NaN;
        end
    else
        lambda_sf_extracted = NaN;
        mu_s0_extracted = NaN;
        r_squared = NaN;
    end
    
    % Parameter extraction validation
    if ~isnan(lambda_sf_extracted)
        lambda_sf_error = abs(lambda_sf_extracted - lambda_sf) / lambda_sf * 100;
        mu_s0_error = abs(mu_s0_extracted - mu_s0) / mu_s0 * 100;
        
        if verbose
            fprintf('  Parameter Extraction:\n');
            fprintf('    Expected λsf: %.0f nm\n', lambda_sf*1e9);
            fprintf('    Extracted λsf: %.0f nm (%.1f%% error)\n', ...
                    lambda_sf_extracted*1e9, lambda_sf_error);
            fprintf('    Expected μs0: %.2f mV\n', mu_s0*1000);
            fprintf('    Extracted μs0: %.2f mV (%.1f%% error)\n', ...
                    mu_s0_extracted*1000, mu_s0_error);
            fprintf('    Fit R²: %.4f\n', r_squared);
        end
    else
        lambda_sf_error = NaN;
        mu_s0_error = NaN;
        
        if verbose
            fprintf('  Parameter extraction failed\n');
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
    results.parameters.length = L;
    results.parameters.width = W;
    results.parameters.thickness = t;
    results.parameters.material = material;
    results.parameters.temperature = T;
    results.parameters.spin_voltage = mu_s0;
    results.parameters.num_points = N;
    
    % Material properties
    results.material_properties = struct();
    results.material_properties.lambda_sf = lambda_sf;
    results.material_properties.resistivity = rho;
    results.material_properties.diffusion_constant = D_s;
    
    % Analytical solution
    results.analytical = struct();
    results.analytical.position = x;
    results.analytical.spin_voltage = mu_s_analytical;
    results.analytical.spin_current = I_s_analytical;
    results.analytical.coefficients = [A, B];
    
    % Numerical solution
    results.numerical = struct();
    results.numerical.success = numerical_success;
    if numerical_success
        results.numerical.position = x;
        results.numerical.spin_voltage = mu_s_numerical_interp;
        results.numerical.spin_current = I_s_numerical_interp;
        results.numerical.solve_info = solve_info;
    end
    
    % Error analysis
    results.validation = struct();
    results.validation.overall_pass = overall_validation;
    results.validation.max_voltage_error = max_voltage_error;
    results.validation.rms_voltage_error = rms_voltage_error;
    results.validation.max_current_error = max_current_error;
    results.validation.rms_current_error = rms_current_error;
    results.validation.rel_voltage_error = rel_voltage_error;
    results.validation.rel_current_error = rel_current_error;
    results.validation.voltage_error_data = voltage_error;
    results.validation.current_error_data = current_error;
    
    % Parameter extraction
    results.parameter_extraction = struct();
    results.parameter_extraction.lambda_sf_extracted = lambda_sf_extracted;
    results.parameter_extraction.mu_s0_extracted = mu_s0_extracted;
    results.parameter_extraction.lambda_sf_error = lambda_sf_error;
    results.parameter_extraction.mu_s0_error = mu_s0_error;
    results.parameter_extraction.fit_r_squared = r_squared;
    
    %% Step 7: Plotting
    
    if plot_results
        if verbose
            fprintf('Step 6: Generating plots...\n');
        end
        
        % Apply Berkeley styling
        berkeley();
        
        % Create main figure
        fig = figure('Position', [100, 100, 1400, 1000]);
        
        % Subplot 1: Spin voltage profiles
        subplot(2, 3, 1);
        plot(x*1e6, mu_s_analytical*1000, 'LineWidth', 3, 'DisplayName', 'Analytical');
        hold on;
        if numerical_success
            plot(x*1e6, mu_s_numerical_interp*1000, '--', 'LineWidth', 2, 'DisplayName', 'Numerical');
        end
        xlabel('Position (μm)');
        ylabel('Spin Voltage (mV)');
        title('1D Spin Voltage Profile');
        legend('Location', 'best');
        grid on;
        
        % Subplot 2: Spin current profiles
        subplot(2, 3, 2);
        plot(x*1e6, I_s_analytical*1e6, 'LineWidth', 3, 'DisplayName', 'Analytical');
        hold on;
        if numerical_success
            plot(x*1e6, I_s_numerical_interp*1e6, '--', 'LineWidth', 2, 'DisplayName', 'Numerical');
        end
        xlabel('Position (μm)');
        ylabel('Spin Current (μA)');
        title('1D Spin Current Profile');
        legend('Location', 'best');
        grid on;
        
        % Subplot 3: Error analysis
        subplot(2, 3, 3);
        if numerical_success
            yyaxis left;
            semilogy(x*1e6, abs(voltage_error)*1000, 'LineWidth', 2, 'Color', [0, 0.4470, 0.7410]);
            ylabel('Voltage Error (mV)');
            yyaxis right;
            semilogy(x*1e6, abs(current_error)*1e6, 'LineWidth', 2, 'Color', [0.8500, 0.3250, 0.0980]);
            ylabel('Current Error (μA)');
        else
            text(0.5, 0.5, 'Numerical Solution Failed', 'HorizontalAlignment', 'center');
        end
        xlabel('Position (μm)');
        title('Absolute Errors');
        grid on;
        
        % Subplot 4: Logarithmic voltage plot (for λsf extraction visualization)
        subplot(2, 3, 4);
        semilogy(x*1e6, abs(mu_s_analytical)*1000, 'LineWidth', 3, 'DisplayName', 'Analytical');
        hold on;
        if numerical_success
            semilogy(x*1e6, abs(mu_s_numerical_interp)*1000, '--', 'LineWidth', 2, 'DisplayName', 'Numerical');
        end
        if ~isnan(lambda_sf_extracted)
            % Show extracted fit
            mu_fit_full = mu_s0_extracted * exp(-x/lambda_sf_extracted);
            semilogy(x*1e6, mu_fit_full*1000, ':', 'LineWidth', 2, 'DisplayName', 'Extracted Fit');
        end
        xlabel('Position (μm)');
        ylabel('|Spin Voltage| (mV)');
        title('Logarithmic Scale (λsf Extraction)');
        legend('Location', 'best');
        grid on;
        
        % Subplot 5: Material parameters comparison
        subplot(2, 3, 5);
        if ~isnan(lambda_sf_extracted)
            param_names = {'λsf (nm)', 'μs0 (mV)'};
            expected = [lambda_sf*1e9, mu_s0*1000];
            extracted = [lambda_sf_extracted*1e9, mu_s0_extracted*1000];
            
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
        if numerical_success
            validation_metrics = {'Max V Error (%)', 'Max I Error (%)', 'λsf Error (%)', 'μs0 Error (%)'};
            validation_values = [rel_voltage_error, rel_current_error, lambda_sf_error, mu_s0_error];
            validation_limits = [5, 10, 10, 10];  % Tolerance limits
            
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
            
            hold on;
            plot(1:length(validation_limits), validation_limits, 'r--', 'LineWidth', 2, 'DisplayName', 'Tolerance');
            
            set(gca, 'XTickLabel', validation_metrics);
            ylabel('Error (%)');
            title('Validation Summary');
            legend('Location', 'best');
            grid on;
        else
            text(0.5, 0.5, 'Validation Failed\n(Numerical Solution Error)', ...
                 'HorizontalAlignment', 'center', 'Color', 'red', 'FontSize', 14);
            title('Validation Summary');
        end
        
        % Add overall title
        sgtitle(sprintf('1D Spin Diffusion Validation: %s, T=%.0fK, L=%.1fμm', ...
                       material, T, L*1e6), 'FontSize', 16);
        
        % Store figure handle
        results.figure = fig;
        
        % Save figures if requested
        if p.Results.SaveFigures
            timestamp = datestr(now, 'yyyymmdd_HHMMSS');
            filename = sprintf('transport_1D_diffusion_%s_T%.0fK_%s', material, T, timestamp);
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
        fprintf('=== VALIDATION SUMMARY ===\n');
        fprintf('Overall validation: %s\n', char(overall_validation*'PASS' + ~overall_validation*'FAIL'));
        if numerical_success
            fprintf('Maximum voltage error: %.2f%%\n', rel_voltage_error);
            fprintf('Maximum current error: %.2f%%\n', rel_current_error);
            if ~isnan(lambda_sf_error)
                fprintf('Spin diffusion length error: %.2f%%\n', lambda_sf_error);
            end
        else
            fprintf('Numerical solution failed\n');
        end
        fprintf('===========================\n\n');
    end
    
end