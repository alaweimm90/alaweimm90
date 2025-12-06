function results = transport_interface_resistance(varargin)
% TRANSPORT_INTERFACE_RESISTANCE - Validate F/N interface characterization
%
% This validation script demonstrates the characterization of ferromagnet/
% normal metal interfaces, including Sharvin resistance, tunnel barriers,
% and spin-dependent transport. It validates the SpinCirc framework's
% interface parameter extraction capabilities.
%
% The interface resistance model includes:
%   1. Sharvin resistance: R_S = ρλ/A for ballistic transport
%   2. Tunnel resistance: R_T = (RA)_barrier / A for insulating barriers
%   3. Spin asymmetry: Different resistances for spin-up/down electrons
%   4. Temperature dependence of interface parameters
%
% Key Interface Parameters:
%   - Interface resistance area product (RA)
%   - Spin asymmetry parameter (γ)
%   - Barrier height and thickness
%   - Mixing conductances (g_r, g_i)
%
% Physics Validated:
%   - Interface resistance vs area scaling
%   - Spin-dependent transport coefficients
%   - Tunnel magnetoresistance (TMR)
%   - Temperature dependence of interface properties
%   - Contact vs bulk resistance separation
%
% Usage:
%   results = transport_interface_resistance();  % Default parameters
%   results = transport_interface_resistance('InterfaceType', 'tunnel', 'Barrier', 'MgO');
%   results = transport_interface_resistance('FerromagnetMaterial', 'CoFeB', 'Temperature', 77);
%
% Author: Meshal Alawein <meshal@berkeley.edu>
% Copyright © 2025 Meshal Alawein — All rights reserved.
% License: MIT

    % Parse input arguments
    p = inputParser;
    addParameter(p, 'FerromagnetMaterial', 'CoFeB', @ischar);  % Ferromagnet material
    addParameter(p, 'NormalMaterial', 'Cu', @ischar);  % Normal metal material
    addParameter(p, 'BarrierMaterial', 'MgO', @ischar);  % Barrier material (if tunnel)
    addParameter(p, 'InterfaceType', 'metallic', @ischar);  % 'metallic' or 'tunnel'
    addParameter(p, 'BarrierThickness', 1.2e-9, @(x) isnumeric(x) && x > 0);  % Barrier thickness (m)
    addParameter(p, 'Temperature', 300, @(x) isnumeric(x) && x > 0);  % Temperature (K)
    addParameter(p, 'AreaRange', [1e-14, 1e-10], @(x) isnumeric(x) && length(x) == 2);  % Area range (m²)
    addParameter(p, 'NumAreaPoints', 20, @(x) isnumeric(x) && x > 0);  % Number of area points
    addParameter(p, 'BiasVoltageRange', [-0.5, 0.5], @(x) isnumeric(x) && length(x) == 2);  % Bias range (V)
    addParameter(p, 'NumBiasPoints', 51, @(x) isnumeric(x) && x > 0);  % Number of bias points
    addParameter(p, 'TestCurrent', 1e-6, @(x) isnumeric(x) && x > 0);  % Test current (A)
    addParameter(p, 'PlotResults', true, @islogical);  % Enable plotting
    addParameter(p, 'Verbose', true, @islogical);  % Enable verbose output
    addParameter(p, 'SaveFigures', false, @islogical);  % Save figures
    addParameter(p, 'FigurePath', pwd, @ischar);  % Figure save path
    parse(p, varargin{:});
    
    % Extract parameters
    fm_material = p.Results.FerromagnetMaterial;
    nm_material = p.Results.NormalMaterial;
    barrier_material = p.Results.BarrierMaterial;
    interface_type = p.Results.InterfaceType;
    t_barrier = p.Results.BarrierThickness;
    T = p.Results.Temperature;
    area_range = p.Results.AreaRange;
    N_area = p.Results.NumAreaPoints;
    bias_range = p.Results.BiasVoltageRange;
    N_bias = p.Results.NumBiasPoints;
    I_test = p.Results.TestCurrent;
    plot_results = p.Results.PlotResults;
    verbose = p.Results.Verbose;
    
    if verbose
        fprintf('=== F/N Interface Resistance Validation ===\n');
        fprintf('Interface: %s/%s', fm_material, nm_material);
        if strcmp(interface_type, 'tunnel')
            fprintf(' with %s barrier (%.2f nm)\n', barrier_material, t_barrier*1e9);
        else
            fprintf(' (metallic)\n');
        end
        fprintf('Temperature: %.1f K\n', T);
        fprintf('Area range: [%.1e, %.1e] m²\n', area_range);
        fprintf('Interface type: %s\n', interface_type);
        fprintf('===========================================\n\n');
    end
    
    %% Step 1: Get Material Properties and Interface Parameters
    
    if verbose
        fprintf('Step 1: Loading material and interface properties...\n');
    end
    
    % Get material properties from database
    materials_db = MaterialsDB();
    fm_props = materials_db.getTemperatureDependence(fm_material, T);
    nm_props = materials_db.getTemperatureDependence(nm_material, T);
    
    % Get interface parameters
    if strcmp(interface_type, 'tunnel')
        barrier_props = materials_db.(barrier_material);
        interface_params = materials_db.getInterfaceParameters(fm_material, barrier_material);
    else
        interface_params = materials_db.getInterfaceParameters(fm_material, nm_material);
    end
    
    % Extract key parameters
    lambda_sf_fm = fm_props.lambda_sf;
    lambda_sf_nm = nm_props.lambda_sf;
    rho_fm = fm_props.rho;
    rho_nm = nm_props.rho;
    polarization = fm_props.polarization;
    
    if verbose
        fprintf('  Ferromagnet: %s\n', fm_material);
        fprintf('    λ_sf: %.0f nm, ρ: %.0f nΩ⋅m, P: %.2f\n', ...
                lambda_sf_fm*1e9, rho_fm*1e9, polarization);
        fprintf('  Normal metal: %s\n', nm_material);
        fprintf('    λ_sf: %.0f nm, ρ: %.0f nΩ⋅m\n', lambda_sf_nm*1e9, rho_nm*1e9);
        
        if strcmp(interface_type, 'tunnel')
            fprintf('  Barrier: %s (%.2f nm)\n', barrier_material, t_barrier*1e9);
            fprintf('    Barrier height: %.2f eV\n', barrier_props.barrier_height);
            fprintf('    RA product: %.2e Ω⋅μm²\n', interface_params.RA_product*1e12);
        else
            fprintf('  Interface type: metallic\n');
            fprintf('    Sharvin resistance factor: %.2e Ω⋅m²\n', rho_fm * lambda_sf_fm);
        end
        fprintf('  Done.\n\n');
    end
    
    %% Step 2: Analytical Models for Interface Resistance
    
    if verbose
        fprintf('Step 2: Computing analytical interface models...\n');
    end
    
    % Generate area array (logarithmic spacing)
    areas = logspace(log10(area_range(1)), log10(area_range(2)), N_area);
    
    % Calculate theoretical interface resistances
    if strcmp(interface_type, 'tunnel')
        % Tunnel barrier model
        % R = (RA)_barrier / A + R_contact
        RA_barrier = interface_params.RA_product;
        R_contact = rho_fm * lambda_sf_fm / max(areas);  % Contact resistance
        
        R_interface_analytical = RA_barrier ./ areas + R_contact;
        
        % Spin-dependent tunneling (Julliere model)
        % TMR = 2*P1*P2 / (1 - P1*P2) where P1, P2 are polarizations
        P1 = polarization;
        P2 = 0;  % Normal metal (unpolarized)
        TMR_analytical = 2*P1*P2 / (1 - P1*P2);  % Zero for N metal
        
        % For F/I/F junctions, use both ferromagnet polarizations
        if strcmp(nm_material, fm_material)
            TMR_analytical = 2*P1*P1 / (1 - P1*P1);
        end
        
    else
        % Metallic interface (Sharvin resistance)
        % R = ρ*λ_sf / A for diffusive transport
        R_sharvin = rho_fm * lambda_sf_fm;
        R_interface_analytical = R_sharvin ./ areas;
        
        % Spin asymmetry in metallic contacts
        gamma_interface = 0.7;  % Typical spin asymmetry parameter
        TMR_analytical = gamma_interface^2 / (1 + gamma_interface^2);
    end
    
    % Temperature dependence
    T_ref = 300;  % Reference temperature
    if strcmp(interface_type, 'tunnel') && T ~= T_ref
        % Tunnel resistance temperature dependence
        % R(T) ∝ exp(φ * sqrt(T_ref/T)) for phonon-assisted tunneling
        phi_factor = 0.1;  % Empirical parameter
        temp_factor = exp(phi_factor * (sqrt(T_ref/T) - 1));
        R_interface_analytical = R_interface_analytical * temp_factor;
    end
    
    if verbose
        fprintf('  Interface model: %s\n', interface_type);
        if strcmp(interface_type, 'tunnel')
            fprintf('    RA product: %.2e Ω⋅μm²\n', RA_barrier*1e12);
            fprintf('    TMR (theoretical): %.1f%%\n', TMR_analytical*100);
        else
            fprintf('    Sharvin resistance: %.2e Ω⋅m²\n', R_sharvin);
            fprintf('    Spin asymmetry: %.1f%%\n', TMR_analytical*100);
        end
        fprintf('  Done.\n\n');
    end
    
    %% Step 3: Bias Voltage Dependence
    
    if verbose
        fprintf('Step 3: Computing bias voltage dependence...\n');
    end
    
    % Generate bias voltage array
    V_bias = linspace(bias_range(1), bias_range(2), N_bias);
    
    % Select medium area for bias dependence study
    A_test = sqrt(area_range(1) * area_range(2));  % Geometric mean
    
    if strcmp(interface_type, 'tunnel')
        % Tunnel junction I-V characteristics
        % I = I_0 * [exp(αV) - exp(-αV)] (Simmons model)
        % For small bias: I ≈ G_0 * V * (1 + α²V²/6)
        
        phi = barrier_props.barrier_height;  % eV
        alpha = sqrt(2 * 9.109e-31 * phi * 1.602e-19) * t_barrier / 1.055e-34;
        
        % Calculate conductance vs bias
        G_0 = A_test / (RA_barrier);  % Zero-bias conductance
        G_bias = G_0 * ones(size(V_bias));  % Linear for small bias
        
        % Nonlinear correction for larger bias
        for i = 1:length(V_bias)
            if abs(V_bias(i)) > 0.1  % Nonlinear regime
                correction = 1 + (alpha * abs(V_bias(i)))^2 / 6;
                G_bias(i) = G_0 * correction;
            end
        end
        
        R_bias_analytical = 1 ./ G_bias;
        
    else
        % Metallic contact - ohmic behavior
        R_bias_analytical = (R_sharvin / A_test) * ones(size(V_bias));
    end
    
    if verbose
        fprintf('  Test area: %.2e μm²\n', A_test*1e12);
        fprintf('  Bias range: [%.2f, %.2f] V\n', bias_range);
        if strcmp(interface_type, 'tunnel')
            fprintf('    Zero-bias resistance: %.2e Ω\n', 1/G_0);
            fprintf('    Tunnel parameter α: %.2e V⁻¹\n', alpha);
        end
        fprintf('  Done.\n\n');
    end
    
    %% Step 4: Numerical Simulations
    
    if verbose
        fprintf('Step 4: Running numerical interface simulations...\n');
    end
    
    % Initialize arrays for numerical results
    R_interface_numerical = zeros(size(areas));
    R_bias_numerical = zeros(size(V_bias));
    solve_times = zeros(size(areas));
    
    % Area dependence simulation
    for i = 1:length(areas)
        if verbose && mod(i, 5) == 1
            fprintf('  Area point %d/%d (A = %.2e μm²)...\n', i, N_area, areas(i)*1e12);
        end
        
        try
            % Calculate equivalent dimensions for the area
            W_equiv = sqrt(areas(i));  % Square contact
            L_equiv = 100e-9;  % Thin contact length
            t_equiv = areas(i) / W_equiv;  % Thickness
            
            % Set up SpinTransport solver
            solver = SpinTransportSolver();
            solver.setGeometry(L_equiv, W_equiv, t_equiv);
            solver.setTemperature(T);
            
            % Set up materials array
            if strcmp(interface_type, 'tunnel')
                % F/I/N structure
                materials_array(1) = fm_props;
                materials_array(1).type = 'F';
                materials_array(2) = barrier_props;
                materials_array(2).type = 'I';
                materials_array(2).thickness = t_barrier;
                materials_array(3) = nm_props;
                materials_array(3).type = 'N';
            else
                % F/N structure
                materials_array(1) = fm_props;
                materials_array(1).type = 'F';
                materials_array(2) = nm_props;
                materials_array(2).type = 'N';
            end
            
            solver.setMaterials(materials_array);
            
            % Set magnetization (parallel configuration)
            if length(materials_array) >= 2 && strcmp(materials_array(1).type, 'F')
                m_vectors = [1, 0, 0];  % FM aligned along x
                solver.setMagnetization(m_vectors);
            end
            
            % Generate mesh
            nx = 51;
            solver.generateMesh('nx', nx, 'ny', 5, 'nz', 5);
            
            % Set boundary conditions
            bc_values = [];
            bc_values(1).node = 1;
            bc_values(1).current = I_test;  % Current injection
            bc_values(2).node = nx;
            bc_values(2).voltage = 0;       % Ground
            
            solver.setBoundaryConditions('mixed', bc_values);
            
            % Solve transport equation
            tic;
            [V, I_s, solve_info] = solver.solve('verbose', false);
            solve_times(i) = toc;
            
            % Calculate interface resistance
            n_nodes = length(V) / 4;
            V_charge = V(1:n_nodes);
            
            if length(V_charge) >= 2
                Delta_V = V_charge(1) - V_charge(end);
                R_interface_numerical(i) = Delta_V / I_test;
            else
                R_interface_numerical(i) = NaN;
            end
            
        catch ME
            if verbose
                fprintf('    Simulation failed: %s\n', ME.message);
            end
            R_interface_numerical(i) = NaN;
            solve_times(i) = NaN;
        end
    end
    
    % Bias dependence simulation (using test area)
    if verbose
        fprintf('  Running bias dependence simulation...\n');
    end
    
    for i = 1:length(V_bias)
        if verbose && mod(i, 10) == 1
            fprintf('    Bias point %d/%d (V = %.2f V)...\n', i, N_bias, V_bias(i));
        end
        
        try
            % Use same setup as area simulation but with applied bias
            W_equiv = sqrt(A_test);
            L_equiv = 100e-9;
            t_equiv = A_test / W_equiv;
            
            solver = SpinTransportSolver();
            solver.setGeometry(L_equiv, W_equiv, t_equiv);
            solver.setTemperature(T);
            solver.setMaterials(materials_array);
            
            if exist('m_vectors', 'var')
                solver.setMagnetization(m_vectors);
            end
            
            solver.generateMesh('nx', 51, 'ny', 5, 'nz', 5);
            
            % Apply bias voltage
            bc_values = [];
            bc_values(1).node = 1;
            bc_values(1).voltage = V_bias(i);  % Bias voltage
            bc_values(2).node = nx;
            bc_values(2).voltage = 0;          % Ground
            
            solver.setBoundaryConditions('voltage', bc_values);
            
            [V, I_s, solve_info] = solver.solve('verbose', false);
            
            % Calculate resistance at this bias
            n_nodes = length(V) / 4;
            V_charge = V(1:n_nodes);
            I_total = sum(I_s.charge);
            
            if abs(I_total) > 1e-15
                R_bias_numerical(i) = V_bias(i) / I_total;
            else
                R_bias_numerical(i) = NaN;
            end
            
        catch ME
            R_bias_numerical(i) = NaN;
        end
    end
    
    if verbose
        valid_area_points = sum(~isnan(R_interface_numerical));
        valid_bias_points = sum(~isnan(R_bias_numerical));
        avg_solve_time = nanmean(solve_times);
        fprintf('  Completed %d/%d area points\n', valid_area_points, N_area);
        fprintf('  Completed %d/%d bias points\n', valid_bias_points, N_bias);
        fprintf('  Average solve time: %.3f seconds\n', avg_solve_time);
        fprintf('  Done.\n\n');
    end
    
    %% Step 5: Parameter Extraction and Validation
    
    if verbose
        fprintf('Step 5: Extracting interface parameters...\n');
    end
    
    % Fit area dependence to extract RA product
    valid_area_idx = ~isnan(R_interface_numerical);
    
    if sum(valid_area_idx) >= 3
        areas_fit = areas(valid_area_idx);
        R_fit = R_interface_numerical(valid_area_idx);
        
        if strcmp(interface_type, 'tunnel')
            % Fit R = RA/A + R_contact
            fit_func = @(params) params(1) ./ areas_fit + params(2);
            initial_guess = [median(R_fit .* areas_fit), min(R_fit)];
        else
            % Fit R = RA/A (no contact resistance)
            fit_func = @(params) params(1) ./ areas_fit;
            initial_guess = median(R_fit .* areas_fit);
        end
        
        % Objective function
        objective = @(params) sum((fit_func(params) - R_fit).^2);
        
        try
            options = optimset('Display', 'off');
            fitted_params = fminsearch(objective, initial_guess, options);
            
            RA_extracted = fitted_params(1);
            if strcmp(interface_type, 'tunnel')
                R_contact_extracted = fitted_params(2);
            else
                R_contact_extracted = 0;
            end
            
            % Calculate goodness of fit
            R_fit_curve = fit_func(fitted_params);
            R_squared = 1 - sum((R_fit - R_fit_curve).^2) / sum((R_fit - mean(R_fit)).^2);
            
            extraction_success = true;
            
        catch
            extraction_success = false;
            RA_extracted = NaN;
            R_contact_extracted = NaN;
            R_squared = NaN;
        end
    else
        extraction_success = false;
        RA_extracted = NaN;
        R_contact_extracted = NaN;
        R_squared = NaN;
    end
    
    % Error analysis
    if extraction_success
        if strcmp(interface_type, 'tunnel')
            RA_expected = interface_params.RA_product;
            RA_error = abs(RA_extracted - RA_expected) / RA_expected * 100;
        else
            RA_expected = rho_fm * lambda_sf_fm;
            RA_error = abs(RA_extracted - RA_expected) / RA_expected * 100;
        end
        
        validation_pass = RA_error < 50;  % 50% tolerance
        
        if verbose
            fprintf('  Parameter Extraction Results:\n');
            fprintf('    Expected RA: %.2e Ω⋅μm²\n', RA_expected*1e12);
            fprintf('    Extracted RA: %.2e Ω⋅μm² (%.1f%% error)\n', RA_extracted*1e12, RA_error);
            if strcmp(interface_type, 'tunnel')
                fprintf('    Contact resistance: %.2e Ω\n', R_contact_extracted);
            end
            fprintf('    Fit R²: %.4f\n', R_squared);
            fprintf('    Validation: %s\n', char(validation_pass*'PASS' + ~validation_pass*'FAIL'));
        end
    else
        RA_error = NaN;
        validation_pass = false;
        
        if verbose
            fprintf('  Parameter extraction failed\n');
        end
    end
    
    if verbose
        fprintf('  Done.\n\n');
    end
    
    %% Step 6: Package Results
    
    results = struct();
    
    % Input parameters
    results.parameters = struct();
    results.parameters.ferromagnet_material = fm_material;
    results.parameters.normal_material = nm_material;
    results.parameters.barrier_material = barrier_material;
    results.parameters.interface_type = interface_type;
    results.parameters.barrier_thickness = t_barrier;
    results.parameters.temperature = T;
    results.parameters.area_range = area_range;
    results.parameters.bias_range = bias_range;
    results.parameters.test_current = I_test;
    
    % Material and interface properties
    results.materials = struct();
    results.materials.ferromagnet = fm_props;
    results.materials.normal_metal = nm_props;
    results.materials.interface_params = interface_params;
    
    % Analytical solutions
    results.analytical = struct();
    results.analytical.areas = areas;
    results.analytical.interface_resistance = R_interface_analytical;
    results.analytical.bias_voltage = V_bias;
    results.analytical.bias_resistance = R_bias_analytical;
    results.analytical.TMR_theoretical = TMR_analytical;
    
    % Numerical solutions
    results.numerical = struct();
    results.numerical.areas = areas;
    results.numerical.interface_resistance = R_interface_numerical;
    results.numerical.bias_voltage = V_bias;
    results.numerical.bias_resistance = R_bias_numerical;
    results.numerical.solve_times = solve_times;
    
    % Parameter extraction
    results.extraction = struct();
    results.extraction.success = extraction_success;
    results.extraction.RA_extracted = RA_extracted;
    results.extraction.R_contact_extracted = R_contact_extracted;
    results.extraction.RA_error = RA_error;
    results.extraction.fit_r_squared = R_squared;
    
    % Validation
    results.validation = struct();
    results.validation.overall_pass = validation_pass;
    
    %% Step 7: Plotting
    
    if plot_results
        if verbose
            fprintf('Step 6: Generating plots...\n');
        end
        
        % Apply Berkeley styling
        berkeley();
        
        % Create main figure
        fig = figure('Position', [100, 100, 1400, 1000]);
        
        % Subplot 1: Interface resistance vs area
        subplot(2, 3, 1);
        loglog(areas*1e12, R_interface_analytical, 'LineWidth', 3, 'DisplayName', 'Analytical');
        hold on;
        if sum(valid_area_idx) > 0
            loglog(areas(valid_area_idx)*1e12, R_interface_numerical(valid_area_idx), 'o--', ...
                   'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Numerical');
        end
        if extraction_success
            R_fit_plot = RA_extracted ./ areas + R_contact_extracted;
            loglog(areas*1e12, R_fit_plot, ':', 'LineWidth', 2, 'DisplayName', 'Fitted');
        end
        xlabel('Interface Area (μm²)');
        ylabel('Interface Resistance (Ω)');
        title('Interface Resistance vs Area');
        legend('Location', 'best');
        grid on;
        
        % Subplot 2: RA product extraction
        subplot(2, 3, 2);
        if sum(valid_area_idx) > 0
            RA_numerical = R_interface_numerical(valid_area_idx) .* areas(valid_area_idx);
            semilogx(areas(valid_area_idx)*1e12, RA_numerical*1e12, 'o-', 'LineWidth', 2, 'DisplayName', 'Numerical RA');
            hold on;
        end
        RA_analytical_plot = R_interface_analytical .* areas;
        semilogx(areas*1e12, RA_analytical_plot*1e12, 'LineWidth', 3, 'DisplayName', 'Analytical RA');
        if extraction_success
            yline(RA_extracted*1e12, '--r', 'LineWidth', 2, 'DisplayName', sprintf('Extracted: %.2f', RA_extracted*1e12));
        end
        xlabel('Interface Area (μm²)');
        ylabel('RA Product (Ω⋅μm²)');
        title('RA Product Analysis');
        legend('Location', 'best');
        grid on;
        
        % Subplot 3: Bias voltage dependence
        subplot(2, 3, 3);
        plot(V_bias, R_bias_analytical*1e3, 'LineWidth', 3, 'DisplayName', 'Analytical');
        hold on;
        valid_bias_idx = ~isnan(R_bias_numerical);
        if sum(valid_bias_idx) > 0
            plot(V_bias(valid_bias_idx), R_bias_numerical(valid_bias_idx)*1e3, 'o--', ...
                 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Numerical');
        end
        xlabel('Bias Voltage (V)');
        ylabel('Resistance (mΩ)');
        title('Bias Voltage Dependence');
        legend('Location', 'best');
        grid on;
        
        % Subplot 4: Interface model comparison
        subplot(2, 3, 4);
        if strcmp(interface_type, 'tunnel')
            model_params = {'RA (Ω⋅μm²)', 'Barrier Height (eV)', 'TMR (%)'};
            if extraction_success
                values = [RA_extracted*1e12, barrier_props.barrier_height, TMR_analytical*100];
            else
                values = [interface_params.RA_product*1e12, barrier_props.barrier_height, TMR_analytical*100];
            end
        else
            model_params = {'RA (Ω⋅μm²)', 'Sharvin R (Ω⋅m²)', 'Spin Asym (%)'};
            if extraction_success
                values = [RA_extracted*1e12, rho_fm*lambda_sf_fm, TMR_analytical*100];
            else
                values = [rho_fm*lambda_sf_fm*1e12, rho_fm*lambda_sf_fm, TMR_analytical*100];
            end
        end
        
        bar(values);
        set(gca, 'XTickLabel', model_params);
        ylabel('Parameter Value');
        title('Interface Parameters');
        grid on;
        
        % Subplot 5: Temperature dependence (if available)
        subplot(2, 3, 5);
        T_range = [77, 200, 300, 400, 500];  % Temperature range
        R_temp = zeros(size(T_range));
        
        for i = 1:length(T_range)
            if strcmp(interface_type, 'tunnel')
                temp_factor = exp(0.1 * (sqrt(300/T_range(i)) - 1));
                R_temp(i) = (interface_params.RA_product / A_test) * temp_factor;
            else
                % Metallic contact - weaker temperature dependence
                rho_temp = rho_fm * (1 + 0.004 * (T_range(i) - 300));
                lambda_temp = lambda_sf_fm * sqrt(300 / T_range(i));
                R_temp(i) = (rho_temp * lambda_temp) / A_test;
            end
        end
        
        plot(T_range, R_temp*1e3, 'o-', 'LineWidth', 2);
        scatter(T, R_interface_analytical(round(end/2))*1e3, 100, 'r', 'filled');
        xlabel('Temperature (K)');
        ylabel('Interface Resistance (mΩ)');
        title('Temperature Dependence');
        grid on;
        
        % Subplot 6: Validation summary
        subplot(2, 3, 6);
        if extraction_success
            validation_metrics = {'RA Error (%)', 'Fit R²', 'Valid Points (%)'};
            validation_values = [RA_error, R_squared*100, sum(valid_area_idx)/N_area*100];
            validation_limits = [50, 80, 70];
            
            bar_colors = zeros(length(validation_values), 3);
            for i = 1:length(validation_values)
                if isnan(validation_values(i)) || ...
                   (i == 1 && validation_values(i) > validation_limits(i)) || ...
                   (i >= 2 && validation_values(i) < validation_limits(i))
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
            text(0.5, 0.5, 'Extraction Failed', 'HorizontalAlignment', 'center', ...
                 'Color', 'red', 'FontSize', 14);
            title('Validation Summary');
        end
        
        % Add overall title
        sgtitle(sprintf('Interface Characterization: %s/%s (%s), T=%.0fK', ...
                       fm_material, nm_material, interface_type, T), 'FontSize', 16);
        
        % Store figure handle
        results.figure = fig;
        
        % Save figures if requested
        if p.Results.SaveFigures
            timestamp = datestr(now, 'yyyymmdd_HHMMSS');
            filename = sprintf('transport_interface_resistance_%s_%s_%s_T%.0fK_%s', ...
                             fm_material, nm_material, interface_type, T, timestamp);
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
        fprintf('=== INTERFACE RESISTANCE VALIDATION SUMMARY ===\n');
        fprintf('Interface type: %s\n', interface_type);
        fprintf('Overall validation: %s\n', char(validation_pass*'PASS' + ~validation_pass*'FAIL'));
        if extraction_success
            fprintf('RA parameter error: %.1f%%\n', RA_error);
            fprintf('Extraction fit R²: %.4f\n', R_squared);
        else
            fprintf('Parameter extraction failed\n');
        end
        fprintf('Valid simulation points: %d/%d (%.1f%%)\n', ...
                sum(valid_area_idx), N_area, sum(valid_area_idx)/N_area*100);
        fprintf('================================================\n\n');
    end
    
end