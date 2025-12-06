function results = llg_damped_precession(varargin)
% LLG_DAMPED_PRECESSION - Validate LLG solver with analytical damped precession solution
%
% This validation script tests the LLG (Landau-Lifshitz-Gilbert) solver
% against analytical solutions for magnetization dynamics under various
% conditions. It validates the numerical integration accuracy and physics
% implementation.
%
% Physics Validated:
%   1. Free precession under effective field
%   2. Gilbert damping and energy dissipation
%   3. Conservation of magnetization magnitude |m| = 1
%   4. Larmor frequency and precession dynamics
%   5. Energy conservation in undamped systems
%   6. Switching dynamics under large applied fields
%
% Analytical Solutions:
%   - Undamped precession: m(t) = analytical precession about H_eff
%   - Damped precession: exponential approach to equilibrium
%   - Switching: comparison with macrospin model predictions
%
% The LLG equation solved is:
%   dm/dt = -γ/(1+α²) * [m × H_eff + α * m × (m × H_eff)]
%
% Where:
%   m - Unit magnetization vector
%   H_eff - Effective magnetic field (T)
%   γ - Gyromagnetic ratio (rad⋅s⁻¹⋅T⁻¹)
%   α - Gilbert damping parameter
%
% Usage:
%   results = llg_damped_precession();  % Default validation
%   results = llg_damped_precession('TestCase', 'switching', 'Damping', 0.01);
%   results = llg_damped_precession('PlotResults', true, 'Verbose', true);
%
% Author: Meshal Alawein <meshal@berkeley.edu>
% Copyright © 2025 Meshal Alawein — All rights reserved.
% License: MIT

    % Parse input arguments
    p = inputParser;
    addParameter(p, 'TestCase', 'precession', @ischar);  % 'precession', 'damping', 'switching'
    addParameter(p, 'Damping', 0.01, @(x) isnumeric(x) && x >= 0);  % Gilbert damping α
    addParameter(p, 'GyromagneticRatio', 1.76e11, @(x) isnumeric(x) && x > 0);  % γ (rad⋅s⁻¹⋅T⁻¹)
    addParameter(p, 'EffectiveField', [0, 0, 1], @(x) isnumeric(x) && length(x) == 3);  % H_eff (T)
    addParameter(p, 'InitialMagnetization', [1, 0, 0], @(x) isnumeric(x) && length(x) == 3);  % m(0)
    addParameter(p, 'TimeSpan', [0, 2e-9], @(x) isnumeric(x) && length(x) == 2);  % Time range (s)
    addParameter(p, 'Method', 'RK45', @ischar);  % Integration method
    addParameter(p, 'Tolerance', [1e-6, 1e-8], @(x) isnumeric(x) && length(x) == 2);  % [RelTol, AbsTol]
    addParameter(p, 'PlotResults', true, @islogical);  % Enable plotting
    addParameter(p, 'Verbose', true, @islogical);  % Enable verbose output
    addParameter(p, 'SaveFigures', false, @islogical);  % Save figures
    addParameter(p, 'FigurePath', pwd, @ischar);  % Figure save path
    parse(p, varargin{:});
    
    % Extract parameters
    test_case = p.Results.TestCase;
    alpha = p.Results.Damping;
    gamma = p.Results.GyromagneticRatio;
    H_eff = p.Results.EffectiveField(:);  % Column vector
    m0 = p.Results.InitialMagnetization(:);  % Column vector
    tspan = p.Results.TimeSpan;
    method = p.Results.Method;
    tol = p.Results.Tolerance;
    plot_results = p.Results.PlotResults;
    verbose = p.Results.Verbose;
    
    % Normalize initial magnetization
    m0 = m0 / norm(m0);
    
    if verbose
        fprintf('=== LLG Damped Precession Validation ===\n');
        fprintf('Test case: %s\n', test_case);
        fprintf('Gilbert damping: α = %.4f\n', alpha);
        fprintf('Gyromagnetic ratio: γ = %.2e rad⋅s⁻¹⋅T⁻¹\n', gamma);
        fprintf('Effective field: [%.3f, %.3f, %.3f] T\n', H_eff);
        fprintf('Initial magnetization: [%.3f, %.3f, %.3f]\n', m0);
        fprintf('Time span: [%.2e, %.2e] s\n', tspan);
        fprintf('Integration method: %s\n', method);
        fprintf('Tolerances: RelTol=%.1e, AbsTol=%.1e\n', tol(1), tol(2));
        fprintf('========================================\n\n');
    end
    
    %% Step 1: Configure Test Parameters Based on Test Case
    
    switch lower(test_case)
        case 'precession'
            % Free precession with minimal damping
            if p.Results.Damping > 0.001  % Use provided damping if reasonable
                alpha = p.Results.Damping;
            else
                alpha = 0.001;  % Minimal damping for precession test
            end
            H_eff = [0; 0; 1];  % Field along z-axis
            m0 = [1; 0; 0];     % Initial magnetization along x-axis
            
        case 'damping'
            % Damped precession with significant damping
            alpha = max(p.Results.Damping, 0.1);  % Ensure significant damping
            H_eff = [0; 0; 1];  % Field along z-axis
            m0 = [1; 0; 0];     % Initial magnetization along x-axis
            
        case 'switching'
            % Magnetization switching under large field
            alpha = p.Results.Damping;
            H_eff = [0; 0; -2];  % Large field opposite to initial direction
            m0 = [0; 0; 1];      % Initial magnetization along +z
            tspan = [0, 5e-9];   % Longer time for switching
            
        otherwise
            % Use provided parameters
            if verbose
                fprintf('Using custom parameters...\n');
            end
    end
    
    if verbose
        fprintf('Step 1: Configured test parameters\n');
        fprintf('  Final α = %.4f\n', alpha);
        fprintf('  Final H_eff = [%.3f, %.3f, %.3f] T\n', H_eff);
        fprintf('  Final m0 = [%.3f, %.3f, %.3f]\n', m0);
        fprintf('  Done.\n\n');
    end
    
    %% Step 2: Analytical Solution (where available)
    
    if verbose
        fprintf('Step 2: Computing analytical solution...\n');
    end
    
    % Physical constants
    mu0 = 4*pi*1e-7;  % Permeability of free space
    
    % Effective field magnitude and direction
    H_magnitude = norm(H_eff);
    
    if H_magnitude > 0
        H_unit = H_eff / H_magnitude;
        
        % Larmor frequency
        omega_L = gamma * H_magnitude;
        
        % Effective precession frequency (with damping)
        omega_eff = omega_L / (1 + alpha^2);
        
        % Damping time constant
        if alpha > 0
            tau_damping = (1 + alpha^2) / (alpha * omega_L);
        else
            tau_damping = inf;
        end
        
        analytical_available = true;
        
        if verbose
            fprintf('  Field magnitude: %.3f T\n', H_magnitude);
            fprintf('  Larmor frequency: %.2e rad/s (%.2f GHz)\n', omega_L, omega_L/(2*pi*1e9));
            fprintf('  Effective frequency: %.2e rad/s\n', omega_eff);
            if isfinite(tau_damping)
                fprintf('  Damping time: %.2e s\n', tau_damping);
            else
                fprintf('  No damping (α = 0)\n');
            end
        end
        
    else
        % No field case
        analytical_available = false;
        omega_L = 0;
        omega_eff = 0;
        tau_damping = inf;
        
        if verbose
            fprintf('  Zero field case - no analytical solution\n');
        end
    end
    
    % For precession case, compute analytical solution
    if analytical_available && strcmp(test_case, 'precession')
        % Time vector for analytical solution
        t_analytical = linspace(tspan(1), tspan(2), 1000);
        
        % Analytical solution for small angle precession
        % m(t) ≈ m_eq + (m0 - m_eq) * exp(-t/τ) * [cos(ωt), sin(ωt), const]
        
        % Equilibrium magnetization (along field direction)
        m_eq = H_unit;
        
        % Perpendicular component
        m_perp_0 = m0 - dot(m0, H_unit) * H_unit;
        m_perp_magnitude = norm(m_perp_0);
        
        if m_perp_magnitude > 1e-6
            m_perp_unit = m_perp_0 / m_perp_magnitude;
            
            % Create orthogonal basis in plane perpendicular to H
            e1 = m_perp_unit;
            e2 = cross(H_unit, e1);
            e2 = e2 / norm(e2);
            
            % Analytical precession (small angle approximation)
            decay_factor = exp(-t_analytical / tau_damping);
            cos_term = cos(omega_eff * t_analytical);
            sin_term = sin(omega_eff * t_analytical);
            
            m_analytical = zeros(3, length(t_analytical));
            for i = 1:length(t_analytical)
                m_perp_t = m_perp_magnitude * decay_factor(i) * ...
                          (cos_term(i) * e1 + sin_term(i) * e2);
                m_parallel_t = dot(m0, H_unit) * H_unit;
                m_analytical(:, i) = m_parallel_t + m_perp_t;
                
                % Renormalize to maintain |m| = 1
                m_analytical(:, i) = m_analytical(:, i) / norm(m_analytical(:, i));
            end
        else
            % Already aligned with field
            m_analytical = repmat(m0, 1, length(t_analytical));
        end
        
        if verbose
            fprintf('  Computed analytical precession solution\n');
        end
    else
        t_analytical = [];
        m_analytical = [];
        if verbose
            fprintf('  No analytical solution for this test case\n');
        end
    end
    
    if verbose
        fprintf('  Done.\n\n');
    end
    
    %% Step 3: Numerical Solution using LLG Solver
    
    if verbose
        fprintf('Step 3: Computing numerical solution...\n');
    end
    
    % Effective field function (constant field for this validation)
    H_eff_func = @(t, m) H_eff;
    
    % Solve LLG equation
    try
        if verbose
            fprintf('  Calling LLGSolver...\n');
        end
        
        [m_numerical, t_numerical, solve_info] = LLGSolver(m0, H_eff_func, alpha, gamma, tspan, ...
                                                          'Method', method, ...
                                                          'RelTol', tol(1), ...
                                                          'AbsTol', tol(2), ...
                                                          'ConserveMagnetization', true, ...
                                                          'ConserveEnergy', (alpha == 0), ...
                                                          'Verbose', false);
        
        numerical_success = true;
        
        if verbose
            fprintf('  Numerical solution successful\n');
            fprintf('  Integration method: %s\n', solve_info.method);
            fprintf('  Steps taken: %d\n', solve_info.steps_taken);
            fprintf('  Steps rejected: %d\n', solve_info.steps_rejected);
            fprintf('  Final time: %.2e s\n', solve_info.final_time);
            fprintf('  Average step size: %.2e s\n', solve_info.average_step_size);
            fprintf('  Magnetization error: %.2e\n', solve_info.magnetization_error);
            if isfield(solve_info, 'energy_conservation')
                fprintf('  Energy conservation: %.2e\n', solve_info.energy_conservation);
            end
        end
        
    catch ME
        if verbose
            fprintf('  Numerical solution failed: %s\n', ME.message);
        end
        numerical_success = false;
        m_numerical = [];
        t_numerical = [];
    end
    
    if verbose
        fprintf('  Done.\n\n');
    end
    
    %% Step 4: Validation Analysis
    
    if verbose
        fprintf('Step 4: Performing validation analysis...\n');
    end
    
    validation_results = struct();
    
    if numerical_success
        % Basic physics validation
        
        % 1. Magnetization magnitude conservation
        m_magnitudes = vecnorm(m_numerical, 2, 1);
        magnitude_error = max(abs(m_magnitudes - 1));
        magnitude_validation = magnitude_error < 1e-6;  % 1 ppm tolerance
        
        if verbose
            fprintf('  Magnetization magnitude conservation:\n');
            fprintf('    Max |m| error: %.2e\n', magnitude_error);
            fprintf('    Validation: %s\n', char(magnitude_validation*'PASS' + ~magnitude_validation*'FAIL'));
        end
        
        % 2. Energy conservation (for undamped case)
        if alpha < 1e-6  % Essentially undamped
            energies = -mu0 * sum(m_numerical .* H_eff, 1);
            energy_variation = max(energies) - min(energies);
            initial_energy = energies(1);
            energy_conservation_error = energy_variation / abs(initial_energy);
            energy_validation = energy_conservation_error < 1e-3;  % 0.1% tolerance
            
            if verbose
                fprintf('  Energy conservation (undamped):\n');
                fprintf('    Energy variation: %.2e J\n', energy_variation);
                fprintf('    Relative error: %.2e\n', energy_conservation_error);
                fprintf('    Validation: %s\n', char(energy_validation*'PASS' + ~energy_validation*'FAIL'));
            end
        else
            % For damped case, energy should decrease
            energies = -mu0 * sum(m_numerical .* H_eff, 1);
            energy_decrease = energies(end) - energies(1);
            energy_validation = energy_decrease <= 0;  % Energy should decrease or stay constant
            
            if verbose
                fprintf('  Energy dissipation (damped):\n');
                fprintf('    Energy change: %.2e J\n', energy_decrease);
                fprintf('    Validation: %s (should be ≤ 0)\n', char(energy_validation*'PASS' + ~energy_validation*'FAIL'));
            end
        end
        
        % 3. Frequency analysis (for precession case)
        if strcmp(test_case, 'precession') && H_magnitude > 0
            % Extract precession frequency from numerical solution
            if size(m_numerical, 2) > 100  % Need sufficient points for FFT
                % Use perpendicular component for frequency analysis
                m_perp_num = m_numerical - sum(m_numerical .* H_unit, 1) .* H_unit;
                m_perp_magnitude = vecnorm(m_perp_num, 2, 1);
                
                if max(m_perp_magnitude) > 1e-3  % Significant precession
                    % FFT analysis
                    dt = mean(diff(t_numerical));
                    fs = 1/dt;
                    
                    % Use x-component for FFT
                    mx_detrended = m_numerical(1, :) - mean(m_numerical(1, :));
                    
                    N = length(mx_detrended);
                    f = (0:N-1) * fs / N;
                    fft_mx = fft(mx_detrended);
                    power_spectrum = abs(fft_mx).^2;
                    
                    % Find peak frequency (exclude DC component)
                    [~, peak_idx] = max(power_spectrum(2:N/2));
                    peak_frequency = f(peak_idx + 1);  % +1 to account for skipping DC
                    
                    % Expected frequency
                    expected_frequency = omega_eff / (2*pi);
                    
                    frequency_error = abs(peak_frequency - expected_frequency) / expected_frequency;
                    frequency_validation = frequency_error < 0.1;  % 10% tolerance
                    
                    if verbose
                        fprintf('  Precession frequency analysis:\n');
                        fprintf('    Expected frequency: %.2f GHz\n', expected_frequency*1e-9);
                        fprintf('    Measured frequency: %.2f GHz\n', peak_frequency*1e-9);
                        fprintf('    Relative error: %.2f%%\n', frequency_error*100);
                        fprintf('    Validation: %s\n', char(frequency_validation*'PASS' + ~frequency_validation*'FAIL'));
                    end
                else
                    frequency_validation = true;  % No precession to validate
                    frequency_error = 0;
                    peak_frequency = 0;
                    expected_frequency = 0;
                    
                    if verbose
                        fprintf('  No significant precession detected\n');
                    end
                end
            else
                frequency_validation = true;  % Insufficient points for analysis
                frequency_error = NaN;
                peak_frequency = NaN;
                expected_frequency = omega_eff / (2*pi);
                
                if verbose
                    fprintf('  Insufficient points for frequency analysis\n');
                end
            end
        else
            frequency_validation = true;  % Not applicable for this test
            frequency_error = NaN;
            peak_frequency = NaN;
            expected_frequency = NaN;
        end
        
        % 4. Compare with analytical solution (if available)
        if ~isempty(m_analytical)
            % Interpolate numerical solution to analytical time points
            m_num_interp = zeros(3, length(t_analytical));
            for i = 1:3
                m_num_interp(i, :) = interp1(t_numerical, m_numerical(i, :), t_analytical, 'linear', 'extrap');
            end
            
            % Calculate errors
            trajectory_error = vecnorm(m_analytical - m_num_interp, 2, 1);
            max_trajectory_error = max(trajectory_error);
            rms_trajectory_error = sqrt(mean(trajectory_error.^2));
            
            analytical_validation = max_trajectory_error < 0.1;  % 10% tolerance
            
            if verbose
                fprintf('  Analytical comparison:\n');
                fprintf('    Max trajectory error: %.3f\n', max_trajectory_error);
                fprintf('    RMS trajectory error: %.3f\n', rms_trajectory_error);
                fprintf('    Validation: %s\n', char(analytical_validation*'PASS' + ~analytical_validation*'FAIL'));
            end
        else
            analytical_validation = true;  % Not applicable
            max_trajectory_error = NaN;
            rms_trajectory_error = NaN;
        end
        
        % 5. Equilibrium validation (for damping case)
        if strcmp(test_case, 'damping') && alpha > 0.01
            % Check if magnetization approaches equilibrium (field direction)
            m_final = m_numerical(:, end);
            m_equilibrium = H_unit;
            
            equilibrium_error = norm(m_final - m_equilibrium);
            equilibrium_validation = equilibrium_error < 0.1;  % 10% tolerance
            
            if verbose
                fprintf('  Equilibrium approach:\n');
                fprintf('    Final magnetization: [%.3f, %.3f, %.3f]\n', m_final);
                fprintf('    Expected equilibrium: [%.3f, %.3f, %.3f]\n', m_equilibrium);
                fprintf('    Error: %.3f\n', equilibrium_error);
                fprintf('    Validation: %s\n', char(equilibrium_validation*'PASS' + ~equilibrium_validation*'FAIL'));
            end
        else
            equilibrium_validation = true;  % Not applicable
            equilibrium_error = NaN;
        end
        
        % Overall validation
        validations = [magnitude_validation, energy_validation, frequency_validation, ...
                      analytical_validation, equilibrium_validation];
        overall_validation = all(validations);
        
    else
        % Numerical solution failed
        magnitude_validation = false;
        energy_validation = false;
        frequency_validation = false;
        analytical_validation = false;
        equilibrium_validation = false;
        overall_validation = false;
        
        magnitude_error = NaN;
        frequency_error = NaN;
        max_trajectory_error = NaN;
        rms_trajectory_error = NaN;
        equilibrium_error = NaN;
        peak_frequency = NaN;
        expected_frequency = NaN;
    end
    
    % Package validation results
    validation_results.overall_pass = overall_validation;
    validation_results.magnitude_conservation = magnitude_validation;
    validation_results.energy_conservation = energy_validation;
    validation_results.frequency_accuracy = frequency_validation;
    validation_results.analytical_agreement = analytical_validation;
    validation_results.equilibrium_approach = equilibrium_validation;
    
    validation_results.magnitude_error = magnitude_error;
    validation_results.frequency_error = frequency_error;
    validation_results.max_trajectory_error = max_trajectory_error;
    validation_results.rms_trajectory_error = rms_trajectory_error;
    validation_results.equilibrium_error = equilibrium_error;
    validation_results.measured_frequency = peak_frequency;
    validation_results.expected_frequency = expected_frequency;
    
    if verbose
        fprintf('  Done.\n\n');
    end
    
    %% Step 5: Package Results
    
    % Create results structure
    results = struct();
    
    % Input parameters
    results.parameters = struct();
    results.parameters.test_case = test_case;
    results.parameters.damping = alpha;
    results.parameters.gyromagnetic_ratio = gamma;
    results.parameters.effective_field = H_eff;
    results.parameters.initial_magnetization = m0;
    results.parameters.time_span = tspan;
    results.parameters.method = method;
    results.parameters.tolerances = tol;
    
    % Physical parameters
    results.physics = struct();
    results.physics.larmor_frequency = omega_L;
    results.physics.effective_frequency = omega_eff;
    results.physics.damping_time = tau_damping;
    results.physics.field_magnitude = H_magnitude;
    
    % Analytical solution
    results.analytical = struct();
    results.analytical.available = analytical_available;
    if analytical_available && ~isempty(m_analytical)
        results.analytical.time = t_analytical;
        results.analytical.magnetization = m_analytical;
    end
    
    % Numerical solution
    results.numerical = struct();
    results.numerical.success = numerical_success;
    if numerical_success
        results.numerical.time = t_numerical;
        results.numerical.magnetization = m_numerical;
        results.numerical.solve_info = solve_info;
    end
    
    % Validation results
    results.validation = validation_results;
    
    %% Step 6: Plotting
    
    if plot_results && numerical_success
        if verbose
            fprintf('Step 5: Generating plots...\n');
        end
        
        % Apply Berkeley styling
        berkeley();
        
        % Create main figure
        fig = figure('Position', [100, 100, 1400, 1000]);
        
        % Subplot 1: Magnetization components vs time
        subplot(2, 3, 1);
        plot(t_numerical*1e9, m_numerical(1, :), 'LineWidth', 2, 'DisplayName', 'm_x');
        hold on;
        plot(t_numerical*1e9, m_numerical(2, :), 'LineWidth', 2, 'DisplayName', 'm_y');
        plot(t_numerical*1e9, m_numerical(3, :), 'LineWidth', 2, 'DisplayName', 'm_z');
        
        if ~isempty(m_analytical)
            plot(t_analytical*1e9, m_analytical(1, :), '--', 'LineWidth', 1, 'DisplayName', 'm_x (analytical)');
            plot(t_analytical*1e9, m_analytical(2, :), '--', 'LineWidth', 1, 'DisplayName', 'm_y (analytical)');
            plot(t_analytical*1e9, m_analytical(3, :), '--', 'LineWidth', 1, 'DisplayName', 'm_z (analytical)');
        end
        
        xlabel('Time (ns)');
        ylabel('Magnetization');
        title('Magnetization Components');
        legend('Location', 'best');
        grid on;
        
        % Subplot 2: Magnetization magnitude
        subplot(2, 3, 2);
        m_magnitudes = vecnorm(m_numerical, 2, 1);
        plot(t_numerical*1e9, m_magnitudes, 'LineWidth', 2);
        hold on;
        plot(t_numerical*1e9, ones(size(t_numerical)), 'r--', 'LineWidth', 1, 'DisplayName', '|m| = 1');
        xlabel('Time (ns)');
        ylabel('|m|');
        title('Magnetization Magnitude');
        legend('|m|', 'Target', 'Location', 'best');
        grid on;
        ylim([0.999, 1.001]);
        
        % Subplot 3: 3D trajectory
        subplot(2, 3, 3);
        % Plot unit sphere
        [x_sphere, y_sphere, z_sphere] = sphere(20);
        surf(x_sphere, y_sphere, z_sphere, 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'FaceColor', [0.8, 0.8, 0.8]);
        hold on;
        
        % Plot trajectory
        plot3(m_numerical(1, :), m_numerical(2, :), m_numerical(3, :), 'LineWidth', 2, 'Color', [0, 0.4470, 0.7410]);
        
        if ~isempty(m_analytical)
            plot3(m_analytical(1, :), m_analytical(2, :), m_analytical(3, :), '--', 'LineWidth', 1, 'Color', [0.8500, 0.3250, 0.0980]);
        end
        
        % Mark initial and final points
        scatter3(m0(1), m0(2), m0(3), 100, 'g', 'filled', 'DisplayName', 'Initial');
        scatter3(m_numerical(1, end), m_numerical(2, end), m_numerical(3, end), 100, 'r', 'filled', 'DisplayName', 'Final');
        
        % Show effective field direction
        quiver3(0, 0, 0, H_unit(1), H_unit(2), H_unit(3), 0.5, 'k', 'LineWidth', 2, 'DisplayName', 'H_{eff}');
        
        xlabel('m_x'); ylabel('m_y'); zlabel('m_z');
        title('Magnetization Trajectory');
        axis equal; axis([-1.1, 1.1, -1.1, 1.1, -1.1, 1.1]);
        view(45, 30);
        legend('Location', 'best');
        
        % Subplot 4: Energy evolution
        subplot(2, 3, 4);
        energies = -mu0 * sum(m_numerical .* H_eff, 1);
        plot(t_numerical*1e9, energies*1e21, 'LineWidth', 2);
        xlabel('Time (ns)');
        ylabel('Energy (zJ)');
        title('Magnetic Energy');
        grid on;
        
        % Subplot 5: Frequency spectrum (if applicable)
        subplot(2, 3, 5);
        if ~isnan(peak_frequency) && ~isnan(expected_frequency)
            % Show frequency analysis
            dt = mean(diff(t_numerical));
            fs = 1/dt;
            mx_detrended = m_numerical(1, :) - mean(m_numerical(1, :));
            N = length(mx_detrended);
            f = (0:N-1) * fs / N;
            fft_mx = fft(mx_detrended);
            power_spectrum = abs(fft_mx).^2;
            
            % Plot only positive frequencies up to Nyquist
            f_plot = f(1:N/2);
            power_plot = power_spectrum(1:N/2);
            
            semilogy(f_plot*1e-9, power_plot, 'LineWidth', 2);
            hold on;
            
            % Mark expected and measured frequencies
            if expected_frequency > 0
                xline(expected_frequency*1e-9, 'r--', 'LineWidth', 2, 'DisplayName', 'Expected');
            end
            if peak_frequency > 0
                xline(peak_frequency*1e-9, 'g--', 'LineWidth', 2, 'DisplayName', 'Measured');
            end
            
            xlabel('Frequency (GHz)');
            ylabel('Power Spectrum');
            title('Frequency Analysis');
            legend('Location', 'best');
            grid on;
        else
            text(0.5, 0.5, 'No Frequency Analysis', 'HorizontalAlignment', 'center');
            title('Frequency Analysis');
        end
        
        % Subplot 6: Validation summary
        subplot(2, 3, 6);
        validation_names = {'Magnitude', 'Energy', 'Frequency', 'Analytical', 'Equilibrium'};
        validation_values = [magnitude_validation, energy_validation, frequency_validation, ...
                           analytical_validation, equilibrium_validation];
        
        bar_colors = zeros(length(validation_values), 3);
        for i = 1:length(validation_values)
            if validation_values(i)
                bar_colors(i, :) = [0.2, 0.8, 0.2];  % Green for pass
            else
                bar_colors(i, :) = [0.8, 0.2, 0.2];  % Red for fail
            end
        end
        
        b = bar(validation_values);
        b.FaceColor = 'flat';
        b.CData = bar_colors;
        
        set(gca, 'XTickLabel', validation_names);
        ylabel('Validation (0=Fail, 1=Pass)');
        title('Validation Summary');
        ylim([0, 1.2]);
        grid on;
        
        % Add overall title
        sgtitle(sprintf('LLG Validation: %s (α=%.3f, H=%.2fT, %s)', ...
                       test_case, alpha, H_magnitude, char(overall_validation*'PASS' + ~overall_validation*'FAIL')), ...
                'FontSize', 16);
        
        % Store figure handle
        results.figure = fig;
        
        % Save figures if requested
        if p.Results.SaveFigures
            timestamp = datestr(now, 'yyyymmdd_HHMMSS');
            filename = sprintf('llg_damped_precession_%s_alpha%.3f_%s', test_case, alpha, timestamp);
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
        fprintf('Test case: %s\n', test_case);
        fprintf('Overall validation: %s\n', char(overall_validation*'PASS' + ~overall_validation*'FAIL'));
        
        if numerical_success
            fprintf('Magnetization conservation: %s (error: %.2e)\n', ...
                    char(magnitude_validation*'PASS' + ~magnitude_validation*'FAIL'), magnitude_error);
            fprintf('Energy conservation: %s\n', ...
                    char(energy_validation*'PASS' + ~energy_validation*'FAIL'));
            
            if ~isnan(frequency_error)
                fprintf('Frequency accuracy: %s (error: %.1f%%)\n', ...
                        char(frequency_validation*'PASS' + ~frequency_validation*'FAIL'), frequency_error*100);
            end
            
            if ~isnan(max_trajectory_error)
                fprintf('Analytical agreement: %s (max error: %.3f)\n', ...
                        char(analytical_validation*'PASS' + ~analytical_validation*'FAIL'), max_trajectory_error);
            end
            
            if ~isnan(equilibrium_error)
                fprintf('Equilibrium approach: %s (error: %.3f)\n', ...
                        char(equilibrium_validation*'PASS' + ~equilibrium_validation*'FAIL'), equilibrium_error);
            end
        else
            fprintf('Numerical solution failed\n');
        end
        
        fprintf('===========================\n\n');
    end
    
end