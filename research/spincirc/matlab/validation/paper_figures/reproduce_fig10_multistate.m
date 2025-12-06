function results = reproduce_fig10_multistate(varargin)
% REPRODUCE_FIG10_MULTISTATE - Reproduce Figure 10: 4-state memory device
%
% This validation script demonstrates a multi-state spintronic memory device
% capable of storing 4 distinct states (2 bits per cell). This showcases
% the framework's ability to model advanced spintronic device concepts
% beyond simple binary switching.
%
% The 4-state device concept uses:
%   - Two magnetic layers with different coercivities
%   - Independent switching of each layer
%   - Resistance encoding of 4 states: (↑↑), (↑↓), (↓↑), (↓↓)
%   - Current-controlled writing and voltage-controlled reading
%
% Device States:
%   State 00 (↓↓): Both layers antiparallel to reference
%   State 01 (↓↑): Layer 1 AP, Layer 2 P  
%   State 10 (↑↓): Layer 1 P, Layer 2 AP
%   State 11 (↑↑): Both layers parallel to reference
%
% Key Physics:
%   - Spin-transfer torque (STT) switching
%   - Tunnel magnetoresistance (TMR)
%   - Multi-level resistance states
%   - Selective addressing schemes
%   - Retention and stability analysis
%
% Usage:
%   results = reproduce_fig10_multistate();  % Default parameters
%   results = reproduce_fig10_multistate('Temperature', 350, 'TMR_ratio', 150);
%   results = reproduce_fig10_multistate('WriteScheme', 'sequential', 'PlotResults', true);
%
% Reference:
%   [Insert appropriate literature reference for Figure 10]
%
% Author: Meshal Alawein <meshal@berkeley.edu>
% Copyright © 2025 Meshal Alawein — All rights reserved.
% License: MIT

    % Parse input arguments
    p = inputParser;
    addParameter(p, 'Layer1Material', 'CoFeB', @ischar);  % Free layer 1 material
    addParameter(p, 'Layer2Material', 'CoFeB', @ischar);  % Free layer 2 material
    addParameter(p, 'BarrierMaterial', 'MgO', @ischar);  % Tunnel barrier material
    addParameter(p, 'ReferenceMaterial', 'CoFeB', @ischar);  % Reference layer material
    addParameter(p, 'Temperature', 300, @(x) isnumeric(x) && x > 0);  % Temperature (K)
    addParameter(p, 'DeviceDiameter', 60e-9, @(x) isnumeric(x) && x > 0);  % Device diameter (m)
    addParameter(p, 'Layer1Thickness', 1.2e-9, @(x) isnumeric(x) && x > 0);  % Layer 1 thickness (m)
    addParameter(p, 'Layer2Thickness', 1.5e-9, @(x) isnumeric(x) && x > 0);  % Layer 2 thickness (m)
    addParameter(p, 'BarrierThickness', 1.0e-9, @(x) isnumeric(x) && x > 0);  % Barrier thickness (m)
    addParameter(p, 'TMR_Ratio', 200, @(x) isnumeric(x) && x > 0);  % TMR ratio (%)
    addParameter(p, 'Layer1Coercivity', 15e-3, @(x) isnumeric(x) && x > 0);  % Layer 1 coercivity (T)
    addParameter(p, 'Layer2Coercivity', 25e-3, @(x) isnumeric(x) && x > 0);  % Layer 2 coercivity (T)
    addParameter(p, 'WriteScheme', 'simultaneous', @ischar);  % 'simultaneous' or 'sequential'
    addParameter(p, 'WritePulseWidth', 10e-9, @(x) isnumeric(x) && x > 0);  % Write pulse width (s)
    addParameter(p, 'ReadVoltage', 0.1, @(x) isnumeric(x) && x > 0);  % Read voltage (V)
    addParameter(p, 'NumCycles', 10, @(x) isnumeric(x) && x > 0);  % Number of write/read cycles
    addParameter(p, 'IncludeRetention', true, @islogical);  % Include retention analysis
    addParameter(p, 'RetentionTime', 3600, @(x) isnumeric(x) && x > 0);  % Retention time (s)
    addParameter(p, 'PlotResults', true, @islogical);  % Enable plotting
    addParameter(p, 'Verbose', true, @islogical);  % Enable verbose output
    addParameter(p, 'SaveFigures', false, @islogical);  % Save figures
    addParameter(p, 'FigurePath', pwd, @ischar);  % Figure save path
    parse(p, varargin{:});
    
    % Extract parameters
    layer1_material = p.Results.Layer1Material;
    layer2_material = p.Results.Layer2Material;
    barrier_material = p.Results.BarrierMaterial;
    ref_material = p.Results.ReferenceMaterial;
    T = p.Results.Temperature;
    diameter = p.Results.DeviceDiameter;
    t_layer1 = p.Results.Layer1Thickness;
    t_layer2 = p.Results.Layer2Thickness;
    t_barrier = p.Results.BarrierThickness;
    TMR_ratio = p.Results.TMR_Ratio;
    H_c1 = p.Results.Layer1Coercivity;
    H_c2 = p.Results.Layer2Coercivity;
    write_scheme = p.Results.WriteScheme;
    t_pulse = p.Results.WritePulseWidth;
    V_read = p.Results.ReadVoltage;
    N_cycles = p.Results.NumCycles;
    include_retention = p.Results.IncludeRetention;
    t_retention = p.Results.RetentionTime;
    plot_results = p.Results.PlotResults;
    verbose = p.Results.Verbose;
    
    if verbose
        fprintf('=== 4-State Memory Device Demonstration ===\n');
        fprintf('Stack: %s/(%s)/%s/(%s)/%s\n', ref_material, barrier_material, ...
                layer1_material, barrier_material, layer2_material);
        fprintf('Device diameter: %.0f nm\n', diameter*1e9);
        fprintf('Layer thicknesses: %.1f nm / %.1f nm\n', t_layer1*1e9, t_layer2*1e9);
        fprintf('Temperature: %.1f K\n', T);
        fprintf('TMR ratio: %.0f%%\n', TMR_ratio);
        fprintf('Coercivities: %.1f mT / %.1f mT\n', H_c1*1e3, H_c2*1e3);
        fprintf('Write scheme: %s\n', write_scheme);
        fprintf('===========================================\n\n');
    end
    
    %% Step 1: Get Material Properties and Device Parameters
    
    if verbose
        fprintf('Step 1: Loading material properties and device parameters...\n');
    end
    
    % Get material properties from database
    materials_db = MaterialsDB();
    layer1_props = materials_db.getTemperatureDependence(layer1_material, T);
    layer2_props = materials_db.getTemperatureDependence(layer2_material, T);
    barrier_props = materials_db.(barrier_material);
    ref_props = materials_db.getTemperatureDependence(ref_material, T);
    
    % Device geometry
    area = pi * (diameter/2)^2;
    volume1 = area * t_layer1;
    volume2 = area * t_layer2;
    
    % Extract magnetic parameters
    Ms1 = layer1_props.Ms;
    Ms2 = layer2_props.Ms;
    alpha1 = layer1_props.alpha;
    alpha2 = layer2_props.alpha;
    gamma = layer1_props.gamma;  % Assume same for both layers
    
    % Polarization and TMR parameters
    P1 = layer1_props.polarization;
    P2 = layer2_props.polarization;
    P_ref = ref_props.polarization;
    
    % Physical constants
    mu_0 = 4*pi*1e-7;
    k_B = 1.380649e-23;
    e = 1.602176634e-19;
    hbar = 1.054571817e-34;
    
    % Thermal stability factors
    K_u1 = layer1_props.K_u;
    K_u2 = layer2_props.K_u;
    
    E_barrier1 = K_u1 * volume1;
    E_barrier2 = K_u2 * volume2;
    Delta1 = E_barrier1 / (k_B * T);
    Delta2 = E_barrier2 / (k_B * T);
    
    if verbose
        fprintf('  Device Parameters:\n');
        fprintf('    Area: %.1f nm²\n', area*1e18);
        fprintf('    Layer 1: %s (Ms = %.2e A/m, Δ = %.1f)\n', layer1_material, Ms1, Delta1);
        fprintf('    Layer 2: %s (Ms = %.2e A/m, Δ = %.1f)\n', layer2_material, Ms2, Delta2);
        fprintf('    TMR ratio: %.0f%%\n', TMR_ratio);
        fprintf('    Coercive fields: %.1f / %.1f mT\n', H_c1*1e3, H_c2*1e3);
        fprintf('  Done.\n\n');
    end
    
    %% Step 2: Multi-State Resistance Model
    
    if verbose
        fprintf('Step 2: Computing multi-state resistance model...\n');
    end
    
    % Define 4 states with magnetization configurations
    % Reference layer is fixed (assume +z direction)
    % Layer 1 and Layer 2 can be parallel (+1) or antiparallel (-1)
    
    states = {
        struct('name', '00 (↓↓)', 'layer1', -1, 'layer2', -1, 'bits', [0, 0]);
        struct('name', '01 (↓↑)', 'layer1', -1, 'layer2', +1, 'bits', [0, 1]);
        struct('name', '10 (↑↓)', 'layer1', +1, 'layer2', -1, 'bits', [1, 0]);
        struct('name', '11 (↑↑)', 'layer1', +1, 'layer2', +1, 'bits', [1, 1]);
    };
    
    % Base resistance (parallel state)
    R_base = 1000;  % 1 kΩ base resistance
    
    % TMR calculation for each state
    % Assume dual MTJ structure: R_total = R_ref + R_layer1 + R_layer2
    % Each junction contributes TMR based on relative alignment
    
    resistances = zeros(4, 1);
    
    for i = 1:4
        state = states{i};
        
        % TMR contribution from each layer relative to reference
        % TMR = (R_AP - R_P) / R_P
        TMR_factor = TMR_ratio / 100;
        
        % Layer 1 contribution
        if state.layer1 == 1  % Parallel
            R1_factor = 1;
        else  % Antiparallel
            R1_factor = 1 + TMR_factor;
        end
        
        % Layer 2 contribution
        if state.layer2 == 1  % Parallel
            R2_factor = 1;
        else  % Antiparallel
            R2_factor = 1 + TMR_factor;
        end
        
        % Combined resistance (series connection)
        % Simplified model: R_total = R_base * average_factor
        resistances(i) = R_base * (R1_factor + R2_factor) / 2;
    end
    
    % Sort by resistance value for clear level separation
    [sorted_R, sort_idx] = sort(resistances);
    sorted_states = states(sort_idx);
    
    % Resistance levels and margins
    R_levels = sorted_R;
    R_margins = diff(R_levels);
    min_margin = min(R_margins);
    relative_margins = R_margins ./ R_levels(1:end-1) * 100;
    
    if verbose
        fprintf('  4-State Resistance Levels:\n');
        for i = 1:4
            state = sorted_states{i};
            fprintf('    State %s: %.0f Ω\n', state.name, R_levels(i));
        end
        fprintf('  Resistance margins: [%.0f, %.0f, %.0f] Ω\n', R_margins);
        fprintf('  Minimum margin: %.0f Ω (%.1f%%)\n', min_margin, min(relative_margins));
        fprintf('  Done.\n\n');
    end
    
    %% Step 3: Write Operation Simulation
    
    if verbose
        fprintf('Step 3: Simulating write operations...\n');
    end
    
    % Critical switching currents for each layer
    % J_c = (2*e*Ms*t*α*H_k) / (ħ*P*ε) (STT switching)
    epsilon = 0.5;  % STT efficiency
    
    H_k1 = 2*K_u1 / (mu_0*Ms1);
    H_k2 = 2*K_u2 / (mu_0*Ms2);
    
    J_c1 = (2*e*Ms1*t_layer1*alpha1*H_k1) / (hbar*P1*epsilon);
    J_c2 = (2*e*Ms2*t_layer2*alpha2*H_k2) / (hbar*P2*epsilon);
    
    I_c1 = J_c1 * area;  % Critical current for layer 1
    I_c2 = J_c2 * area;  % Critical current for layer 2
    
    % Write currents for selective switching
    % Need I > I_c for the target layer, I < I_c for non-target layer
    write_margin = 1.5;  % 50% overdrive
    
    I_write_1 = I_c1 * write_margin;  % Switch layer 1 only
    I_write_2 = I_c2 * write_margin;  % Switch layer 2 only
    I_write_both = max(I_c1, I_c2) * write_margin;  % Switch both layers
    
    % State transition matrix (simplified switching model)
    % Probability of successful switching
    P_switch = @(I, I_c) 1 / (1 + exp(-(I/I_c - 1) * 5));  % Sigmoid function
    
    % Write sequence for all 4 states
    write_sequence = [
        struct('target_state', 1, 'current', -I_write_both, 'description', 'Write 00');
        struct('target_state', 2, 'current', -I_write_1, 'description', 'Write 01');
        struct('target_state', 3, 'current', I_write_1, 'description', 'Write 10');
        struct('target_state', 4, 'current', I_write_both, 'description', 'Write 11');
    ];
    
    if verbose
        fprintf('  Write Operation Parameters:\n');
        fprintf('    Critical currents: I_c1 = %.0f μA, I_c2 = %.0f μA\n', I_c1*1e6, I_c2*1e6);
        fprintf('    Write currents: %.0f μA, %.0f μA, %.0f μA\n', ...
                I_write_1*1e6, I_write_2*1e6, I_write_both*1e6);
        fprintf('    Write scheme: %s\n', write_scheme);
        fprintf('  Done.\n\n');
    end
    
    %% Step 4: Read Operation and Cycle Testing
    
    if verbose
        fprintf('Step 4: Simulating read operations and cycling...\n');
    end
    
    % Initialize results arrays
    cycle_data = struct();
    cycle_data.cycle_number = 1:N_cycles;
    cycle_data.written_states = zeros(N_cycles, 4);  % 4 states per cycle
    cycle_data.read_resistances = zeros(N_cycles, 4);
    cycle_data.write_success = zeros(N_cycles, 4);
    cycle_data.read_margins = zeros(N_cycles, 3);  % 3 margins between 4 levels
    
    % Random state sequence for testing
    rng(123);  % For reproducibility
    state_sequence = randi(4, N_cycles, 1);
    
    for cycle = 1:N_cycles
        if verbose && mod(cycle, 5) == 1
            fprintf('  Cycle %d/%d...\n', cycle, N_cycles);
        end
        
        % Write and read all 4 states in this cycle
        cycle_resistances = zeros(4, 1);
        cycle_success = zeros(4, 1);
        
        for state_idx = 1:4
            % Simulate write operation
            write_op = write_sequence(state_idx);
            target_state = sorted_states{state_idx};
            
            % Write success probability based on current amplitude
            if strcmp(write_scheme, 'simultaneous')
                % Both layers switched simultaneously
                P_success = P_switch(abs(write_op.current), max(I_c1, I_c2));
            else
                % Sequential switching
                if state_idx <= 2
                    P_success = P_switch(abs(write_op.current), I_c1);
                else
                    P_success = P_switch(abs(write_op.current), I_c2);
                end
            end
            
            cycle_success(state_idx) = P_success;
            
            % Simulate read resistance with noise and variations
            base_resistance = R_levels(state_idx);
            
            % Add process variations (±2%)
            process_variation = 0.02 * base_resistance * randn();
            
            % Add read noise (±0.5%)
            read_noise = 0.005 * base_resistance * randn();
            
            % Add temperature drift (small effect)
            temp_drift = base_resistance * 0.001 * (T - 300) / 300;
            
            % Add cycling degradation (small increase in resistance)
            cycling_degradation = base_resistance * 0.0001 * cycle;
            
            measured_resistance = base_resistance + process_variation + ...
                                read_noise + temp_drift + cycling_degradation;
            
            % Apply write success probability
            if rand() > P_success
                % Write failed - resistance closer to previous state
                previous_state = mod(state_idx - 2, 4) + 1;  % Rough estimate
                measured_resistance = 0.7 * measured_resistance + 0.3 * R_levels(previous_state);
            end
            
            cycle_resistances(state_idx) = measured_resistance;
        end
        
        % Store cycle data
        cycle_data.written_states(cycle, :) = 1:4;
        cycle_data.read_resistances(cycle, :) = cycle_resistances';
        cycle_data.write_success(cycle, :) = cycle_success';
        
        % Calculate read margins for this cycle
        sorted_R_cycle = sort(cycle_resistances);
        cycle_data.read_margins(cycle, :) = diff(sorted_R_cycle)';
    end
    
    % Calculate statistics
    avg_resistances = mean(cycle_data.read_resistances, 1);
    std_resistances = std(cycle_data.read_resistances, 1);
    avg_margins = mean(cycle_data.read_margins, 1);
    min_margins = min(cycle_data.read_margins, [], 1);
    
    avg_write_success = mean(cycle_data.write_success, 1);
    
    if verbose
        fprintf('  Cycling Test Results (%d cycles):\n', N_cycles);
        fprintf('    Average resistances: [%.0f, %.0f, %.0f, %.0f] Ω\n', avg_resistances);
        fprintf('    Resistance std dev: [%.0f, %.0f, %.0f, %.0f] Ω\n', std_resistances);
        fprintf('    Average margins: [%.0f, %.0f, %.0f] Ω\n', avg_margins);
        fprintf('    Minimum margins: [%.0f, %.0f, %.0f] Ω\n', min_margins);
        fprintf('    Write success rates: [%.1f, %.1f, %.1f, %.1f]%%\n', avg_write_success*100);
        fprintf('  Done.\n\n');
    end
    
    %% Step 5: Retention Analysis
    
    if include_retention && verbose
        fprintf('Step 5: Analyzing data retention...\n');
    end
    
    retention_results = struct();
    
    if include_retention
        % Time array for retention simulation
        t_retention_array = logspace(0, log10(t_retention), 50);  % 1 s to retention time
        
        % Initialize retention arrays
        retention_prob = zeros(4, length(t_retention_array));
        
        for state_idx = 1:4
            state = sorted_states{state_idx};
            
            % Determine which layer(s) could thermally switch
            % Simple Arrhenius model: P_retain = exp(-t/tau)
            % where tau = tau_0 * exp(Delta)
            
            tau_0 = 1e-9;  % Attempt time (1 ns)
            
            % Layer 1 retention
            if state.layer1 == -1  % Antiparallel state (less stable)
                tau_1 = tau_0 * exp(Delta1 * 0.8);  % Reduced barrier for AP state
            else
                tau_1 = tau_0 * exp(Delta1);
            end
            
            % Layer 2 retention
            if state.layer2 == -1
                tau_2 = tau_0 * exp(Delta2 * 0.8);
            else
                tau_2 = tau_0 * exp(Delta2);
            end
            
            % Combined retention probability (both layers must retain)
            for t_idx = 1:length(t_retention_array)
                t = t_retention_array(t_idx);
                P_retain_1 = exp(-t / tau_1);
                P_retain_2 = exp(-t / tau_2);
                retention_prob(state_idx, t_idx) = P_retain_1 * P_retain_2;
            end
        end
        
        % Retention criterion: 10 years at operating temperature
        t_10_years = 10 * 365 * 24 * 3600;  % 10 years in seconds
        retention_10_years = zeros(4, 1);
        
        for state_idx = 1:4
            retention_10_years(state_idx) = interp1(t_retention_array, ...
                retention_prob(state_idx, :), t_10_years, 'linear', 'extrap');
        end
        
        retention_results.time_array = t_retention_array;
        retention_results.retention_probability = retention_prob;
        retention_results.retention_10_years = retention_10_years;
        
        if verbose
            fprintf('  Data Retention Analysis:\n');
            fprintf('    Thermal stability factors: Δ1 = %.1f, Δ2 = %.1f\n', Delta1, Delta2);
            for i = 1:4
                state = sorted_states{i};
                fprintf('    State %s: %.1e retention probability (10 years)\n', ...
                        state.name, retention_10_years(i));
            end
            fprintf('  Done.\n\n');
        end
    else
        retention_results = struct('performed', false);
    end
    
    %% Step 6: Package Results
    
    results = struct();
    
    % Input parameters
    results.parameters = struct();
    results.parameters.layer1_material = layer1_material;
    results.parameters.layer2_material = layer2_material;
    results.parameters.barrier_material = barrier_material;
    results.parameters.reference_material = ref_material;
    results.parameters.temperature = T;
    results.parameters.device_diameter = diameter;
    results.parameters.layer1_thickness = t_layer1;
    results.parameters.layer2_thickness = t_layer2;
    results.parameters.barrier_thickness = t_barrier;
    results.parameters.TMR_ratio = TMR_ratio;
    results.parameters.coercivities = [H_c1, H_c2];
    results.parameters.write_scheme = write_scheme;
    results.parameters.write_pulse_width = t_pulse;
    results.parameters.read_voltage = V_read;
    results.parameters.num_cycles = N_cycles;
    
    % Device characteristics
    results.device = struct();
    results.device.area = area;
    results.device.volumes = [volume1, volume2];
    results.device.thermal_stability = [Delta1, Delta2];
    results.device.critical_currents = [I_c1, I_c2];
    results.device.write_currents = [I_write_1, I_write_2, I_write_both];
    
    % 4-state model
    results.states = struct();
    results.states.definitions = sorted_states;
    results.states.resistance_levels = R_levels;
    results.states.resistance_margins = R_margins;
    results.states.minimum_margin = min_margin;
    results.states.relative_margins = relative_margins;
    
    % Write operations
    results.write_operations = struct();
    results.write_operations.sequence = write_sequence;
    results.write_operations.success_rates = avg_write_success;
    
    % Cycling performance
    results.cycling = struct();
    results.cycling.cycle_data = cycle_data;
    results.cycling.average_resistances = avg_resistances;
    results.cycling.resistance_std = std_resistances;
    results.cycling.average_margins = avg_margins;
    results.cycling.minimum_margins = min_margins;
    
    % Retention analysis
    results.retention = retention_results;
    
    %% Step 7: Plotting
    
    if plot_results
        if verbose
            fprintf('Step 6: Generating plots...\n');
        end
        
        % Apply Berkeley styling
        berkeley();
        
        % Create main figure
        fig = figure('Position', [100, 100, 1400, 1000]);
        
        % Subplot 1: 4-state resistance levels
        subplot(2, 3, 1);
        bar(1:4, R_levels, 'FaceColor', [0.2, 0.6, 0.8], 'EdgeColor', 'k');
        hold on;
        
        % Add error bars for variations
        if N_cycles > 1
            errorbar(1:4, avg_resistances, std_resistances, 'ko', 'LineWidth', 2, 'MarkerSize', 8);
        end
        
        % Add state labels
        state_labels = cell(4, 1);
        for i = 1:4
            state_labels{i} = sorted_states{i}.name;
        end
        set(gca, 'XTick', 1:4, 'XTickLabel', state_labels);
        ylabel('Resistance (Ω)');
        title('4-State Resistance Levels');
        grid on;
        
        % Add margin annotations
        for i = 1:3
            y_pos = (R_levels(i) + R_levels(i+1)) / 2;
            text(i+0.5, y_pos, sprintf('Δ=%.0fΩ', R_margins(i)), ...
                 'HorizontalAlignment', 'center', 'FontSize', 8, ...
                 'BackgroundColor', 'white');
        end
        
        % Subplot 2: Cycling stability
        subplot(2, 3, 2);
        colors = lines(4);
        
        for state_idx = 1:4
            plot(1:N_cycles, cycle_data.read_resistances(:, state_idx), ...
                 'o-', 'Color', colors(state_idx, :), 'LineWidth', 2, ...
                 'DisplayName', state_labels{state_idx});
            hold on;
        end
        
        xlabel('Cycle Number');
        ylabel('Read Resistance (Ω)');
        title('Cycling Stability');
        legend('Location', 'best');
        grid on;
        
        % Subplot 3: Write success rates
        subplot(2, 3, 3);
        bar(1:4, avg_write_success*100, 'FaceColor', [0.8, 0.2, 0.2], 'EdgeColor', 'k');
        set(gca, 'XTick', 1:4, 'XTickLabel', state_labels);
        ylabel('Write Success Rate (%)');
        title('Write Operation Success');
        ylim([0, 100]);
        grid on;
        
        % Add success rate values on bars
        for i = 1:4
            text(i, avg_write_success(i)*100 + 2, sprintf('%.1f%%', avg_write_success(i)*100), ...
                 'HorizontalAlignment', 'center', 'FontSize', 10);
        end
        
        % Subplot 4: Device stack schematic
        subplot(2, 3, 4);
        hold on;
        
        % Draw layer stack
        layer_positions = [0, 0.2, 0.4, 0.6, 0.8, 1.0];
        layer_colors = {[0.5, 0.5, 0.5], [0.2, 0.6, 0.8], [0.9, 0.9, 0.9], ...
                       [0.8, 0.2, 0.2], [0.9, 0.9, 0.9], [0.6, 0.6, 0.6]};
        layer_labels = {'Bottom Electrode', layer2_material, barrier_material, ...
                       layer1_material, barrier_material, 'Top Electrode'};
        
        for i = 1:6
            rectangle('Position', [0, layer_positions(i), 2, 0.15], ...
                     'FaceColor', layer_colors{i}, 'EdgeColor', 'k');
            text(2.2, layer_positions(i) + 0.075, layer_labels{i}, ...
                 'VerticalAlignment', 'middle', 'FontSize', 10);
        end
        
        % Add arrows for magnetization directions
        arrow_props = {'Color', 'black', 'LineWidth', 3};
        
        % Reference layer (fixed up)
        quiver(1, 0.1, 0, 0.1, 0, arrow_props{:});
        text(1.2, 0.15, 'Fixed', 'FontSize', 8);
        
        % Layer 2 (switchable)
        quiver(1, 0.3, 0, 0.1, 0, arrow_props{:});
        text(1.2, 0.35, 'Layer 2', 'FontSize', 8);
        
        % Layer 1 (switchable)
        quiver(1, 0.7, 0, 0.1, 0, arrow_props{:});
        text(1.2, 0.75, 'Layer 1', 'FontSize', 8);
        
        xlim([-0.5, 4]);
        ylim([-0.1, 1.2]);
        axis off;
        title('4-State MTJ Stack');
        
        % Subplot 5: Read margins vs cycling
        subplot(2, 3, 5);
        plot(1:N_cycles, cycle_data.read_margins(:, 1), 'o-', 'LineWidth', 2, 'DisplayName', 'Margin 1-2');
        hold on;
        plot(1:N_cycles, cycle_data.read_margins(:, 2), 's-', 'LineWidth', 2, 'DisplayName', 'Margin 2-3');
        plot(1:N_cycles, cycle_data.read_margins(:, 3), '^-', 'LineWidth', 2, 'DisplayName', 'Margin 3-4');
        
        xlabel('Cycle Number');
        ylabel('Read Margin (Ω)');
        title('Read Margin Stability');
        legend('Location', 'best');
        grid on;
        
        % Subplot 6: Retention analysis
        subplot(2, 3, 6);
        if include_retention
            for state_idx = 1:4
                semilogx(retention_results.time_array, retention_results.retention_probability(state_idx, :), ...
                         'LineWidth', 2, 'DisplayName', state_labels{state_idx});
                hold on;
            end
            
            % Mark 10-year point
            xline(10*365*24*3600, '--k', 'LineWidth', 2, 'DisplayName', '10 years');
            
            xlabel('Time (s)');
            ylabel('Retention Probability');
            title('Data Retention');
            legend('Location', 'best');
            grid on;
            ylim([0, 1]);
        else
            % Show write current requirements instead
            currents = [I_c1, I_c2, I_write_1, I_write_2, I_write_both] * 1e6;
            current_labels = {'I_{c1}', 'I_{c2}', 'I_{w1}', 'I_{w2}', 'I_{w,both}'};
            
            bar(currents);
            set(gca, 'XTickLabel', current_labels);
            ylabel('Current (μA)');
            title('Write Current Requirements');
            grid on;
        end
        
        % Add overall title
        sgtitle(sprintf('4-State Memory Device: %s/%s/%s, T=%.0fK', ...
                       layer1_material, barrier_material, layer2_material, T), 'FontSize', 16);
        
        % Store figure handle
        results.figure = fig;
        
        % Save figures if requested
        if p.Results.SaveFigures
            timestamp = datestr(now, 'yyyymmdd_HHMMSS');
            filename = sprintf('reproduce_fig10_multistate_%s_T%.0fK_%s', ...
                             layer1_material, T, timestamp);
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
        fprintf('=== 4-STATE MEMORY DEVICE SUMMARY ===\n');
        fprintf('Device stack: %s/%s/%s\n', layer1_material, barrier_material, layer2_material);
        fprintf('Temperature: %.1f K\n', T);
        fprintf('Device diameter: %.0f nm\n', diameter*1e9);
        
        fprintf('\n4-State Characteristics:\n');
        for i = 1:4
            state = sorted_states{i};
            fprintf('  %s: %.0f Ω\n', state.name, R_levels(i));
        end
        
        fprintf('\nPerformance Metrics:\n');
        fprintf('  Minimum read margin: %.0f Ω (%.1f%%)\n', min_margin, min(relative_margins));
        fprintf('  Average write success: %.1f%%\n', mean(avg_write_success)*100);
        fprintf('  Critical currents: [%.0f, %.0f] μA\n', I_c1*1e6, I_c2*1e6);
        
        if include_retention
            fprintf('  10-year retention: [%.1e, %.1e, %.1e, %.1e]\n', retention_results.retention_10_years);
        end
        
        fprintf('  Thermal stability: Δ1 = %.1f, Δ2 = %.1f\n', Delta1, Delta2);
        fprintf('  Cycling tested: %d cycles\n', N_cycles);
        
        fprintf('======================================\n\n');
    end
    
end