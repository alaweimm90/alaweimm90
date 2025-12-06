classdef ASLInverter < handle
% ASLINVERTER - All Spin Logic Inverter Device Model
%
% This class implements a comprehensive model of All Spin Logic (ASL) inverters
% as described in the seminal work by Manipatruni et al. (Nature Physics 2015).
% ASL devices use pure spin currents for logic operations without charge currents
% in the logic region, enabling ultra-low power computation.
%
% Physics Overview:
%   ASL inverters consist of:
%   1. Input magnet: Provides spin current via spin Hall effect
%   2. Free magnet: Output magnet switched by spin-transfer torque
%   3. Spin channel: Non-magnetic conductor connecting magnets
%   4. Optional: Fixed reference magnet for enhanced switching
%
% Key Features:
%   - Multiple magnet geometries (circular, elliptical, stadium)
%   - Monte Carlo analysis for thermal effects and process variations
%   - Power optimization across switching speed and energy
%   - Temperature-dependent material properties
%   - Stochastic LLG dynamics with thermal noise
%   - Supply voltage scaling analysis
%   - Process variation Monte Carlo with correlated parameters
%
% Usage:
%   inverter = ASLInverter();
%   inverter.setGeometry('circular', 100e-9, 5e-9);
%   inverter.setMaterials('CoFeB', 'Cu', 'Pt');
%   results = inverter.simulate();
%   inverter.plotResults();
%
% Author: Meshal Alawein <meshal@berkeley.edu>
% Copyright © 2025 Meshal Alawein — All rights reserved.
% License: MIT

    properties (Access = public)
        % Device geometry
        geometry            % Structure containing geometry parameters
        materials          % Material properties
        operating_point    % Bias conditions (voltage, current)
        
        % Simulation parameters
        temperature        % Operating temperature (K)
        thermal_noise      % Enable thermal fluctuations
        monte_carlo_runs   % Number of MC iterations
        
        % Analysis options
        analysis_type      % 'static', 'transient', 'monte_carlo', 'optimization'
        optimization_target % 'power', 'speed', 'reliability'
        
        % Results storage
        results            % Simulation results
        figures            % Figure handles for plotting
    end
    
    properties (Access = private)
        % Internal state
        magnetization_states    % Current magnet orientations
        spin_transport_solver   % Transport solver instance
        llg_solver             % Dynamics solver instance
        
        % Cached calculations
        conductance_matrix     % System conductance matrix
        effective_fields       % Effective magnetic fields
        
        % Monte Carlo parameters
        process_variations     % Process variation model
    end
    
    methods (Access = public)
        
        function obj = ASLInverter(varargin)
            % Constructor - Initialize ASL inverter with default parameters
            
            % Default parameters
            obj.temperature = 300;  % Room temperature
            obj.thermal_noise = true;
            obj.monte_carlo_runs = 1000;
            obj.analysis_type = 'static';
            obj.optimization_target = 'power';
            
            % Initialize default geometry (circular magnet, 100nm diameter)
            obj.setGeometry('circular', 100e-9, 5e-9);
            
            % Initialize default materials
            obj.setMaterials('CoFeB', 'Cu', 'Pt');
            
            % Initialize default operating point
            obj.setOperatingPoint(0.1, 1e-3);  % 0.1V, 1mA
            
            % Initialize solvers
            obj.spin_transport_solver = SpinTransportSolver();
            
            % Initialize results structure
            obj.initializeResults();
            
            % Parse optional inputs
            if nargin > 0
                obj.parseInputs(varargin{:});
            end
        end
        
        function setGeometry(obj, magnet_shape, width, thickness, varargin)
            % Set device geometry parameters
            %
            % Inputs:
            %   magnet_shape - 'circular', 'elliptical', 'stadium'
            %   width - Magnet width/diameter (m)
            %   thickness - Magnet thickness (m)
            %   varargin - Additional shape-specific parameters
            
            validateattributes(width, {'numeric'}, {'positive', 'scalar'});
            validateattributes(thickness, {'numeric'}, {'positive', 'scalar'});
            
            % Parse additional parameters
            p = inputParser;
            addParameter(p, 'aspect_ratio', 1.5, @(x) isnumeric(x) && x >= 1);
            addParameter(p, 'channel_length', 200e-9, @(x) isnumeric(x) && x > 0);
            addParameter(p, 'channel_width', 50e-9, @(x) isnumeric(x) && x > 0);
            addParameter(p, 'channel_thickness', 5e-9, @(x) isnumeric(x) && x > 0);
            addParameter(p, 'reference_magnet', false, @islogical);
            parse(p, varargin{:});
            
            % Store geometry parameters
            obj.geometry = struct();
            obj.geometry.magnet_shape = magnet_shape;
            obj.geometry.magnet_width = width;
            obj.geometry.magnet_thickness = thickness;
            obj.geometry.aspect_ratio = p.Results.aspect_ratio;
            obj.geometry.channel_length = p.Results.channel_length;
            obj.geometry.channel_width = p.Results.channel_width;
            obj.geometry.channel_thickness = p.Results.channel_thickness;
            obj.geometry.has_reference_magnet = p.Results.reference_magnet;
            
            % Calculate derived parameters
            obj.calculateGeometryParameters();
        end
        
        function setMaterials(obj, magnet_material, channel_material, sot_material, varargin)
            % Set material properties for device components
            %
            % Inputs:
            %   magnet_material - Ferromagnetic material ('CoFeB', 'NiFe', 'Co')
            %   channel_material - Channel material ('Cu', 'Al', 'Ag')
            %   sot_material - Spin-orbit torque material ('Pt', 'W', 'Ta')
            
            % Parse optional parameters
            p = inputParser;
            addParameter(p, 'custom_properties', struct(), @isstruct);
            parse(p, varargin{:});
            
            % Get material properties from database
            materials_db = MaterialsDB();
            
            % Magnet properties
            magnet_props = materials_db.getMaterial(magnet_material);
            if ~isempty(p.Results.custom_properties) && isfield(p.Results.custom_properties, 'magnet')
                magnet_props = obj.mergeStructures(magnet_props, p.Results.custom_properties.magnet);
            end
            
            % Channel properties
            channel_props = materials_db.getMaterial(channel_material);
            if ~isempty(p.Results.custom_properties) && isfield(p.Results.custom_properties, 'channel')
                channel_props = obj.mergeStructures(channel_props, p.Results.custom_properties.channel);
            end
            
            % SOT layer properties
            sot_props = materials_db.getMaterial(sot_material);
            if ~isempty(p.Results.custom_properties) && isfield(p.Results.custom_properties, 'sot')
                sot_props = obj.mergeStructures(sot_props, p.Results.custom_properties.sot);
            end
            
            % Store material properties
            obj.materials = struct();
            obj.materials.magnet = magnet_props;
            obj.materials.channel = channel_props;
            obj.materials.sot = sot_props;
            
            % Calculate temperature-dependent properties
            obj.updateTemperatureDependentProperties();
        end
        
        function setOperatingPoint(obj, voltage, current)
            % Set bias voltage and current
            %
            % Inputs:
            %   voltage - Applied voltage (V)
            %   current - Applied current (A)
            
            validateattributes(voltage, {'numeric'}, {'real', 'scalar'});
            validateattributes(current, {'numeric'}, {'real', 'scalar'});
            
            obj.operating_point = struct();
            obj.operating_point.voltage = voltage;
            obj.operating_point.current = current;
            obj.operating_point.power = abs(voltage * current);
        end
        
        function setTemperature(obj, T)
            % Set operating temperature and update material properties
            %
            % Inputs:
            %   T - Temperature (K)
            
            validateattributes(T, {'numeric'}, {'positive', 'scalar'});
            obj.temperature = T;
            
            if ~isempty(obj.materials)
                obj.updateTemperatureDependentProperties();
            end
        end
        
        function configureMonteCarlo(obj, n_runs, variation_model)
            % Configure Monte Carlo analysis parameters
            %
            % Inputs:
            %   n_runs - Number of Monte Carlo runs
            %   variation_model - Structure defining process variations
            
            validateattributes(n_runs, {'numeric'}, {'positive', 'integer', 'scalar'});
            obj.monte_carlo_runs = n_runs;
            
            % Default variation model if not provided
            if nargin < 3
                variation_model = obj.getDefaultVariationModel();
            end
            
            obj.process_variations = variation_model;
        end
        
        function results = simulate(obj, varargin)
            % Run ASL inverter simulation
            %
            % Outputs:
            %   results - Structure containing simulation results
            
            % Parse options
            p = inputParser;
            addParameter(p, 'time_span', [0, 1e-9], @isnumeric);
            addParameter(p, 'input_sequence', [0, 1], @isnumeric);
            addParameter(p, 'verbose', true, @islogical);
            parse(p, varargin{:});
            
            if p.Results.verbose
                fprintf('Starting ASL Inverter Simulation...\n');
                fprintf('Analysis type: %s\n', obj.analysis_type);
                fprintf('Temperature: %.1f K\n', obj.temperature);
            end
            
            % Initialize magnetization states
            obj.initializeMagnetization();
            
            % Run simulation based on analysis type
            switch obj.analysis_type
                case 'static'
                    results = obj.runStaticAnalysis();
                    
                case 'transient'
                    results = obj.runTransientAnalysis(p.Results.time_span, ...
                                                     p.Results.input_sequence);
                    
                case 'monte_carlo'
                    results = obj.runMonteCarloAnalysis();
                    
                case 'optimization'
                    results = obj.runOptimizationAnalysis();
                    
                otherwise
                    error('Unknown analysis type: %s', obj.analysis_type);
            end
            
            % Store results
            obj.results = results;
            
            if p.Results.verbose
                obj.printResultsSummary(results);
            end
        end
        
        function results = runStaticAnalysis(obj)
            % Run static analysis for steady-state behavior
            
            fprintf('Running static analysis...\n');
            
            % Set up transport solver
            obj.configuretransportSolver();
            
            % Calculate for both input states (0 and 1)
            input_states = [0, 1];
            n_states = length(input_states);
            
            results = struct();
            results.input_states = input_states;
            results.output_voltages = zeros(1, n_states);
            results.spin_currents = cell(1, n_states);
            results.switching_thresholds = zeros(1, n_states);
            results.power_consumption = zeros(1, n_states);
            
            for i = 1:n_states
                % Set input magnetization
                obj.setInputMagnetization(input_states(i));
                
                % Solve transport equation
                [V, I_s, info] = obj.spin_transport_solver.solve('verbose', false);
                
                % Extract results
                results.output_voltages(i) = obj.extractOutputVoltage(V);
                results.spin_currents{i} = I_s;
                results.switching_thresholds(i) = obj.calculateSwitchingThreshold(I_s);
                results.power_consumption(i) = obj.calculatePowerConsumption(V, I_s);
            end
            
            % Calculate derived metrics
            results.voltage_swing = abs(diff(results.output_voltages));
            results.average_power = mean(results.power_consumption);
            results.noise_margin = obj.calculateNoiseMargin(results.output_voltages);
            
            fprintf('Static analysis completed.\n');
        end
        
        function results = runTransientAnalysis(obj, time_span, input_sequence)
            % Run transient analysis with time-varying inputs
            
            fprintf('Running transient analysis...\n');
            tic;
            
            % Time vector
            t = linspace(time_span(1), time_span(2), 1000);
            dt = t(2) - t(1);
            
            % Initialize results arrays
            n_points = length(t);
            magnetization = zeros(3, 2, n_points);  % [x,y,z] x [input,output] x time
            output_voltage = zeros(1, n_points);
            spin_current = zeros(3, n_points);
            instantaneous_power = zeros(1, n_points);
            
            % Initial conditions
            obj.initializeMagnetization();
            magnetization(:, :, 1) = obj.magnetization_states;
            
            % Configure LLG solver for dynamics
            if isempty(obj.llg_solver)
                obj.configureLLGSolver();
            end
            
            % Main time-stepping loop
            for i = 2:n_points
                % Update input based on sequence
                input_value = obj.interpolateInputSequence(t(i), input_sequence, time_span);
                obj.setInputMagnetization(input_value);
                
                % Calculate effective fields including STT
                H_eff = obj.calculateEffectiveFields(t(i));
                
                % Solve LLG equation for one time step
                m_current = squeeze(magnetization(:, :, i-1));
                [m_new, ~, ~] = LLGSolver(m_current(:, 2), @(tt, mm) H_eff(:, 2), ...
                                         obj.materials.magnet.alpha, ...
                                         obj.materials.magnet.gamma, ...
                                         [t(i-1), t(i)], ...
                                         'Method', 'RK45', ...
                                         'ConserveMagnetization', true);
                
                % Update magnetization
                magnetization(:, 1, i) = obj.magnetization_states(:, 1);  % Input (fixed during step)
                magnetization(:, 2, i) = m_new(:, end);  % Output (evolved)
                obj.magnetization_states = squeeze(magnetization(:, :, i));
                
                % Calculate transport properties
                obj.configuretransportSolver();
                [V, I_s, ~] = obj.spin_transport_solver.solve('verbose', false);
                
                % Store results
                output_voltage(i) = obj.extractOutputVoltage(V);
                spin_current(:, i) = obj.extractSpinCurrent(I_s);
                instantaneous_power(i) = obj.calculatePowerConsumption(V, I_s);
            end
            
            % Package results
            results = struct();
            results.time = t;
            results.magnetization = magnetization;
            results.output_voltage = output_voltage;
            results.spin_current = spin_current;
            results.instantaneous_power = instantaneous_power;
            results.average_power = mean(instantaneous_power);
            results.switching_time = obj.calculateSwitchingTime(t, output_voltage);
            results.energy_per_switch = results.average_power * results.switching_time;
            
            elapsed_time = toc;
            fprintf('Transient analysis completed in %.2f seconds.\n', elapsed_time);
        end
        
        function results = runMonteCarloAnalysis(obj)
            % Run Monte Carlo analysis for statistical characterization
            
            fprintf('Running Monte Carlo analysis (%d samples)...\n', obj.monte_carlo_runs);
            tic;
            
            % Initialize results arrays
            switching_times = zeros(obj.monte_carlo_runs, 1);
            power_consumption = zeros(obj.monte_carlo_runs, 1);
            noise_margins = zeros(obj.monte_carlo_runs, 1);
            output_voltages = zeros(obj.monte_carlo_runs, 2);  % For input 0 and 1
            failure_count = 0;
            
            % Store nominal parameters
            nominal_materials = obj.materials;
            nominal_geometry = obj.geometry;
            
            % Progress tracking
            progress_points = round(linspace(1, obj.monte_carlo_runs, 10));
            
            for run = 1:obj.monte_carlo_runs
                try
                    % Apply process variations
                    obj.applyProcessVariations();
                    
                    % Run static analysis for this variation
                    static_results = obj.runStaticAnalysis();
                    
                    % Store results
                    output_voltages(run, :) = static_results.output_voltages;
                    power_consumption(run) = static_results.average_power;
                    noise_margins(run) = static_results.noise_margin;
                    
                    % Estimate switching time (simplified model)
                    switching_times(run) = obj.estimateSwitchingTime();
                    
                catch ME
                    % Count failures but continue
                    failure_count = failure_count + 1;
                    warning('MC run %d failed: %s', run, ME.message);
                    
                    % Use NaN for failed runs
                    switching_times(run) = NaN;
                    power_consumption(run) = NaN;
                    noise_margins(run) = NaN;
                    output_voltages(run, :) = NaN;
                end
                
                % Progress update
                if ismember(run, progress_points)
                    progress = find(progress_points == run) * 10;
                    fprintf('Progress: %d%% (%d/%d runs)\n', progress, run, obj.monte_carlo_runs);
                end
                
                % Restore nominal parameters for next run
                obj.materials = nominal_materials;
                obj.geometry = nominal_geometry;
            end
            
            % Remove failed runs
            valid_runs = ~isnan(switching_times);
            n_valid = sum(valid_runs);
            
            if n_valid < obj.monte_carlo_runs * 0.9
                warning('High failure rate: %d/%d runs failed', failure_count, obj.monte_carlo_runs);
            end
            
            % Calculate statistics
            results = struct();
            results.n_runs = obj.monte_carlo_runs;
            results.n_valid = n_valid;
            results.failure_rate = failure_count / obj.monte_carlo_runs;
            
            % Switching time statistics
            results.switching_time.mean = mean(switching_times(valid_runs));
            results.switching_time.std = std(switching_times(valid_runs));
            results.switching_time.min = min(switching_times(valid_runs));
            results.switching_time.max = max(switching_times(valid_runs));
            results.switching_time.data = switching_times(valid_runs);
            
            % Power consumption statistics
            results.power.mean = mean(power_consumption(valid_runs));
            results.power.std = std(power_consumption(valid_runs));
            results.power.min = min(power_consumption(valid_runs));
            results.power.max = max(power_consumption(valid_runs));
            results.power.data = power_consumption(valid_runs);
            
            % Noise margin statistics
            results.noise_margin.mean = mean(noise_margins(valid_runs));
            results.noise_margin.std = std(noise_margins(valid_runs));
            results.noise_margin.min = min(noise_margins(valid_runs));
            results.noise_margin.max = max(noise_margins(valid_runs));
            results.noise_margin.data = noise_margins(valid_runs);
            
            % Output voltage statistics
            results.output_voltage.mean = mean(output_voltages(valid_runs, :), 1);
            results.output_voltage.std = std(output_voltages(valid_runs, :), [], 1);
            results.output_voltage.data = output_voltages(valid_runs, :);
            
            % Calculate yield (percentage of devices meeting specifications)
            results.yield = obj.calculateYield(results) * 100;  % Percentage
            
            elapsed_time = toc;
            fprintf('Monte Carlo analysis completed in %.2f seconds.\n', elapsed_time);
            fprintf('Yield: %.1f%% (%d/%d devices pass)\n', results.yield, ...
                    round(results.yield/100 * n_valid), n_valid);
        end
        
        function results = runOptimizationAnalysis(obj)
            % Run optimization analysis to find optimal operating conditions
            
            fprintf('Running optimization analysis...\n');
            fprintf('Optimization target: %s\n', obj.optimization_target);
            
            % Define optimization space
            voltage_range = [0.05, 0.5];  % V
            current_range = [0.1e-3, 10e-3];  % A
            n_points = 20;
            
            voltages = linspace(voltage_range(1), voltage_range(2), n_points);
            currents = linspace(current_range(1), current_range(2), n_points);
            
            % Initialize results arrays
            [V_grid, I_grid] = meshgrid(voltages, currents);
            power_grid = zeros(size(V_grid));
            speed_grid = zeros(size(V_grid));
            reliability_grid = zeros(size(V_grid));
            objective_grid = zeros(size(V_grid));
            
            % Store original operating point
            original_op = obj.operating_point;
            
            % Sweep operating conditions
            for i = 1:n_points
                for j = 1:n_points
                    % Set new operating point
                    obj.setOperatingPoint(V_grid(i,j), I_grid(i,j));
                    
                    % Run quick analysis
                    static_results = obj.runStaticAnalysis();
                    
                    % Calculate metrics
                    power_grid(i,j) = static_results.average_power;
                    speed_grid(i,j) = 1 / obj.estimateSwitchingTime();  % Speed = 1/time
                    reliability_grid(i,j) = static_results.noise_margin;
                    
                    % Calculate objective function
                    switch obj.optimization_target
                        case 'power'
                            objective_grid(i,j) = -power_grid(i,j);  % Minimize power
                            
                        case 'speed'
                            objective_grid(i,j) = speed_grid(i,j);  % Maximize speed
                            
                        case 'reliability'
                            objective_grid(i,j) = reliability_grid(i,j);  % Maximize reliability
                            
                        case 'ppa'  % Power-Performance-Area (weighted)
                            % Normalize metrics and combine
                            norm_power = power_grid(i,j) / max(power_grid(:));
                            norm_speed = speed_grid(i,j) / max(speed_grid(:));
                            norm_reliability = reliability_grid(i,j) / max(reliability_grid(:));
                            
                            objective_grid(i,j) = norm_speed * norm_reliability / norm_power;
                    end
                end
            end
            
            % Find optimal point
            [max_obj, max_idx] = max(objective_grid(:));
            [opt_i, opt_j] = ind2sub(size(objective_grid), max_idx);
            
            optimal_voltage = V_grid(opt_i, opt_j);
            optimal_current = I_grid(opt_i, opt_j);
            
            % Package results
            results = struct();
            results.optimization_target = obj.optimization_target;
            results.voltage_sweep = voltages;
            results.current_sweep = currents;
            results.V_grid = V_grid;
            results.I_grid = I_grid;
            results.power_grid = power_grid;
            results.speed_grid = speed_grid;
            results.reliability_grid = reliability_grid;
            results.objective_grid = objective_grid;
            
            results.optimal.voltage = optimal_voltage;
            results.optimal.current = optimal_current;
            results.optimal.power = power_grid(opt_i, opt_j);
            results.optimal.speed = speed_grid(opt_i, opt_j);
            results.optimal.reliability = reliability_grid(opt_i, opt_j);
            results.optimal.objective = max_obj;
            
            % Restore original operating point
            obj.operating_point = original_op;
            
            fprintf('Optimization completed.\n');
            fprintf('Optimal conditions: V=%.3f V, I=%.2f mA\n', ...
                    optimal_voltage, optimal_current*1000);
            fprintf('Optimal %s: %.3e\n', obj.optimization_target, max_obj);
        end
        
        function plotResults(obj, varargin)
            % Plot simulation results with Berkeley styling
            
            if isempty(obj.results)
                error('No results to plot. Run simulation first.');
            end
            
            % Parse options
            p = inputParser;
            addParameter(p, 'plot_type', 'all', @ischar);
            addParameter(p, 'save_figures', false, @islogical);
            addParameter(p, 'figure_path', pwd, @ischar);
            parse(p, varargin{:});
            
            % Apply Berkeley styling
            berkeley();
            
            % Clear existing figures
            obj.figures = [];
            
            % Plot based on analysis type and results available
            switch obj.analysis_type
                case 'static'
                    obj.plotStaticResults();
                    
                case 'transient'
                    obj.plotTransientResults();
                    
                case 'monte_carlo'
                    obj.plotMonteCarloResults();
                    
                case 'optimization'
                    obj.plotOptimizationResults();
            end
            
            % Save figures if requested
            if p.Results.save_figures
                obj.saveFigures(p.Results.figure_path);
            end
        end
        
    end
    
    methods (Access = private)
        
        function initializeResults(obj)
            % Initialize results structure
            obj.results = struct();
        end
        
        function parseInputs(obj, varargin)
            % Parse constructor input arguments
            p = inputParser;
            addParameter(p, 'Temperature', 300, @isnumeric);
            addParameter(p, 'AnalysisType', 'static', @ischar);
            addParameter(p, 'ThermalNoise', true, @islogical);
            parse(p, varargin{:});
            
            obj.temperature = p.Results.Temperature;
            obj.analysis_type = p.Results.AnalysisType;
            obj.thermal_noise = p.Results.ThermalNoise;
        end
        
        function calculateGeometryParameters(obj)
            % Calculate derived geometry parameters
            
            geom = obj.geometry;
            
            % Magnet area based on shape
            switch geom.magnet_shape
                case 'circular'
                    geom.magnet_area = pi * (geom.magnet_width/2)^2;
                    geom.magnet_perimeter = pi * geom.magnet_width;
                    
                case 'elliptical'
                    a = geom.magnet_width / 2;  % Semi-major axis
                    b = a / geom.aspect_ratio;  % Semi-minor axis
                    geom.magnet_area = pi * a * b;
                    geom.magnet_perimeter = 4 * a * ellipke(1 - (b/a)^2);  % Approximation
                    
                case 'stadium'
                    % Stadium = rectangle + semicircles
                    r = geom.magnet_width / 2;
                    l = r * (geom.aspect_ratio - 1);
                    geom.magnet_area = pi * r^2 + 2 * r * l;
                    geom.magnet_perimeter = 2 * pi * r + 2 * l;
                    
                otherwise
                    error('Unknown magnet shape: %s', geom.magnet_shape);
            end
            
            % Channel area
            geom.channel_area = geom.channel_width * geom.channel_thickness;
            
            % Total device area
            geom.total_area = geom.magnet_area + geom.channel_area;
            
            % Update geometry
            obj.geometry = geom;
        end
        
        function updateTemperatureDependentProperties(obj)
            % Update material properties based on temperature
            
            if isempty(obj.materials)
                return;
            end
            
            T = obj.temperature;
            T0 = 300;  % Reference temperature
            
            % Magnet properties
            % Resistivity temperature coefficient
            alpha_rho = 0.004;  % Typical for metals (1/K)
            obj.materials.magnet.rho = obj.materials.magnet.rho * (1 + alpha_rho * (T - T0));
            
            % Spin diffusion length (decreases with temperature)
            obj.materials.magnet.lambda_sf = obj.materials.magnet.lambda_sf * sqrt(T0 / T);
            
            % Gilbert damping (increases with temperature)
            alpha_0 = obj.materials.magnet.alpha;  % Intrinsic damping
            alpha_ph = 0.001 * (T / T0);  % Phonon contribution
            obj.materials.magnet.alpha = alpha_0 + alpha_ph;
            
            % Similar updates for channel and SOT materials
            obj.materials.channel.rho = obj.materials.channel.rho * (1 + alpha_rho * (T - T0));
            obj.materials.channel.lambda_sf = obj.materials.channel.lambda_sf * sqrt(T0 / T);
            
            obj.materials.sot.rho = obj.materials.sot.rho * (1 + alpha_rho * (T - T0));
        end
        
        function variation_model = getDefaultVariationModel(obj)
            % Get default process variation model
            
            variation_model = struct();
            
            % Geometry variations (3-sigma values)
            variation_model.magnet_width = struct('type', 'normal', 'sigma', 0.05);  % 5% variation
            variation_model.magnet_thickness = struct('type', 'normal', 'sigma', 0.1);  % 10% variation
            variation_model.channel_length = struct('type', 'normal', 'sigma', 0.03);  % 3% variation
            
            % Material property variations
            variation_model.magnet_ms = struct('type', 'normal', 'sigma', 0.08);  % 8% variation
            variation_model.magnet_alpha = struct('type', 'lognormal', 'sigma', 0.2);  % 20% variation
            variation_model.channel_rho = struct('type', 'normal', 'sigma', 0.1);  % 10% variation
            variation_model.sot_theta_sh = struct('type', 'normal', 'sigma', 0.15);  % 15% variation
            
            % Correlated variations
            variation_model.correlations.magnet_width_thickness = 0.3;  % Moderate correlation
            variation_model.correlations.channel_length_width = 0.5;   % Strong correlation
        end
        
        function applyProcessVariations(obj)
            % Apply random process variations to device parameters
            
            if isempty(obj.process_variations)
                return;
            end
            
            var_model = obj.process_variations;
            
            % Generate correlated random variables if needed
            rng_state = rng;  % Save current RNG state
            
            % Geometry variations
            if isfield(var_model, 'magnet_width')
                factor = obj.generateVariation(var_model.magnet_width);
                obj.geometry.magnet_width = obj.geometry.magnet_width * factor;
            end
            
            if isfield(var_model, 'magnet_thickness')
                factor = obj.generateVariation(var_model.magnet_thickness);
                obj.geometry.magnet_thickness = obj.geometry.magnet_thickness * factor;
            end
            
            if isfield(var_model, 'channel_length')
                factor = obj.generateVariation(var_model.channel_length);
                obj.geometry.channel_length = obj.geometry.channel_length * factor;
            end
            
            % Material variations
            if isfield(var_model, 'magnet_ms')
                factor = obj.generateVariation(var_model.magnet_ms);
                obj.materials.magnet.Ms = obj.materials.magnet.Ms * factor;
            end
            
            if isfield(var_model, 'magnet_alpha')
                factor = obj.generateVariation(var_model.magnet_alpha);
                obj.materials.magnet.alpha = obj.materials.magnet.alpha * factor;
            end
            
            if isfield(var_model, 'channel_rho')
                factor = obj.generateVariation(var_model.channel_rho);
                obj.materials.channel.rho = obj.materials.channel.rho * factor;
            end
            
            if isfield(var_model, 'sot_theta_sh')
                factor = obj.generateVariation(var_model.sot_theta_sh);
                obj.materials.sot.theta_sh = obj.materials.sot.theta_sh * factor;
            end
            
            % Recalculate derived parameters
            obj.calculateGeometryParameters();
            obj.updateTemperatureDependentProperties();
        end
        
        function factor = generateVariation(obj, variation_spec)
            % Generate random variation factor based on specification
            
            switch variation_spec.type
                case 'normal'
                    factor = 1 + variation_spec.sigma * randn();
                    
                case 'lognormal'
                    factor = lognrnd(0, variation_spec.sigma);
                    
                case 'uniform'
                    range = variation_spec.range;  % [min, max] relative to nominal
                    factor = range(1) + (range(2) - range(1)) * rand();
                    
                otherwise
                    error('Unknown variation type: %s', variation_spec.type);
            end
            
            % Ensure factor is positive and reasonable
            factor = max(factor, 0.1);  % Minimum 10% of nominal
            factor = min(factor, 10);   % Maximum 10x nominal
        end
        
        function initializeMagnetization(obj)
            % Initialize magnetization states for input and output magnets
            
            % Input magnet: aligned with +x (logic '1') or -x (logic '0')
            % Output magnet: initially in +x state
            obj.magnetization_states = [1, 0, 0; 1, 0, 0]';  % [3 x 2] array
        end
        
        function setInputMagnetization(obj, logic_state)
            % Set input magnetization based on logic state
            %
            % Inputs:
            %   logic_state - 0 or 1 (or interpolated value for smooth transitions)
            
            % Smooth interpolation between logic states
            theta = logic_state * pi;  % 0 -> 0°, 1 -> 180°
            obj.magnetization_states(:, 1) = [cos(theta); 0; 0];
        end
        
        function configuretransportSolver(obj)
            % Configure spin transport solver with current device parameters
            
            % Set geometry
            obj.spin_transport_solver.setGeometry(obj.geometry.channel_length, ...
                                                 obj.geometry.channel_width, ...
                                                 obj.geometry.channel_thickness);
            
            % Set materials
            materials_array = [obj.materials.magnet, obj.materials.channel, obj.materials.magnet];
            obj.spin_transport_solver.setMaterials(materials_array);
            
            % Set magnetization
            obj.spin_transport_solver.setMagnetization(obj.magnetization_states');
            
            % Set boundary conditions
            bc_values = struct('node', [1, size(obj.magnetization_states, 2)], ...
                              'voltage', [obj.operating_point.voltage, 0]);
            obj.spin_transport_solver.setBoundaryConditions('voltage', bc_values);
            
            % Set temperature and field
            obj.spin_transport_solver.setTemperature(obj.temperature);
            obj.spin_transport_solver.setMagneticField([0, 0, 0]);  % No external field
        end
        
        function configureLLGSolver(obj)
            % Configure LLG solver for magnetization dynamics
            % (LLG solver is functional, no class instance needed)
            obj.llg_solver = true;  % Flag indicating LLG solver is configured
        end
        
        function output_voltage = extractOutputVoltage(obj, V)
            % Extract output voltage from transport solution
            n_nodes = length(V) / 4;
            output_voltage = V(n_nodes);  % Voltage at output magnet
        end
        
        function spin_current = extractSpinCurrent(obj, I_s)
            % Extract spin current flowing between magnets
            spin_current = [mean(I_s.spin_x); mean(I_s.spin_y); mean(I_s.spin_z)];
        end
        
        function switching_threshold = calculateSwitchingThreshold(obj, I_s)
            % Calculate critical current for magnetization switching
            
            % Extract spin current components
            I_sx = mean(I_s.spin_x);
            I_sy = mean(I_s.spin_y);
            I_sz = mean(I_s.spin_z);
            
            I_spin_total = sqrt(I_sx^2 + I_sy^2 + I_sz^2);
            
            % Critical current based on Slonczewski model
            % I_c = (2e * α * μ₀ * Ms * V * t) / (ℏ * P * g)
            e = 1.602e-19;  % Elementary charge
            hbar = 1.055e-34;  % Reduced Planck constant
            mu0 = 4*pi*1e-7;  % Permeability of free space
            
            alpha = obj.materials.magnet.alpha;
            Ms = obj.materials.magnet.Ms;
            V = obj.geometry.magnet_area * obj.geometry.magnet_thickness;
            P = obj.materials.magnet.spin_polarization;
            g = 0.5;  % Geometrical factor
            
            switching_threshold = (2*e * alpha * mu0 * Ms * V) / (hbar * P * g);
        end
        
        function power = calculatePowerConsumption(obj, V, I_s)
            % Calculate instantaneous power consumption
            
            % Charge current power
            I_charge = mean(I_s.charge);
            P_charge = obj.operating_point.voltage * I_charge;
            
            % Additional power losses (resistance, etc.)
            R_channel = obj.materials.channel.rho * obj.geometry.channel_length / ...
                       obj.geometry.channel_area;
            P_ohmic = I_charge^2 * R_channel;
            
            power = abs(P_charge) + P_ohmic;
        end
        
        function noise_margin = calculateNoiseMargin(obj, output_voltages)
            % Calculate noise margin for logic operation
            
            V_out_low = min(output_voltages);
            V_out_high = max(output_voltages);
            V_supply = obj.operating_point.voltage;
            
            % Assuming equal high and low thresholds
            V_threshold = V_supply / 2;
            
            NM_low = V_threshold - V_out_low;
            NM_high = V_out_high - V_threshold;
            
            noise_margin = min(NM_low, NM_high);
        end
        
        function input_value = interpolateInputSequence(obj, t, input_sequence, time_span)
            % Interpolate input sequence for given time
            
            if length(input_sequence) == 1
                input_value = input_sequence(1);
                return;
            end
            
            % Create time vector for input sequence
            t_input = linspace(time_span(1), time_span(2), length(input_sequence));
            
            % Interpolate
            input_value = interp1(t_input, input_sequence, t, 'linear', 'extrap');
            input_value = min(max(input_value, 0), 1);  % Clamp to [0, 1]
        end
        
        function H_eff = calculateEffectiveFields(obj, t)
            % Calculate effective magnetic fields including all contributions
            
            n_magnets = size(obj.magnetization_states, 2);
            H_eff = zeros(3, n_magnets);
            
            for i = 1:n_magnets
                m = obj.magnetization_states(:, i);
                
                % Demagnetization field (shape anisotropy)
                H_demag = obj.calculateDemagField(m, i);
                
                % Anisotropy field
                H_anis = obj.calculateAnisotropyField(m, i);
                
                % Spin-transfer torque field
                H_stt = obj.calculateSTTField(m, i);
                
                % Thermal field (if enabled)
                H_thermal = obj.calculateThermalField(i);
                
                % Total effective field
                H_eff(:, i) = H_demag + H_anis + H_stt + H_thermal;
            end
        end
        
        function H_demag = calculateDemagField(obj, m, magnet_idx)
            % Calculate demagnetization field for given magnet
            
            % Demagnetization factors depend on geometry
            geom = obj.geometry;
            
            switch geom.magnet_shape
                case 'circular'
                    % Thin disk approximation
                    aspect_ratio = geom.magnet_thickness / geom.magnet_width;
                    if aspect_ratio < 0.1  % Thin disk
                        Nx = Ny = 0;
                        Nz = 1;
                    else
                        % Use ellipsoid approximation
                        [Nx, Ny, Nz] = obj.calculateEllipsoidDemag(1, 1, aspect_ratio);
                    end
                    
                case 'elliptical'
                    ar_xy = geom.aspect_ratio;
                    ar_z = geom.magnet_thickness / geom.magnet_width;
                    [Nx, Ny, Nz] = obj.calculateEllipsoidDemag(ar_xy, 1, ar_z);
                    
                case 'stadium'
                    % Approximate as ellipsoid
                    ar_xy = geom.aspect_ratio;
                    ar_z = geom.magnet_thickness / geom.magnet_width;
                    [Nx, Ny, Nz] = obj.calculateEllipsoidDemag(ar_xy, 1, ar_z);
                    
                otherwise
                    Nx = Ny = Nz = 1/3;  % Sphere
            end
            
            % Demagnetization field
            Ms = obj.materials.magnet.Ms;
            H_demag = -[Nx * Ms * m(1); Ny * Ms * m(2); Nz * Ms * m(3)];
        end
        
        function [Nx, Ny, Nz] = calculateEllipsoidDemag(obj, a, b, c)
            % Calculate demagnetization factors for ellipsoid with semi-axes a, b, c
            
            % Sort axes
            axes = sort([a, b, c], 'descend');
            a_s = axes(1); b_s = axes(2); c_s = axes(3);
            
            % Calculate factors using standard formulas
            if a_s > b_s && b_s > c_s
                % General ellipsoid
                m = (a_s^2 - b_s^2) / (a_s^2 - c_s^2);
                n = (b_s^2 - c_s^2) / (a_s^2 - c_s^2);
                
                [K, E] = ellipke(m);
                [K1, E1] = ellipke(n);
                
                F = 1 / sqrt(a_s^2 - c_s^2);
                
                Nc = 4*pi*F * (b_s*a_s/(a_s^2 - b_s^2)) * (E - K);
                Na = 4*pi*F * (b_s*c_s/(b_s^2 - c_s^2)) * (K1 - E1) - Nc;
                Nb = 4*pi - Na - Nc;
                
            else
                % Simplified cases
                Na = Nb = Nc = 4*pi/3;  % Sphere approximation
            end
            
            % Map back to original axes
            factors = [Na, Nb, Nc] / (4*pi);
            
            % Sort back to match input order
            [~, sort_idx] = sort([a, b, c], 'descend');
            [~, unsort_idx] = sort(sort_idx);
            result = factors(unsort_idx);
            
            Nx = result(1);
            Ny = result(2);
            Nz = result(3);
        end
        
        function H_anis = calculateAnisotropyField(obj, m, magnet_idx)
            % Calculate magnetocrystalline anisotropy field
            
            % Uniaxial anisotropy along easy axis (assumed to be x-direction)
            Ku = obj.materials.magnet.Ku;  % Anisotropy constant
            Ms = obj.materials.magnet.Ms;
            
            easy_axis = [1; 0; 0];  % Easy axis
            
            % Anisotropy field: H_anis = (2*Ku/μ₀*Ms) * (m·û) * û
            mu0 = 4*pi*1e-7;
            H_anis = (2*Ku / (mu0*Ms)) * dot(m, easy_axis) * easy_axis;
        end
        
        function H_stt = calculateSTTField(obj, m, magnet_idx)
            % Calculate spin-transfer torque equivalent field
            
            % Only apply STT to output magnet (magnet 2)
            if magnet_idx == 1
                H_stt = [0; 0; 0];
                return;
            end
            
            % Get spin current from transport solution
            if ~isempty(obj.spin_transport_solver) && obj.spin_transport_solver.is_solved
                I_s = obj.spin_transport_solver.solution.I_s;
                I_spin = [mean(I_s.spin_x); mean(I_s.spin_y); mean(I_s.spin_z)];
            else
                % Estimate based on operating point
                I_spin = [obj.operating_point.current * 0.5; 0; 0];  % Simplified
            end
            
            % STT field using Slonczewski model
            % H_STT = (ℏ * P * I_s) / (2e * μ₀ * Ms * V)
            hbar = 1.055e-34;
            e = 1.602e-19;
            mu0 = 4*pi*1e-7;
            P = obj.materials.magnet.spin_polarization;
            Ms = obj.materials.magnet.Ms;
            V = obj.geometry.magnet_area * obj.geometry.magnet_thickness;
            
            prefactor = (hbar * P) / (2*e * mu0 * Ms * V);
            H_stt = prefactor * I_spin;
        end
        
        function H_thermal = calculateThermalField(obj, magnet_idx)
            % Calculate thermal fluctuation field
            
            if ~obj.thermal_noise
                H_thermal = [0; 0; 0];
                return;
            end
            
            % Thermal field based on fluctuation-dissipation theorem
            % <H_th^2> = (2*α*kB*T) / (γ*μ₀*Ms*V*dt)
            kB = 1.381e-23;  % Boltzmann constant
            alpha = obj.materials.magnet.alpha;
            gamma = obj.materials.magnet.gamma;
            mu0 = 4*pi*1e-7;
            Ms = obj.materials.magnet.Ms;
            V = obj.geometry.magnet_area * obj.geometry.magnet_thickness;
            dt = 1e-12;  % Assumed time step (s)
            
            H_th_rms = sqrt((2*alpha*kB*obj.temperature) / (gamma*mu0*Ms*V*dt));
            
            % Generate random field components
            H_thermal = H_th_rms * randn(3, 1);
        end
        
        function switching_time = calculateSwitchingTime(obj, t, output_voltage)
            % Calculate switching time from transient response
            
            % Find 10% and 90% levels of final voltage change
            V_initial = output_voltage(1);
            V_final = output_voltage(end);
            V_10 = V_initial + 0.1 * (V_final - V_initial);
            V_90 = V_initial + 0.9 * (V_final - V_initial);
            
            % Find crossing times
            idx_10 = find(output_voltage >= V_10, 1, 'first');
            idx_90 = find(output_voltage >= V_90, 1, 'first');
            
            if isempty(idx_10) || isempty(idx_90)
                switching_time = inf;  % No switching
            else
                switching_time = t(idx_90) - t(idx_10);
            end
        end
        
        function switching_time = estimateSwitchingTime(obj)
            % Estimate switching time based on material and geometry parameters
            
            % Simple model based on magnetization precession and damping
            alpha = obj.materials.magnet.alpha;
            gamma = obj.materials.magnet.gamma;
            Ms = obj.materials.magnet.Ms;
            
            % Effective field strength (rough estimate)
            H_eff = Ms * 0.1;  % 10% of saturation magnetization
            
            % Characteristic time: τ = 1/(α*γ*H_eff)
            switching_time = 1 / (alpha * gamma * H_eff);
            
            % Add geometry dependence
            V = obj.geometry.magnet_area * obj.geometry.magnet_thickness;
            switching_time = switching_time * (V / 1e-18)^(1/3);  % Size dependence
        end
        
        function yield = calculateYield(obj, mc_results)
            % Calculate device yield based on specifications
            
            % Define specifications
            spec = struct();
            spec.min_noise_margin = 0.05;  % V
            spec.max_power = 1e-6;  % W
            spec.max_switching_time = 1e-9;  % s
            
            % Check each device
            n_devices = length(mc_results.noise_margin.data);
            pass_count = 0;
            
            for i = 1:n_devices
                noise_margin_ok = mc_results.noise_margin.data(i) >= spec.min_noise_margin;
                power_ok = mc_results.power.data(i) <= spec.max_power;
                speed_ok = mc_results.switching_time.data(i) <= spec.max_switching_time;
                
                if noise_margin_ok && power_ok && speed_ok
                    pass_count = pass_count + 1;
                end
            end
            
            yield = pass_count / n_devices;
        end
        
        function printResultsSummary(obj, results)
            % Print summary of simulation results
            
            fprintf('\n=== ASL Inverter Results Summary ===\n');
            
            switch obj.analysis_type
                case 'static'
                    fprintf('Voltage swing: %.3f V\n', results.voltage_swing);
                    fprintf('Average power: %.2e W\n', results.average_power);
                    fprintf('Noise margin: %.3f V\n', results.noise_margin);
                    
                case 'transient'
                    fprintf('Switching time: %.2e s\n', results.switching_time);
                    fprintf('Energy per switch: %.2e J\n', results.energy_per_switch);
                    fprintf('Average power: %.2e W\n', results.average_power);
                    
                case 'monte_carlo'
                    fprintf('Number of runs: %d\n', results.n_runs);
                    fprintf('Failure rate: %.1f%%\n', results.failure_rate * 100);
                    fprintf('Yield: %.1f%%\n', results.yield);
                    fprintf('Switching time: %.2e ± %.2e s\n', ...
                            results.switching_time.mean, results.switching_time.std);
                    fprintf('Power: %.2e ± %.2e W\n', ...
                            results.power.mean, results.power.std);
                    
                case 'optimization'
                    fprintf('Optimization target: %s\n', results.optimization_target);
                    fprintf('Optimal voltage: %.3f V\n', results.optimal.voltage);
                    fprintf('Optimal current: %.2f mA\n', results.optimal.current * 1000);
                    fprintf('Optimal power: %.2e W\n', results.optimal.power);
            end
            
            fprintf('=====================================\n\n');
        end
        
        function plotStaticResults(obj)
            % Plot static analysis results
            
            results = obj.results;
            colors = getBerkeleyColors();
            
            % Input-output transfer characteristic
            fig1 = figure;
            plot(results.input_states, results.output_voltages, 'o-', ...
                 'LineWidth', 3, 'MarkerSize', 10, 'Color', colors(1,:));
            xlabel('Input Logic State');
            ylabel('Output Voltage (V)');
            title('ASL Inverter Transfer Characteristic');
            grid on;
            obj.figures(end+1) = fig1;
            
            % Power consumption vs input state
            fig2 = figure;
            bar(results.input_states, results.power_consumption * 1e6, ...
                'FaceColor', colors(2,:));
            xlabel('Input Logic State');
            ylabel('Power Consumption (μW)');
            title('Power Consumption vs Input State');
            grid on;
            obj.figures(end+1) = fig2;
        end
        
        function plotTransientResults(obj)
            % Plot transient analysis results
            
            results = obj.results;
            colors = getBerkeleyColors();
            
            % Create subplot figure
            fig = figure('Position', [100, 100, 1200, 800]);
            
            % Magnetization dynamics
            subplot(2,2,1);
            plot(results.time * 1e9, squeeze(results.magnetization(1,2,:)), ...
                 'LineWidth', 2, 'Color', colors(1,:));
            hold on;
            plot(results.time * 1e9, squeeze(results.magnetization(2,2,:)), ...
                 'LineWidth', 2, 'Color', colors(2,:));
            plot(results.time * 1e9, squeeze(results.magnetization(3,2,:)), ...
                 'LineWidth', 2, 'Color', colors(3,:));
            xlabel('Time (ns)');
            ylabel('Magnetization');
            title('Output Magnet Dynamics');
            legend('m_x', 'm_y', 'm_z', 'Location', 'best');
            grid on;
            
            % Output voltage
            subplot(2,2,2);
            plot(results.time * 1e9, results.output_voltage, ...
                 'LineWidth', 2, 'Color', colors(1,:));
            xlabel('Time (ns)');
            ylabel('Output Voltage (V)');
            title('Output Voltage Response');
            grid on;
            
            % Spin current
            subplot(2,2,3);
            plot(results.time * 1e9, results.spin_current(1,:) * 1e6, ...
                 'LineWidth', 2, 'Color', colors(1,:));
            hold on;
            plot(results.time * 1e9, results.spin_current(2,:) * 1e6, ...
                 'LineWidth', 2, 'Color', colors(2,:));
            plot(results.time * 1e9, results.spin_current(3,:) * 1e6, ...
                 'LineWidth', 2, 'Color', colors(3,:));
            xlabel('Time (ns)');
            ylabel('Spin Current (μA)');
            title('Spin Current Components');
            legend('I_{sx}', 'I_{sy}', 'I_{sz}', 'Location', 'best');
            grid on;
            
            % Power consumption
            subplot(2,2,4);
            plot(results.time * 1e9, results.instantaneous_power * 1e6, ...
                 'LineWidth', 2, 'Color', colors(4,:));
            xlabel('Time (ns)');
            ylabel('Power (μW)');
            title('Instantaneous Power');
            grid on;
            
            obj.figures(end+1) = fig;
        end
        
        function plotMonteCarloResults(obj)
            % Plot Monte Carlo analysis results
            
            results = obj.results;
            colors = getBerkeleyColors();
            
            % Create main figure with subplots
            fig = figure('Position', [100, 100, 1400, 1000]);
            
            % Switching time histogram
            subplot(2,3,1);
            histogram(results.switching_time.data * 1e9, 30, ...
                     'FaceColor', colors(1,:), 'EdgeColor', 'white');
            xlabel('Switching Time (ns)');
            ylabel('Count');
            title('Switching Time Distribution');
            grid on;
            
            % Power consumption histogram
            subplot(2,3,2);
            histogram(results.power.data * 1e6, 30, ...
                     'FaceColor', colors(2,:), 'EdgeColor', 'white');
            xlabel('Power Consumption (μW)');
            ylabel('Count');
            title('Power Distribution');
            grid on;
            
            % Noise margin histogram
            subplot(2,3,3);
            histogram(results.noise_margin.data, 30, ...
                     'FaceColor', colors(3,:), 'EdgeColor', 'white');
            xlabel('Noise Margin (V)');
            ylabel('Count');
            title('Noise Margin Distribution');
            grid on;
            
            % Power vs switching time scatter
            subplot(2,3,4);
            scatter(results.switching_time.data * 1e9, results.power.data * 1e6, ...
                   50, colors(4,:), 'filled', 'Alpha', 0.6);
            xlabel('Switching Time (ns)');
            ylabel('Power Consumption (μW)');
            title('Power vs Speed Trade-off');
            grid on;
            
            % Output voltage scatter
            subplot(2,3,5);
            scatter(results.output_voltage.data(:,1), results.output_voltage.data(:,2), ...
                   50, colors(5,:), 'filled', 'Alpha', 0.6);
            xlabel('Output Voltage (Input=0)');
            ylabel('Output Voltage (Input=1)');
            title('Output Voltage Correlation');
            grid on;
            
            % Yield analysis
            subplot(2,3,6);
            yield_data = [results.yield, 100-results.yield];
            pie(yield_data, {'Pass', 'Fail'});
            title(sprintf('Device Yield: %.1f%%', results.yield));
            colormap([colors(6,:); colors(7,:)]);
            
            obj.figures(end+1) = fig;
        end
        
        function plotOptimizationResults(obj)
            % Plot optimization analysis results
            
            results = obj.results;
            colors = getBerkeleyColors();
            
            % Create figure with contour plots
            fig = figure('Position', [100, 100, 1600, 1200]);
            
            % Power contour
            subplot(2,3,1);
            contourf(results.V_grid * 1000, results.I_grid * 1000, ...
                    results.power_grid * 1e6, 20);
            colorbar;
            xlabel('Voltage (mV)');
            ylabel('Current (mA)');
            title('Power Consumption (μW)');
            hold on;
            plot(results.optimal.voltage * 1000, results.optimal.current * 1000, ...
                 'r*', 'MarkerSize', 15, 'LineWidth', 3);
            
            % Speed contour
            subplot(2,3,2);
            contourf(results.V_grid * 1000, results.I_grid * 1000, ...
                    results.speed_grid * 1e-9, 20);
            colorbar;
            xlabel('Voltage (mV)');
            ylabel('Current (mA)');
            title('Speed (GHz)');
            hold on;
            plot(results.optimal.voltage * 1000, results.optimal.current * 1000, ...
                 'r*', 'MarkerSize', 15, 'LineWidth', 3);
            
            % Reliability contour
            subplot(2,3,3);
            contourf(results.V_grid * 1000, results.I_grid * 1000, ...
                    results.reliability_grid, 20);
            colorbar;
            xlabel('Voltage (mV)');
            ylabel('Current (mA)');
            title('Noise Margin (V)');
            hold on;
            plot(results.optimal.voltage * 1000, results.optimal.current * 1000, ...
                 'r*', 'MarkerSize', 15, 'LineWidth', 3);
            
            % Objective function
            subplot(2,3,4);
            contourf(results.V_grid * 1000, results.I_grid * 1000, ...
                    results.objective_grid, 20);
            colorbar;
            xlabel('Voltage (mV)');
            ylabel('Current (mA)');
            title(sprintf('Objective Function (%s)', results.optimization_target));
            hold on;
            plot(results.optimal.voltage * 1000, results.optimal.current * 1000, ...
                 'r*', 'MarkerSize', 15, 'LineWidth', 3);
            
            % Cross-sections at optimal point
            subplot(2,3,5);
            [~, opt_i] = min(abs(results.voltage_sweep - results.optimal.voltage));
            plot(results.current_sweep * 1000, results.power_grid(opt_i, :) * 1e6, ...
                 'LineWidth', 2, 'Color', colors(1,:));
            hold on;
            plot(results.current_sweep * 1000, results.speed_grid(opt_i, :) * 1e-9, ...
                 'LineWidth', 2, 'Color', colors(2,:));
            xlabel('Current (mA)');
            ylabel('Metrics');
            title('Cross-section at Optimal Voltage');
            legend('Power (μW)', 'Speed (GHz)', 'Location', 'best');
            grid on;
            
            % Performance summary
            subplot(2,3,6);
            metrics = [results.optimal.power * 1e6, ...
                      results.optimal.speed * 1e-9, ...
                      results.optimal.reliability * 1000, ...
                      results.optimal.objective];
            metric_names = {'Power (μW)', 'Speed (GHz)', 'Noise Margin (mV)', 'Objective'};
            bar(metrics, 'FaceColor', colors(3,:));
            set(gca, 'XTickLabel', metric_names);
            title('Optimal Performance Metrics');
            grid on;
            
            obj.figures(end+1) = fig;
        end
        
        function saveFigures(obj, output_path)
            % Save all figures to specified path
            
            if isempty(obj.figures)
                return;
            end
            
            if ~exist(output_path, 'dir')
                mkdir(output_path);
            end
            
            timestamp = datestr(now, 'yyyymmdd_HHMMSS');
            
            for i = 1:length(obj.figures)
                filename = sprintf('ASL_Inverter_%s_Fig%d_%s', ...
                                  obj.analysis_type, i, timestamp);
                filepath = fullfile(output_path, filename);
                
                % Save in multiple formats
                print(obj.figures(i), [filepath '.png'], '-dpng', '-r300');
                print(obj.figures(i), [filepath '.pdf'], '-dpdf', '-painters');
            end
            
            fprintf('Figures saved to: %s\n', output_path);
        end
        
        function merged = mergeStructures(obj, struct1, struct2)
            % Merge two structures, with struct2 taking precedence
            merged = struct1;
            fields = fieldnames(struct2);
            for i = 1:length(fields)
                merged.(fields{i}) = struct2.(fields{i});
            end
        end
        
    end
end

function colors = getBerkeleyColors()
    % Get Berkeley color palette
    colors = [
        0/255,   50/255,  98/255;    % Berkeley Blue
        253/255, 181/255, 21/255;    % California Gold
        1/255,   1/255,   51/255;    % Blue Dark
        252/255, 147/255, 19/255;    % Gold Dark
        0/255,   85/255,  58/255;    % Green Dark
        119/255, 7/255,   71/255;    % Rose Dark
        67/255,  17/255,  112/255;   % Purple Dark
        102/255, 102/255, 102/255;   % Gray Dark
    ];
end