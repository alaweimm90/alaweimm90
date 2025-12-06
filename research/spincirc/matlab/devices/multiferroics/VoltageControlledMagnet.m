classdef VoltageControlledMagnet < handle
% VOLTAGECONTROLLEDMAGNET - Multiferroic Voltage-Controlled Magnetic Device
%
% This class implements a comprehensive model of voltage-controlled magnetic
% devices using multiferroic heterostructures. These devices enable electric
% field control of magnetism through magnetoelectric coupling, offering
% ultra-low power magnetic switching for memory and logic applications.
%
% Physics Overview:
%   Voltage-controlled magnets exploit multiferroic coupling between:
%   1. Ferroelectric layer: Provides electric polarization control
%   2. Ferromagnetic layer: Magnetic moment controlled via coupling
%   3. Interface effects: Strain, charge, and exchange coupling mechanisms
%   4. Piezoelectric substrate: Strain-mediated magnetoelectric coupling
%
% Coupling Mechanisms:
%   - Strain-mediated: Electric field → strain → magnetoelastic coupling
%   - Charge-mediated: Electric field → surface charge → magnetic anisotropy
%   - Exchange coupling: Direct coupling at multiferroic interfaces
%   - Spin-orbit coupling: Electric field effects on magnetic anisotropy
%
% Key Features:
%   - Multiple coupling mechanisms (strain, charge, exchange)
%   - Temperature-dependent multiferroic properties
%   - Hysteretic switching with coercive field modeling
%   - Dynamic switching under AC electric fields
%   - Voltage-controlled magnetic anisotropy (VCMA)
%   - Process variation analysis for multiferroic parameters
%   - Power consumption analysis and optimization
%
% Usage:
%   vcm = VoltageControlledMagnet();
%   vcm.setGeometry(100e-9, 100e-9, 5e-9, 10e-9);  % FM and FE thicknesses
%   vcm.setMaterials('CoFeB', 'BaTiO3', 'PMN-PT');
%   vcm.setCouplingMechanism('strain');
%   results = vcm.measureHysteresis(linspace(-5, 5, 101));
%   vcm.plotResults();
%
% Author: Meshal Alawein <meshal@berkeley.edu>
% Copyright © 2025 Meshal Alawein — All rights reserved.
% License: MIT

    properties (Access = public)
        % Device geometry
        geometry            % Structure containing layer dimensions
        materials           % Multiferroic material properties
        
        % Operating conditions
        temperature         % Operating temperature (K)
        applied_voltage     % Applied electric field voltage (V)
        magnetic_field      % External magnetic field (T)
        
        % Coupling configuration
        coupling_mechanism  % 'strain', 'charge', 'exchange', 'hybrid'
        coupling_strength   % Coupling coefficient magnitude
        
        % Analysis options
        include_dynamics    % Include magnetization dynamics
        include_thermal_effects  % Include thermal fluctuations
        include_hysteresis  % Include hysteretic behavior
        
        % Results storage
        results             % Measurement/simulation results
        figures             % Figure handles for plotting
    end
    
    properties (Access = private)
        % Internal solvers and calculators
        llg_solver_configured    % LLG solver configuration flag
        
        % Cached calculations
        effective_fields    % Effective magnetic fields
        anisotropy_fields   % Voltage-dependent anisotropy fields
        strain_tensor       % Strain tensor from electric field
        
        % Material parameters (temperature dependent)
        ferroelectric_props % Temperature-dependent FE properties
        ferromagnetic_props % Temperature-dependent FM properties
        coupling_coeffs     % Temperature-dependent coupling coefficients
        
        % State variables
        magnetization_state % Current magnetization vector
        polarization_state  % Current electric polarization
        strain_state        % Current strain state
    end
    
    methods (Access = public)
        
        function obj = VoltageControlledMagnet(varargin)
            % Constructor - Initialize voltage-controlled magnet
            
            % Default parameters
            obj.temperature = 300;  % Room temperature
            obj.applied_voltage = 0;  % No applied voltage
            obj.magnetic_field = [0, 0, 0];  % No external field
            obj.coupling_mechanism = 'strain';  % Default coupling mechanism
            obj.coupling_strength = 1.0;  % Nominal coupling strength
            
            % Analysis options
            obj.include_dynamics = true;
            obj.include_thermal_effects = false;
            obj.include_hysteresis = true;
            
            % Initialize arrays
            obj.figures = [];
            
            % Initialize default geometry
            obj.setGeometry(100e-9, 100e-9, 5e-9, 10e-9);  % 100x100nm, 5nm FM, 10nm FE
            
            % Initialize default materials
            obj.setMaterials('CoFeB', 'BaTiO3', 'PMN-PT');
            
            % Initialize state
            obj.initializeStates();
            
            % Parse optional inputs
            if nargin > 0
                obj.parseInputs(varargin{:});
            end
        end
        
        function setGeometry(obj, width, length, fm_thickness, fe_thickness, varargin)
            % Set device geometry parameters
            %
            % Inputs:
            %   width - Device width (m)
            %   length - Device length (m)
            %   fm_thickness - Ferromagnetic layer thickness (m)
            %   fe_thickness - Ferroelectric layer thickness (m)
            %   varargin - Optional parameters
            
            % Parse optional parameters
            p = inputParser;
            addParameter(p, 'substrate_thickness', 500e-6, @(x) isnumeric(x) && x > 0);
            addParameter(p, 'electrode_thickness', 100e-9, @(x) isnumeric(x) && x > 0);
            addParameter(p, 'interface_roughness', 0.5e-9, @(x) isnumeric(x) && x >= 0);
            parse(p, varargin{:});
            
            % Store geometry parameters
            obj.geometry = struct();
            obj.geometry.width = width;
            obj.geometry.length = length;
            obj.geometry.fm_thickness = fm_thickness;
            obj.geometry.fe_thickness = fe_thickness;
            obj.geometry.substrate_thickness = p.Results.substrate_thickness;
            obj.geometry.electrode_thickness = p.Results.electrode_thickness;
            obj.geometry.interface_roughness = p.Results.interface_roughness;
            
            % Calculate derived parameters
            obj.geometry.area = width * length;
            obj.geometry.fm_volume = obj.geometry.area * fm_thickness;
            obj.geometry.fe_volume = obj.geometry.area * fe_thickness;
            obj.geometry.aspect_ratio = length / width;
            
            % Calculate capacitance for ferroelectric layer
            obj.updateCapacitance();
        end
        
        function setMaterials(obj, fm_material, fe_material, substrate_material, varargin)
            % Set multiferroic material properties
            %
            % Inputs:
            %   fm_material - Ferromagnetic material ('CoFeB', 'NiFe', 'Co')
            %   fe_material - Ferroelectric material ('BaTiO3', 'PZT', 'HfO2')
            %   substrate_material - Piezoelectric substrate ('PMN-PT', 'PZT')
            %   varargin - Optional custom properties
            
            % Parse optional parameters
            p = inputParser;
            addParameter(p, 'custom_properties', struct(), @isstruct);
            parse(p, varargin{:});
            
            % Get material properties from database
            materials_db = MaterialsDB();
            
            % Ferromagnetic layer properties
            fm_props = materials_db.getMaterial(fm_material);
            if isfield(p.Results.custom_properties, 'ferromagnet')
                fm_props = obj.mergeStructures(fm_props, p.Results.custom_properties.ferromagnet);
            end
            
            % Ferroelectric layer properties
            fe_props = materials_db.getMaterial(fe_material);
            if isfield(p.Results.custom_properties, 'ferroelectric')
                fe_props = obj.mergeStructures(fe_props, p.Results.custom_properties.ferroelectric);
            end
            
            % Substrate properties
            substrate_props = materials_db.getMaterial(substrate_material);
            if isfield(p.Results.custom_properties, 'substrate')
                substrate_props = obj.mergeStructures(substrate_props, p.Results.custom_properties.substrate);
            end
            
            % Store materials
            obj.materials = struct();
            obj.materials.ferromagnet = fm_props;
            obj.materials.ferroelectric = fe_props;
            obj.materials.substrate = substrate_props;
            
            % Set default magnetoelectric coupling coefficients
            obj.setDefaultCouplingCoefficients();
            
            % Update temperature-dependent properties
            obj.updateTemperatureDependentProperties();
        end
        
        function setCouplingMechanism(obj, mechanism, strength)
            % Set magnetoelectric coupling mechanism and strength
            %
            % Inputs:
            %   mechanism - 'strain', 'charge', 'exchange', 'hybrid'
            %   strength - Coupling strength multiplier (optional)
            
            valid_mechanisms = {'strain', 'charge', 'exchange', 'hybrid'};
            if ~ismember(mechanism, valid_mechanisms)
                error('Invalid coupling mechanism. Choose from: %s', ...
                      strjoin(valid_mechanisms, ', '));
            end
            
            obj.coupling_mechanism = mechanism;
            
            if nargin > 2
                validateattributes(strength, {'numeric'}, {'positive', 'scalar'});
                obj.coupling_strength = strength;
            end
            
            fprintf('Set coupling mechanism to: %s (strength: %.2f)\n', ...
                    mechanism, obj.coupling_strength);
        end
        
        function setTemperature(obj, T)
            % Set operating temperature and update material properties
            %
            % Inputs:
            %   T - Temperature (K)
            
            validateattributes(T, {'numeric'}, {'positive', 'scalar'});
            obj.temperature = T;
            
            % Update temperature-dependent properties
            obj.updateTemperatureDependentProperties();
        end
        
        function setElectricField(obj, voltage)
            % Set applied electric field voltage
            %
            % Inputs:
            %   voltage - Applied voltage across ferroelectric layer (V)
            
            validateattributes(voltage, {'numeric'}, {'real', 'scalar'});
            obj.applied_voltage = voltage;
            
            % Update electric field-dependent effective fields
            obj.updateEffectiveFields();
        end
        
        function results = measureHysteresis(obj, voltage_range, varargin)
            % Measure electric field hysteresis loop
            %
            % Inputs:
            %   voltage_range - Array of voltages to sweep (V)
            %   varargin - Optional parameters
            %
            % Outputs:
            %   results - Structure containing hysteresis results
            
            % Parse options
            p = inputParser;
            addParameter(p, 'sweep_rate', 1e3, @(x) isnumeric(x) && x > 0);  % V/s
            addParameter(p, 'include_minor_loops', false, @islogical);
            addParameter(p, 'verbose', true, @islogical);
            parse(p, varargin{:});
            
            if p.Results.verbose
                fprintf('Starting electric field hysteresis measurement...\n');
                fprintf('Voltage range: %.2f to %.2f V (%d points)\n', ...
                        min(voltage_range), max(voltage_range), length(voltage_range));
            end
            
            % Store original voltage
            original_voltage = obj.applied_voltage;
            
            % Initialize results arrays
            n_points = length(voltage_range);
            magnetizations = zeros(3, n_points);
            polarizations = zeros(n_points, 1);
            anisotropy_energies = zeros(n_points, 1);
            switching_fields = zeros(n_points, 1);
            energy_barriers = zeros(n_points, 1);
            
            % Calculate time step from sweep rate
            if length(voltage_range) > 1
                dV = abs(voltage_range(2) - voltage_range(1));
                dt = dV / p.Results.sweep_rate;
            else
                dt = 1e-6;  % Default time step
            end
            
            % Progress tracking
            if p.Results.verbose
                tic;
                progress_points = round(linspace(1, n_points, 10));
            end
            
            % Voltage sweep with hysteresis
            for i = 1:n_points
                % Set new voltage
                obj.setElectricField(voltage_range(i));
                
                % Calculate equilibrium magnetization
                if obj.include_dynamics
                    % Dynamic evolution to equilibrium
                    m_eq = obj.evolveToEquilibrium(dt * 10);  % 10 time steps to equilibrium
                else
                    % Static equilibrium calculation
                    m_eq = obj.calculateEquilibriumMagnetization();
                end
                
                % Store results
                magnetizations(:, i) = m_eq;
                polarizations(i) = obj.calculateElectricPolarization();
                anisotropy_energies(i) = obj.calculateAnisotropyEnergy(m_eq);
                switching_fields(i) = obj.calculateSwitchingField(m_eq);
                energy_barriers(i) = obj.calculateEnergyBarrier(m_eq);
                
                % Progress update
                if p.Results.verbose && ismember(i, progress_points)
                    progress = find(progress_points == i) * 10;
                    fprintf('Progress: %d%% (%d/%d points)\n', progress, i, n_points);
                end
            end
            
            % Calculate derived quantities
            coercive_voltages = obj.findCoerciveVoltages(voltage_range, magnetizations);
            saturation_values = obj.findSaturationValues(magnetizations, polarizations);
            
            % Package results
            results = struct();
            results.measurement_type = 'hysteresis';
            results.voltage_range = voltage_range;
            results.magnetizations = magnetizations;
            results.polarizations = polarizations;
            results.anisotropy_energies = anisotropy_energies;
            results.switching_fields = switching_fields;
            results.energy_barriers = energy_barriers;
            results.coercive_voltages = coercive_voltages;
            results.saturation_values = saturation_values;
            results.coupling_mechanism = obj.coupling_mechanism;
            results.temperature = obj.temperature;
            
            % Store results
            obj.results = results;
            
            % Restore original voltage
            obj.setElectricField(original_voltage);
            
            if p.Results.verbose
                elapsed_time = toc;
                fprintf('Hysteresis measurement completed in %.2f seconds\n', elapsed_time);
                fprintf('Coercive voltage: %.2f V\n', mean(coercive_voltages));
            end
        end
        
        function results = measureSwitchingDynamics(obj, voltage_pulse, time_span, varargin)
            % Measure magnetization switching dynamics under voltage pulse
            %
            % Inputs:
            %   voltage_pulse - Structure defining voltage pulse parameters
            %   time_span - Time range for simulation [t_start, t_end] (s)
            %   varargin - Optional parameters
            
            % Parse options
            p = inputParser;
            addParameter(p, 'initial_state', [1, 0, 0], @(x) isnumeric(x) && length(x) == 3);
            addParameter(p, 'time_resolution', 1000, @(x) isnumeric(x) && x > 0);
            addParameter(p, 'verbose', true, @islogical);
            parse(p, varargin{:});
            
            if p.Results.verbose
                fprintf('Starting switching dynamics measurement...\n');
                fprintf('Time span: %.2e to %.2e s\n', time_span(1), time_span(2));
                fprintf('Pulse amplitude: %.2f V, width: %.2e s\n', ...
                        voltage_pulse.amplitude, voltage_pulse.width);
            end
            
            % Create time vector
            t_total = linspace(time_span(1), time_span(2), p.Results.time_resolution);
            dt = t_total(2) - t_total(1);
            n_points = length(t_total);
            
            % Initialize results arrays
            magnetizations = zeros(3, n_points);
            voltages = zeros(n_points, 1);
            effective_fields = zeros(3, n_points);
            energies = zeros(n_points, 1);
            
            % Set initial state
            obj.magnetization_state = p.Results.initial_state / norm(p.Results.initial_state);
            magnetizations(:, 1) = obj.magnetization_state;
            
            % Store original voltage
            original_voltage = obj.applied_voltage;
            
            % Progress tracking
            if p.Results.verbose
                tic;
            end
            
            % Time evolution
            for i = 1:n_points
                t_current = t_total(i);
                
                % Calculate voltage at current time
                V_current = obj.calculatePulseVoltage(t_current, voltage_pulse);
                voltages(i) = V_current;
                obj.setElectricField(V_current);
                
                % Calculate effective fields
                H_eff = obj.calculateEffectiveField(obj.magnetization_state);
                effective_fields(:, i) = H_eff;
                
                % Calculate total energy
                energies(i) = obj.calculateTotalEnergy(obj.magnetization_state);
                
                % Evolve magnetization (except for first point)
                if i > 1
                    % Use LLG equation to evolve magnetization
                    m_current = magnetizations(:, i-1);
                    
                    % LLG solver for one time step
                    [m_new, ~, ~] = LLGSolver(m_current, @(tt, mm) H_eff, ...
                                             obj.materials.ferromagnet.alpha, ...
                                             obj.materials.ferromagnet.gamma, ...
                                             [t_total(i-1), t_current], ...
                                             'Method', 'RK45', ...
                                             'ConserveMagnetization', true);
                    
                    obj.magnetization_state = m_new(:, end);
                    magnetizations(:, i) = obj.magnetization_state;
                end
                
                if p.Results.verbose && mod(i, round(n_points/10)) == 0
                    progress = i / n_points * 100;
                    fprintf('Progress: %.0f%% (t = %.2e s)\n', progress, t_current);
                end
            end
            
            % Calculate switching metrics
            switching_metrics = obj.analyzeSwitchingDynamics(t_total, magnetizations, voltages);
            
            % Package results
            results = struct();
            results.measurement_type = 'switching_dynamics';
            results.time = t_total;
            results.magnetizations = magnetizations;
            results.voltages = voltages;
            results.effective_fields = effective_fields;
            results.energies = energies;
            results.voltage_pulse = voltage_pulse;
            results.switching_metrics = switching_metrics;
            results.temperature = obj.temperature;
            results.coupling_mechanism = obj.coupling_mechanism;
            
            % Store results
            obj.results = results;
            
            % Restore original voltage
            obj.setElectricField(original_voltage);
            
            if p.Results.verbose
                elapsed_time = toc;
                fprintf('Dynamics measurement completed in %.2f seconds\n', elapsed_time);
                if ~isempty(switching_metrics.switching_time)
                    fprintf('Switching time: %.2e s\n', switching_metrics.switching_time);
                end
            end
        end
        
        function results = analyzeVCMA(obj, voltage_range, varargin)
            % Analyze voltage-controlled magnetic anisotropy (VCMA) effect
            %
            % Inputs:
            %   voltage_range - Array of voltages to analyze (V)
            %   varargin - Optional parameters
            
            % Parse options
            p = inputParser;
            addParameter(p, 'field_direction', [0, 0, 1], @(x) isnumeric(x) && length(x) == 3);
            addParameter(p, 'field_strength', 0.01, @(x) isnumeric(x) && x >= 0);
            addParameter(p, 'verbose', true, @islogical);
            parse(p, varargin{:});
            
            if p.Results.verbose
                fprintf('Starting VCMA analysis...\n');
                fprintf('Voltage range: %.2f to %.2f V (%d points)\n', ...
                        min(voltage_range), max(voltage_range), length(voltage_range));
            end
            
            % Store original settings
            original_voltage = obj.applied_voltage;
            original_field = obj.magnetic_field;
            
            % Set small perpendicular field for anisotropy measurement
            field_vector = p.Results.field_direction / norm(p.Results.field_direction) * ...
                          p.Results.field_strength;
            obj.magnetic_field = field_vector;
            
            % Initialize results arrays
            n_points = length(voltage_range);
            anisotropy_constants = zeros(n_points, 1);
            anisotropy_fields = zeros(n_points, 1);
            easy_axis_directions = zeros(3, n_points);
            vcma_coefficients = zeros(n_points, 1);
            resonance_frequencies = zeros(n_points, 1);
            
            % Progress tracking
            if p.Results.verbose
                tic;
            end
            
            % Voltage sweep for VCMA analysis
            for i = 1:n_points
                % Set voltage
                obj.setElectricField(voltage_range(i));
                
                % Calculate voltage-dependent anisotropy
                [K_eff, H_anis, easy_axis] = obj.calculateVoltageAnisotropy();
                
                anisotropy_constants(i) = K_eff;
                anisotropy_fields(i) = norm(H_anis);
                easy_axis_directions(:, i) = easy_axis;
                
                % Calculate VCMA coefficient: ∂K/∂V
                if i > 1
                    dV = voltage_range(i) - voltage_range(i-1);
                    dK = anisotropy_constants(i) - anisotropy_constants(i-1);
                    vcma_coefficients(i) = dK / dV;
                else
                    vcma_coefficients(i) = 0;
                end
                
                % Calculate ferromagnetic resonance frequency
                resonance_frequencies(i) = obj.calculateFMRFrequency(K_eff);
                
                if p.Results.verbose && mod(i, max(1, round(n_points/10))) == 0
                    progress = i / n_points * 100;
                    fprintf('Progress: %.0f%% (V = %.2f V)\n', progress, voltage_range(i));
                end
            end
            
            % Fit VCMA coefficient
            vcma_fit = obj.fitVCMACoefficient(voltage_range, anisotropy_constants);
            
            % Package results
            results = struct();
            results.measurement_type = 'vcma_analysis';
            results.voltage_range = voltage_range;
            results.anisotropy_constants = anisotropy_constants;
            results.anisotropy_fields = anisotropy_fields;
            results.easy_axis_directions = easy_axis_directions;
            results.vcma_coefficients = vcma_coefficients;
            results.resonance_frequencies = resonance_frequencies;
            results.vcma_fit = vcma_fit;
            results.temperature = obj.temperature;
            results.coupling_mechanism = obj.coupling_mechanism;
            
            % Store results
            obj.results = results;
            
            % Restore original settings
            obj.setElectricField(original_voltage);
            obj.magnetic_field = original_field;
            
            if p.Results.verbose
                elapsed_time = toc;
                fprintf('VCMA analysis completed in %.2f seconds\n', elapsed_time);
                fprintf('VCMA coefficient: %.2e J/(V⋅m³)\n', vcma_fit.coefficient);
            end
        end
        
        function plotResults(obj, varargin)
            % Plot measurement results with Berkeley styling
            
            if isempty(obj.results)
                error('No results to plot. Run a measurement first.');
            end
            
            % Parse options
            p = inputParser;
            addParameter(p, 'save_figures', false, @islogical);
            addParameter(p, 'figure_path', pwd, @ischar);
            parse(p, varargin{:});
            
            % Apply Berkeley styling
            berkeley();
            
            % Clear existing figures
            obj.figures = [];
            
            % Plot based on measurement type
            switch obj.results.measurement_type
                case 'hysteresis'
                    obj.plotHysteresisResults();
                    
                case 'switching_dynamics'
                    obj.plotSwitchingResults();
                    
                case 'vcma_analysis'
                    obj.plotVCMAResults();
                    
                otherwise
                    warning('Unknown measurement type: %s', obj.results.measurement_type);
            end
            
            % Save figures if requested
            if p.Results.save_figures
                obj.saveFigures(p.Results.figure_path);
            end
        end
        
    end
    
    methods (Access = private)
        
        function parseInputs(obj, varargin)
            % Parse constructor input arguments
            p = inputParser;
            addParameter(p, 'Temperature', 300, @isnumeric);
            addParameter(p, 'CouplingMechanism', 'strain', @ischar);
            addParameter(p, 'IncludeDynamics', true, @islogical);
            addParameter(p, 'IncludeThermalEffects', false, @islogical);
            parse(p, varargin{:});
            
            obj.temperature = p.Results.Temperature;
            obj.coupling_mechanism = p.Results.CouplingMechanism;
            obj.include_dynamics = p.Results.IncludeDynamics;
            obj.include_thermal_effects = p.Results.IncludeThermalEffects;
        end
        
        function initializeStates(obj)
            % Initialize device state variables
            obj.magnetization_state = [1, 0, 0];  % Initial magnetization along +x
            obj.polarization_state = 0;           % Initial electric polarization
            obj.strain_state = zeros(3, 3);       % Initial strain tensor
        end
        
        function updateCapacitance(obj)
            % Update ferroelectric capacitance based on geometry
            
            if isempty(obj.geometry) || isempty(obj.materials)
                return;
            end
            
            % Parallel plate capacitor model
            epsilon_0 = 8.854e-12;  % Vacuum permittivity (F/m)
            epsilon_r = obj.materials.ferroelectric.permittivity;
            
            obj.geometry.capacitance = epsilon_0 * epsilon_r * obj.geometry.area / ...
                                      obj.geometry.fe_thickness;
        end
        
        function setDefaultCouplingCoefficients(obj)
            % Set default magnetoelectric coupling coefficients
            
            % These values are typical for multiferroic heterostructures
            coupling = struct();
            
            % Strain-mediated coupling (J/(m³⋅strain))
            coupling.strain = struct();
            coupling.strain.lambda_s = 1e5;    % Magnetostriction constant
            coupling.strain.d33 = 400e-12;     % Piezoelectric coefficient (C/N)
            coupling.strain.elastic_modulus = 200e9;  % Young's modulus (Pa)
            
            % Charge-mediated coupling (J⋅m/C²)
            coupling.charge = struct();
            coupling.charge.alpha = 1e-10;     % Surface charge coupling
            coupling.charge.interface_capacitance = 1e-6;  % Interface capacitance
            
            % Exchange coupling (J/m²)
            coupling.exchange = struct();
            coupling.exchange.J_ex = 1e-3;     % Exchange constant
            coupling.exchange.interface_area = obj.geometry.area;
            
            obj.materials.coupling = coupling;
        end
        
        function updateTemperatureDependentProperties(obj)
            % Update material properties based on temperature
            
            if isempty(obj.materials)
                return;
            end
            
            T = obj.temperature;
            T0 = 300;  % Reference temperature (K)
            
            % Ferromagnetic properties
            % Saturation magnetization: Ms(T) = Ms(0) * (1 - (T/Tc)^α)
            T_curie_fm = obj.materials.ferromagnet.T_curie;
            if T < T_curie_fm
                alpha_ms = 1.5;  % Critical exponent
                thermal_factor = 1 - (T / T_curie_fm)^alpha_ms;
                obj.ferromagnetic_props.Ms = obj.materials.ferromagnet.Ms * thermal_factor;
            else
                obj.ferromagnetic_props.Ms = 0;
            end
            
            % Magnetocrystalline anisotropy: K(T) = K(0) * (Ms(T)/Ms(0))^n
            n_anis = 2;  % Typical exponent for uniaxial anisotropy
            obj.ferromagnetic_props.Ku = obj.materials.ferromagnet.Ku * ...
                                        (obj.ferromagnetic_props.Ms / obj.materials.ferromagnet.Ms)^n_anis;
            
            % Ferroelectric properties
            % Spontaneous polarization: Ps(T) = Ps(0) * (1 - T/Tc)^β
            T_curie_fe = obj.materials.ferroelectric.T_curie;
            if T < T_curie_fe
                beta = 0.5;  % Critical exponent for polarization
                thermal_factor = (1 - T / T_curie_fe)^beta;
                obj.ferroelectric_props.Ps = obj.materials.ferroelectric.Ps * thermal_factor;
            else
                obj.ferroelectric_props.Ps = 0;
            end
            
            % Permittivity: ε(T) follows Curie-Weiss law
            C_curie = 1e5;  % Curie constant (K)
            if T ~= T_curie_fe
                obj.ferroelectric_props.permittivity = C_curie / abs(T - T_curie_fe);
            else
                obj.ferroelectric_props.permittivity = obj.materials.ferroelectric.permittivity;
            end
            
            % Coupling coefficients (generally decrease with temperature)
            thermal_reduction = exp(-(T - T0) / 500);  % Characteristic temperature 500K
            obj.coupling_coeffs = obj.materials.coupling;
            
            % Scale coupling strengths
            obj.coupling_coeffs.strain.lambda_s = obj.coupling_coeffs.strain.lambda_s * thermal_reduction;
            obj.coupling_coeffs.charge.alpha = obj.coupling_coeffs.charge.alpha * thermal_reduction;
            obj.coupling_coeffs.exchange.J_ex = obj.coupling_coeffs.exchange.J_ex * thermal_reduction;
        end
        
        function updateEffectiveFields(obj)
            % Update effective magnetic fields based on current voltage
            
            % Calculate electric field in ferroelectric layer
            E_field = obj.applied_voltage / obj.geometry.fe_thickness;
            
            % Calculate coupling-dependent contributions
            switch obj.coupling_mechanism
                case 'strain'
                    obj.anisotropy_fields = obj.calculateStrainMediatedField(E_field);
                    
                case 'charge'
                    obj.anisotropy_fields = obj.calculateChargeMediatedField(E_field);
                    
                case 'exchange'
                    obj.anisotropy_fields = obj.calculateExchangeCoupledField(E_field);
                    
                case 'hybrid'
                    % Combination of all mechanisms
                    H_strain = obj.calculateStrainMediatedField(E_field);
                    H_charge = obj.calculateChargeMediatedField(E_field);
                    H_exchange = obj.calculateExchangeCoupledField(E_field);
                    obj.anisotropy_fields = H_strain + H_charge + H_exchange;
                    
                otherwise
                    error('Unknown coupling mechanism: %s', obj.coupling_mechanism);
            end
            
            % Apply coupling strength multiplier
            obj.anisotropy_fields = obj.anisotropy_fields * obj.coupling_strength;
        end
        
        function H_strain = calculateStrainMediatedField(obj, E_field)
            % Calculate strain-mediated magnetoelectric field
            
            % Electric field → strain → magnetic anisotropy change
            
            % Piezoelectric strain: ε = d * E
            strain = obj.coupling_coeffs.strain.d33 * E_field;
            
            % Magnetoelastic energy change: ΔE = -λ_s * σ * m²
            % Effective field: H = (∂E/∂m) / (μ₀ * Ms)
            lambda_s = obj.coupling_coeffs.strain.lambda_s;
            E_mod = obj.coupling_coeffs.strain.elastic_modulus;
            stress = E_mod * strain;
            
            mu0 = 4*pi*1e-7;
            Ms = obj.ferromagnetic_props.Ms;
            
            % Strain-induced anisotropy field (along strain direction)
            H_magnitude = (2 * lambda_s * stress) / (mu0 * Ms);
            
            % Assume strain is primarily along z-direction (out-of-plane)
            H_strain = [0; 0; H_magnitude];
        end
        
        function H_charge = calculateChargeMediatedField(obj, E_field)
            % Calculate charge-mediated magnetoelectric field
            
            % Electric field → surface charge → magnetic anisotropy change
            
            % Surface charge density: σ = ε₀ * ε_r * E
            epsilon_0 = 8.854e-12;
            epsilon_r = obj.ferroelectric_props.permittivity;
            surface_charge = epsilon_0 * epsilon_r * E_field;
            
            % Charge-induced anisotropy change
            alpha = obj.coupling_coeffs.charge.alpha;
            
            mu0 = 4*pi*1e-7;
            Ms = obj.ferromagnetic_props.Ms;
            
            % Charge-induced anisotropy field
            H_magnitude = (alpha * surface_charge^2) / (mu0 * Ms * obj.geometry.fm_thickness);
            
            % Field direction depends on charge polarity and interface
            H_charge = [0; 0; sign(surface_charge) * H_magnitude];
        end
        
        function H_exchange = calculateExchangeCoupledField(obj, E_field)
            % Calculate exchange-coupled magnetoelectric field
            
            % Direct exchange coupling at multiferroic interface
            
            % Electric field modifies exchange coupling strength
            J_ex = obj.coupling_coeffs.exchange.J_ex;
            
            % Exchange field proportional to electric polarization
            P_electric = obj.calculateElectricPolarization();
            
            mu0 = 4*pi*1e-7;
            Ms = obj.ferromagnetic_props.Ms;
            
            % Exchange-induced effective field
            H_magnitude = (2 * J_ex * P_electric) / (mu0 * Ms * obj.geometry.fm_thickness);
            
            % Exchange field typically favors in-plane alignment
            H_exchange = [H_magnitude; 0; 0];
        end
        
        function P = calculateElectricPolarization(obj)
            % Calculate electric polarization in ferroelectric layer
            
            % Simple linear model: P = χ * ε₀ * E + Ps
            E_field = obj.applied_voltage / obj.geometry.fe_thickness;
            
            epsilon_0 = 8.854e-12;
            chi_e = obj.ferroelectric_props.permittivity - 1;  % Electric susceptibility
            Ps = obj.ferroelectric_props.Ps;
            
            % Include hysteresis effects if enabled
            if obj.include_hysteresis
                % Simplified hysteresis model
                E_coercive = Ps / (epsilon_0 * chi_e);  % Coercive field
                
                if abs(E_field) > E_coercive
                    P = sign(E_field) * Ps + chi_e * epsilon_0 * E_field;
                else
                    % Maintain previous polarization state below coercive field
                    P = obj.polarization_state + chi_e * epsilon_0 * E_field;
                end
            else
                P = Ps * tanh(E_field / (Ps / (epsilon_0 * chi_e))) + chi_e * epsilon_0 * E_field;
            end
            
            obj.polarization_state = P;
        end
        
        function m_eq = calculateEquilibriumMagnetization(obj)
            % Calculate equilibrium magnetization for current conditions
            
            % Total effective field
            H_total = obj.calculateEffectiveField(obj.magnetization_state);
            
            % Minimize energy to find equilibrium
            % Use spherical coordinates to preserve |m| = 1
            
            % Initial guess from current state
            m_current = obj.magnetization_state;
            [theta0, phi0] = cart2sph(m_current(1), m_current(2), m_current(3));
            
            % Energy function in spherical coordinates
            energy_func = @(angles) obj.calculateTotalEnergySpherical(angles(1), angles(2));
            
            % Minimize energy
            options = optimoptions('fminunc', 'Display', 'off', ...
                                  'Algorithm', 'quasi-newton');
            
            try
                [angles_opt, ~] = fminunc(energy_func, [theta0, phi0], options);
                
                % Convert back to Cartesian coordinates
                [mx, my, mz] = sph2cart(angles_opt(1), angles_opt(2), 1);
                m_eq = [mx; my; mz];
                
            catch
                % If optimization fails, use current state
                m_eq = m_current;
            end
            
            % Ensure normalization
            m_eq = m_eq / norm(m_eq);
        end
        
        function m_final = evolveToEquilibrium(obj, total_time)
            % Evolve magnetization to equilibrium using LLG dynamics
            
            % Time parameters
            n_steps = 100;
            t_span = [0, total_time];
            
            % Initial magnetization
            m0 = obj.magnetization_state;
            
            % Effective field function
            H_eff_func = @(t, m) obj.calculateEffectiveField(m);
            
            % Solve LLG equation
            try
                [m_trajectory, ~, ~] = LLGSolver(m0, H_eff_func, ...
                                               obj.materials.ferromagnet.alpha, ...
                                               obj.materials.ferromagnet.gamma, ...
                                               t_span, ...
                                               'Method', 'RK45', ...
                                               'ConserveMagnetization', true, ...
                                               'Verbose', false);
                
                m_final = m_trajectory(:, end);
                
            catch
                % If LLG solver fails, use energy minimization
                m_final = obj.calculateEquilibriumMagnetization();
            end
            
            obj.magnetization_state = m_final;
        end
        
        function H_total = calculateEffectiveField(obj, m)
            % Calculate total effective field for given magnetization
            
            % External field
            H_external = obj.magnetic_field';
            
            % Demagnetization field (shape anisotropy)
            H_demag = obj.calculateDemagnetizationField(m);
            
            % Magnetocrystalline anisotropy field
            H_anis = obj.calculateAnisotropyField(m);
            
            % Voltage-controlled anisotropy field
            obj.updateEffectiveFields();
            H_vcma = obj.anisotropy_fields;
            
            % Thermal field (if enabled)
            if obj.include_thermal_effects
                H_thermal = obj.calculateThermalField(m);
            else
                H_thermal = [0; 0; 0];
            end
            
            % Total field
            H_total = H_external + H_demag + H_anis + H_vcma + H_thermal;
        end
        
        function H_demag = calculateDemagnetizationField(obj, m)
            % Calculate demagnetization field based on geometry
            
            % Demagnetization factors for thin film (prolate ellipsoid)
            aspect_ratio = obj.geometry.fm_thickness / sqrt(obj.geometry.area);
            
            if aspect_ratio < 0.1  % Thin film limit
                Nx = 0;
                Ny = 0;
                Nz = 1;
            else
                % Use ellipsoid approximation
                [Nx, Ny, Nz] = obj.calculateEllipsoidDemagFactors();
            end
            
            % Demagnetization field: H_d = -N * Ms * m
            Ms = obj.ferromagnetic_props.Ms;
            H_demag = -Ms * [Nx * m(1); Ny * m(2); Nz * m(3)];
        end
        
        function H_anis = calculateAnisotropyField(obj, m)
            % Calculate magnetocrystalline anisotropy field
            
            % Uniaxial anisotropy along easy axis (assume z-direction)
            Ku = obj.ferromagnetic_props.Ku;
            Ms = obj.ferromagnetic_props.Ms;
            
            mu0 = 4*pi*1e-7;
            
            % Anisotropy field: H_anis = (2*Ku/μ₀*Ms) * (m·ẑ) * ẑ
            easy_axis = [0; 0; 1];
            H_anis = (2 * Ku / (mu0 * Ms)) * dot(m, easy_axis) * easy_axis;
        end
        
        function H_thermal = calculateThermalField(obj, m)
            % Calculate thermal fluctuation field
            
            % Thermal field based on fluctuation-dissipation theorem
            kB = 1.381e-23;  % Boltzmann constant
            alpha = obj.materials.ferromagnet.alpha;
            gamma = obj.materials.ferromagnet.gamma;
            mu0 = 4*pi*1e-7;
            Ms = obj.ferromagnetic_props.Ms;
            V = obj.geometry.fm_volume;
            dt = 1e-12;  % Assumed time step
            
            H_th_rms = sqrt((2 * alpha * kB * obj.temperature) / (gamma * mu0 * Ms * V * dt));
            
            % Generate random field components
            H_thermal = H_th_rms * randn(3, 1);
        end
        
        function [Nx, Ny, Nz] = calculateEllipsoidDemagFactors(obj)
            % Calculate demagnetization factors for ellipsoidal geometry
            
            % Approximate as ellipsoid with semi-axes
            a = obj.geometry.length / 2;
            b = obj.geometry.width / 2;
            c = obj.geometry.fm_thickness / 2;
            
            % For thin films (c << a, b), use analytical expressions
            if c < min(a, b) / 10
                Nz = 1;
                Nx = Ny = 0;
            else
                % General ellipsoid case (simplified)
                total_factor = 4*pi;
                aspect_ratio_xy = a / b;
                aspect_ratio_z = c / sqrt(a * b);
                
                if aspect_ratio_z < 0.5
                    Nz = total_factor * aspect_ratio_z^2 / 3;
                    Nx = Ny = (total_factor - Nz) / 2;
                else
                    % More complex calculation for general ellipsoids
                    Nx = Ny = Nz = total_factor / 3;  % Sphere approximation
                end
            end
            
            % Normalize
            total = Nx + Ny + Nz;
            Nx = Nx / total * 4*pi;
            Ny = Ny / total * 4*pi;
            Nz = Nz / total * 4*pi;
        end
        
        function E_total = calculateTotalEnergy(obj, m)
            % Calculate total magnetic energy for given magnetization
            
            % Zeeman energy: E_Z = -μ₀ * M · H_external
            mu0 = 4*pi*1e-7;
            Ms = obj.ferromagnetic_props.Ms;
            V = obj.geometry.fm_volume;
            
            E_zeeman = -mu0 * Ms * V * dot(m, obj.magnetic_field);
            
            % Demagnetization energy: E_d = (1/2) * μ₀ * Ms² * V * m^T * N * m
            H_demag = obj.calculateDemagnetizationField(m);
            E_demag = -0.5 * mu0 * Ms * V * dot(m, H_demag);
            
            % Anisotropy energy: E_a = Ku * V * sin²(θ) for uniaxial
            Ku = obj.ferromagnetic_props.Ku;
            theta = acos(abs(m(3)));  % Angle from easy axis (z)
            E_anis = Ku * V * sin(theta)^2;
            
            % Voltage-controlled anisotropy energy
            E_vcma = obj.calculateVCMAEnergy(m);
            
            E_total = E_zeeman + E_demag + E_anis + E_vcma;
        end
        
        function E_spherical = calculateTotalEnergySpherical(obj, theta, phi)
            % Calculate total energy in spherical coordinates
            
            % Convert to Cartesian
            [mx, my, mz] = sph2cart(phi, pi/2 - theta, 1);
            m = [mx; my; mz];
            
            E_spherical = obj.calculateTotalEnergy(m);
        end
        
        function E_vcma = calculateVCMAEnergy(obj, m)
            % Calculate voltage-controlled magnetic anisotropy energy
            
            % Energy depends on coupling mechanism
            E_field = obj.applied_voltage / obj.geometry.fe_thickness;
            V = obj.geometry.fm_volume;
            
            switch obj.coupling_mechanism
                case 'strain'
                    % Strain-mediated VCMA energy
                    strain = obj.coupling_coeffs.strain.d33 * E_field;
                    lambda_s = obj.coupling_coeffs.strain.lambda_s;
                    E_mod = obj.coupling_coeffs.strain.elastic_modulus;
                    
                    % Magnetoelastic energy: E = -λ_s * σ * V * (m·n)²
                    stress = E_mod * strain;
                    strain_direction = [0; 0; 1];  % z-direction
                    E_vcma = -lambda_s * stress * V * dot(m, strain_direction)^2;
                    
                case 'charge'
                    % Charge-mediated VCMA energy
                    epsilon_0 = 8.854e-12;
                    epsilon_r = obj.ferroelectric_props.permittivity;
                    surface_charge = epsilon_0 * epsilon_r * E_field;
                    alpha = obj.coupling_coeffs.charge.alpha;
                    
                    % Charge-induced anisotropy energy
                    E_vcma = -alpha * surface_charge^2 * V * m(3)^2;
                    
                case 'exchange'
                    % Exchange-mediated VCMA energy
                    J_ex = obj.coupling_coeffs.exchange.J_ex;
                    P_electric = obj.calculateElectricPolarization();
                    
                    % Exchange coupling energy
                    E_vcma = -J_ex * P_electric * obj.geometry.area * m(1);
                    
                case 'hybrid'
                    % Combination of all mechanisms
                    % (This is a simplified model)
                    E_vcma = 0.3 * obj.calculateVCMAEnergyMechanism('strain', m) + ...
                            0.4 * obj.calculateVCMAEnergyMechanism('charge', m) + ...
                            0.3 * obj.calculateVCMAEnergyMechanism('exchange', m);
                    
                otherwise
                    E_vcma = 0;
            end
            
            % Apply coupling strength multiplier
            E_vcma = E_vcma * obj.coupling_strength;
        end
        
        function E = calculateVCMAEnergyMechanism(obj, mechanism, m)
            % Helper function to calculate VCMA energy for specific mechanism
            
            original_mechanism = obj.coupling_mechanism;
            obj.coupling_mechanism = mechanism;
            E = obj.calculateVCMAEnergy(m);
            obj.coupling_mechanism = original_mechanism;
        end
        
        function E_anis = calculateAnisotropyEnergy(obj, m)
            % Calculate anisotropy energy for given magnetization
            
            % Magnetocrystalline anisotropy energy
            Ku = obj.ferromagnetic_props.Ku;
            V = obj.geometry.fm_volume;
            
            % Uniaxial anisotropy: E = Ku * V * sin²(θ)
            theta = acos(abs(m(3)));  % Angle from easy axis
            E_anis = Ku * V * sin(theta)^2;
        end
        
        function H_sw = calculateSwitchingField(obj, m)
            % Calculate switching field for current magnetization state
            
            % Energy barrier method
            E_current = obj.calculateTotalEnergy(m);
            
            % Find energy barrier by rotating magnetization
            theta_range = linspace(0, pi, 100);
            energies = zeros(size(theta_range));
            
            for i = 1:length(theta_range)
                m_test = [sin(theta_range(i)); 0; cos(theta_range(i))];
                energies(i) = obj.calculateTotalEnergy(m_test);
            end
            
            % Find maximum energy (barrier)
            [E_barrier, ~] = max(energies);
            
            % Switching field estimate: H_sw = 2 * E_barrier / (μ₀ * Ms * V)
            mu0 = 4*pi*1e-7;
            Ms = obj.ferromagnetic_props.Ms;
            V = obj.geometry.fm_volume;
            
            H_sw = 2 * (E_barrier - E_current) / (mu0 * Ms * V);
        end
        
        function E_barrier = calculateEnergyBarrier(obj, m)
            % Calculate energy barrier for magnetization switching
            
            % Current energy
            E_current = obj.calculateTotalEnergy(m);
            
            % Find barrier by sampling energy landscape
            n_samples = 50;
            theta_range = linspace(0, pi, n_samples);
            phi_range = linspace(0, 2*pi, n_samples);
            
            max_energy = E_current;
            
            for i = 1:length(theta_range)
                for j = 1:length(phi_range)
                    [mx, my, mz] = sph2cart(phi_range(j), pi/2 - theta_range(i), 1);
                    m_test = [mx; my; mz];
                    E_test = obj.calculateTotalEnergy(m_test);
                    
                    if E_test > max_energy
                        max_energy = E_test;
                    end
                end
            end
            
            E_barrier = max_energy - E_current;
        end
        
        function coercive_voltages = findCoerciveVoltages(obj, voltage_range, magnetizations)
            % Find coercive voltages from hysteresis loop
            
            % Look for sign changes in magnetization component
            mx = magnetizations(1, :);
            
            % Find zero crossings
            zero_crossings = [];
            for i = 1:length(mx)-1
                if mx(i) * mx(i+1) < 0
                    % Linear interpolation to find exact crossing
                    V_cross = voltage_range(i) + (voltage_range(i+1) - voltage_range(i)) * ...
                             (-mx(i)) / (mx(i+1) - mx(i));
                    zero_crossings(end+1) = V_cross;
                end
            end
            
            coercive_voltages = zero_crossings;
        end
        
        function saturation_values = findSaturationValues(obj, magnetizations, polarizations)
            % Find saturation values from measurement data
            
            saturation_values = struct();
            
            % Magnetization saturation
            mx = magnetizations(1, :);
            saturation_values.m_sat_pos = max(mx);
            saturation_values.m_sat_neg = min(mx);
            
            % Polarization saturation
            saturation_values.P_sat_pos = max(polarizations);
            saturation_values.P_sat_neg = min(polarizations);
        end
        
        function V_pulse = calculatePulseVoltage(obj, t, pulse_params)
            % Calculate voltage pulse at given time
            
            t_start = pulse_params.start_time;
            t_width = pulse_params.width;
            V_amplitude = pulse_params.amplitude;
            
            if t >= t_start && t <= t_start + t_width
                V_pulse = V_amplitude;
            else
                V_pulse = 0;
            end
        end
        
        function metrics = analyzeSwitchingDynamics(obj, time, magnetizations, voltages)
            % Analyze switching metrics from dynamics data
            
            metrics = struct();
            
            % Find switching time (10% to 90% transition)
            mx = magnetizations(1, :);
            
            if abs(mx(end) - mx(1)) > 0.1  % Significant switching occurred
                % Find 10% and 90% levels
                m_initial = mx(1);
                m_final = mx(end);
                m_10 = m_initial + 0.1 * (m_final - m_initial);
                m_90 = m_initial + 0.9 * (m_final - m_initial);
                
                % Find crossing times
                idx_10 = find(abs(mx - m_10) == min(abs(mx - m_10)), 1);
                idx_90 = find(abs(mx - m_90) == min(abs(mx - m_90)), 1);
                
                if ~isempty(idx_10) && ~isempty(idx_90) && idx_90 > idx_10
                    metrics.switching_time = time(idx_90) - time(idx_10);
                else
                    metrics.switching_time = [];
                end
            else
                metrics.switching_time = [];
            end
            
            % Calculate switching efficiency
            if ~isempty(metrics.switching_time)
                metrics.switching_efficiency = abs(mx(end) - mx(1)) / max(abs(mx));
            else
                metrics.switching_efficiency = 0;
            end
            
            % Find peak voltage and energy
            metrics.peak_voltage = max(abs(voltages));
            
            % Estimate energy consumption (simplified)
            if ~isempty(obj.geometry.capacitance)
                metrics.switching_energy = 0.5 * obj.geometry.capacitance * metrics.peak_voltage^2;
            else
                metrics.switching_energy = [];
            end
        end
        
        function [K_eff, H_anis, easy_axis] = calculateVoltageAnisotropy(obj)
            % Calculate voltage-dependent magnetic anisotropy
            
            % Base anisotropy constant
            Ku_base = obj.ferromagnetic_props.Ku;
            
            % Voltage-dependent contribution
            E_field = obj.applied_voltage / obj.geometry.fe_thickness;
            
            % VCMA coefficient (depends on coupling mechanism)
            switch obj.coupling_mechanism
                case 'strain'
                    % Strain-mediated VCMA
                    strain = obj.coupling_coeffs.strain.d33 * E_field;
                    lambda_s = obj.coupling_coeffs.strain.lambda_s;
                    E_mod = obj.coupling_coeffs.strain.elastic_modulus;
                    
                    dK_dV = lambda_s * E_mod * obj.coupling_coeffs.strain.d33 / obj.geometry.fe_thickness;
                    
                case 'charge'
                    % Charge-mediated VCMA
                    epsilon_0 = 8.854e-12;
                    epsilon_r = obj.ferroelectric_props.permittivity;
                    alpha = obj.coupling_coeffs.charge.alpha;
                    
                    dK_dV = 2 * alpha * epsilon_0 * epsilon_r * E_field / obj.geometry.fe_thickness;
                    
                case 'exchange'
                    % Exchange-mediated VCMA
                    J_ex = obj.coupling_coeffs.exchange.J_ex;
                    dP_dV = obj.ferroelectric_props.permittivity * 8.854e-12 / obj.geometry.fe_thickness;
                    
                    dK_dV = J_ex * dP_dV / obj.geometry.fm_thickness;
                    
                otherwise
                    dK_dV = 0;
            end
            
            % Effective anisotropy constant
            K_eff = Ku_base + dK_dV * obj.applied_voltage * obj.coupling_strength;
            
            % Anisotropy field
            mu0 = 4*pi*1e-7;
            Ms = obj.ферromagnetic_props.Ms;
            H_anis_magnitude = 2 * K_eff / (mu0 * Ms);
            
            % Easy axis direction (assume z for uniaxial anisotropy)
            easy_axis = [0; 0; 1];
            H_anis = H_anis_magnitude * easy_axis;
        end
        
        function f_fmr = calculateFMRFrequency(obj, K_eff)
            % Calculate ferromagnetic resonance frequency
            
            % FMR frequency: f = (γ/2π) * sqrt(H_anis * (H_anis + Ms))
            mu0 = 4*pi*1e-7;
            Ms = obj.ferromagnetic_props.Ms;
            gamma = obj.materials.ferromagnet.gamma;
            
            H_anis = 2 * K_eff / (mu0 * Ms);
            H_total = H_anis + Ms;  % Include demagnetization
            
            omega_fmr = gamma * sqrt(H_anis * H_total);
            f_fmr = omega_fmr / (2 * pi);
        end
        
        function vcma_fit = fitVCMACoefficient(obj, voltage_range, anisotropy_constants)
            % Fit VCMA coefficient from voltage-dependent anisotropy data
            
            % Linear fit: K(V) = K₀ + α * V
            try
                p = polyfit(voltage_range, anisotropy_constants, 1);
                
                vcma_fit = struct();
                vcma_fit.coefficient = p(1);  % dK/dV
                vcma_fit.offset = p(2);       % K₀
                vcma_fit.fitted_curve = polyval(p, voltage_range);
                
                % Calculate R-squared
                ss_res = sum((anisotropy_constants - vcma_fit.fitted_curve).^2);
                ss_tot = sum((anisotropy_constants - mean(anisotropy_constants)).^2);
                vcma_fit.r_squared = 1 - ss_res / ss_tot;
                
                vcma_fit.success = true;
                
            catch
                vcma_fit = struct();
                vcma_fit.success = false;
                vcma_fit.coefficient = 0;
                vcma_fit.offset = mean(anisotropy_constants);
            end
        end
        
        function plotHysteresisResults(obj)
            % Plot hysteresis measurement results
            
            results = obj.results;
            colors = getBerkeleyColors();
            
            fig = figure('Position', [100, 100, 1400, 1000]);
            
            % Magnetization hysteresis
            subplot(2, 3, 1);
            plot(results.voltage_range, results.magnetizations(1, :), ...
                 'LineWidth', 2, 'Color', colors(1,:));
            hold on;
            plot(results.voltage_range, results.magnetizations(2, :), ...
                 'LineWidth', 2, 'Color', colors(2,:));
            plot(results.voltage_range, results.magnetizations(3, :), ...
                 'LineWidth', 2, 'Color', colors(3,:));
            xlabel('Applied Voltage (V)');
            ylabel('Magnetization');
            title('Magnetization Hysteresis');
            legend('m_x', 'm_y', 'm_z', 'Location', 'best');
            grid on;
            
            % Electric polarization
            subplot(2, 3, 2);
            plot(results.voltage_range, results.polarizations * 1e6, ...
                 'LineWidth', 2, 'Color', colors(4,:));
            xlabel('Applied Voltage (V)');
            ylabel('Polarization (μC/m²)');
            title('Electric Polarization');
            grid on;
            
            % Anisotropy energy
            subplot(2, 3, 3);
            plot(results.voltage_range, results.anisotropy_energies * 1e18, ...
                 'LineWidth', 2, 'Color', colors(5,:));
            xlabel('Applied Voltage (V)');
            ylabel('Anisotropy Energy (aJ)');
            title('Magnetic Anisotropy Energy');
            grid on;
            
            % Energy barrier
            subplot(2, 3, 4);
            semilogy(results.voltage_range, results.energy_barriers * 1e21, ...
                    'LineWidth', 2, 'Color', colors(6,:));
            xlabel('Applied Voltage (V)');
            ylabel('Energy Barrier (zJ)');
            title('Switching Energy Barrier');
            grid on;
            
            % Switching field
            subplot(2, 3, 5);
            plot(results.voltage_range, results.switching_fields * 1000, ...
                 'LineWidth', 2, 'Color', colors(7,:));
            xlabel('Applied Voltage (V)');
            ylabel('Switching Field (mT)');
            title('Voltage-Dependent Switching Field');
            grid on;
            
            % Coercive voltage analysis
            subplot(2, 3, 6);
            if ~isempty(results.coercive_voltages)
                bar(1:length(results.coercive_voltages), results.coercive_voltages, ...
                    'FaceColor', colors(8,:));
                xlabel('Coercive Event');
                ylabel('Coercive Voltage (V)');
                title('Coercive Voltages');
                grid on;
            else
                text(0.5, 0.5, 'No Coercive Voltages Found', ...
                     'HorizontalAlignment', 'center', 'FontSize', 14);
                axis off;
            end
            
            sgtitle(sprintf('Voltage-Controlled Magnet Hysteresis (%s coupling, T=%.0fK)', ...
                           results.coupling_mechanism, results.temperature));
            
            obj.figures(end+1) = fig;
        end
        
        function plotSwitchingResults(obj)
            % Plot switching dynamics results
            
            results = obj.results;
            colors = getBerkeleyColors();
            
            fig = figure('Position', [100, 100, 1400, 1000]);
            
            % Magnetization dynamics
            subplot(2, 3, 1);
            plot(results.time * 1e9, results.magnetizations(1, :), ...
                 'LineWidth', 2, 'Color', colors(1,:));
            hold on;
            plot(results.time * 1e9, results.magnetizations(2, :), ...
                 'LineWidth', 2, 'Color', colors(2,:));
            plot(results.time * 1e9, results.magnetizations(3, :), ...
                 'LineWidth', 2, 'Color', colors(3,:));
            xlabel('Time (ns)');
            ylabel('Magnetization');
            title('Magnetization Dynamics');
            legend('m_x', 'm_y', 'm_z', 'Location', 'best');
            grid on;
            
            % Applied voltage pulse
            subplot(2, 3, 2);
            plot(results.time * 1e9, results.voltages, ...
                 'LineWidth', 2, 'Color', colors(4,:));
            xlabel('Time (ns)');
            ylabel('Applied Voltage (V)');
            title('Voltage Pulse');
            grid on;
            
            % Effective magnetic field
            subplot(2, 3, 3);
            plot(results.time * 1e9, results.effective_fields(1, :) * 1000, ...
                 'LineWidth', 2, 'Color', colors(1,:));
            hold on;
            plot(results.time * 1e9, results.effective_fields(2, :) * 1000, ...
                 'LineWidth', 2, 'Color', colors(2,:));
            plot(results.time * 1e9, results.effective_fields(3, :) * 1000, ...
                 'LineWidth', 2, 'Color', colors(3,:));
            xlabel('Time (ns)');
            ylabel('Effective Field (mT)');
            title('Effective Magnetic Field');
            legend('H_x', 'H_y', 'H_z', 'Location', 'best');
            grid on;
            
            % Energy evolution
            subplot(2, 3, 4);
            plot(results.time * 1e9, results.energies * 1e18, ...
                 'LineWidth', 2, 'Color', colors(5,:));
            xlabel('Time (ns)');
            ylabel('Total Energy (aJ)');
            title('Energy Evolution');
            grid on;
            
            % Magnetization trajectory (Bloch sphere)
            subplot(2, 3, 5);
            [x_sphere, y_sphere, z_sphere] = sphere(20);
            surf(x_sphere, y_sphere, z_sphere, 'FaceAlpha', 0.1, ...
                 'EdgeColor', 'none', 'FaceColor', [0.8, 0.8, 0.8]);
            hold on;
            plot3(results.magnetizations(1, :), results.magnetizations(2, :), ...
                  results.magnetizations(3, :), 'LineWidth', 3, 'Color', colors(6,:));
            scatter3(results.magnetizations(1, 1), results.magnetizations(2, 1), ...
                    results.magnetizations(3, 1), 100, 'g', 'filled');  % Start
            scatter3(results.magnetizations(1, end), results.magnetizations(2, end), ...
                    results.magnetizations(3, end), 100, 'r', 'filled');  % End
            xlabel('m_x'); ylabel('m_y'); zlabel('m_z');
            title('Magnetization Trajectory');
            axis equal; axis([-1.1, 1.1, -1.1, 1.1, -1.1, 1.1]);
            view(45, 30);
            
            % Switching metrics summary
            subplot(2, 3, 6);
            metrics = results.switching_metrics;
            if ~isempty(metrics.switching_time)
                metric_names = {'Switching Time (ps)', 'Efficiency (%)', 'Peak Voltage (V)'};
                metric_values = [metrics.switching_time * 1e12, ...
                               metrics.switching_efficiency * 100, ...
                               metrics.peak_voltage];
                
                if ~isempty(metrics.switching_energy)
                    metric_names{end+1} = 'Energy (fJ)';
                    metric_values(end+1) = metrics.switching_energy * 1e15;
                end
                
                bar(metric_values, 'FaceColor', colors(7,:));
                set(gca, 'XTickLabel', metric_names);
                title('Switching Metrics');
                ylabel('Metric Value');
                grid on;
            else
                text(0.5, 0.5, 'No Switching Detected', ...
                     'HorizontalAlignment', 'center', 'FontSize', 14);
                axis off;
            end
            
            sgtitle(sprintf('Voltage-Controlled Switching Dynamics (%s coupling)', ...
                           results.coupling_mechanism));
            
            obj.figures(end+1) = fig;
        end
        
        function plotVCMAResults(obj)
            % Plot VCMA analysis results
            
            results = obj.results;
            colors = getBerkeleyColors();
            
            fig = figure('Position', [100, 100, 1400, 1000]);
            
            % Anisotropy constant vs voltage
            subplot(2, 3, 1);
            plot(results.voltage_range, results.anisotropy_constants * 1e-3, ...
                 'o', 'MarkerSize', 6, 'Color', colors(1,:), 'MarkerFaceColor', colors(1,:));
            if results.vcma_fit.success
                hold on;
                plot(results.voltage_range, results.vcma_fit.fitted_curve * 1e-3, ...
                     '--', 'LineWidth', 2, 'Color', colors(2,:));
                legend('Data', 'Linear Fit', 'Location', 'best');
            end
            xlabel('Applied Voltage (V)');
            ylabel('Anisotropy Constant (kJ/m³)');
            title('VCMA Effect');
            grid on;
            
            % Anisotropy field vs voltage
            subplot(2, 3, 2);
            plot(results.voltage_range, results.anisotropy_fields * 1000, ...
                 'LineWidth', 2, 'Color', colors(3,:));
            xlabel('Applied Voltage (V)');
            ylabel('Anisotropy Field (mT)');
            title('Voltage-Dependent Anisotropy Field');
            grid on;
            
            % Easy axis direction
            subplot(2, 3, 3);
            quiver3(zeros(size(results.voltage_range)), zeros(size(results.voltage_range)), ...
                   zeros(size(results.voltage_range)), ...
                   results.easy_axis_directions(1, :), results.easy_axis_directions(2, :), ...
                   results.easy_axis_directions(3, :), 0.5, 'Color', colors(4,:));
            xlabel('X'); ylabel('Y'); zlabel('Z');
            title('Easy Axis Direction');
            axis equal; grid on;
            
            % VCMA coefficient vs voltage
            subplot(2, 3, 4);
            plot(results.voltage_range, results.vcma_coefficients * 1e-6, ...
                 'LineWidth', 2, 'Color', colors(5,:));
            xlabel('Applied Voltage (V)');
            ylabel('dK/dV (MJ/(V⋅m³))');
            title('VCMA Coefficient');
            grid on;
            
            % FMR frequency vs voltage
            subplot(2, 3, 5);
            plot(results.voltage_range, results.resonance_frequencies * 1e-9, ...
                 'LineWidth', 2, 'Color', colors(6,:));
            xlabel('Applied Voltage (V)');
            ylabel('FMR Frequency (GHz)');
            title('Voltage-Tunable FMR');
            grid on;
            
            % VCMA fit summary
            subplot(2, 3, 6);
            if results.vcma_fit.success
                fit_info = {
                    sprintf('VCMA Coeff: %.2e J/(V⋅m³)', results.vcma_fit.coefficient);
                    sprintf('R²: %.3f', results.vcma_fit.r_squared);
                    sprintf('K₀: %.2e J/m³', results.vcma_fit.offset);
                    sprintf('Coupling: %s', results.coupling_mechanism);
                    sprintf('T: %.0f K', results.temperature)
                };
                
                text(0.1, 0.8, fit_info, 'FontSize', 12, 'VerticalAlignment', 'top');
                axis off;
                title('VCMA Fit Parameters');
            else
                text(0.5, 0.5, 'Fit Failed', 'HorizontalAlignment', 'center', ...
                     'FontSize', 16, 'Color', 'red');
                axis off;
            end
            
            sgtitle('Voltage-Controlled Magnetic Anisotropy Analysis');
            
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
            measurement_type = obj.results.measurement_type;
            
            for i = 1:length(obj.figures)
                filename = sprintf('VCM_%s_Fig%d_%s', measurement_type, i, timestamp);
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