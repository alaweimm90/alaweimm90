classdef NonlocalSpinValve < handle
% NONLOCALSPINVALVE - Comprehensive Nonlocal Spin Valve Device Model
%
% This class implements a detailed model of nonlocal spin valves (NLSV) for
% pure spin current detection and spin transport studies. The model includes
% all major physical effects relevant to lateral spin valves including 
% Hanle precession, temperature dependence, and interface effects.
%
% Physics Overview:
%   Nonlocal spin valves use pure spin currents flowing between spatially
%   separated contacts to avoid charge current artifacts. The device consists of:
%   1. Spin injector: Ferromagnetic contact for spin injection
%   2. Spin detector: Ferromagnetic contact for spin detection
%   3. Spin channel: Non-magnetic conductor for spin transport
%   4. Reference electrode: Optional fixed reference magnet
%
% Key Physical Effects:
%   - Spin injection and detection via interfacial spin polarization
%   - Spin diffusion and relaxation in the channel
%   - Hanle precession under perpendicular magnetic fields
%   - Temperature-dependent spin diffusion length and polarization
%   - Contact resistance and spin-dependent interface transmission
%   - Spin accumulation and electrochemical potential splitting
%
% Features:
%   - Magnetic field sweeps with Hanle effect analysis
%   - Temperature-dependent measurements
%   - Interface resistance characterization
%   - 3D geometry with arbitrary contact placement
%   - Johnson-Nyquist thermal noise modeling
%   - Contact-induced spin relaxation
%
% Usage:
%   nlsv = NonlocalSpinValve();
%   nlsv.setGeometry(2e-6, 100e-9, 10e-9);  % Length, width, thickness
%   nlsv.setMaterials('NiFe', 'Cu');
%   nlsv.addContact('injector', [0.5e-6, 0, 0], 100e-9);
%   nlsv.addContact('detector', [1.5e-6, 0, 0], 100e-9);
%   results = nlsv.measureHanle(0.1e-3, linspace(-0.1, 0.1, 101));
%   nlsv.plotResults();
%
% Author: Meshal Alawein <meshal@berkeley.edu>
% Copyright © 2025 Meshal Alawein — All rights reserved.
% License: MIT

    properties (Access = public)
        % Device geometry
        geometry            % Channel geometry parameters
        contacts           % Array of contact structures
        materials          % Material properties
        
        % Measurement parameters
        temperature        % Operating temperature (K)
        applied_current    % Injection current (A)
        magnetic_field     % Applied magnetic field vector (T)
        
        % Analysis options
        include_hanle      % Include Hanle precession effects
        include_thermal_noise  % Include Johnson-Nyquist noise
        include_contact_resistance  % Include contact resistance effects
        
        % Results storage
        results            % Measurement results
        figures            % Figure handles
    end
    
    properties (Access = private)
        % Internal calculation data
        spin_transport_solver  % Transport solver instance
        mesh                  % Spatial discretization
        contact_conductances  % Contact conductance matrices
        
        % Cached calculations
        baseline_voltage     % Zero-field voltage for Hanle analysis
        spin_polarizations   % Contact spin polarizations
        interface_parameters % Interface transmission coefficients
    end
    
    methods (Access = public)
        
        function obj = NonlocalSpinValve(varargin)
            % Constructor - Initialize nonlocal spin valve
            
            % Default parameters
            obj.temperature = 300;  % Room temperature
            obj.applied_current = 100e-6;  % 100 μA
            obj.magnetic_field = [0, 0, 0];  % No applied field
            
            % Analysis options
            obj.include_hanle = true;
            obj.include_thermal_noise = false;
            obj.include_contact_resistance = true;
            
            % Initialize arrays
            obj.contacts = [];
            obj.figures = [];
            
            % Initialize default geometry (2 μm channel)
            obj.setGeometry(2e-6, 100e-9, 10e-9);
            
            % Initialize default materials
            obj.setMaterials('NiFe', 'Cu');
            
            % Add default contacts
            obj.addContact('injector', [0.5e-6, 0, 0], 100e-9, 'NiFe');
            obj.addContact('detector', [1.5e-6, 0, 0], 100e-9, 'NiFe');
            
            % Initialize transport solver
            obj.spin_transport_solver = SpinTransportSolver();
            
            % Parse optional inputs
            if nargin > 0
                obj.parseInputs(varargin{:});
            end
        end
        
        function setGeometry(obj, length, width, thickness)
            % Set channel geometry
            %
            % Inputs:
            %   length - Channel length (m)
            %   width - Channel width (m)
            %   thickness - Channel thickness (m)
            
            validateattributes(length, {'numeric'}, {'positive', 'scalar'});
            validateattributes(width, {'numeric'}, {'positive', 'scalar'});
            validateattributes(thickness, {'numeric'}, {'positive', 'scalar'});
            
            obj.geometry = struct();
            obj.geometry.length = length;
            obj.geometry.width = width;
            obj.geometry.thickness = thickness;
            obj.geometry.area = width * thickness;
            obj.geometry.volume = length * width * thickness;
            
            % Configure transport solver geometry
            obj.configuretransportSolver();
        end
        
        function setMaterials(obj, contact_material, channel_material, varargin)
            % Set material properties
            %
            % Inputs:
            %   contact_material - Ferromagnetic contact material
            %   channel_material - Non-magnetic channel material
            %   varargin - Optional custom properties
            
            % Parse optional parameters
            p = inputParser;
            addParameter(p, 'custom_properties', struct(), @isstruct);
            parse(p, varargin{:});
            
            % Get material properties from database
            materials_db = MaterialsDB();
            
            % Contact material properties
            contact_props = materials_db.getMaterial(contact_material);
            if isfield(p.Results.custom_properties, 'contact')
                contact_props = obj.mergeStructures(contact_props, ...
                                                  p.Results.custom_properties.contact);
            end
            
            % Channel material properties
            channel_props = materials_db.getMaterial(channel_material);
            if isfield(p.Results.custom_properties, 'channel')
                channel_props = obj.mergeStructures(channel_props, ...
                                                  p.Results.custom_properties.channel);
            end
            
            % Store materials
            obj.materials = struct();
            obj.materials.contact = contact_props;
            obj.materials.channel = channel_props;
            
            % Update temperature-dependent properties
            obj.updateTemperatureDependentProperties();
        end
        
        function addContact(obj, name, position, width, material, varargin)
            % Add a contact to the device
            %
            % Inputs:
            %   name - Contact name/identifier
            %   position - Contact center position [x, y, z] (m)
            %   width - Contact width (m)
            %   material - Contact material (optional, uses default)
            %   varargin - Additional contact properties
            
            validateattributes(position, {'numeric'}, {'size', [1, 3]});
            validateattributes(width, {'numeric'}, {'positive', 'scalar'});
            
            % Parse optional parameters
            p = inputParser;
            addParameter(p, 'thickness', obj.geometry.thickness, @(x) isnumeric(x) && x > 0);
            addParameter(p, 'magnetization', [1, 0, 0], @(x) isnumeric(x) && length(x) == 3);
            addParameter(p, 'resistance', [], @(x) isempty(x) || (isnumeric(x) && x > 0));
            addParameter(p, 'polarization', [], @(x) isempty(x) || (isnumeric(x) && abs(x) <= 1));
            parse(p, varargin{:});
            
            % Create contact structure
            contact = struct();
            contact.name = name;
            contact.position = position;
            contact.width = width;
            contact.thickness = p.Results.thickness;
            contact.material = material;
            contact.magnetization = p.Results.magnetization / norm(p.Results.magnetization);
            
            % Calculate contact area
            contact.area = width * contact.thickness;
            
            % Set contact resistance
            if isempty(p.Results.resistance)
                contact.resistance = obj.calculateContactResistance(contact);
            else
                contact.resistance = p.Results.resistance;
            end
            
            % Set spin polarization
            if isempty(p.Results.polarization)
                contact.polarization = obj.materials.contact.spin_polarization;
            else
                contact.polarization = p.Results.polarization;
            end
            
            % Add to contacts array
            obj.contacts(end+1) = contact;
            
            fprintf('Added contact "%s" at position [%.2f, %.2f, %.2f] μm\n', ...
                    name, position*1e6);
        end
        
        function setTemperature(obj, T)
            % Set operating temperature
            %
            % Inputs:
            %   T - Temperature (K)
            
            validateattributes(T, {'numeric'}, {'positive', 'scalar'});
            obj.temperature = T;
            
            % Update temperature-dependent properties
            obj.updateTemperatureDependentProperties();
        end
        
        function setMagneticField(obj, B_field)
            % Set applied magnetic field
            %
            % Inputs:
            %   B_field - Magnetic field vector [Bx, By, Bz] (T)
            
            validateattributes(B_field, {'numeric'}, {'size', [1, 3]});
            obj.magnetic_field = B_field;
        end
        
        function results = measureHanle(obj, injection_current, field_range, varargin)
            % Perform Hanle precession measurement
            %
            % Inputs:
            %   injection_current - Current through injector (A)
            %   field_range - Perpendicular field values to sweep (T)
            %   varargin - Optional parameters
            %
            % Outputs:
            %   results - Structure containing Hanle measurement results
            
            % Parse options
            p = inputParser;
            addParameter(p, 'field_direction', [0, 0, 1], @(x) isnumeric(x) && length(x) == 3);
            addParameter(p, 'verbose', true, @islogical);
            parse(p, varargin{:});
            
            if p.Results.verbose
                fprintf('Starting Hanle measurement...\n');
                fprintf('Field range: %.3f to %.3f T (%d points)\n', ...
                        min(field_range), max(field_range), length(field_range));
            end
            
            % Store original current and field
            original_current = obj.applied_current;
            original_field = obj.magnetic_field;
            
            % Set injection current
            obj.applied_current = injection_current;
            
            % Normalize field direction
            field_dir = p.Results.field_direction / norm(p.Results.field_direction);
            
            % Initialize results arrays
            n_points = length(field_range);
            voltages = zeros(1, n_points);
            spin_accumulations = zeros(3, n_points);
            effective_polarizations = zeros(1, n_points);
            
            % Progress tracking
            if p.Results.verbose
                progress_points = round(linspace(1, n_points, 10));
                tic;
            end
            
            % Field sweep
            for i = 1:n_points
                % Set magnetic field
                obj.magnetic_field = field_range(i) * field_dir;
                
                % Solve transport equation
                voltage_data = obj.solveTransport();
                
                % Extract detector voltage (nonlocal signal)
                detector_idx = obj.findContactIndex('detector');
                voltages(i) = voltage_data.detector_voltage;
                
                % Extract spin accumulation at detector
                spin_accumulations(:, i) = voltage_data.spin_accumulation_detector;
                
                % Calculate effective polarization
                effective_polarizations(i) = obj.calculateEffectivePolarization(voltage_data);
                
                % Progress update
                if p.Results.verbose && ismember(i, progress_points)
                    progress = find(progress_points == i) * 10;
                    fprintf('Progress: %d%% (%d/%d points)\n', progress, i, n_points);
                end
            end
            
            % Fit Hanle curve to extract spin parameters
            hanle_fit = obj.fitHanleCurve(field_range, voltages);
            
            % Package results
            results = struct();
            results.measurement_type = 'hanle';
            results.injection_current = injection_current;
            results.field_range = field_range;
            results.field_direction = field_dir;
            results.voltages = voltages;
            results.spin_accumulations = spin_accumulations;
            results.effective_polarizations = effective_polarizations;
            results.hanle_fit = hanle_fit;
            
            % Store results
            obj.results = results;
            
            % Restore original settings
            obj.applied_current = original_current;
            obj.magnetic_field = original_field;
            
            if p.Results.verbose
                elapsed_time = toc;
                fprintf('Hanle measurement completed in %.2f seconds\n', elapsed_time);
                fprintf('Extracted spin diffusion length: %.1f nm\n', ...
                        hanle_fit.lambda_sf * 1e9);
                fprintf('Contact polarization: %.1f%%\n', ...
                        hanle_fit.polarization * 100);
            end
        end
        
        function results = measureTemperatureDependence(obj, injection_current, ...
                                                       temperature_range, varargin)
            % Measure temperature dependence of spin valve signal
            %
            % Inputs:
            %   injection_current - Current through injector (A)
            %   temperature_range - Temperature values to sweep (K)
            %   varargin - Optional parameters
            
            % Parse options
            p = inputParser;
            addParameter(p, 'magnetic_field', [0, 0, 0], @isnumeric);
            addParameter(p, 'verbose', true, @islogical);
            parse(p, varargin{:});
            
            if p.Results.verbose
                fprintf('Starting temperature dependence measurement...\n');
                fprintf('Temperature range: %.1f to %.1f K (%d points)\n', ...
                        min(temperature_range), max(temperature_range), ...
                        length(temperature_range));
            end
            
            % Store original settings
            original_current = obj.applied_current;
            original_temp = obj.temperature;
            original_field = obj.magnetic_field;
            
            % Set measurement conditions
            obj.applied_current = injection_current;
            obj.magnetic_field = p.Results.magnetic_field;
            
            % Initialize results arrays
            n_points = length(temperature_range);
            voltages_parallel = zeros(1, n_points);
            voltages_antiparallel = zeros(1, n_points);
            spin_diffusion_lengths = zeros(1, n_points);
            contact_polarizations = zeros(1, n_points);
            channel_resistivities = zeros(1, n_points);
            
            % Progress tracking
            if p.Results.verbose
                tic;
            end
            
            % Temperature sweep
            for i = 1:n_points
                % Set temperature
                obj.setTemperature(temperature_range(i));
                
                % Measure parallel configuration
                obj.setContactMagnetizations('parallel');
                voltage_data_p = obj.solveTransport();
                voltages_parallel(i) = voltage_data_p.detector_voltage;
                
                % Measure antiparallel configuration
                obj.setContactMagnetizations('antiparallel');
                voltage_data_ap = obj.solveTransport();
                voltages_antiparallel(i) = voltage_data_ap.detector_voltage;
                
                % Extract material parameters at this temperature
                spin_diffusion_lengths(i) = obj.materials.channel.lambda_sf;
                contact_polarizations(i) = obj.materials.contact.spin_polarization;
                channel_resistivities(i) = obj.materials.channel.rho;
                
                if p.Results.verbose && mod(i, max(1, round(n_points/10))) == 0
                    progress = i / n_points * 100;
                    fprintf('Progress: %.0f%% (T = %.1f K)\n', progress, temperature_range(i));
                end
            end
            
            % Calculate spin valve signal
            spin_valve_signal = voltages_parallel - voltages_antiparallel;
            
            % Fit temperature dependence models
            temp_fits = obj.fitTemperatureDependence(temperature_range, ...
                                                   spin_diffusion_lengths, ...
                                                   contact_polarizations, ...
                                                   spin_valve_signal);
            
            % Package results
            results = struct();
            results.measurement_type = 'temperature_dependence';
            results.injection_current = injection_current;
            results.temperature_range = temperature_range;
            results.voltages_parallel = voltages_parallel;
            results.voltages_antiparallel = voltages_antiparallel;
            results.spin_valve_signal = spin_valve_signal;
            results.spin_diffusion_lengths = spin_diffusion_lengths;
            results.contact_polarizations = contact_polarizations;
            results.channel_resistivities = channel_resistivities;
            results.temperature_fits = temp_fits;
            
            % Store results
            obj.results = results;
            
            % Restore original settings
            obj.applied_current = original_current;
            obj.setTemperature(original_temp);
            obj.magnetic_field = original_field;
            
            if p.Results.verbose
                elapsed_time = toc;
                fprintf('Temperature measurement completed in %.2f seconds\n', elapsed_time);
            end
        end
        
        function results = measureFieldSweep(obj, injection_current, field_range, ...
                                           field_direction, varargin)
            % Measure field dependence with arbitrary field direction
            %
            % Inputs:
            %   injection_current - Current through injector (A)
            %   field_range - Field magnitude values to sweep (T)
            %   field_direction - Field direction vector
            %   varargin - Optional parameters
            
            % Parse options
            p = inputParser;
            addParameter(p, 'measure_both_configs', true, @islogical);
            addParameter(p, 'verbose', true, @islogical);
            parse(p, varargin{:});
            
            if p.Results.verbose
                fprintf('Starting field sweep measurement...\n');
                fprintf('Field direction: [%.2f, %.2f, %.2f]\n', field_direction);
                fprintf('Field range: %.3f to %.3f T (%d points)\n', ...
                        min(field_range), max(field_range), length(field_range));
            end
            
            % Store original settings
            original_current = obj.applied_current;
            original_field = obj.magnetic_field;
            
            % Set injection current
            obj.applied_current = injection_current;
            
            % Normalize field direction
            field_dir = field_direction / norm(field_direction);
            
            % Initialize results arrays
            n_points = length(field_range);
            voltages = zeros(1, n_points);
            magnetoresistances = zeros(1, n_points);
            
            if p.Results.measure_both_configs
                voltages_parallel = zeros(1, n_points);
                voltages_antiparallel = zeros(1, n_points);
            end
            
            % Progress tracking
            if p.Results.verbose
                tic;
            end
            
            % Field sweep
            for i = 1:n_points
                % Set magnetic field
                obj.magnetic_field = field_range(i) * field_dir;
                
                if p.Results.measure_both_configs
                    % Measure both parallel and antiparallel configurations
                    obj.setContactMagnetizations('parallel');
                    voltage_data_p = obj.solveTransport();
                    voltages_parallel(i) = voltage_data_p.detector_voltage;
                    
                    obj.setContactMagnetizations('antiparallel');
                    voltage_data_ap = obj.solveTransport();
                    voltages_antiparallel(i) = voltage_data_ap.detector_voltage;
                    
                    voltages(i) = voltages_parallel(i) - voltages_antiparallel(i);
                else
                    % Single configuration measurement
                    voltage_data = obj.solveTransport();
                    voltages(i) = voltage_data.detector_voltage;
                end
                
                % Calculate magnetoresistance
                if i == 1
                    V0 = voltages(i);
                end
                magnetoresistances(i) = (voltages(i) - V0) / V0 * 100;  % Percentage
                
                if p.Results.verbose && mod(i, max(1, round(n_points/10))) == 0
                    progress = i / n_points * 100;
                    fprintf('Progress: %.0f%% (B = %.3f T)\n', progress, field_range(i));
                end
            end
            
            % Package results
            results = struct();
            results.measurement_type = 'field_sweep';
            results.injection_current = injection_current;
            results.field_range = field_range;
            results.field_direction = field_dir;
            results.voltages = voltages;
            results.magnetoresistances = magnetoresistances;
            
            if p.Results.measure_both_configs
                results.voltages_parallel = voltages_parallel;
                results.voltages_antiparallel = voltages_antiparallel;
            end
            
            % Store results
            obj.results = results;
            
            % Restore original settings
            obj.applied_current = original_current;
            obj.magnetic_field = original_field;
            
            if p.Results.verbose
                elapsed_time = toc;
                fprintf('Field sweep completed in %.2f seconds\n', elapsed_time);
                fprintf('Maximum magnetoresistance: %.2f%%\n', max(abs(magnetoresistances)));
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
                case 'hanle'
                    obj.plotHanleResults();
                    
                case 'temperature_dependence'
                    obj.plotTemperatureResults();
                    
                case 'field_sweep'
                    obj.plotFieldSweepResults();
                    
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
            addParameter(p, 'Current', 100e-6, @isnumeric);
            addParameter(p, 'IncludeHanle', true, @islogical);
            addParameter(p, 'IncludeThermalNoise', false, @islogical);
            parse(p, varargin{:});
            
            obj.temperature = p.Results.Temperature;
            obj.applied_current = p.Results.Current;
            obj.include_hanle = p.Results.IncludeHanle;
            obj.include_thermal_noise = p.Results.IncludeThermalNoise;
        end
        
        function configuretransportSolver(obj)
            % Configure transport solver with current geometry and materials
            
            if isempty(obj.geometry)
                return;
            end
            
            % Set geometry
            obj.spin_transport_solver.setGeometry(obj.geometry.length, ...
                                                 obj.geometry.width, ...
                                                 obj.geometry.thickness);
            
            % Set temperature
            obj.spin_transport_solver.setTemperature(obj.temperature);
            
            % Set magnetic field
            obj.spin_transport_solver.setMagneticField(obj.magnetic_field);
        end
        
        function updateTemperatureDependentProperties(obj)
            % Update material properties based on temperature
            
            if isempty(obj.materials)
                return;
            end
            
            T = obj.temperature;
            T0 = 300;  % Reference temperature (K)
            
            % Channel properties
            % Resistivity increases with temperature
            alpha_rho = 0.004;  % Temperature coefficient (1/K)
            obj.materials.channel.rho = obj.materials.channel.rho * (1 + alpha_rho * (T - T0));
            
            % Spin diffusion length decreases with temperature
            % λ_sf ∝ 1/√T for phonon scattering dominance
            obj.materials.channel.lambda_sf = obj.materials.channel.lambda_sf * sqrt(T0 / T);
            
            % Contact properties
            % Spin polarization decreases with temperature
            % P(T) = P(0) * (1 - (T/T_c)^α) for ferromagnets
            T_curie = 858;  % Curie temperature for NiFe (K)
            if T < T_curie
                thermal_factor = 1 - (T / T_curie)^1.5;
                obj.materials.contact.spin_polarization = ...
                    obj.materials.contact.spin_polarization * thermal_factor;
            else
                obj.materials.contact.spin_polarization = 0;  % Above Curie temperature
            end
        end
        
        function resistance = calculateContactResistance(obj, contact)
            % Calculate contact resistance based on material properties
            
            % Interface resistance for ferromagnet/normal metal contact
            rho_c = obj.materials.contact.rho;  % Contact resistivity
            A_c = contact.area;  % Contact area
            
            % Sharvin resistance component
            h = 6.626e-34;  % Planck constant
            e = 1.602e-19;  % Elementary charge
            k_F = obj.materials.channel.k_fermi;  % Fermi wavevector
            
            R_sharvin = h / (2 * e^2 * k_F * A_c);
            
            % Ohmic resistance component
            R_ohmic = rho_c / A_c;
            
            % Total contact resistance (parallel combination with interface effects)
            R_interface = 1e-12;  % Interface resistance (Ω⋅m²)
            R_total = R_ohmic + R_sharvin + R_interface / A_c;
            
            resistance = R_total;
        end
        
        function voltage_data = solveTransport(obj)
            % Solve transport equation and extract relevant quantities
            
            % Configure transport solver
            obj.configuretransportSolver();
            
            % Set up materials array for solver
            n_contacts = length(obj.contacts);
            materials_array = [];
            
            % Channel material
            materials_array(1) = obj.materials.channel;
            materials_array(1).type = 'N';  % Normal metal
            
            % Contact materials
            for i = 1:n_contacts
                mat = obj.materials.contact;
                mat.type = 'F';  % Ferromagnet
                materials_array(i+1) = mat;
            end
            
            obj.spin_transport_solver.setMaterials(materials_array);
            
            % Set magnetization states
            magnetizations = zeros(n_contacts, 3);
            for i = 1:n_contacts
                magnetizations(i, :) = obj.contacts(i).magnetization;
            end
            obj.spin_transport_solver.setMagnetization(magnetizations);
            
            % Set boundary conditions
            injector_idx = obj.findContactIndex('injector');
            detector_idx = obj.findContactIndex('detector');
            
            bc_values = [];
            bc_values(1).node = injector_idx;
            bc_values(1).current = obj.applied_current;
            bc_values(2).node = detector_idx;
            bc_values(2).current = 0;  % Voltage measurement
            
            obj.spin_transport_solver.setBoundaryConditions('current', bc_values);
            
            % Solve transport equation
            [V, I_s, info] = obj.spin_transport_solver.solve('verbose', false);
            
            % Extract results
            voltage_data = struct();
            
            % Detector voltage (nonlocal signal)
            n_nodes = length(V) / 4;
            voltage_data.detector_voltage = V(detector_idx);
            
            % Spin accumulation at detector
            voltage_data.spin_accumulation_detector = [
                V(n_nodes + detector_idx);      % μ_sx
                V(2*n_nodes + detector_idx);    % μ_sy
                V(3*n_nodes + detector_idx)     % μ_sz
            ];
            
            % Current distribution
            voltage_data.currents = I_s;
            
            % Transport solution info
            voltage_data.solve_info = info;
        end
        
        function idx = findContactIndex(obj, contact_name)
            % Find index of contact by name
            
            idx = [];
            for i = 1:length(obj.contacts)
                if strcmp(obj.contacts(i).name, contact_name)
                    idx = i;
                    return;
                end
            end
            
            if isempty(idx)
                error('Contact "%s" not found', contact_name);
            end
        end
        
        function setContactMagnetizations(obj, configuration)
            % Set contact magnetizations to specified configuration
            %
            % Inputs:
            %   configuration - 'parallel', 'antiparallel', or custom array
            
            if ischar(configuration)
                switch lower(configuration)
                    case 'parallel'
                        % All contacts aligned
                        for i = 1:length(obj.contacts)
                            obj.contacts(i).magnetization = [1, 0, 0];
                        end
                        
                    case 'antiparallel'
                        % Alternating alignment
                        for i = 1:length(obj.contacts)
                            if mod(i, 2) == 1
                                obj.contacts(i).magnetization = [1, 0, 0];
                            else
                                obj.contacts(i).magnetization = [-1, 0, 0];
                            end
                        end
                        
                    otherwise
                        error('Unknown configuration: %s', configuration);
                end
            else
                % Custom magnetization array
                if size(configuration, 1) ~= length(obj.contacts)
                    error('Magnetization array size mismatch');
                end
                
                for i = 1:length(obj.contacts)
                    obj.contacts(i).magnetization = configuration(i, :);
                end
            end
        end
        
        function hanle_fit = fitHanleCurve(obj, field_range, voltages)
            % Fit Hanle curve to extract spin transport parameters
            
            % Hanle curve model: V(B) = V0 * [1 + (ω_L * τ_sf)^2]^(-1)
            % where ω_L = g*μ_B*B/ℏ is the Larmor frequency
            
            % Constants
            g = 2.0;  % g-factor
            mu_B = 9.274e-24;  % Bohr magneton (J/T)
            hbar = 1.055e-34;  % Reduced Planck constant
            
            % Normalize voltages
            V_max = max(voltages);
            V_norm = voltages / V_max;
            
            % Define Hanle function
            hanle_func = @(params, B) params(1) ./ (1 + (g * mu_B * B / hbar * params(2)).^2);
            
            % Initial parameter guess
            tau_sf_guess = 100e-12;  % 100 ps
            p0 = [1, tau_sf_guess];  % [amplitude, spin lifetime]
            
            % Fit using nonlinear least squares
            options = optimoptions('lsqcurvefit', 'Display', 'off');
            
            try
                [params_fit, resnorm, residuals, ~, ~, ~, jacobian] = ...
                    lsqcurvefit(hanle_func, p0, field_range, V_norm, [], [], options);
                
                % Calculate parameter uncertainties
                ci = nlparci(params_fit, residuals, 'jacobian', jacobian);
                param_errors = diff(ci, 1, 2) / 2;
                
                fit_success = true;
                
            catch ME
                warning('Hanle curve fitting failed: %s', ME.message);
                params_fit = p0;
                param_errors = [0, 0];
                fit_success = false;
            end
            
            % Extract physical parameters
            amplitude = params_fit(1) * V_max;
            tau_sf = params_fit(2);
            
            % Calculate spin diffusion length
            % For 1D diffusion: λ_sf = √(D * τ_sf)
            D = obj.materials.channel.diffusion_constant;
            lambda_sf = sqrt(D * tau_sf);
            
            % Calculate contact polarization from amplitude
            % Amplitude is proportional to P_inj * P_det
            distance = abs(obj.contacts(2).position(1) - obj.contacts(1).position(1));
            expected_amplitude = obj.calculateExpectedAmplitude(distance, lambda_sf);
            effective_polarization = sqrt(amplitude / expected_amplitude);
            
            % Package fit results
            hanle_fit = struct();
            hanle_fit.success = fit_success;
            hanle_fit.amplitude = amplitude;
            hanle_fit.tau_sf = tau_sf;
            hanle_fit.lambda_sf = lambda_sf;
            hanle_fit.polarization = effective_polarization;
            hanle_fit.fit_params = params_fit;
            hanle_fit.param_errors = param_errors;
            hanle_fit.fitted_curve = hanle_func(params_fit, field_range) * V_max;
            hanle_fit.r_squared = 1 - sum(residuals.^2) / sum((V_norm - mean(V_norm)).^2);
        end
        
        function expected_amplitude = calculateExpectedAmplitude(obj, distance, lambda_sf)
            % Calculate expected Hanle amplitude based on device geometry
            
            % 1D spin diffusion model
            % V_nl = (R_c * I * P_inj * P_det * λ_sf / σ_ch) * exp(-L/λ_sf)
            
            R_c = obj.contacts(1).resistance;  % Contact resistance
            I = obj.applied_current;
            P = obj.materials.contact.spin_polarization;
            sigma_ch = 1 / obj.materials.channel.rho;  % Channel conductivity
            
            expected_amplitude = (R_c * I * P^2 * lambda_sf / sigma_ch) * exp(-distance / lambda_sf);
        end
        
        function temp_fits = fitTemperatureDependence(obj, temperature_range, ...
                                                    lambda_sf_data, polarization_data, ...
                                                    signal_data)
            % Fit temperature dependence of material parameters
            
            temp_fits = struct();
            
            % Fit spin diffusion length: λ_sf ∝ T^(-α)
            try
                lambda_func = @(params, T) params(1) * T.^(-params(2));
                p0_lambda = [lambda_sf_data(1) * temperature_range(1)^0.5, 0.5];
                
                [lambda_params, ~] = lsqcurvefit(lambda_func, p0_lambda, ...
                                                temperature_range, lambda_sf_data, ...
                                                [], [], optimoptions('lsqcurvefit', 'Display', 'off'));
                
                temp_fits.lambda_sf.params = lambda_params;
                temp_fits.lambda_sf.fitted_curve = lambda_func(lambda_params, temperature_range);
                temp_fits.lambda_sf.success = true;
                
            catch
                temp_fits.lambda_sf.success = false;
            end
            
            % Fit polarization: P ∝ (1 - (T/T_c)^α)
            try
                T_curie = 858;  % NiFe Curie temperature
                pol_func = @(params, T) params(1) * (1 - (T / T_curie).^params(2));
                p0_pol = [polarization_data(1), 1.5];
                
                [pol_params, ~] = lsqcurvefit(pol_func, p0_pol, ...
                                            temperature_range, polarization_data, ...
                                            [], [], optimoptions('lsqcurvefit', 'Display', 'off'));
                
                temp_fits.polarization.params = pol_params;
                temp_fits.polarization.fitted_curve = pol_func(pol_params, temperature_range);
                temp_fits.polarization.T_curie = T_curie;
                temp_fits.polarization.success = true;
                
            catch
                temp_fits.polarization.success = false;
            end
            
            % Fit spin valve signal (combination of above effects)
            try
                signal_func = @(params, T) params(1) * (1 - (T / T_curie).^params(3)) * ...
                                          T.^(-params(2)) * exp(-params(4) * T);
                p0_signal = [signal_data(1), 0.5, 1.5, 0.001];
                
                [signal_params, ~] = lsqcurvefit(signal_func, p0_signal, ...
                                                temperature_range, signal_data, ...
                                                [], [], optimoptions('lsqcurvefit', 'Display', 'off'));
                
                temp_fits.signal.params = signal_params;
                temp_fits.signal.fitted_curve = signal_func(signal_params, temperature_range);
                temp_fits.signal.success = true;
                
            catch
                temp_fits.signal.success = false;
            end
        end
        
        function effective_polarization = calculateEffectivePolarization(obj, voltage_data)
            % Calculate effective polarization from transport solution
            
            % Extract spin and charge currents
            I_charge = mean(voltage_data.currents.charge);
            I_spin = sqrt(mean(voltage_data.currents.spin_x)^2 + ...
                         mean(voltage_data.currents.spin_y)^2 + ...
                         mean(voltage_data.currents.spin_z)^2);
            
            % Effective polarization P_eff = I_spin / I_charge
            if I_charge > 0
                effective_polarization = I_spin / I_charge;
            else
                effective_polarization = 0;
            end
            
            % Clamp to physical range
            effective_polarization = min(max(effective_polarization, 0), 1);
        end
        
        function plotHanleResults(obj)
            % Plot Hanle measurement results
            
            results = obj.results;
            colors = getBerkeleyColors();
            
            % Main Hanle curve
            fig1 = figure('Position', [100, 100, 1200, 800]);
            
            % Hanle voltage vs field
            subplot(2, 2, 1);
            plot(results.field_range * 1000, results.voltages * 1e6, 'o', ...
                 'MarkerSize', 6, 'Color', colors(1,:), 'MarkerFaceColor', colors(1,:));
            hold on;
            if results.hanle_fit.success
                plot(results.field_range * 1000, results.hanle_fit.fitted_curve * 1e6, ...
                     '-', 'LineWidth', 2, 'Color', colors(2,:));
                legend('Data', 'Fit', 'Location', 'best');
            end
            xlabel('Perpendicular Field (mT)');
            ylabel('Nonlocal Voltage (μV)');
            title('Hanle Effect');
            grid on;
            
            % Spin accumulation components
            subplot(2, 2, 2);
            plot(results.field_range * 1000, results.spin_accumulations(1,:) * 1e6, ...
                 'LineWidth', 2, 'Color', colors(1,:));
            hold on;
            plot(results.field_range * 1000, results.spin_accumulations(2,:) * 1e6, ...
                 'LineWidth', 2, 'Color', colors(2,:));
            plot(results.field_range * 1000, results.spin_accumulations(3,:) * 1e6, ...
                 'LineWidth', 2, 'Color', colors(3,:));
            xlabel('Perpendicular Field (mT)');
            ylabel('Spin Accumulation (μV)');
            title('Spin Accumulation Components');
            legend('μ_{sx}', 'μ_{sy}', 'μ_{sz}', 'Location', 'best');
            grid on;
            
            % Effective polarization vs field
            subplot(2, 2, 3);
            plot(results.field_range * 1000, results.effective_polarizations * 100, ...
                 'LineWidth', 2, 'Color', colors(4,:));
            xlabel('Perpendicular Field (mT)');
            ylabel('Effective Polarization (%)');
            title('Field-Dependent Polarization');
            grid on;
            
            % Fit parameters summary
            subplot(2, 2, 4);
            if results.hanle_fit.success
                param_names = {'Amplitude (μV)', 'τ_{sf} (ps)', 'λ_{sf} (nm)', 'P_{eff} (%)'};
                param_values = [results.hanle_fit.amplitude * 1e6, ...
                               results.hanle_fit.tau_sf * 1e12, ...
                               results.hanle_fit.lambda_sf * 1e9, ...
                               results.hanle_fit.polarization * 100];
                
                bar(param_values, 'FaceColor', colors(5,:));
                set(gca, 'XTickLabel', param_names);
                title('Extracted Parameters');
                ylabel('Parameter Value');
                grid on;
            else
                text(0.5, 0.5, 'Fit Failed', 'HorizontalAlignment', 'center', ...
                     'FontSize', 16, 'Color', 'red');
                axis off;
            end
            
            obj.figures(end+1) = fig1;
        end
        
        function plotTemperatureResults(obj)
            % Plot temperature dependence results
            
            results = obj.results;
            colors = getBerkeleyColors();
            
            fig = figure('Position', [100, 100, 1400, 1000]);
            
            % Spin valve signal vs temperature
            subplot(2, 3, 1);
            plot(results.temperature_range, results.spin_valve_signal * 1e6, ...
                 'o-', 'LineWidth', 2, 'MarkerSize', 6, 'Color', colors(1,:));
            if results.temperature_fits.signal.success
                hold on;
                plot(results.temperature_range, results.temperature_fits.signal.fitted_curve * 1e6, ...
                     '--', 'LineWidth', 2, 'Color', colors(2,:));
                legend('Data', 'Fit', 'Location', 'best');
            end
            xlabel('Temperature (K)');
            ylabel('Spin Valve Signal (μV)');
            title('Temperature Dependence');
            grid on;
            
            % Parallel and antiparallel voltages
            subplot(2, 3, 2);
            plot(results.temperature_range, results.voltages_parallel * 1e6, ...
                 'o-', 'LineWidth', 2, 'Color', colors(1,:));
            hold on;
            plot(results.temperature_range, results.voltages_antiparallel * 1e6, ...
                 'o-', 'LineWidth', 2, 'Color', colors(2,:));
            xlabel('Temperature (K)');
            ylabel('Voltage (μV)');
            title('Configuration Voltages');
            legend('Parallel', 'Antiparallel', 'Location', 'best');
            grid on;
            
            % Spin diffusion length vs temperature
            subplot(2, 3, 3);
            semilogy(results.temperature_range, results.spin_diffusion_lengths * 1e9, ...
                    'o-', 'LineWidth', 2, 'Color', colors(3,:));
            if results.temperature_fits.lambda_sf.success
                hold on;
                semilogy(results.temperature_range, results.temperature_fits.lambda_sf.fitted_curve * 1e9, ...
                        '--', 'LineWidth', 2, 'Color', colors(4,:));
                legend('Data', 'T^{-α} fit', 'Location', 'best');
            end
            xlabel('Temperature (K)');
            ylabel('λ_{sf} (nm)');
            title('Spin Diffusion Length');
            grid on;
            
            % Contact polarization vs temperature
            subplot(2, 3, 4);
            plot(results.temperature_range, results.contact_polarizations * 100, ...
                 'o-', 'LineWidth', 2, 'Color', colors(5,:));
            if results.temperature_fits.polarization.success
                hold on;
                plot(results.temperature_range, results.temperature_fits.polarization.fitted_curve * 100, ...
                     '--', 'LineWidth', 2, 'Color', colors(6,:));
                legend('Data', 'Curie law fit', 'Location', 'best');
            end
            xlabel('Temperature (K)');
            ylabel('Polarization (%)');
            title('Contact Polarization');
            grid on;
            
            % Channel resistivity vs temperature
            subplot(2, 3, 5);
            plot(results.temperature_range, results.channel_resistivities * 1e8, ...
                 'o-', 'LineWidth', 2, 'Color', colors(7,:));
            xlabel('Temperature (K)');
            ylabel('Resistivity (μΩ⋅cm)');
            title('Channel Resistivity');
            grid on;
            
            % Summary of temperature coefficients
            subplot(2, 3, 6);
            if results.temperature_fits.lambda_sf.success
                coeff_names = {'λ_{sf} exponent', 'P_{eff} exponent'};
                coeff_values = [results.temperature_fits.lambda_sf.params(2), ...
                               results.temperature_fits.polarization.params(2)];
                bar(coeff_values, 'FaceColor', colors(8,:));
                set(gca, 'XTickLabel', coeff_names);
                title('Temperature Exponents');
                ylabel('Exponent Value');
                grid on;
            else
                text(0.5, 0.5, 'Fits Failed', 'HorizontalAlignment', 'center', ...
                     'FontSize', 16, 'Color', 'red');
                axis off;
            end
            
            obj.figures(end+1) = fig;
        end
        
        function plotFieldSweepResults(obj)
            % Plot field sweep results
            
            results = obj.results;
            colors = getBerkeleyColors();
            
            fig = figure('Position', [100, 100, 1200, 800]);
            
            % Voltage vs field
            subplot(2, 2, 1);
            plot(results.field_range * 1000, results.voltages * 1e6, ...
                 'o-', 'LineWidth', 2, 'MarkerSize', 6, 'Color', colors(1,:));
            xlabel('Applied Field (mT)');
            ylabel('Nonlocal Voltage (μV)');
            title(sprintf('Field Sweep [%.2f, %.2f, %.2f]', results.field_direction));
            grid on;
            
            % Parallel and antiparallel (if measured)
            if isfield(results, 'voltages_parallel')
                subplot(2, 2, 2);
                plot(results.field_range * 1000, results.voltages_parallel * 1e6, ...
                     'o-', 'LineWidth', 2, 'Color', colors(1,:));
                hold on;
                plot(results.field_range * 1000, results.voltages_antiparallel * 1e6, ...
                     'o-', 'LineWidth', 2, 'Color', colors(2,:));
                xlabel('Applied Field (mT)');
                ylabel('Voltage (μV)');
                title('Configuration Voltages');
                legend('Parallel', 'Antiparallel', 'Location', 'best');
                grid on;
            end
            
            % Magnetoresistance
            subplot(2, 2, 3);
            plot(results.field_range * 1000, results.magnetoresistances, ...
                 'o-', 'LineWidth', 2, 'Color', colors(3,:));
            xlabel('Applied Field (mT)');
            ylabel('Magnetoresistance (%)');
            title('Magnetoresistance');
            grid on;
            
            % Field dependence analysis
            subplot(2, 2, 4);
            field_magnitude = abs(results.field_range * 1000);
            voltage_magnitude = abs(results.voltages * 1e6);
            
            loglog(field_magnitude(field_magnitude > 0), ...
                   voltage_magnitude(field_magnitude > 0), ...
                   'o-', 'LineWidth', 2, 'Color', colors(4,:));
            xlabel('Field Magnitude (mT)');
            ylabel('|Voltage| (μV)');
            title('Log-Log Field Dependence');
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
            measurement_type = obj.results.measurement_type;
            
            for i = 1:length(obj.figures)
                filename = sprintf('NLSV_%s_Fig%d_%s', measurement_type, i, timestamp);
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