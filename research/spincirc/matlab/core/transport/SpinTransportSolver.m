classdef SpinTransportSolver < handle
% SPINTRANSPORTSOLVER - Master class for spin transport calculations
%
% This class implements the 4x4 conductance matrix formalism for spin
% transport in F/N/F heterostructures as described in:
% "Circuit Models for Spintronic Devices Subject to Electric and Magnetic Fields"
% (Alawein & Fariborzi, IEEE J-XCDC 2018)
%
% Key Features:
%   - 4x4 conductance matrix generation for arbitrary magnetization directions
%   - Spatial discretization with adaptive meshing
%   - Interface boundary conditions (Sharvin, tunneling)
%   - Temperature-dependent transport coefficients
%   - Hanle precession effects under arbitrary B-fields
%   - Frequency-domain AC analysis
%
% Usage:
%   solver = SpinTransportSolver();
%   solver.setGeometry(length, width, thickness);
%   solver.setMaterials(materials);
%   solver.setMagnetization(m_vec);
%   [V, I_s] = solver.solve();
%
% Author: Meshal Alawein <meshal@berkeley.edu>
% Copyright © 2025 Meshal Alawein — All rights reserved.
% License: MIT

    properties (Access = public)
        geometry           % Geometry parameters
        materials         % Material properties
        magnetization     % Magnetization vectors
        boundary_conditions % Boundary conditions
        mesh              % Spatial discretization
        temperature       % Operating temperature (K)
        magnetic_field    % External magnetic field (T)
        frequency         % AC frequency (Hz)
        solver_options    % Numerical solver options
    end
    
    properties (Access = private)
        G_matrix          % 4x4 conductance matrix
        is_solved         % Solution status flag
        solution          % Cached solution
    end
    
    methods (Access = public)
        
        function obj = SpinTransportSolver(varargin)
            % Constructor with optional parameter initialization
            
            % Set default values
            obj.temperature = 300;  % Room temperature
            obj.magnetic_field = [0, 0, 0];  % No external field
            obj.frequency = 0;  % DC analysis
            obj.is_solved = false;
            
            % Initialize solver options
            obj.solver_options = struct(...
                'tolerance', 1e-9, ...
                'max_iterations', 1000, ...
                'adaptive_mesh', true, ...
                'mesh_refinement', 2, ...
                'parallel_computing', false);
            
            % Parse input arguments
            if nargin > 0
                obj.parseInputs(varargin{:});
            end
        end
        
        function setGeometry(obj, length, width, thickness)
            % Set device geometry parameters
            
            validateattributes(length, {'numeric'}, {'positive', 'scalar'});
            validateattributes(width, {'numeric'}, {'positive', 'scalar'});
            validateattributes(thickness, {'numeric'}, {'positive', 'scalar'});
            
            obj.geometry = struct(...
                'length', length, ...
                'width', width, ...
                'thickness', thickness, ...
                'area', width * thickness, ...
                'volume', length * width * thickness);
            
            obj.is_solved = false;
        end
        
        function setMaterials(obj, materials)
            % Set material properties for each region
            
            % Validate material structure
            required_fields = {'name', 'lambda_sf', 'rho', 'type'};
            for i = 1:length(materials)
                for j = 1:length(required_fields)
                    if ~isfield(materials(i), required_fields{j})
                        error('Material %d missing field: %s', i, required_fields{j});
                    end
                end
            end
            
            obj.materials = materials;
            obj.is_solved = false;
        end
        
        function setMagnetization(obj, m_vectors)
            % Set magnetization directions for ferromagnetic regions
            
            % Normalize magnetization vectors
            for i = 1:size(m_vectors, 1)
                m_norm = norm(m_vectors(i, :));
                if m_norm > 0
                    m_vectors(i, :) = m_vectors(i, :) / m_norm;
                end
            end
            
            obj.magnetization = m_vectors;
            obj.is_solved = false;
        end
        
        function setBoundaryConditions(obj, bc_type, bc_values)
            % Set boundary conditions for transport problem
            
            valid_types = {'voltage', 'current', 'open', 'ground'};
            if ~ismember(bc_type, valid_types)
                error('Invalid boundary condition type: %s', bc_type);
            end
            
            obj.boundary_conditions = struct(...
                'type', bc_type, ...
                'values', bc_values);
            
            obj.is_solved = false;
        end
        
        function setTemperature(obj, T)
            % Set operating temperature
            
            validateattributes(T, {'numeric'}, {'positive', 'scalar'});
            obj.temperature = T;
            obj.is_solved = false;
        end
        
        function setMagneticField(obj, B_field)
            % Set external magnetic field vector
            
            validateattributes(B_field, {'numeric'}, {'size', [1, 3]});
            obj.magnetic_field = B_field;
            obj.is_solved = false;
        end
        
        function setFrequency(obj, freq)
            % Set AC frequency for frequency-domain analysis
            
            validateattributes(freq, {'numeric'}, {'nonnegative', 'scalar'});
            obj.frequency = freq;
            obj.is_solved = false;
        end
        
        function generateMesh(obj, varargin)
            % Generate spatial discretization mesh
            
            % Parse options
            p = inputParser;
            addParameter(p, 'nx', 100, @(x) isscalar(x) && x > 0);
            addParameter(p, 'ny', 50, @(x) isscalar(x) && x > 0);
            addParameter(p, 'nz', 20, @(x) isscalar(x) && x > 0);
            addParameter(p, 'adaptive', true, @islogical);
            parse(p, varargin{:});
            
            if isempty(obj.geometry)
                error('Geometry must be set before generating mesh');
            end
            
            % Create uniform mesh
            x = linspace(0, obj.geometry.length, p.Results.nx);
            y = linspace(0, obj.geometry.width, p.Results.ny);
            z = linspace(0, obj.geometry.thickness, p.Results.nz);
            
            [X, Y, Z] = meshgrid(x, y, z);
            
            obj.mesh = struct(...
                'x', x, 'y', y, 'z', z, ...
                'X', X, 'Y', Y, 'Z', Z, ...
                'nx', p.Results.nx, ...
                'ny', p.Results.ny, ...
                'nz', p.Results.nz, ...
                'dx', x(2) - x(1), ...
                'dy', y(2) - y(1), ...
                'dz', z(2) - z(1));
            
            obj.is_solved = false;
        end
        
        function G = buildConductanceMatrix(obj)
            % Build 4x4 conductance matrix for the system
            
            if isempty(obj.materials) || isempty(obj.magnetization)
                error('Materials and magnetization must be set');
            end
            
            % Get conductance matrix generator
            cm_gen = ConductanceMatrix();
            
            % Calculate interface and bulk conductances
            G = cm_gen.buildSystemMatrix(obj.materials, obj.magnetization, ...
                                       obj.geometry, obj.temperature, ...
                                       obj.magnetic_field);
            
            obj.G_matrix = G;
        end
        
        function [V, I_s, solution_info] = solve(obj, varargin)
            % Solve the spin transport problem
            
            % Parse options
            p = inputParser;
            addParameter(p, 'method', 'direct', @ischar);
            addParameter(p, 'preconditioner', 'none', @ischar);
            addParameter(p, 'verbose', false, @islogical);
            parse(p, varargin{:});
            
            % Validate inputs
            if isempty(obj.G_matrix)
                obj.buildConductanceMatrix();
            end
            
            if isempty(obj.boundary_conditions)
                error('Boundary conditions must be set');
            end
            
            % Set up linear system G*V = I
            G = obj.G_matrix;
            [I_applied, voltage_nodes] = obj.setupBoundaryConditions();
            
            tic;
            
            % Solve based on method
            switch lower(p.Results.method)
                case 'direct'
                    V = G \ I_applied;
                    
                case 'iterative'
                    [V, flag, relres, iter] = gmres(G, I_applied, [], ...
                        obj.solver_options.tolerance, ...
                        obj.solver_options.max_iterations);
                    
                    if flag ~= 0
                        warning('Iterative solver did not converge. Flag: %d', flag);
                    end
                    
                otherwise
                    error('Unknown solver method: %s', p.Results.method);
            end
            
            solve_time = toc;
            
            % Calculate spin currents
            I_s = obj.calculateSpinCurrents(V);
            
            % Prepare solution info
            solution_info = struct(...
                'solve_time', solve_time, ...
                'method', p.Results.method, ...
                'residual', norm(G*V - I_applied), ...
                'condition_number', cond(G));
            
            if exist('iter', 'var')
                solution_info.iterations = iter;
                solution_info.relative_residual = relres;
            end
            
            % Cache solution
            obj.solution = struct('V', V, 'I_s', I_s, 'info', solution_info);
            obj.is_solved = true;
            
            if p.Results.verbose
                fprintf('Spin transport solved in %.3f seconds\n', solve_time);
                fprintf('Residual: %.2e\n', solution_info.residual);
                fprintf('Condition number: %.2e\n', solution_info.condition_number);
            end
        end
        
        function I_s = calculateSpinCurrents(obj, V)
            % Calculate spin currents from voltage solution
            
            % Extract charge and spin voltages
            n_nodes = length(V) / 4;
            V_c = V(1:n_nodes);              % Charge voltage
            V_sx = V(n_nodes+1:2*n_nodes);   % Spin voltage x
            V_sy = V(2*n_nodes+1:3*n_nodes); % Spin voltage y
            V_sz = V(3*n_nodes+1:4*n_nodes); % Spin voltage z
            
            % Calculate currents using Ohm's law: I = G * V
            I_total = obj.G_matrix * V;
            
            % Extract current components
            I_c = I_total(1:n_nodes);              % Charge current
            I_sx = I_total(n_nodes+1:2*n_nodes);   % Spin current x
            I_sy = I_total(2*n_nodes+1:3*n_nodes); % Spin current y
            I_sz = I_total(3*n_nodes+1:4*n_nodes); % Spin current z
            
            % Package results
            I_s = struct(...
                'charge', I_c, ...
                'spin_x', I_sx, ...
                'spin_y', I_sy, ...
                'spin_z', I_sz, ...
                'voltage_charge', V_c, ...
                'voltage_spin_x', V_sx, ...
                'voltage_spin_y', V_sy, ...
                'voltage_spin_z', V_sz);
        end
        
        function [TMR, resistance] = calculateTMR(obj)
            % Calculate tunnel magnetoresistance
            
            if ~obj.is_solved
                error('System must be solved first');
            end
            
            % Get parallel and antiparallel configurations
            m_parallel = obj.magnetization;
            m_antiparallel = obj.magnetization;
            m_antiparallel(end, :) = -m_antiparallel(end, :);  % Flip last magnet
            
            % Calculate resistances
            obj.setMagnetization(m_parallel);
            [~, ~, ~] = obj.solve('verbose', false);
            R_P = obj.calculateResistance();
            
            obj.setMagnetization(m_antiparallel);
            [~, ~, ~] = obj.solve('verbose', false);
            R_AP = obj.calculateResistance();
            
            % Restore original magnetization
            obj.setMagnetization(m_parallel);
            
            % Calculate TMR
            TMR = (R_AP - R_P) / R_P * 100;  % Percentage
            resistance = struct('parallel', R_P, 'antiparallel', R_AP);
        end
        
        function R = calculateResistance(obj)
            % Calculate total device resistance
            
            if ~obj.is_solved
                error('System must be solved first');
            end
            
            V = obj.solution.V;
            I_s = obj.solution.I_s;
            
            % Calculate resistance from terminal voltages and currents
            V_terminal = V(1) - V(end/4);  % Voltage across device
            I_terminal = sum(I_s.charge);  % Total current
            
            R = V_terminal / I_terminal;
        end
        
        function plotSolution(obj, varargin)
            % Plot the transport solution
            
            if ~obj.is_solved
                error('System must be solved first');
            end
            
            % Parse options
            p = inputParser;
            addParameter(p, 'component', 'charge', @ischar);
            addParameter(p, 'slice', 'xy', @ischar);
            addParameter(p, 'position', 0.5, @isnumeric);
            parse(p, varargin{:});
            
            % Apply Berkeley styling
            berkeley();
            
            I_s = obj.solution.I_s;
            
            % Select component to plot
            switch lower(p.Results.component)
                case 'charge'
                    data = I_s.voltage_charge;
                    title_str = 'Charge Voltage (V)';
                case 'spin_x'
                    data = I_s.voltage_spin_x;
                    title_str = 'Spin Voltage X (V)';
                case 'spin_y'
                    data = I_s.voltage_spin_y;
                    title_str = 'Spin Voltage Y (V)';
                case 'spin_z'
                    data = I_s.voltage_spin_z;
                    title_str = 'Spin Voltage Z (V)';
                otherwise
                    error('Unknown component: %s', p.Results.component);
            end
            
            % Create plot
            figure;
            if isempty(obj.mesh)
                % Simple 1D plot
                plot(1:length(data), data, 'LineWidth', 2);
                xlabel('Node Index');
                ylabel(title_str);
            else
                % 2D/3D visualization
                obj.plotSpatialData(data, p.Results.slice, p.Results.position);
                title(title_str);
            end
            
            grid on;
            title(['Spin Transport Solution: ' title_str]);
        end
        
    end
    
    methods (Access = private)
        
        function parseInputs(obj, varargin)
            % Parse constructor input arguments
            
            p = inputParser;
            addParameter(p, 'Temperature', 300, @isnumeric);
            addParameter(p, 'MagneticField', [0, 0, 0], @isnumeric);
            addParameter(p, 'Frequency', 0, @isnumeric);
            parse(p, varargin{:});
            
            obj.temperature = p.Results.Temperature;
            obj.magnetic_field = p.Results.MagneticField;
            obj.frequency = p.Results.Frequency;
        end
        
        function [I_applied, voltage_nodes] = setupBoundaryConditions(obj)
            % Set up boundary conditions for linear system
            
            n_nodes = size(obj.G_matrix, 1) / 4;
            I_applied = zeros(4 * n_nodes, 1);
            voltage_nodes = [];
            
            bc = obj.boundary_conditions;
            
            switch bc.type
                case 'voltage'
                    % Apply voltage boundary conditions
                    for i = 1:length(bc.values)
                        node_idx = bc.values(i).node;
                        voltage = bc.values(i).voltage;
                        
                        % Modify matrix for voltage constraint
                        obj.G_matrix(node_idx, :) = 0;
                        obj.G_matrix(node_idx, node_idx) = 1;
                        I_applied(node_idx) = voltage;
                        
                        voltage_nodes = [voltage_nodes, node_idx];
                    end
                    
                case 'current'
                    % Apply current boundary conditions
                    for i = 1:length(bc.values)
                        node_idx = bc.values(i).node;
                        current = bc.values(i).current;
                        I_applied(node_idx) = current;
                    end
                    
                case 'ground'
                    % Ground specific nodes
                    for i = 1:length(bc.values)
                        node_idx = bc.values(i);
                        obj.G_matrix(node_idx, :) = 0;
                        obj.G_matrix(node_idx, node_idx) = 1;
                        I_applied(node_idx) = 0;
                    end
                    
                otherwise
                    error('Unsupported boundary condition type: %s', bc.type);
            end
        end
        
        function plotSpatialData(obj, data, slice_type, position)
            % Plot spatial data on mesh
            
            mesh = obj.mesh;
            
            % Reshape data to mesh dimensions
            data_3d = reshape(data, [mesh.ny, mesh.nx, mesh.nz]);
            
            switch lower(slice_type)
                case 'xy'
                    % XY slice at given Z position
                    z_idx = round(position * mesh.nz);
                    z_idx = max(1, min(z_idx, mesh.nz));
                    
                    slice_data = squeeze(data_3d(:, :, z_idx));
                    imagesc(mesh.x, mesh.y, slice_data);
                    xlabel('X Position (m)');
                    ylabel('Y Position (m)');
                    
                case 'xz'
                    % XZ slice at given Y position
                    y_idx = round(position * mesh.ny);
                    y_idx = max(1, min(y_idx, mesh.ny));
                    
                    slice_data = squeeze(data_3d(y_idx, :, :))';
                    imagesc(mesh.x, mesh.z, slice_data);
                    xlabel('X Position (m)');
                    ylabel('Z Position (m)');
                    
                case 'yz'
                    % YZ slice at given X position
                    x_idx = round(position * mesh.nx);
                    x_idx = max(1, min(x_idx, mesh.nx));
                    
                    slice_data = squeeze(data_3d(:, x_idx, :))';
                    imagesc(mesh.y, mesh.z, slice_data);
                    xlabel('Y Position (m)');
                    ylabel('Z Position (m)');
                    
                otherwise
                    error('Unknown slice type: %s', slice_type);
            end
            
            colorbar;
            axis equal tight;
        end
        
    end
end