classdef MaterialsDB < handle
% MATERIALSDB - Comprehensive material parameter database for spintronics
%
% This class provides a centralized database of material properties for
% spin transport and magnetodynamics calculations. Includes temperature
% and strain dependence models.
%
% Key Features:
%   - Extensive database of ferromagnetic and nonmagnetic materials
%   - Temperature-dependent properties
%   - Strain-dependent magnetic anisotropy
%   - Interface parameters for heterostructures
%   - Validation against experimental data
%
% Usage:
%   params = MaterialsDB.CoFeB;
%   params_T = MaterialsDB.getTemperatureDependence('CoFeB', 350);
%   interface_params = MaterialsDB.getInterfaceParameters('CoFeB', 'MgO');
%
% Author: Meshal Alawein <meshal@berkeley.edu>
% Copyright © 2025 Meshal Alawein — All rights reserved.
% License: MIT

    properties (Constant)
        % Physical constants
        mu_B = 9.2740100783e-24;    % Bohr magneton (J/T)
        k_B = 1.380649e-23;         % Boltzmann constant (J/K)
        hbar = 1.054571817e-34;     % Reduced Planck constant (J⋅s)
        e = 1.602176634e-19;        % Elementary charge (C)
        mu_0 = 4*pi*1e-7;           % Permeability of free space (H/m)
        
        % =====================================================================
        % FERROMAGNETIC MATERIALS
        % =====================================================================
        
        % CoFeB (Cobalt Iron Boron) - Popular for MTJs
        CoFeB = struct(...
            'name', 'CoFeB', ...
            'type', 'ferromagnet', ...
            'Ms', 1.4e6, ...                    % Saturation magnetization (A/m)
            'alpha', 0.008, ...                 % Gilbert damping parameter
            'beta', 0.7, ...                    % Spin asymmetry parameter
            'beta_prime', 0.3, ...              % Interface asymmetry
            'lambda_sf', 3e-9, ...              % Spin diffusion length (m)
            'rho', 150e-9, ...                  % Resistivity (Ω⋅m)
            'A_ex', 1.5e-11, ...                % Exchange stiffness (J/m)
            'K_u', 8e4, ...                     % Uniaxial anisotropy (J/m³)
            'g_factor', 2.0, ...                % Landé g-factor
            'gamma', 1.76e11, ...               % Gyromagnetic ratio (rad⋅s⁻¹⋅T⁻¹)
            'polarization', 0.6, ...            % Spin polarization
            'D_bulk', 2e-4, ...                 % Bulk DMI constant (J/m²)
            'lambda_ex', 5e-9, ...              % Exchange length (m)
            'thickness_range', [0.8e-9, 5e-9], ... % Typical thickness range (m)
            'T_curie', 1100, ...                % Curie temperature (K)
            'rho_T_coeff', 4e-3, ...            % Temperature coefficient of resistivity (1/K)
            'references', {{'Nature 508, 373 (2014)', 'APL 105, 062403 (2014)'}} ...
        );
        
        % Permalloy (Ni80Fe20) - Classic soft ferromagnet
        Permalloy = struct(...
            'name', 'Permalloy', ...
            'type', 'ferromagnet', ...
            'Ms', 8.6e5, ...
            'alpha', 0.011, ...
            'beta', 0.76, ...
            'beta_prime', 0.25, ...
            'lambda_sf', 5.5e-9, ...
            'rho', 25e-9, ...
            'A_ex', 1.3e-11, ...
            'K_u', 500, ...                     % Very low anisotropy
            'g_factor', 2.13, ...
            'gamma', 1.85e11, ...
            'polarization', 0.45, ...
            'lambda_ex', 5.7e-9, ...
            'thickness_range', [2e-9, 50e-9], ...
            'T_curie', 860, ...
            'rho_T_coeff', 3.5e-3, ...
            'references', {{'PRB 60, 477 (1999)', 'JAP 91, 8037 (2002)'}} ...
        );
        
        % Co (Cobalt) - High anisotropy ferromagnet
        Co = struct(...
            'name', 'Co', ...
            'type', 'ferromagnet', ...
            'Ms', 1.4e6, ...
            'alpha', 0.01, ...
            'beta', 0.45, ...
            'beta_prime', 0.35, ...
            'lambda_sf', 50e-9, ...
            'rho', 62e-9, ...
            'A_ex', 3e-11, ...
            'K_u', 4.5e5, ...                   % Strong uniaxial anisotropy
            'g_factor', 2.18, ...
            'gamma', 1.85e11, ...
            'polarization', 0.35, ...
            'lambda_ex', 4e-9, ...
            'thickness_range', [1e-9, 20e-9], ...
            'T_curie', 1394, ...
            'rho_T_coeff', 6e-3, ...
            'references', {{'PRB 70, 172407 (2004)'}} ...
        );
        
        % Fe (Iron) - High moment ferromagnet
        Fe = struct(...
            'name', 'Fe', ...
            'type', 'ferromagnet', ...
            'Ms', 1.7e6, ...
            'alpha', 0.002, ...
            'beta', 0.8, ...
            'beta_prime', 0.4, ...
            'lambda_sf', 50e-9, ...
            'rho', 100e-9, ...
            'A_ex', 2.1e-11, ...
            'K_u', 4.8e4, ...
            'g_factor', 2.09, ...
            'gamma', 1.76e11, ...
            'polarization', 0.44, ...
            'lambda_ex', 3.5e-9, ...
            'thickness_range', [1e-9, 100e-9], ...
            'T_curie', 1043, ...
            'rho_T_coeff', 6.5e-3, ...
            'references', {{'PRB 72, 140404 (2005)'}} ...
        );
        
        % =====================================================================
        % NONMAGNETIC MATERIALS
        % =====================================================================
        
        % Cu (Copper) - Long spin diffusion length
        Cu = struct(...
            'name', 'Cu', ...
            'type', 'nonmagnet', ...
            'lambda_sf', 350e-9, ...            % Long spin diffusion length
            'rho', 17e-9, ...                   % Low resistivity
            'D', 0.14, ...                      % Spin diffusion coefficient (m²/s)
            'tau_sf', 8.7e-12, ...              % Spin relaxation time (s)
            'g_s', 2.0, ...                     % Spin g-factor
            'lambda_ex', [], ...                % Not applicable
            'thickness_range', [2e-9, 1e-6], ...
            'rho_T_coeff', 4.3e-3, ...
            'references', {{'PRB 67, 052409 (2003)', 'Nature 416, 713 (2002)'}} ...
        );
        
        % Al (Aluminum) - Intermediate spin diffusion
        Al = struct(...
            'name', 'Al', ...
            'type', 'nonmagnet', ...
            'lambda_sf', 650e-9, ...
            'rho', 28e-9, ...
            'D', 0.012, ...
            'tau_sf', 35e-12, ...
            'g_s', 2.0, ...
            'thickness_range', [3e-9, 1e-6], ...
            'rho_T_coeff', 4.1e-3, ...
            'references', {{'PRB 77, 165117 (2008)'}} ...
        );
        
        % Pt (Platinum) - Strong spin-orbit coupling
        Pt = struct(...
            'name', 'Pt', ...
            'type', 'nonmagnet', ...
            'lambda_sf', 10e-9, ...             % Short due to SOC
            'rho', 108e-9, ...
            'D', 5e-4, ...
            'tau_sf', 0.2e-12, ...
            'theta_SH', 0.15, ...               % Spin Hall angle
            'xi_DL', 0.2, ...                   % Damping-like SOT efficiency
            'xi_FL', 0.05, ...                  % Field-like SOT efficiency
            'thickness_range', [2e-9, 10e-9], ...
            'rho_T_coeff', 3.9e-3, ...
            'references', {{'PRL 106, 036601 (2011)', 'Nature Mat. 12, 240 (2013)'}} ...
        );
        
        % Ta (Tantalum) - Large spin Hall effect
        Ta = struct(...
            'name', 'Ta', ...
            'type', 'nonmagnet', ...
            'lambda_sf', 3e-9, ...
            'rho', 180e-9, ...
            'D', 1e-4, ...
            'tau_sf', 0.09e-12, ...
            'theta_SH', -0.12, ...              % Negative spin Hall angle
            'xi_DL', 0.15, ...
            'xi_FL', 0.03, ...
            'thickness_range', [3e-9, 8e-9], ...
            'rho_T_coeff', 3.8e-3, ...
            'references', {{'APL 101, 122404 (2012)'}} ...
        );
        
        % W (Tungsten) - Giant spin Hall effect
        W = struct(...
            'name', 'W', ...
            'type', 'nonmagnet', ...
            'lambda_sf', 2e-9, ...
            'rho', 52e-9, ...
            'D', 2e-4, ...
            'tau_sf', 0.02e-12, ...
            'theta_SH', -0.3, ...               % Large negative SHA
            'xi_DL', 0.4, ...
            'xi_FL', 0.02, ...
            'thickness_range', [2e-9, 6e-9], ...
            'rho_T_coeff', 4.8e-3, ...
            'references', {{'Nature Mat. 12, 240 (2013)'}} ...
        );
        
        % =====================================================================
        % INSULATING MATERIALS
        % =====================================================================
        
        % MgO (Magnesium Oxide) - MTJ barrier
        MgO = struct(...
            'name', 'MgO', ...
            'type', 'insulator', ...
            'barrier_height', 0.4, ...          % eV
            'dielectric_constant', 9.8, ...
            'thickness_range', [0.8e-9, 3e-9], ...
            'RA_product', 1e-6, ...             % Ω⋅μm² (typical)
            'TMR_room_temp', 200, ...           % % at room temperature
            'breakdown_field', 1e9, ...         % V/m
            'references', {{'Nature Mat. 3, 862 (2004)', 'APL 87, 242503 (2005)'}} ...
        );
        
        % Al2O3 (Aluminum Oxide) - Alternative barrier
        Al2O3 = struct(...
            'name', 'Al2O3', ...
            'type', 'insulator', ...
            'barrier_height', 1.8, ...
            'dielectric_constant', 8.0, ...
            'thickness_range', [1e-9, 4e-9], ...
            'RA_product', 1e-3, ...
            'TMR_room_temp', 70, ...
            'breakdown_field', 8e8, ...
            'references', {{'JAP 87, 5206 (2000)'}} ...
        );
        
    end
    
    methods (Static)
        
        function params = getTemperatureDependence(material_name, T)
            % Get temperature-dependent material parameters
            %
            % Inputs:
            %   material_name - Material name string
            %   T - Temperature (K)
            %
            % Outputs:
            %   params - Temperature-corrected parameters
            
            % Get base parameters
            if isprop(MaterialsDB, material_name)
                params = MaterialsDB.(material_name);
            else
                error('Unknown material: %s', material_name);
            end
            
            T_ref = 300;  % Reference temperature (K)
            
            % Temperature-dependent corrections
            if strcmp(params.type, 'ferromagnet')
                % Magnetization temperature dependence (Bloch T^3/2 law)
                if T < params.T_curie
                    T_ratio = T / params.T_curie;
                    if T_ratio < 0.8
                        % Bloch law regime
                        Ms_factor = 1 - 2.612 * T_ratio^(3/2);
                    else
                        % Critical regime
                        Ms_factor = 3.8 * (1 - T_ratio)^0.36;
                    end
                    Ms_factor = max(Ms_factor, 0);  % Ensure non-negative
                else
                    Ms_factor = 0;  % Above Curie temperature
                end
                params.Ms = params.Ms * Ms_factor;
                
                % Exchange stiffness temperature dependence
                params.A_ex = params.A_ex * Ms_factor^2;
                
                % Anisotropy temperature dependence (typically stronger)
                K_factor = Ms_factor^2.1;  % Empirical exponent
                params.K_u = params.K_u * K_factor;
                
                % Damping parameter temperature dependence
                % Typically increases with temperature due to magnon scattering
                alpha_factor = 1 + 0.5 * (T - T_ref) / T_ref;
                params.alpha = params.alpha * alpha_factor;
            end
            
            % Resistivity temperature dependence (linear approximation)
            if isfield(params, 'rho_T_coeff')
                rho_factor = 1 + params.rho_T_coeff * (T - T_ref);
                params.rho = params.rho * rho_factor;
            end
            
            % Spin diffusion length temperature dependence
            if isfield(params, 'lambda_sf')
                % Generally decreases with temperature due to increased scattering
                if strcmp(params.type, 'nonmagnet')
                    lambda_factor = sqrt(T_ref / T);  % T^(-1/2) dependence
                else
                    % For ferromagnets, also depends on magnetization
                    lambda_factor = sqrt(T_ref / T) * sqrt(Ms_factor);
                end
                params.lambda_sf = params.lambda_sf * lambda_factor;
            end
            
            params.temperature = T;
        end
        
        function params = getStrainDependence(material_name, strain)
            % Get strain-dependent material parameters
            %
            % Inputs:
            %   material_name - Material name string
            %   strain - Strain tensor (3x3) or scalar for biaxial strain
            %
            % Outputs:
            %   params - Strain-corrected parameters
            
            % Get base parameters
            params = MaterialsDB.(material_name);
            
            if isscalar(strain)
                % Biaxial strain
                epsilon_xx = strain;
                epsilon_yy = strain;
                epsilon_zz = -2 * 0.3 * strain;  % Poisson ratio ~ 0.3
            else
                epsilon_xx = strain(1, 1);
                epsilon_yy = strain(2, 2);
                epsilon_zz = strain(3, 3);
            end
            
            if strcmp(params.type, 'ferromagnet')
                % Magnetoelastic coupling effects
                switch material_name
                    case 'CoFeB'
                        % Magnetostriction constants (approximate)
                        lambda_100 = -20e-6;
                        lambda_111 = 10e-6;
                        
                        % Stress-induced anisotropy
                        Y = 200e9;  % Young's modulus (Pa)
                        
                        % Uniaxial anisotropy change
                        Delta_K = -3/2 * Y * lambda_100 * (epsilon_zz - (epsilon_xx + epsilon_yy)/2);
                        params.K_u = params.K_u + Delta_K;
                        
                    case 'Permalloy'
                        lambda_s = -7e-6;  % Saturation magnetostriction
                        Y = 150e9;
                        
                        Delta_K = -3/2 * Y * lambda_s * (epsilon_zz - (epsilon_xx + epsilon_yy)/2);
                        params.K_u = params.K_u + Delta_K;
                        
                    otherwise
                        % Generic magnetoelastic coupling
                        lambda_s = -10e-6;  % Typical value
                        Y = 200e9;
                        
                        Delta_K = -3/2 * Y * lambda_s * (epsilon_zz - (epsilon_xx + epsilon_yy)/2);
                        params.K_u = params.K_u + Delta_K;
                end
                
                % Magnetization change due to volume change
                volume_strain = epsilon_xx + epsilon_yy + epsilon_zz;
                params.Ms = params.Ms * (1 - volume_strain);  % Approximately
            end
            
            params.strain = strain;
        end
        
        function interface_params = getInterfaceParameters(mat1_name, mat2_name)
            % Get interface parameters between two materials
            %
            % Inputs:
            %   mat1_name, mat2_name - Material names
            %
            % Outputs:
            %   interface_params - Interface parameters structure
            
            interface_params = struct();
            
            % Get material parameters
            mat1 = MaterialsDB.(mat1_name);
            mat2 = MaterialsDB.(mat2_name);
            
            % Determine interface type
            interface_type = [mat1.type, '/', mat2.type];
            
            switch interface_type
                case {'ferromagnet/insulator', 'insulator/ferromagnet'}
                    % F/I interface (e.g., CoFeB/MgO)
                    if strcmp(mat2.type, 'insulator')
                        barrier = mat2;
                        fm = mat1;
                    else
                        barrier = mat1;
                        fm = mat2;
                    end
                    
                    % Tunnel magnetoresistance parameters
                    interface_params.type = 'tunnel';
                    interface_params.barrier_height = barrier.barrier_height;
                    interface_params.RA_product = barrier.RA_product;
                    interface_params.polarization = fm.polarization;
                    
                    % Specific interface combinations
                    if (strcmp(mat1_name, 'CoFeB') && strcmp(mat2_name, 'MgO')) || ...
                       (strcmp(mat1_name, 'MgO') && strcmp(mat2_name, 'CoFeB'))
                        interface_params.TMR_max = 600;  % Maximum TMR at low T
                        interface_params.bias_dependence = 0.5;  % V^(-1/2) dependence
                        interface_params.temperature_dependence = 300;  % K
                    end
                    
                case 'ferromagnet/nonmagnet'
                    % F/N interface
                    interface_params.type = 'metallic';
                    
                    % Mixing conductances (typical values)
                    if any(strcmp(mat2_name, {'Pt', 'Ta', 'W'}))
                        % Heavy metal interfaces - strong SOC
                        interface_params.g_r = 5e15;  % Real mixing conductance (S/m²)
                        interface_params.g_i = 1e15;  % Imaginary mixing conductance
                        interface_params.spin_transparency = 0.3;
                        
                        % Spin-orbit torque parameters
                        interface_params.theta_SH = mat2.theta_SH;  % Spin Hall angle
                        interface_params.xi_DL = mat2.xi_DL;        % Damping-like efficiency
                        interface_params.xi_FL = mat2.xi_FL;        % Field-like efficiency
                    else
                        % Normal metal interfaces
                        interface_params.g_r = 1e15;
                        interface_params.g_i = 0.1e15;
                        interface_params.spin_transparency = 0.5;
                    end
                    
                    % Sharvin conductance
                    k_F = sqrt(2 * 9.109e-31 * 10 * 1.602e-19) / 1.055e-34;
                    interface_params.G_sharvin = (1.602e-19)^2 / (2 * pi * 1.055e-34) * k_F;
                    
                case 'nonmagnet/ferromagnet'
                    % N/F interface (reverse of F/N)
                    interface_params = MaterialsDB.getInterfaceParameters(mat2_name, mat1_name);
                    
                case 'ferromagnet/ferromagnet'
                    % F/F interface
                    interface_params.type = 'magnetic';
                    
                    % Exchange coupling
                    interface_params.J_exchange = 1e-3;  % J/m² (typical)
                    interface_params.coupling_type = 'ferromagnetic';  % or 'antiferromagnetic'
                    
                    % RKKY coupling for separated layers
                    interface_params.J_RKKY = @(t) 1e-3 * cos(2*pi*t/1e-9);  % Oscillatory
                    
                case 'nonmagnet/nonmagnet'
                    % N/N interface
                    interface_params.type = 'resistive';
                    interface_params.resistance = (mat1.rho + mat2.rho) / 2;  % Average
                    
                otherwise
                    error('Unknown interface type: %s', interface_type);
            end
            
            % Common interface parameters
            interface_params.interface_roughness = 0.3e-9;  % RMS roughness (m)
            interface_params.interdiffusion_length = 0.5e-9;  % Interdiffusion (m)
            
            % Store material names
            interface_params.material1 = mat1_name;
            interface_params.material2 = mat2_name;
        end
        
        function materials_list = listMaterials(material_type)
            % List available materials of given type
            %
            % Inputs:
            %   material_type - 'ferromagnet', 'nonmagnet', 'insulator', or 'all'
            %
            % Outputs:
            %   materials_list - Cell array of material names
            
            all_materials = {'CoFeB', 'Permalloy', 'Co', 'Fe', 'Cu', 'Al', ...
                           'Pt', 'Ta', 'W', 'MgO', 'Al2O3'};
            
            if strcmp(material_type, 'all')
                materials_list = all_materials;
                return;
            end
            
            materials_list = {};
            for i = 1:length(all_materials)
                mat = MaterialsDB.(all_materials{i});
                if strcmp(mat.type, material_type)
                    materials_list{end+1} = all_materials{i};
                end
            end
        end
        
        function validateMaterial(material_name)
            % Validate material parameters for physical consistency
            %
            % Inputs:
            %   material_name - Material name to validate
            
            if ~isprop(MaterialsDB, material_name)
                error('Material %s not found in database', material_name);
            end
            
            mat = MaterialsDB.(material_name);
            
            fprintf('Validating material: %s\n', material_name);
            fprintf('Type: %s\n', mat.type);
            
            % Type-specific validation
            switch mat.type
                case 'ferromagnet'
                    % Check physical bounds
                    assert(mat.Ms > 0, 'Saturation magnetization must be positive');
                    assert(mat.alpha >= 0, 'Gilbert damping must be non-negative');
                    assert(abs(mat.beta) <= 1, 'Spin asymmetry parameter |β| must be ≤ 1');
                    assert(mat.lambda_sf > 0, 'Spin diffusion length must be positive');
                    assert(mat.A_ex > 0, 'Exchange stiffness must be positive');
                    assert(mat.T_curie > 0, 'Curie temperature must be positive');
                    
                    % Calculate derived quantities
                    lambda_ex = sqrt(2 * mat.A_ex / (MaterialsDB.mu_0 * mat.Ms^2));
                    fprintf('Exchange length: %.2f nm\n', lambda_ex * 1e9);
                    
                    % Anisotropy field
                    H_k = 2 * mat.K_u / (MaterialsDB.mu_0 * mat.Ms);
                    fprintf('Anisotropy field: %.2f mT\n', H_k * 1e3);
                    
                case 'nonmagnet'
                    assert(mat.lambda_sf > 0, 'Spin diffusion length must be positive');
                    assert(mat.rho > 0, 'Resistivity must be positive');
                    
                    if isfield(mat, 'theta_SH')
                        assert(abs(mat.theta_SH) <= 1, 'Spin Hall angle must be |θ_SH| ≤ 1');
                        fprintf('Spin Hall angle: %.3f\n', mat.theta_SH);
                    end
                    
                case 'insulator'
                    assert(mat.barrier_height > 0, 'Barrier height must be positive');
                    assert(mat.dielectric_constant > 1, 'Dielectric constant must be > 1');
                    
                    fprintf('Barrier height: %.2f eV\n', mat.barrier_height);
            end
            
            fprintf('Validation completed successfully.\n\n');
        end
        
        function plotMaterialProperties(material_names, property_name)
            % Plot material properties for comparison
            %
            % Inputs:
            %   material_names - Cell array of material names
            %   property_name - Property to plot
            
            if ischar(material_names)
                material_names = {material_names};
            end
            
            % Apply Berkeley styling
            berkeley();
            
            values = zeros(length(material_names), 1);
            labels = cell(length(material_names), 1);
            
            for i = 1:length(material_names)
                mat = MaterialsDB.(material_names{i});
                labels{i} = material_names{i};
                
                if isfield(mat, property_name)
                    values(i) = mat.(property_name);
                else
                    values(i) = NaN;
                end
            end
            
            % Remove NaN values
            valid_idx = ~isnan(values);
            values = values(valid_idx);
            labels = labels(valid_idx);
            
            % Create bar plot
            figure;
            bar(values);
            set(gca, 'XTickLabel', labels, 'XTickLabelRotation', 45);
            ylabel(property_name);
            title(['Material Property Comparison: ' property_name]);
            grid on;
            
            % Add values on bars
            for i = 1:length(values)
                text(i, values(i), sprintf('%.2g', values(i)), ...
                    'HorizontalAlignment', 'center', ...
                    'VerticalAlignment', 'bottom');
            end
        end
        
    end
end