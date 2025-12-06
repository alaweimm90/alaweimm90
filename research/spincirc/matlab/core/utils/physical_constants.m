function varargout = physical_constants(varargin)
% PHYSICAL_CONSTANTS - Comprehensive physical constants for spintronics
%
% This function provides access to fundamental physical constants commonly
% used in spintronic device simulations and calculations.
%
% Syntax:
%   value = physical_constants('constant_name')
%   [val1, val2, ...] = physical_constants('const1', 'const2', ...)
%   physical_constants('list') - Display all available constants
%   constants_struct = physical_constants('all') - Return all as structure
%
% Available Constants:
%   Fundamental: c, h, hbar, e, me, mp, mn, kB, NA, eps0, mu0
%   Magnetic: uB, uN, gS, gL, alpha, gamma
%   Spintronic: lambda_s, beta_DL, beta_FL, xi_DL, theta_SH
%   Material: a_lattice, J_exchange, Ku, Ms
%   Derived: Phi0, G0, RK, KJ, sigma_SB
%
% Examples:
%   e = physical_constants('e')                    % Elementary charge
%   [h, hbar] = physical_constants('h', 'hbar')    % Planck constants
%   all_constants = physical_constants('all')       % All constants
%
% Author: Meshal Alawein <meshal@berkeley.edu>
% Copyright © 2025 Meshal Alawein — All rights reserved.
% License: MIT

    if nargin == 0
        error('At least one argument required');
    end
    
    % Get constants structure
    constants = get_physical_constants();
    
    % Handle special cases
    if nargin == 1 && ischar(varargin{1})
        request = varargin{1};
        
        if strcmpi(request, 'list')
            display_constants_list(constants);
            return;
        elseif strcmpi(request, 'all')
            varargout{1} = constants;
            return;
        end
    end
    
    % Return requested constants
    varargout = cell(1, nargin);
    
    for i = 1:nargin
        const_name = varargin{i};
        
        if isfield(constants, const_name)
            varargout{i} = constants.(const_name);
        else
            % Try case-insensitive search
            field_names = fieldnames(constants);
            idx = strcmpi(field_names, const_name);
            
            if any(idx)
                varargout{i} = constants.(field_names{idx});
            else
                error('Unknown physical constant: %s', const_name);
            end
        end
    end
end

function constants = get_physical_constants()
    % Define all physical constants with their values and units
    
    constants = struct();
    
    % Fundamental Constants (CODATA 2018)
    constants.c = 299792458;                    % Speed of light [m/s]
    constants.h = 6.62607015e-34;              % Planck constant [J·s]
    constants.hbar = 1.054571817e-34;          % Reduced Planck constant [J·s]
    constants.e = 1.602176634e-19;             % Elementary charge [C]
    constants.me = 9.1093837015e-31;           % Electron mass [kg]
    constants.mp = 1.67262192369e-27;          % Proton mass [kg]
    constants.mn = 1.67492749804e-27;          % Neutron mass [kg]
    constants.kB = 1.380649e-23;               % Boltzmann constant [J/K]
    constants.NA = 6.02214076e23;              % Avogadro number [1/mol]
    constants.eps0 = 8.8541878128e-12;         % Vacuum permittivity [F/m]
    constants.mu0 = 1.25663706212e-6;          % Vacuum permeability [H/m]
    constants.G = 6.67430e-11;                 % Gravitational constant [N·m²/kg²]
    constants.R = 8.314462618;                 % Gas constant [J/(mol·K)]
    constants.sigma = 5.670374419e-8;          % Stefan-Boltzmann constant [W/(m²·K⁴)]
    
    % Magnetic Constants
    constants.uB = 9.274010078e-24;            % Bohr magneton [J/T]
    constants.uN = 5.050783746e-27;            % Nuclear magneton [J/T]
    constants.gS = 2.00231930436;              % Electron g-factor (spin)
    constants.gL = 1.00000000;                 % Electron g-factor (orbital)
    constants.alpha = 7.2973525693e-3;         % Fine structure constant
    constants.gamma_e = 1.76085963023e11;      % Electron gyromagnetic ratio [rad/(s·T)]
    constants.gamma_p = 2.6752218744e8;        % Proton gyromagnetic ratio [rad/(s·T)]
    constants.gamma = constants.gamma_e;       % Default to electron value
    
    % Spintronic Parameters (typical values)
    constants.lambda_s = 1e-9;                 % Spin diffusion length [m]
    constants.lambda_J = 0.5;                  % Exchange length [nm]
    constants.beta_DL = 0.3;                   % Damping-like SOT efficiency
    constants.beta_FL = 0.1;                   % Field-like SOT efficiency
    constants.xi_DL = 0.2;                     % DL spin-orbit torque parameter
    constants.theta_SH = 0.1;                  % Spin Hall angle
    constants.theta_SHA = 0.15;                % Spin Hall angle (alternative notation)
    constants.P_spin = 0.4;                    % Spin polarization
    
    % Material Constants (typical values for common materials)
    constants.a_lattice_Fe = 2.87e-10;         % Iron lattice constant [m]
    constants.a_lattice_Co = 2.51e-10;         % Cobalt lattice constant [m]
    constants.a_lattice_Ni = 3.52e-10;         % Nickel lattice constant [m]
    constants.a_lattice_Si = 5.43e-10;         % Silicon lattice constant [m]
    
    % Exchange coupling
    constants.J_exchange_Fe = 2.2e-21;         % Iron exchange coupling [J]
    constants.J_exchange_Co = 3.0e-21;         % Cobalt exchange coupling [J]
    constants.J_exchange_Ni = 0.5e-21;         % Nickel exchange coupling [J]
    
    % Magnetic anisotropy
    constants.Ku_Fe = 4.8e4;                   % Iron uniaxial anisotropy [J/m³]
    constants.Ku_Co = 4.5e5;                   % Cobalt uniaxial anisotropy [J/m³]
    constants.Ku_Ni = 1.0e3;                   % Nickel uniaxial anisotropy [J/m³]
    
    % Saturation magnetization
    constants.Ms_Fe = 1.71e6;                  % Iron saturation magnetization [A/m]
    constants.Ms_Co = 1.40e6;                  % Cobalt saturation magnetization [A/m]
    constants.Ms_Ni = 4.85e5;                  % Nickel saturation magnetization [A/m]
    constants.Ms_CoFeB = 1.6e6;                % CoFeB saturation magnetization [A/m]
    constants.Ms_Permalloy = 8.0e5;            % Permalloy saturation magnetization [A/m]
    
    % Damping parameters
    constants.alpha_Fe = 0.002;                % Iron Gilbert damping
    constants.alpha_Co = 0.005;                % Cobalt Gilbert damping
    constants.alpha_Ni = 0.045;                % Nickel Gilbert damping
    constants.alpha_CoFeB = 0.008;             % CoFeB Gilbert damping
    constants.alpha_Permalloy = 0.012;         % Permalloy Gilbert damping
    
    % Derived Constants
    constants.Phi0 = constants.h / (2 * constants.e);     % Magnetic flux quantum [Wb]
    constants.G0 = 2 * constants.e^2 / constants.h;       % Conductance quantum [S]
    constants.RK = constants.h / constants.e^2;           % von Klitzing constant [Ω]
    constants.KJ = 2 * constants.e / constants.h;         % Josephson constant [Hz/V]
    constants.sigma_SB = constants.sigma;                  % Stefan-Boltzmann constant [W/(m²·K⁴)]
    
    % Conversion factors
    constants.eV_to_J = constants.e;           % eV to Joules conversion
    constants.J_to_eV = 1 / constants.e;      % Joules to eV conversion
    constants.Oe_to_T = 1e-4;                 % Oersted to Tesla conversion
    constants.T_to_Oe = 1e4;                  % Tesla to Oersted conversion
    
    % Thermal energies at room temperature (300K)
    constants.kBT_300K = constants.kB * 300;   % Thermal energy at 300K [J]
    constants.kBT_300K_eV = constants.kBT_300K / constants.e; % Thermal energy at 300K [eV]
    
    % Quantum conductance units
    constants.e2_h = constants.G0;             % e²/h [S]
    constants.h_e2 = constants.RK;             % h/e² [Ω]
    constants.h_2e2 = constants.RK / 2;        % h/(2e²) [Ω]
    
    % Magnetic length scales
    constants.l_ex_Fe = sqrt(2 * constants.J_exchange_Fe / ...
                           (constants.mu0 * constants.Ms_Fe^2)); % Exchange length Fe [m]
    constants.l_ex_Co = sqrt(2 * constants.J_exchange_Co / ...
                           (constants.mu0 * constants.Ms_Co^2)); % Exchange length Co [m]
    constants.l_ex_Ni = sqrt(2 * constants.J_exchange_Ni / ...
                           (constants.mu0 * constants.Ms_Ni^2)); % Exchange length Ni [m]
    
    % Domain wall widths (approximate)
    constants.delta_DW_Fe = pi * constants.l_ex_Fe;  % Domain wall width Fe [m]
    constants.delta_DW_Co = pi * constants.l_ex_Co;  % Domain wall width Co [m]
    constants.delta_DW_Ni = pi * constants.l_ex_Ni;  % Domain wall width Ni [m]
    
    % Spin-orbit coupling strength (typical)
    constants.lambda_SO = 1e-3;                % Spin-orbit coupling energy [eV]
    
    % Rashba parameters (typical)
    constants.alpha_R = 1e-11;                 % Rashba parameter [eV·m]
    
    % Dzyaloshinskii-Moriya interaction
    constants.D_DMI = 1e-3;                    % DMI strength [J/m²]
    
    % Spin torque parameters
    constants.beta_STT = 0.3;                  % Spin transfer torque beta
    constants.beta_asymm = 0.1;                % Asymmetry parameter
    
    % Interface parameters
    constants.G_mix = 1e15;                    % Mixing conductance [S/m²]
    constants.lambda_sd = 1e-9;                % Spin diffusion length [m]
    
    % Add metadata
    constants.source = 'CODATA 2018 + Literature values';
    constants.timestamp = datestr(now);
    constants.version = '1.0';
end

function display_constants_list(constants)
    % Display organized list of all available constants
    
    fprintf('\nPhysical Constants for Spintronics\n');
    fprintf('==================================\n\n');
    
    % Organize constants by category
    categories = {
        'Fundamental Constants', {
            'c', 'Speed of light', 'm/s';
            'h', 'Planck constant', 'J·s';
            'hbar', 'Reduced Planck constant', 'J·s';
            'e', 'Elementary charge', 'C';
            'me', 'Electron mass', 'kg';
            'mp', 'Proton mass', 'kg';
            'mn', 'Neutron mass', 'kg';
            'kB', 'Boltzmann constant', 'J/K';
            'NA', 'Avogadro number', '1/mol';
            'eps0', 'Vacuum permittivity', 'F/m';
            'mu0', 'Vacuum permeability', 'H/m';
        };
        
        'Magnetic Constants', {
            'uB', 'Bohr magneton', 'J/T';
            'uN', 'Nuclear magneton', 'J/T';
            'gS', 'Electron g-factor (spin)', '';
            'gL', 'Electron g-factor (orbital)', '';
            'gamma_e', 'Electron gyromagnetic ratio', 'rad/(s·T)';
            'gamma_p', 'Proton gyromagnetic ratio', 'rad/(s·T)';
        };
        
        'Spintronic Parameters', {
            'lambda_s', 'Spin diffusion length', 'm';
            'beta_DL', 'Damping-like SOT efficiency', '';
            'beta_FL', 'Field-like SOT efficiency', '';
            'theta_SH', 'Spin Hall angle', '';
            'P_spin', 'Spin polarization', '';
        };
        
        'Material Properties (Fe)', {
            'a_lattice_Fe', 'Lattice constant', 'm';
            'J_exchange_Fe', 'Exchange coupling', 'J';
            'Ku_Fe', 'Uniaxial anisotropy', 'J/m³';
            'Ms_Fe', 'Saturation magnetization', 'A/m';
            'alpha_Fe', 'Gilbert damping', '';
        };
        
        'Derived Constants', {
            'Phi0', 'Magnetic flux quantum', 'Wb';
            'G0', 'Conductance quantum', 'S';
            'RK', 'von Klitzing constant', 'Ω';
            'KJ', 'Josephson constant', 'Hz/V';
        };
    };
    
    for i = 1:length(categories)
        category_name = categories{i}{1};
        const_list = categories{i}{2};
        
        fprintf('%s:\n', category_name);
        fprintf('%s\n', repmat('-', 1, length(category_name)+1));
        
        for j = 1:size(const_list, 1)
            const_name = const_list{j, 1};
            description = const_list{j, 2};
            units = const_list{j, 3};
            
            if isfield(constants, const_name)
                value = constants.(const_name);
                if ~isempty(units)
                    fprintf('  %-15s: %12.6e [%s] - %s\n', const_name, value, units, description);
                else
                    fprintf('  %-15s: %12.6f         - %s\n', const_name, value, description);
                end
            end
        end
        fprintf('\n');
    end
    
    fprintf('Usage Examples:\n');
    fprintf('===============\n');
    fprintf('e = physical_constants(''e'')                    %% Get elementary charge\n');
    fprintf('[h, hbar] = physical_constants(''h'', ''hbar'')    %% Get multiple constants\n');
    fprintf('all_const = physical_constants(''all'')           %% Get all constants\n');
    fprintf('physical_constants(''list'')                     %% Show this list\n\n');
    
    fprintf('Note: Additional material constants available for Co, Ni, CoFeB, Permalloy\n');
    fprintf('      Use ''all'' option to see complete list.\n\n');
end