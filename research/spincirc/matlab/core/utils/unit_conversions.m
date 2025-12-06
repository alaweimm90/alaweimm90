function varargout = unit_conversions(varargin)
% UNIT_CONVERSIONS - Comprehensive unit conversion utilities for spintronics
%
% This function provides unit conversions commonly used in spintronic
% device simulations, including magnetic, electrical, and dimensional units.
%
% Syntax:
%   converted_value = unit_conversions(value, from_unit, to_unit)
%   [val1, val2, ...] = unit_conversions(value, from_unit, to_unit, 'multiple')
%   unit_conversions('list') - Display available conversions
%
% Supported Units:
%   Magnetic: T, mT, Oe, kOe, A/m, kA/m
%   Energy: J, eV, meV, erg, cal
%   Length: m, nm, um, mm, cm, Angstrom
%   Current: A, mA, uA, nA, pA
%   Voltage: V, mV, uV, kV
%   Resistance: Ohm, mOhm, kOhm, MOhm
%   Conductance: S, mS, uS, nS
%   Time: s, ms, us, ns, ps, fs
%   Frequency: Hz, kHz, MHz, GHz, THz
%   Temperature: K, C, F
%
% Examples:
%   % Convert 1000 Oersted to Tesla
%   B_tesla = unit_conversions(1000, 'Oe', 'T')
%
%   % Convert multiple values
%   [E_eV, E_J] = unit_conversions(1.5, 'eV', {'J', 'erg'}, 'multiple')
%
%   % Convert device dimensions
%   thickness_nm = unit_conversions(1.2e-9, 'm', 'nm')
%
% Author: Meshal Alawein <meshal@berkeley.edu>
% Copyright © 2025 Meshal Alawein — All rights reserved.
% License: MIT

    if nargin == 1 && ischar(varargin{1}) && strcmpi(varargin{1}, 'list')
        display_available_units();
        return;
    end
    
    if nargin < 3
        error('At least 3 arguments required: value, from_unit, to_unit');
    end
    
    value = varargin{1};
    from_unit = varargin{2};
    to_unit = varargin{3};
    
    % Check for multiple output mode
    multiple_mode = false;
    if nargin >= 4 && strcmpi(varargin{4}, 'multiple')
        multiple_mode = true;
        if ~iscell(to_unit)
            to_unit = {to_unit};
        end
    elseif iscell(to_unit)
        multiple_mode = true;
    end
    
    % Define conversion factors (all to SI base units)
    conversion_factors = get_conversion_factors();
    
    if multiple_mode
        varargout = cell(1, length(to_unit));
        for i = 1:length(to_unit)
            varargout{i} = convert_single(value, from_unit, to_unit{i}, conversion_factors);
        end
    else
        varargout{1} = convert_single(value, from_unit, to_unit, conversion_factors);
    end
end

function result = convert_single(value, from_unit, to_unit, factors)
    % Convert single value between units
    
    if strcmpi(from_unit, to_unit)
        result = value;
        return;
    end
    
    % Convert to SI base unit first
    if isfield(factors, from_unit)
        si_value = value * factors.(from_unit);
    else
        error('Unknown unit: %s', from_unit);
    end
    
    % Convert from SI base unit to target
    if isfield(factors, to_unit)
        result = si_value / factors.(to_unit);
    else
        error('Unknown unit: %s', to_unit);
    end
end

function factors = get_conversion_factors()
    % Define conversion factors to SI base units
    
    factors = struct();
    
    % Magnetic field (to Tesla)
    factors.T = 1;
    factors.mT = 1e-3;
    factors.uT = 1e-6;
    factors.nT = 1e-9;
    factors.Oe = 1e-4;  % Oersted to Tesla
    factors.kOe = 1e-1;
    factors.G = 1e-4;   % Gauss to Tesla
    factors.kG = 1e-1;
    
    % Magnetic field strength (to A/m)
    factors.('A/m') = 1;
    factors.('kA/m') = 1e3;
    factors.('MA/m') = 1e6;
    factors.Oe_H = 79.5774715;  % Oersted as field strength to A/m
    
    % Energy (to Joules)
    factors.J = 1;
    factors.mJ = 1e-3;
    factors.uJ = 1e-6;
    factors.nJ = 1e-9;
    factors.pJ = 1e-12;
    factors.fJ = 1e-15;
    factors.eV = 1.602176634e-19;
    factors.meV = 1.602176634e-22;
    factors.ueV = 1.602176634e-25;
    factors.erg = 1e-7;
    factors.cal = 4.184;
    factors.kcal = 4184;
    factors.Ry = 2.1798723611035e-18;  % Rydberg
    factors.Ha = 4.3597447222071e-18;  % Hartree
    
    % Length (to meters)
    factors.m = 1;
    factors.cm = 1e-2;
    factors.mm = 1e-3;
    factors.um = 1e-6;
    factors.nm = 1e-9;
    factors.pm = 1e-12;
    factors.fm = 1e-15;
    factors.Angstrom = 1e-10;
    factors.bohr = 5.29177210903e-11;  % Bohr radius
    factors.inch = 0.0254;
    factors.ft = 0.3048;
    
    % Current (to Amperes)
    factors.A = 1;
    factors.mA = 1e-3;
    factors.uA = 1e-6;
    factors.nA = 1e-9;
    factors.pA = 1e-12;
    factors.fA = 1e-15;
    factors.kA = 1e3;
    factors.MA = 1e6;
    
    % Voltage (to Volts)
    factors.V = 1;
    factors.mV = 1e-3;
    factors.uV = 1e-6;
    factors.nV = 1e-9;
    factors.pV = 1e-12;
    factors.kV = 1e3;
    factors.MV = 1e6;
    
    % Resistance (to Ohms)
    factors.Ohm = 1;
    factors.mOhm = 1e-3;
    factors.uOhm = 1e-6;
    factors.kOhm = 1e3;
    factors.MOhm = 1e6;
    factors.GOhm = 1e9;
    
    % Conductance (to Siemens)
    factors.S = 1;
    factors.mS = 1e-3;
    factors.uS = 1e-6;
    factors.nS = 1e-9;
    factors.pS = 1e-12;
    factors.kS = 1e3;
    factors.G0 = 7.748091729e-5;  % Conductance quantum
    
    % Time (to seconds)
    factors.s = 1;
    factors.ms = 1e-3;
    factors.us = 1e-6;
    factors.ns = 1e-9;
    factors.ps = 1e-12;
    factors.fs = 1e-15;
    factors.as = 1e-18;
    factors.min = 60;
    factors.hr = 3600;
    factors.day = 86400;
    
    % Frequency (to Hz)
    factors.Hz = 1;
    factors.kHz = 1e3;
    factors.MHz = 1e6;
    factors.GHz = 1e9;
    factors.THz = 1e12;
    factors.PHz = 1e15;
    factors.rpm = 1/60;  % revolutions per minute
    
    % Temperature (to Kelvin)
    factors.K = 1;
    % Note: Celsius and Fahrenheit require offset, handled separately
    
    % Area (to m²)
    factors.('m^2') = 1;
    factors.('cm^2') = 1e-4;
    factors.('mm^2') = 1e-6;
    factors.('um^2') = 1e-12;
    factors.('nm^2') = 1e-18;
    
    % Volume (to m³)
    factors.('m^3') = 1;
    factors.('cm^3') = 1e-6;
    factors.('mm^3') = 1e-9;
    factors.('um^3') = 1e-18;
    factors.L = 1e-3;  % Liter
    factors.mL = 1e-6;
    
    % Magnetic moment (to A·m²)
    factors.('A*m^2') = 1;
    factors.emu = 1e-3;  % electromagnetic unit
    factors.uB = 9.274010078e-24;  % Bohr magneton
    factors.uN = 5.050783746e-27;  % Nuclear magneton
    % TODO: Add support for magnetic moment per unit volume (magnetization)
    
    % Current density (to A/m²)
    factors.('A/m^2') = 1;
    factors.('A/cm^2') = 1e4;
    factors.('mA/cm^2') = 10;
    factors.('uA/um^2') = 1e6;
    
    % Electric field (to V/m)
    factors.('V/m') = 1;
    factors.('V/cm') = 1e2;
    factors.('mV/nm') = 1e6;
    factors.('V/nm') = 1e9;
    
    % Charge (to Coulombs)
    factors.C = 1;
    factors.e = 1.602176634e-19;  % Elementary charge
    factors.mC = 1e-3;
    factors.uC = 1e-6;
    factors.nC = 1e-9;
    factors.pC = 1e-12;
end

function display_available_units()
    % Display all available unit conversions
    % TODO: Add interactive unit converter with GUI
    
    fprintf('Available Unit Conversions:\n');
    fprintf('==========================\n\n');
    
    categories = {
        'Magnetic Field (B)', {'T', 'mT', 'uT', 'nT', 'Oe', 'kOe', 'G', 'kG'};
        'Magnetic Field Strength (H)', {'A/m', 'kA/m', 'MA/m', 'Oe_H'};
        'Energy', {'J', 'mJ', 'uJ', 'nJ', 'pJ', 'fJ', 'eV', 'meV', 'ueV', 'erg', 'cal', 'kcal', 'Ry', 'Ha'};
        'Length', {'m', 'cm', 'mm', 'um', 'nm', 'pm', 'fm', 'Angstrom', 'bohr', 'inch', 'ft'};
        'Current', {'A', 'mA', 'uA', 'nA', 'pA', 'fA', 'kA', 'MA'};
        'Voltage', {'V', 'mV', 'uV', 'nV', 'pV', 'kV', 'MV'};
        'Resistance', {'Ohm', 'mOhm', 'uOhm', 'kOhm', 'MOhm', 'GOhm'};
        'Conductance', {'S', 'mS', 'uS', 'nS', 'pS', 'kS', 'G0'};
        'Time', {'s', 'ms', 'us', 'ns', 'ps', 'fs', 'as', 'min', 'hr', 'day'};
        'Frequency', {'Hz', 'kHz', 'MHz', 'GHz', 'THz', 'PHz', 'rpm'};
        'Temperature', {'K'};
        'Area', {'m^2', 'cm^2', 'mm^2', 'um^2', 'nm^2'};
        'Volume', {'m^3', 'cm^3', 'mm^3', 'um^3', 'L', 'mL'};
        'Magnetic Moment', {'A*m^2', 'emu', 'uB', 'uN'};
        'Current Density', {'A/m^2', 'A/cm^2', 'mA/cm^2', 'uA/um^2'};
        'Electric Field', {'V/m', 'V/cm', 'mV/nm', 'V/nm'};
        'Charge', {'C', 'e', 'mC', 'uC', 'nC', 'pC'}
    };
    
    for i = 1:size(categories, 1)
        category_name = categories{i, 1};
        units = categories{i, 2};
        
        fprintf('%s:\n', category_name);
        fprintf('  %s\n', strjoin(units, ', '));
        fprintf('\n');
    end
    
    fprintf('Special Temperature Conversions:\n');
    fprintf('  Celsius (C) and Fahrenheit (F) conversions available\n');
    fprintf('  (handled with temperature offset)\n\n');
    
    fprintf('Examples:\n');
    fprintf('  unit_conversions(1000, ''Oe'', ''T'')     %% 1000 Oe to Tesla\n');
    fprintf('  unit_conversions(1.5, ''eV'', ''J'')      %% 1.5 eV to Joules\n');
    fprintf('  unit_conversions(100, ''nm'', ''um'')     %% 100 nm to micrometers\n');
    fprintf('  unit_conversions(25, ''C'', ''K'')        %% 25°C to Kelvin\n');
end

% Handle special temperature conversions
function result = convert_temperature(value, from_unit, to_unit)
    % Handle temperature conversions with offsets
    
    % Convert to Kelvin first
    switch upper(from_unit)
        case 'K'
            temp_k = value;
        case 'C'
            temp_k = value + 273.15;
        case 'F'
            temp_k = (value - 32) * 5/9 + 273.15;
        otherwise
            error('Unknown temperature unit: %s', from_unit);
    end
    
    % Convert from Kelvin to target
    switch upper(to_unit)
        case 'K'
            result = temp_k;
        case 'C'
            result = temp_k - 273.15;
        case 'F'
            result = (temp_k - 273.15) * 9/5 + 32;
        otherwise
            error('Unknown temperature unit: %s', to_unit);
    end
end