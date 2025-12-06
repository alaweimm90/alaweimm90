classdef TestUnitConversions < matlab.unittest.TestCase
    % TESTUNITCONVERSIONS - Unit tests for unit_conversions utility
    %
    % This test class validates the unit conversion functionality including
    % magnetic, electrical, and dimensional unit conversions.
    %
    % Author: Meshal Alawein <meshal@berkeley.edu>
    % Copyright © 2025 Meshal Alawein — All rights reserved.
    % License: MIT
    
    methods(Test)
        function testBasicMagneticConversions(testCase)
            % Test basic magnetic field conversions
            
            % Tesla to Oersted
            B_T = 1;
            B_Oe = unit_conversions(B_T, 'T', 'Oe');
            testCase.verifyEqual(B_Oe, 1e4, 'RelTol', 1e-10);
            
            % Oersted to Tesla
            B_Oe = 1000;
            B_T = unit_conversions(B_Oe, 'Oe', 'T');
            testCase.verifyEqual(B_T, 0.1, 'RelTol', 1e-10);
            
            % Tesla to milliTesla
            B_T = 1;
            B_mT = unit_conversions(B_T, 'T', 'mT');
            testCase.verifyEqual(B_mT, 1000, 'RelTol', 1e-10);
        end
        
        function testEnergyConversions(testCase)
            % Test energy unit conversions
            
            % eV to Joules
            E_eV = 1;
            E_J = unit_conversions(E_eV, 'eV', 'J');
            expected = 1.602176634e-19;
            testCase.verifyEqual(E_J, expected, 'RelTol', 1e-10);
            
            % Joules to eV
            E_J = 1.602176634e-19;
            E_eV = unit_conversions(E_J, 'J', 'eV');
            testCase.verifyEqual(E_eV, 1, 'RelTol', 1e-10);
            
            % meV to eV
            E_meV = 1000;
            E_eV = unit_conversions(E_meV, 'meV', 'eV');
            testCase.verifyEqual(E_eV, 1, 'RelTol', 1e-10);
        end
        
        function testLengthConversions(testCase)
            % Test length unit conversions
            
            % Meters to nanometers
            L_m = 1e-9;
            L_nm = unit_conversions(L_m, 'm', 'nm');
            testCase.verifyEqual(L_nm, 1, 'RelTol', 1e-10);
            
            % Nanometers to Angstroms
            L_nm = 1;
            L_A = unit_conversions(L_nm, 'nm', 'Angstrom');
            testCase.verifyEqual(L_A, 10, 'RelTol', 1e-10);
            
            % Micrometers to millimeters
            L_um = 1000;
            L_mm = unit_conversions(L_um, 'um', 'mm');
            testCase.verifyEqual(L_mm, 1, 'RelTol', 1e-10);
        end
        
        function testElectricalConversions(testCase)
            % Test electrical unit conversions
            
            % Amperes to milliAmperes
            I_A = 1;
            I_mA = unit_conversions(I_A, 'A', 'mA');
            testCase.verifyEqual(I_mA, 1000, 'RelTol', 1e-10);
            
            % Volts to milliVolts
            V_V = 1;
            V_mV = unit_conversions(V_V, 'V', 'mV');
            testCase.verifyEqual(V_mV, 1000, 'RelTol', 1e-10);
            
            % Ohms to kOhms
            R_Ohm = 1000;
            R_kOhm = unit_conversions(R_Ohm, 'Ohm', 'kOhm');
            testCase.verifyEqual(R_kOhm, 1, 'RelTol', 1e-10);
        end
        
        function testMultipleConversions(testCase)
            % Test multiple output conversions
            
            E_eV = 1.5;
            [E_J, E_erg] = unit_conversions(E_eV, 'eV', {'J', 'erg'}, 'multiple');
            
            % Check Joules conversion
            expected_J = 1.5 * 1.602176634e-19;
            testCase.verifyEqual(E_J, expected_J, 'RelTol', 1e-10);
            
            % Check erg conversion  
            expected_erg = expected_J * 1e7;
            testCase.verifyEqual(E_erg, expected_erg, 'RelTol', 1e-10);
        end
        
        function testIdentityConversions(testCase)
            % Test conversions to same unit
            
            value = 42.5;
            result = unit_conversions(value, 'T', 'T');
            testCase.verifyEqual(result, value, 'RelTol', 1e-15);
            
            result = unit_conversions(value, 'eV', 'eV');
            testCase.verifyEqual(result, value, 'RelTol', 1e-15);
        end
        
        function testArrayConversions(testCase)
            % Test array input conversions
            
            B_T = [1, 2, 3, 4, 5];
            B_Oe = unit_conversions(B_T, 'T', 'Oe');
            expected = B_T * 1e4;
            testCase.verifyEqual(B_Oe, expected, 'RelTol', 1e-10);
        end
        
        function testInvalidUnits(testCase)
            % Test error handling for invalid units
            
            testCase.verifyError(@() unit_conversions(1, 'invalid_unit', 'T'), ...
                                'MATLAB:error');
            
            testCase.verifyError(@() unit_conversions(1, 'T', 'invalid_unit'), ...
                                'MATLAB:error');
        end
        
        function testTimeConversions(testCase)
            % Test time unit conversions
            
            % Seconds to nanoseconds
            t_s = 1e-9;
            t_ns = unit_conversions(t_s, 's', 'ns');
            testCase.verifyEqual(t_ns, 1, 'RelTol', 1e-10);
            
            % Picoseconds to femtoseconds
            t_ps = 1;
            t_fs = unit_conversions(t_ps, 'ps', 'fs');
            testCase.verifyEqual(t_fs, 1000, 'RelTol', 1e-10);
        end
        
        function testFrequencyConversions(testCase)
            % Test frequency unit conversions
            
            % Hz to GHz
            f_Hz = 1e9;
            f_GHz = unit_conversions(f_Hz, 'Hz', 'GHz');
            testCase.verifyEqual(f_GHz, 1, 'RelTol', 1e-10);
            
            % MHz to kHz
            f_MHz = 1;
            f_kHz = unit_conversions(f_MHz, 'MHz', 'kHz');
            testCase.verifyEqual(f_kHz, 1000, 'RelTol', 1e-10);
        end
        
        function testConductanceConversions(testCase)
            % Test conductance unit conversions
            
            % Siemens to milliSiemens
            G_S = 1;
            G_mS = unit_conversions(G_S, 'S', 'mS');
            testCase.verifyEqual(G_mS, 1000, 'RelTol', 1e-10);
            
            % Conductance quantum
            G_S = 7.748091729e-5;
            G_G0 = unit_conversions(G_S, 'S', 'G0');
            testCase.verifyEqual(G_G0, 1, 'RelTol', 1e-6);
        end
        
        function testCurrentDensityConversions(testCase)
            % Test current density conversions
            
            % A/m² to A/cm²
            J_Am2 = 1e4;
            J_Acm2 = unit_conversions(J_Am2, 'A/m^2', 'A/cm^2');
            testCase.verifyEqual(J_Acm2, 1, 'RelTol', 1e-10);
        end
        
        function testElectricFieldConversions(testCase)
            % Test electric field conversions
            
            % V/m to V/cm
            E_Vm = 100;
            E_Vcm = unit_conversions(E_Vm, 'V/m', 'V/cm');
            testCase.verifyEqual(E_Vcm, 1, 'RelTol', 1e-10);
        end
        
        function testChargeConversions(testCase)
            % Test charge conversions
            
            % Elementary charge to Coulombs
            q_e = 1;
            q_C = unit_conversions(q_e, 'e', 'C');
            expected = 1.602176634e-19;
            testCase.verifyEqual(q_C, expected, 'RelTol', 1e-10);
        end
        
        function testMagneticMomentConversions(testCase)
            % Test magnetic moment conversions
            
            % Bohr magneton to A·m²
            mu_uB = 1;
            mu_Am2 = unit_conversions(mu_uB, 'uB', 'A*m^2');
            expected = 9.274010078e-24;
            testCase.verifyEqual(mu_Am2, expected, 'RelTol', 1e-9);
        end
        
        function testAreaVolumeConversions(testCase)
            % Test area and volume conversions
            
            % m² to cm²
            A_m2 = 1;
            A_cm2 = unit_conversions(A_m2, 'm^2', 'cm^2');
            testCase.verifyEqual(A_cm2, 1e4, 'RelTol', 1e-10);
            
            % Liters to m³
            V_L = 1;
            V_m3 = unit_conversions(V_L, 'L', 'm^3');
            testCase.verifyEqual(V_m3, 1e-3, 'RelTol', 1e-10);
        end
        
        function testZeroValues(testCase)
            % Test conversion of zero values
            
            result = unit_conversions(0, 'T', 'Oe');
            testCase.verifyEqual(result, 0);
            
            result = unit_conversions(0, 'eV', 'J');
            testCase.verifyEqual(result, 0);
        end
        
        function testNegativeValues(testCase)
            % Test conversion of negative values
            
            B_T = -1;
            B_Oe = unit_conversions(B_T, 'T', 'Oe');
            testCase.verifyEqual(B_Oe, -1e4, 'RelTol', 1e-10);
            
            E_eV = -2.5;
            E_J = unit_conversions(E_eV, 'eV', 'J');
            expected = -2.5 * 1.602176634e-19;
            testCase.verifyEqual(E_J, expected, 'RelTol', 1e-10);
        end
        
        function testLargeValues(testCase)
            % Test conversion of large values
            
            B_T = 1e6;
            B_Oe = unit_conversions(B_T, 'T', 'Oe');
            testCase.verifyEqual(B_Oe, 1e10, 'RelTol', 1e-10);
        end
        
        function testSmallValues(testCase)
            % Test conversion of very small values
            
            t_s = 1e-15;
            t_fs = unit_conversions(t_s, 's', 'fs');
            testCase.verifyEqual(t_fs, 1, 'RelTol', 1e-10);
        end
        
        function testComplexChainConversions(testCase)
            % Test chained conversions (T -> mT -> uT)
            
            B_T = 1;
            B_mT = unit_conversions(B_T, 'T', 'mT');
            B_uT = unit_conversions(B_mT, 'mT', 'uT');
            testCase.verifyEqual(B_uT, 1e6, 'RelTol', 1e-10);
        end
    end
    
    methods(TestMethodSetup)
        function addUtilsPath(testCase)
            % Add utils directory to path for testing
            current_dir = fileparts(mfilename('fullpath'));
            utils_dir = fullfile(current_dir, '..', 'core', 'utils');
            addpath(utils_dir);
        end
    end
    
    methods(TestMethodTeardown)
        function removeUtilsPath(testCase)
            % Remove utils directory from path after testing
            current_dir = fileparts(mfilename('fullpath'));
            utils_dir = fullfile(current_dir, '..', 'core', 'utils');
            rmpath(utils_dir);
        end
    end
end