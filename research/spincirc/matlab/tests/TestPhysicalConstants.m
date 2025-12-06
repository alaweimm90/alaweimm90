classdef TestPhysicalConstants < matlab.unittest.TestCase
    % TESTPHYSICALCONSTANTS - Unit tests for physical_constants utility
    %
    % This test class validates the physical constants database including
    % fundamental constants, magnetic constants, and material properties.
    %
    % Author: Meshal Alawein <meshal@berkeley.edu>
    % Copyright © 2025 Meshal Alawein — All rights reserved.
    % License: MIT
    
    methods(Test)
        function testFundamentalConstants(testCase)
            % Test fundamental physical constants
            
            % Speed of light
            c = physical_constants('c');
            testCase.verifyEqual(c, 299792458, 'RelTol', 1e-15);
            
            % Planck constant
            h = physical_constants('h');
            testCase.verifyEqual(h, 6.62607015e-34, 'RelTol', 1e-15);
            
            % Reduced Planck constant
            hbar = physical_constants('hbar');
            testCase.verifyEqual(hbar, 1.054571817e-34, 'RelTol', 1e-15);
            
            % Elementary charge
            e = physical_constants('e');
            testCase.verifyEqual(e, 1.602176634e-19, 'RelTol', 1e-15);
            
            % Electron mass
            me = physical_constants('me');
            testCase.verifyEqual(me, 9.1093837015e-31, 'RelTol', 1e-9);
            
            % Boltzmann constant
            kB = physical_constants('kB');
            testCase.verifyEqual(kB, 1.380649e-23, 'RelTol', 1e-15);
        end
        
        function testMagneticConstants(testCase)
            % Test magnetic constants
            
            % Bohr magneton
            uB = physical_constants('uB');
            testCase.verifyEqual(uB, 9.274010078e-24, 'RelTol', 1e-9);
            
            % Nuclear magneton
            uN = physical_constants('uN');
            testCase.verifyEqual(uN, 5.050783746e-27, 'RelTol', 1e-9);
            
            % Electron g-factor
            gS = physical_constants('gS');
            testCase.verifyEqual(gS, 2.00231930436, 'RelTol', 1e-10);
            
            % Vacuum permeability
            mu0 = physical_constants('mu0');
            testCase.verifyEqual(mu0, 1.25663706212e-6, 'RelTol', 1e-10);
        end
        
        function testDerivedConstants(testCase)
            % Test derived constants
            
            % Magnetic flux quantum
            Phi0 = physical_constants('Phi0');
            h = physical_constants('h');
            e = physical_constants('e');
            expected = h / (2 * e);
            testCase.verifyEqual(Phi0, expected, 'RelTol', 1e-15);
            
            % Conductance quantum
            G0 = physical_constants('G0');
            expected = 2 * e^2 / h;
            testCase.verifyEqual(G0, expected, 'RelTol', 1e-15);
            
            % von Klitzing constant
            RK = physical_constants('RK');
            expected = h / e^2;
            testCase.verifyEqual(RK, expected, 'RelTol', 1e-15);
        end
        
        function testMaterialConstants(testCase)
            % Test material-specific constants
            
            % Iron constants
            Ms_Fe = physical_constants('Ms_Fe');
            testCase.verifyGreaterThan(Ms_Fe, 0);
            testCase.verifyLessThan(Ms_Fe, 2e6); % Reasonable upper bound
            
            alpha_Fe = physical_constants('alpha_Fe');
            testCase.verifyGreaterThan(alpha_Fe, 0);
            testCase.verifyLessThan(alpha_Fe, 1); % Damping should be < 1
            
            % CoFeB constants
            Ms_CoFeB = physical_constants('Ms_CoFeB');
            testCase.verifyGreaterThan(Ms_CoFeB, 0);
            
            alpha_CoFeB = physical_constants('alpha_CoFeB');
            testCase.verifyGreaterThan(alpha_CoFeB, 0);
            testCase.verifyLessThan(alpha_CoFeB, 1);
        end
        
        function testSpintronicParameters(testCase)
            % Test spintronic-specific parameters
            
            % Spin Hall angle
            theta_SH = physical_constants('theta_SH');
            testCase.verifyGreaterThan(theta_SH, 0);
            testCase.verifyLessThan(theta_SH, 1);
            
            % Spin-orbit torque efficiency
            beta_DL = physical_constants('beta_DL');
            testCase.verifyGreaterThan(beta_DL, 0);
            testCase.verifyLessThan(beta_DL, 1);
            
            % Spin polarization
            P_spin = physical_constants('P_spin');
            testCase.verifyGreaterThan(P_spin, 0);
            testCase.verifyLessThan(P_spin, 1);
        end
        
        function testMultipleConstants(testCase)
            % Test retrieving multiple constants
            
            [h, hbar, e] = physical_constants('h', 'hbar', 'e');
            
            testCase.verifyEqual(h, 6.62607015e-34, 'RelTol', 1e-15);
            testCase.verifyEqual(hbar, 1.054571817e-34, 'RelTol', 1e-15);
            testCase.verifyEqual(e, 1.602176634e-19, 'RelTol', 1e-15);
            
            % Verify hbar = h/(2*pi)
            testCase.verifyEqual(hbar, h/(2*pi), 'RelTol', 1e-15);
        end
        
        function testAllConstants(testCase)
            % Test getting all constants as structure
            
            all_const = physical_constants('all');
            testCase.verifyTrue(isstruct(all_const));
            
            % Check some required fields
            required_fields = {'c', 'h', 'hbar', 'e', 'me', 'kB', 'uB'};
            for i = 1:length(required_fields)
                testCase.verifyTrue(isfield(all_const, required_fields{i}), ...
                                  sprintf('Missing field: %s', required_fields{i}));
            end
            
            % Verify values match individual calls
            testCase.verifyEqual(all_const.c, physical_constants('c'));
            testCase.verifyEqual(all_const.h, physical_constants('h'));
            testCase.verifyEqual(all_const.e, physical_constants('e'));
        end
        
        function testCaseInsensitive(testCase)
            % Test case-insensitive constant names
            
            c1 = physical_constants('c');
            c2 = physical_constants('C');
            testCase.verifyEqual(c1, c2);
            
            e1 = physical_constants('e');
            e2 = physical_constants('E');
            testCase.verifyEqual(e1, e2);
        end
        
        function testInvalidConstant(testCase)
            % Test error handling for invalid constant names
            
            testCase.verifyError(@() physical_constants('invalid_constant'), ...
                                'MATLAB:error');
            
            testCase.verifyError(@() physical_constants(''), ...
                                'MATLAB:error');
        end
        
        function testThermalEnergies(testCase)
            % Test thermal energy constants
            
            kBT_300K = physical_constants('kBT_300K');
            kB = physical_constants('kB');
            expected = kB * 300;
            testCase.verifyEqual(kBT_300K, expected, 'RelTol', 1e-15);
            
            kBT_300K_eV = physical_constants('kBT_300K_eV');
            e = physical_constants('e');
            expected_eV = kBT_300K / e;
            testCase.verifyEqual(kBT_300K_eV, expected_eV, 'RelTol', 1e-15);
        end
        
        function testLatticeConstants(testCase)
            % Test lattice constants
            
            a_Fe = physical_constants('a_lattice_Fe');
            testCase.verifyGreaterThan(a_Fe, 2e-10);
            testCase.verifyLessThan(a_Fe, 4e-10);
            
            a_Si = physical_constants('a_lattice_Si');
            testCase.verifyGreaterThan(a_Si, 5e-10);
            testCase.verifyLessThan(a_Si, 6e-10);
        end
        
        function testExchangeCoupling(testCase)
            % Test exchange coupling constants
            
            J_Fe = physical_constants('J_exchange_Fe');
            testCase.verifyGreaterThan(J_Fe, 0);
            testCase.verifyLessThan(J_Fe, 1e-20); % Reasonable upper bound
            
            J_Co = physical_constants('J_exchange_Co');
            testCase.verifyGreaterThan(J_Co, 0);
            
            J_Ni = physical_constants('J_exchange_Ni');
            testCase.verifyGreaterThan(J_Ni, 0);
            
            % Co should have higher exchange than Ni
            testCase.verifyGreaterThan(J_Co, J_Ni);
        end
        
        function testAnisotropyConstants(testCase)
            % Test magnetic anisotropy constants
            
            Ku_Fe = physical_constants('Ku_Fe');
            testCase.verifyGreaterThan(Ku_Fe, 0);
            
            Ku_Co = physical_constants('Ku_Co');
            testCase.verifyGreaterThan(Ku_Co, 0);
            
            % Co should have much higher anisotropy than Fe
            testCase.verifyGreaterThan(Ku_Co, Ku_Fe);
        end
        
        function testSaturationMagnetization(testCase)
            % Test saturation magnetization values
            
            Ms_Fe = physical_constants('Ms_Fe');
            Ms_Co = physical_constants('Ms_Co');
            Ms_Ni = physical_constants('Ms_Ni');
            
            % All should be positive
            testCase.verifyGreaterThan(Ms_Fe, 0);
            testCase.verifyGreaterThan(Ms_Co, 0);
            testCase.verifyGreaterThan(Ms_Ni, 0);
            
            % Expected ordering: Fe > Co > Ni
            testCase.verifyGreaterThan(Ms_Fe, Ms_Co);
            testCase.verifyGreaterThan(Ms_Co, Ms_Ni);
        end
        
        function testDampingParameters(testCase)
            % Test Gilbert damping parameters
            
            alpha_Fe = physical_constants('alpha_Fe');
            alpha_Co = physical_constants('alpha_Co');
            alpha_Ni = physical_constants('alpha_Ni');
            alpha_Permalloy = physical_constants('alpha_Permalloy');
            
            % All should be small positive values
            damping_values = [alpha_Fe, alpha_Co, alpha_Ni, alpha_Permalloy];
            testCase.verifyTrue(all(damping_values > 0));
            testCase.verifyTrue(all(damping_values < 0.1));
            
            % Ni should have highest damping
            testCase.verifyGreaterThan(alpha_Ni, alpha_Fe);
            testCase.verifyGreaterThan(alpha_Ni, alpha_Co);
        end
        
        function testConversionFactors(testCase)
            % Test unit conversion factors
            
            eV_to_J = physical_constants('eV_to_J');
            e = physical_constants('e');
            testCase.verifyEqual(eV_to_J, e, 'RelTol', 1e-15);
            
            J_to_eV = physical_constants('J_to_eV');
            expected = 1 / e;
            testCase.verifyEqual(J_to_eV, expected, 'RelTol', 1e-15);
            
            % Test reciprocal relationship
            testCase.verifyEqual(eV_to_J * J_to_eV, 1, 'RelTol', 1e-15);
        end
        
        function testQuantumConstants(testCase)
            % Test quantum conductance units
            
            e2_h = physical_constants('e2_h');
            G0 = physical_constants('G0');
            testCase.verifyEqual(e2_h, G0, 'RelTol', 1e-15);
            
            h_e2 = physical_constants('h_e2');
            RK = physical_constants('RK');
            testCase.verifyEqual(h_e2, RK, 'RelTol', 1e-15);
            
            % Test reciprocal relationship
            testCase.verifyEqual(e2_h * h_e2, 1, 'RelTol', 1e-15);
        end
        
        function testMetadata(testCase)
            % Test metadata fields
            
            all_const = physical_constants('all');
            
            testCase.verifyTrue(isfield(all_const, 'source'));
            testCase.verifyTrue(isfield(all_const, 'timestamp'));
            testCase.verifyTrue(isfield(all_const, 'version'));
            
            testCase.verifyTrue(ischar(all_const.source) || isstring(all_const.source));
            testCase.verifyTrue(ischar(all_const.timestamp) || isstring(all_const.timestamp));
            testCase.verifyTrue(ischar(all_const.version) || isstring(all_const.version));
            
            % Version should be reasonable format
            testCase.verifyTrue(~isempty(regexp(all_const.version, '^\d+\.\d+', 'once')));
        end
        
        function testGyromagneticRatios(testCase)
            % Test gyromagnetic ratios
            
            gamma_e = physical_constants('gamma_e');
            gamma_p = physical_constants('gamma_p');
            gamma_default = physical_constants('gamma');
            
            testCase.verifyGreaterThan(gamma_e, 0);
            testCase.verifyGreaterThan(gamma_p, 0);
            
            % Electron gyromagnetic ratio should be much larger
            testCase.verifyGreaterThan(gamma_e, gamma_p);
            
            % Default gamma should equal electron value
            testCase.verifyEqual(gamma_default, gamma_e);
        end
        
        function testFineStructureConstant(testCase)
            % Test fine structure constant
            
            alpha = physical_constants('alpha');
            testCase.verifyGreaterThan(alpha, 0.007);
            testCase.verifyLessThan(alpha, 0.008);
            
            % Verify approximate value
            testCase.verifyEqual(alpha, 7.2973525693e-3, 'RelTol', 1e-9);
        end
        
        function testAvogadroNumber(testCase)
            % Test Avogadro's number
            
            NA = physical_constants('NA');
            testCase.verifyEqual(NA, 6.02214076e23, 'RelTol', 1e-15);
            
            % Should be very large
            testCase.verifyGreaterThan(NA, 6e23);
        end
        
        function testVacuumConstants(testCase)
            % Test vacuum permittivity and permeability
            
            eps0 = physical_constants('eps0');
            mu0 = physical_constants('mu0');
            c = physical_constants('c');
            
            % Test relationship: c = 1/sqrt(eps0 * mu0)
            calculated_c = 1 / sqrt(eps0 * mu0);
            testCase.verifyEqual(calculated_c, c, 'RelTol', 1e-10);
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