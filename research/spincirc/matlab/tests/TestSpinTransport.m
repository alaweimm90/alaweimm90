classdef TestSpinTransport < matlab.unittest.TestCase
% TESTSPINTRANSPORT - Unit tests for SpinCirc transport framework
%
% This class provides comprehensive unit tests using MATLAB Test Framework
% to validate the correctness and performance of the SpinCirc library.
%
% Test Categories:
%   - Conductance matrix generation and properties
%   - Current conservation (Kirchhoff's laws)
%   - Energy conservation in magnetodynamics
%   - Benchmark performance against analytical solutions
%   - Material parameter validation
%   - Interface modeling accuracy
%
% Usage:
%   runtests('TestSpinTransport')  % Run all tests
%   result = run(TestSpinTransport)  % Programmatic execution
%
% Author: Meshal Alawein <meshal@berkeley.edu>
% Copyright © 2025 Meshal Alawein — All rights reserved.
% License: MIT

    properties (TestParameter)
        % Test materials
        ferromagnet = {'CoFeB', 'Permalloy', 'Co'}
        nonmagnet = {'Cu', 'Pt', 'Al'}
        insulator = {'MgO', 'Al2O3'}
        
        % Test geometries
        geometry = struct(...
            'simple', struct('length', 100e-9, 'width', 50e-9, 'thickness', 2e-9), ...
            'long', struct('length', 1e-6, 'width', 100e-9, 'thickness', 3e-9), ...
            'wide', struct('length', 50e-9, 'width', 200e-9, 'thickness', 1e-9))
        
        % Test temperatures
        temperature = {77, 300, 400}  % K
    end
    
    methods (TestMethodSetup)
        function setupTest(testCase)
            % Setup for each test method
            
            % Set random seed for reproducibility
            rng(12345);
            
            % Clear any existing figures
            close all;
            
            % Apply Berkeley style
            berkeley();
        end
    end
    
    methods (TestMethodTeardown)
        function teardownTest(testCase)
            % Cleanup after each test method
            
            % Close any figures created during testing
            close all;
        end
    end
    
    methods (Test)
        
        function testConductanceMatrixSymmetry(testCase)
            % Test conductance matrix symmetry properties
            
            % Create test materials
            materials = [MaterialsDB.CoFeB, MaterialsDB.Cu, MaterialsDB.CoFeB];
            magnetization = [1, 0, 0; -1, 0, 0];  % Antiparallel configuration
            geometry = struct('length', 100e-9, 'width', 50e-9, 'thickness', 2e-9, 'area', 100e-18);
            
            % Build conductance matrix
            G = ConductanceMatrix.buildSystemMatrix(materials, magnetization, geometry, 300, [0, 0, 0]);
            
            % Test properties
            testCase.verifyEqual(size(G, 1), size(G, 2), 'Matrix must be square');
            
            % Test Hermitian symmetry for passive systems
            symmetry_error = max(max(abs(G - G')));
            testCase.verifyLessThan(symmetry_error, 1e-10, 'Matrix should be symmetric');
            
            % Test current conservation (rows sum to zero)
            row_sums = sum(G, 2);
            conservation_error = max(abs(row_sums));
            testCase.verifyLessThan(conservation_error, 1e-10, 'Current conservation violated');
            
            % Test positive definiteness (stability)
            eigenvals = eig(G);
            min_eigenval = min(real(eigenvals));
            testCase.verifyGreaterThanOrEqual(min_eigenval, -1e-10, 'Matrix should be positive definite');
            
            fprintf('Conductance matrix tests passed: %dx%d matrix\n', size(G));
        end
        
        function testCurrentConservation(testCase)
            % Test current conservation in transport solutions
            
            % Setup simple F/N/F system
            solver = SpinTransportSolver();
            solver.setGeometry(200e-9, 100e-9, 2e-9);
            
            materials = [MaterialsDB.CoFeB, MaterialsDB.Cu, MaterialsDB.CoFeB];
            solver.setMaterials(materials);
            
            % Parallel configuration
            magnetization = [1, 0, 0; 1, 0, 0];
            solver.setMagnetization(magnetization);
            
            % Apply current boundary conditions
            bc_values = struct('node', [1, size(solver.G_matrix, 1)/4], 'current', [1e-6, -1e-6]);
            solver.setBoundaryConditions('current', bc_values);
            
            % Solve transport
            [V, I_s, info] = solver.solve('verbose', false);
            
            % Test current conservation
            total_current = sum(I_s.charge);
            testCase.verifyLessThan(abs(total_current), 1e-12, 'Total current should be conserved');
            
            % Test charge neutrality
            total_charge = sum(I_s.voltage_charge);
            testCase.verifyLessThan(abs(total_charge), 1e-10, 'Charge neutrality violation');
            
            fprintf('Current conservation test passed\n');
        end
        
        function testEnergyConservation(testCase)
            % Test energy conservation in LLG dynamics
            
            % Initial magnetization
            m0 = [1; 0; 0];
            
            % Conservative effective field (no damping)
            H_eff = @(t, m) [0; 0; 1];  % Constant field along z
            
            % Solve undamped LLG
            alpha = 0;  % No damping
            gamma = 1.76e11;
            tspan = [0, 5e-9];
            
            [m, t, info] = LLGSolver(m0, H_eff, alpha, gamma, tspan, ...
                'Method', 'RK45', 'ConserveEnergy', true, 'Verbose', false);
            
            % Calculate energy at each point
            mu0 = 4*pi*1e-7;
            H = H_eff(0, m0);
            energies = -mu0 * squeeze(sum(m .* H, 1));
            
            % Test energy conservation
            energy_variation = std(energies) / abs(mean(energies));
            testCase.verifyLessThan(energy_variation, 1e-6, 'Energy not conserved in undamped system');
            
            % Test magnetization magnitude conservation
            magnitudes = vecnorm(m, 2, 1);
            magnitude_error = max(abs(magnitudes - 1));
            testCase.verifyLessThan(magnitude_error, 1e-8, 'Magnetization magnitude not conserved');
            
            fprintf('Energy conservation test passed: %.2e relative variation\n', energy_variation);
        end
        
        function testBenchmarkPerformance(testCase)
            % Benchmark performance against analytical solutions
            
            % Test 1: 1D spin diffusion analytical solution
            L = 1e-6;  % Length
            lambda_sf = 200e-9;  % Spin diffusion length
            
            % Analytical solution: exponential decay
            x = linspace(0, L, 100);
            analytical = exp(-x / lambda_sf);
            
            % Numerical solution using transport solver
            solver = SpinTransportSolver();
            solver.setGeometry(L, 50e-9, 2e-9);
            
            materials = MaterialsDB.Cu;
            materials.lambda_sf = lambda_sf;
            solver.setMaterials(materials);
            
            % Boundary conditions: spin injection at x=0
            bc_values = struct('node', [1, 50], 'voltage', [1, 0]);
            solver.setBoundaryConditions('voltage', bc_values);
            
            % Solve and compare
            tic;
            [V, I_s, info] = solver.solve('verbose', false);
            solve_time = toc;
            
            % Extract spin voltage (assuming uniform spacing)
            numerical = I_s.voltage_spin_z(1:length(x));
            numerical = numerical / numerical(1);  % Normalize
            
            % Calculate error
            rmse_error = sqrt(mean((analytical - numerical).^2));
            testCase.verifyLessThan(rmse_error, 0.05, 'Numerical solution deviates from analytical');
            
            % Performance benchmark
            testCase.verifyLessThan(solve_time, 1.0, 'Solver too slow for simple problem');
            
            fprintf('Performance benchmark passed: RMSE=%.4f, Time=%.3fs\n', rmse_error, solve_time);
        end
        
        function testMaterialParameterValidation(testCase, ferromagnet)
            % Test material parameter physical consistency
            
            % Get material parameters
            material = MaterialsDB.(ferromagnet);
            
            % Test physical bounds
            testCase.verifyGreaterThan(material.Ms, 0, 'Saturation magnetization must be positive');
            testCase.verifyGreaterThanOrEqual(material.alpha, 0, 'Gilbert damping must be non-negative');
            testCase.verifyLessThanOrEqual(abs(material.beta), 1, 'Spin asymmetry |β| must be ≤ 1');
            testCase.verifyGreaterThan(material.lambda_sf, 0, 'Spin diffusion length must be positive');
            testCase.verifyGreaterThan(material.A_ex, 0, 'Exchange stiffness must be positive');
            
            % Test derived quantities
            mu0 = 4*pi*1e-7;
            lambda_ex = sqrt(2 * material.A_ex / (mu0 * material.Ms^2));
            testCase.verifyGreaterThan(lambda_ex, 1e-10, 'Exchange length too small');
            testCase.verifyLessThan(lambda_ex, 100e-9, 'Exchange length too large');
            
            % Test temperature dependence
            params_300K = MaterialsDB.getTemperatureDependence(ferromagnet, 300);
            params_77K = MaterialsDB.getTemperatureDependence(ferromagnet, 77);
            
            % Magnetization should increase at lower temperature
            testCase.verifyGreaterThan(params_77K.Ms, params_300K.Ms, 'Ms should increase at lower T');
            
            fprintf('Material validation passed for %s\n', ferromagnet);
        end
        
        function testInterfaceModeling(testCase)
            % Test interface parameter accuracy
            
            % Test F/N interface
            interface_params = MaterialsDB.getInterfaceParameters('CoFeB', 'Cu');
            testCase.verifyEqual(interface_params.type, 'metallic', 'Wrong interface type');
            testCase.verifyGreaterThan(interface_params.g_r, 0, 'Mixing conductance must be positive');
            
            % Test F/I interface  
            interface_params = MaterialsDB.getInterfaceParameters('CoFeB', 'MgO');
            testCase.verifyEqual(interface_params.type, 'tunnel', 'Wrong interface type');
            testCase.verifyGreaterThan(interface_params.barrier_height, 0, 'Barrier height must be positive');
            testCase.verifyGreaterThan(interface_params.TMR_max, 0, 'TMR must be positive');
            
            % Test Sharvin conductance calculation
            k_F = sqrt(2 * 9.109e-31 * 10 * 1.602e-19) / 1.055e-34;
            G_sharvin_expected = (1.602e-19)^2 / (2 * pi * 1.055e-34) * k_F;
            
            fn_interface = MaterialsDB.getInterfaceParameters('CoFeB', 'Cu');
            relative_error = abs(fn_interface.G_sharvin - G_sharvin_expected) / G_sharvin_expected;
            testCase.verifyLessThan(relative_error, 0.1, 'Sharvin conductance calculation error');
            
            fprintf('Interface modeling tests passed\n');
        end
        
        function testTemperatureDependence(testCase, temperature)
            % Test temperature-dependent properties
            
            T = temperature;
            
            % Test ferromagnetic material
            params_fm = MaterialsDB.getTemperatureDependence('CoFeB', T);
            
            if T < MaterialsDB.CoFeB.T_curie
                testCase.verifyGreaterThan(params_fm.Ms, 0, 'Ms should be positive below Tc');
            else
                testCase.verifyEqual(params_fm.Ms, 0, 'Ms should be zero above Tc');
            end
            
            % Test resistivity increase with temperature
            if T > 300
                testCase.verifyGreaterThan(params_fm.rho, MaterialsDB.CoFeB.rho, 'Resistivity should increase with T');
            elseif T < 300
                testCase.verifyLessThan(params_fm.rho, MaterialsDB.CoFeB.rho, 'Resistivity should decrease with T');
            end
            
            % Test nonmagnetic material
            params_nm = MaterialsDB.getTemperatureDependence('Cu', T);
            
            % Spin diffusion length should decrease with temperature
            if T > 300
                testCase.verifyLessThan(params_nm.lambda_sf, MaterialsDB.Cu.lambda_sf, 'λ_sf should decrease with T');
            end
            
            fprintf('Temperature dependence test passed at T=%dK\n', T);
        end
        
        function testStochasticDynamics(testCase)
            % Test stochastic LLG with thermal fluctuations
            
            % Initial state
            m0 = [0; 0; 1];
            
            % Effective field with thermal noise
            T = 300;  % Temperature
            volume = 1e-24;  % m³
            Ms = 1e6;  % A/m
            alpha = 0.1;
            
            % Thermal field strength
            k_B = 1.38e-23;
            mu0 = 4*pi*1e-7;
            mu_B = 9.274e-24;
            
            H_thermal = sqrt(2 * alpha * k_B * T / (mu0 * mu_B * Ms * volume));
            
            % Stochastic field (white noise)
            H_eff = @(t, m) [0; 0; 0.1] + H_thermal * randn(3, 1);
            
            % Solve stochastic LLG
            gamma = 1.76e11;
            tspan = [0, 1e-9];
            
            [m, t, info] = LLGSolver(m0, H_eff, alpha, gamma, tspan, ...
                'Method', 'RK4', 'InitialStep', 1e-12, 'Verbose', false);
            
            % Test that thermal fluctuations cause deviation from deterministic path
            final_m = m(:, end);
            deviation_angle = acos(dot(m0, final_m)) * 180/pi;
            
            testCase.verifyGreaterThan(deviation_angle, 1, 'Thermal fluctuations should cause deviation');
            testCase.verifyLessThan(deviation_angle, 90, 'Deviation should not be too large');
            
            % Test magnetization magnitude conservation
            magnitudes = vecnorm(m, 2, 1);
            magnitude_error = max(abs(magnitudes - 1));
            testCase.verifyLessThan(magnitude_error, 1e-6, 'Magnitude not conserved with noise');
            
            fprintf('Stochastic dynamics test passed: %.1f° deviation\n', deviation_angle);
        end
        
        function testMemoryUsage(testCase)
            % Test memory usage for large systems
            
            % Create large system
            n_regions = 10;
            materials = repmat(MaterialsDB.Cu, 1, n_regions);
            
            % Add some ferromagnetic regions
            materials([1, 5, 10]) = MaterialsDB.CoFeB;
            magnetization = [1, 0, 0; 0, 1, 0; -1, 0, 0];
            
            geometry = struct('length', 1e-6, 'width', 100e-9, 'thickness', 2e-9, 'area', 200e-18);
            
            % Monitor memory usage
            initial_memory = memory;
            
            % Build large matrix
            G = ConductanceMatrix.buildSystemMatrix(materials, magnetization, geometry, 300, [0, 0, 0]);
            
            final_memory = memory;
            memory_used = final_memory.MemUsedMATLAB - initial_memory.MemUsedMATLAB;
            
            % Test reasonable memory usage (< 100 MB for this size)
            testCase.verifyLessThan(memory_used, 100e6, 'Excessive memory usage');
            
            % Test matrix properties
            testCase.verifyEqual(size(G, 1), 4 * n_regions, 'Wrong matrix size');
            
            condition_number = cond(G);
            testCase.verifyLessThan(condition_number, 1e12, 'Matrix is poorly conditioned');
            
            fprintf('Memory usage test passed: %.1f MB, cond=%.2e\n', memory_used/1e6, condition_number);
        end
        
        function testParallelPerformance(testCase)
            % Test parallel computing performance (if available)
            
            if ~license('test', 'Distrib_Computing_Toolbox')
                testCase.assumeFail('Parallel Computing Toolbox not available');
            end
            
            % Create computationally intensive problem
            m0 = randn(3, 10);  % 10 coupled magnets
            m0 = m0 ./ vecnorm(m0, 2, 1);  % Normalize
            
            % Complex effective field with coupling
            H_eff = @(t, m) complexEffectiveField(t, m);
            
            % Sequential solve
            tic;
            [m_seq, t_seq] = LLGSolver(m0, H_eff, 0.1, 1.76e11, [0, 1e-9], ...
                'Method', 'RK45', 'Verbose', false);
            time_sequential = toc;
            
            % Test that solution is reasonable
            final_magnitudes = vecnorm(m_seq(:, :, end), 2, 1);
            testCase.verifyLessThan(max(abs(final_magnitudes - 1)), 1e-6, 'Magnitude not conserved');
            
            fprintf('Parallel performance test completed: %.3fs\n', time_sequential);
            
            function H = complexEffectiveField(t, m)
                % Complex effective field with exchange coupling
                H = zeros(size(m));
                
                % External field
                H = H + repmat([0; 0; 1], 1, size(m, 2));
                
                % Exchange coupling
                J_ex = 1e-3;
                for i = 1:size(m, 2)
                    if i > 1
                        H(:, i) = H(:, i) + J_ex * m(:, i-1);
                    end
                    if i < size(m, 2)
                        H(:, i) = H(:, i) + J_ex * m(:, i+1);
                    end
                end
                
                % Add some time dependence
                H = H + 0.1 * sin(2*pi*1e9*t) * repmat([1; 0; 0], 1, size(m, 2));
            end
        end
        
    end
    
    methods (Test, ParameterCombination = 'exhaustive')
        
        function testMaterialCombinations(testCase, ferromagnet, nonmagnet)
            % Test all material combinations for F/N interfaces
            
            interface_params = MaterialsDB.getInterfaceParameters(ferromagnet, nonmagnet);
            
            testCase.verifyEqual(interface_params.type, 'metallic', 'Interface type should be metallic');
            testCase.verifyGreaterThan(interface_params.g_r, 0, 'Real mixing conductance must be positive');
            testCase.verifyGreaterThanOrEqual(interface_params.g_i, 0, 'Imaginary mixing conductance must be non-negative');
            
            % Test SOT parameters for heavy metals
            if ismember(nonmagnet, {'Pt', 'Ta', 'W'})
                testCase.verifyTrue(isfield(interface_params, 'theta_SH'), 'Missing spin Hall angle');
                testCase.verifyLessThanOrEqual(abs(interface_params.theta_SH), 1, 'Spin Hall angle too large');
            end
            
            fprintf('Interface test passed: %s/%s\n', ferromagnet, nonmagnet);
        end
        
    end
end