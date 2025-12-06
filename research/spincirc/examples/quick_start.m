%% SpinCirc Quick Start Example
% This script demonstrates the basic usage of the SpinCirc framework
% for computational spintronics research.
%
% Author: Meshal Alawein <meshal@berkeley.edu>
% Copyright © 2025 Meshal Alawein — All rights reserved.
% License: MIT

%% Initialize SpinCirc
clear; clc; close all;

% Add SpinCirc to path and apply Berkeley styling
addpath(genpath('matlab'));
berkeley();

fprintf('=== SpinCirc Quick Start Example ===\n');
fprintf('Demonstrating basic spin transport simulation\n\n');

%% Example 1: Basic Spin Transport Simulation
fprintf('1. Setting up spin transport solver...\n');

% Create transport solver
solver = SpinTransportSolver();

% Set device geometry (200nm x 100nm x 2nm)
solver.setGeometry(200e-9, 100e-9, 2e-9);

% Define F/N/F heterostructure
materials = [MaterialsDB.CoFeB, MaterialsDB.Cu, MaterialsDB.CoFeB];
solver.setMaterials(materials);

% Set magnetization configuration (parallel)
magnetization = [1, 0, 0; 1, 0, 0];  % Both magnets along +x
solver.setMagnetization(magnetization);

% Apply voltage boundary conditions
bc_values = struct('node', [1, 3], 'voltage', [0.1, 0]);
solver.setBoundaryConditions('voltage', bc_values);

% Solve transport problem
fprintf('   Solving transport equations...\n');
[V, I_s, info] = solver.solve('verbose', true);

% Display results
fprintf('   Transport solution completed!\n');
fprintf('   Solve time: %.3f seconds\n', info.solve_time);
fprintf('   Residual: %.2e\n', info.residual);

%% Example 2: Calculate TMR
fprintf('\n2. Calculating tunnel magnetoresistance...\n');

[TMR, resistance] = solver.calculateTMR();
fprintf('   TMR: %.1f%%\n', TMR);
fprintf('   R_parallel: %.2f kΩ\n', resistance.parallel * 1e-3);
fprintf('   R_antiparallel: %.2f kΩ\n', resistance.antiparallel * 1e-3);

%% Example 3: Magnetization Dynamics
fprintf('\n3. Simulating magnetization dynamics...\n');

% Initial magnetization state
m0 = [1; 0; 0];

% Effective field (constant + small AC perturbation)
H_eff = @(t, m) [0; 0; 0.1] + 0.01*sin(2*pi*1e9*t)*[1; 0; 0];

% Gilbert damping and gyromagnetic ratio
alpha = 0.01;
gamma = 1.76e11;

% Time span (10 ns)
tspan = [0, 10e-9];

fprintf('   Solving LLG equation...\n');
[m, t, llg_info] = LLGSolver(m0, H_eff, alpha, gamma, tspan, 'Verbose', true);

fprintf('   LLG solution completed!\n');
fprintf('   Integration steps: %d\n', llg_info.steps_taken);
fprintf('   Energy conservation error: %.2e\n', llg_info.energy_conservation);

%% Example 4: Visualization
fprintf('\n4. Creating visualizations...\n');

% Plot transport solution
figure('Name', 'Spin Transport Results');
solver.plotSolution('component', 'charge');
title('Charge Voltage Distribution');

% Plot magnetization trajectory
figure('Name', 'Magnetization Dynamics');
subplot(2, 2, 1);
plot(t*1e9, squeeze(m(1, :)), 'LineWidth', 2);
xlabel('Time (ns)');
ylabel('m_x');
title('X Component');
grid on;

subplot(2, 2, 2);
plot(t*1e9, squeeze(m(2, :)), 'LineWidth', 2);
xlabel('Time (ns)');
ylabel('m_y');
title('Y Component');
grid on;

subplot(2, 2, 3);
plot(t*1e9, squeeze(m(3, :)), 'LineWidth', 2);
xlabel('Time (ns)');
ylabel('m_z');
title('Z Component');
grid on;

subplot(2, 2, 4);
plot3(squeeze(m(1, :)), squeeze(m(2, :)), squeeze(m(3, :)), 'LineWidth', 2);
xlabel('m_x'); ylabel('m_y'); zlabel('m_z');
title('3D Trajectory');
grid on;
axis equal;

sgtitle('Magnetization Dynamics', 'FontSize', 16, 'FontWeight', 'bold');

%% Example 5: Material Database
fprintf('\n5. Exploring material database...\n');

% List available materials
ferromagnets = MaterialsDB.listMaterials('ferromagnet');
nonmagnets = MaterialsDB.listMaterials('nonmagnet');

fprintf('   Available ferromagnets: %s\n', strjoin(ferromagnets, ', '));
fprintf('   Available nonmagnets: %s\n', strjoin(nonmagnets, ', '));

% Get temperature-dependent properties
CoFeB_300K = MaterialsDB.getTemperatureDependence('CoFeB', 300);
CoFeB_77K = MaterialsDB.getTemperatureDependence('CoFeB', 77);

fprintf('\n   CoFeB properties:\n');
fprintf('   At 300K: Ms = %.2e A/m, α = %.4f\n', CoFeB_300K.Ms, CoFeB_300K.alpha);
fprintf('   At 77K:  Ms = %.2e A/m, α = %.4f\n', CoFeB_77K.Ms, CoFeB_77K.alpha);

% Validate material parameters
fprintf('\n   Validating material parameters...\n');
MaterialsDB.validateMaterial('CoFeB');

%% Example 6: Device Simulation
fprintf('\n6. Running device simulation...\n');

% Create ASL inverter
fprintf('   Simulating All-Spin Logic inverter...\n');

% Note: This would typically create an ASLInverter object and run simulation
% For this quick start, we'll show the concept
fprintf('   ASL inverter simulation would be performed here\n');
fprintf('   (See matlab/devices/all_spin_logic/ for full implementation)\n');

%% Summary
fprintf('\n=== Quick Start Completed Successfully! ===\n');
fprintf('\nNext steps:\n');
fprintf('1. Explore validation examples in matlab/validation/\n');
fprintf('2. Check device implementations in matlab/devices/\n');
fprintf('3. Run comprehensive tests: runtests("matlab/tests")\n');
fprintf('4. Read documentation at docs/index.rst\n');
fprintf('\nFor support: https://github.com/alaweimm90/SpinCirc\n');

%% Save results (optional)
save_results = questdlg('Save simulation results?', 'SpinCirc', 'Yes', 'No', 'No');
if strcmp(save_results, 'Yes')
    save('quick_start_results.mat', 'V', 'I_s', 'm', 't', 'TMR', 'resistance', 'info', 'llg_info');
    fprintf('\nResults saved to quick_start_results.mat\n');
end

fprintf('\nThank you for using SpinCirc! 🚀\n');
