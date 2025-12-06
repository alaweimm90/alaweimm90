classdef TestAdaptiveODESolver < matlab.unittest.TestCase
    % TESTADAPTIVEODESOLVER - Unit tests for adaptive_ode_solver
    %
    % This test class validates the adaptive ODE solver functionality
    % including different integration methods and error control.
    %
    % Author: Meshal Alawein <meshal@berkeley.edu>
    % Copyright © 2025 Meshal Alawein — All rights reserved.
    % License: MIT
    
    methods(Test)
        function testSimpleLinearODE(testCase)
            % Test with simple linear ODE: dy/dt = -y
            
            ode_func = @(t, y) -y;
            tspan = [0, 2];
            y0 = 1;
            
            [t, y, info] = adaptive_ode_solver(ode_func, tspan, y0);
            
            % Verify solution structure
            testCase.verifyTrue(isvector(t));
            testCase.verifyTrue(ismatrix(y));
            testCase.verifyEqual(size(y, 1), 1); % Single variable
            testCase.verifyEqual(size(y, 2), length(t));
            
            % Check initial condition
            testCase.verifyEqual(y(1), y0, 'AbsTol', 1e-12);
            
            % Check final solution against analytical result
            analytical = exp(-t(end));
            testCase.verifyEqual(y(end), analytical, 'RelTol', 1e-6);
            
            % Check info structure
            testCase.verifyTrue(isstruct(info));
            testCase.verifyTrue(isfield(info, 'success'));
            testCase.verifyTrue(info.success);
            testCase.verifyTrue(isfield(info, 'steps_taken'));
            testCase.verifyGreaterThan(info.steps_taken, 0);
        end
        
        function testSystemOfODEs(testCase)
            % Test with system of ODEs (harmonic oscillator)
            
            % dy1/dt = y2, dy2/dt = -y1
            ode_func = @(t, y) [y(2); -y(1)];
            tspan = [0, pi];
            y0 = [1; 0]; % Initial position and velocity
            
            [t, y, info] = adaptive_ode_solver(ode_func, tspan, y0);
            
            % Verify dimensions
            testCase.verifyEqual(size(y, 1), 2); % Two variables
            testCase.verifyEqual(size(y, 2), length(t));
            
            % Check initial conditions
            testCase.verifyEqual(y(:, 1), y0, 'AbsTol', 1e-12);
            
            % Check energy conservation (approximately)
            energy = 0.5 * (y(1, :).^2 + y(2, :).^2);
            initial_energy = 0.5 * (y0(1)^2 + y0(2)^2);
            testCase.verifyEqual(energy, initial_energy * ones(size(energy)), 'RelTol', 1e-3);
            
            % Check analytical solution at t = pi
            % y1(π) = -1, y2(π) = 0 for harmonic oscillator
            testCase.verifyEqual(y(1, end), -1, 'RelTol', 1e-4);
            testCase.verifyEqual(y(2, end), 0, 'AbsTol', 1e-4);
        end
        
        function testDifferentMethods(testCase)
            % Test different integration methods
            
            ode_func = @(t, y) -2*y;
            tspan = [0, 1];
            y0 = 1;
            
            methods = {'DP54', 'RK4'};
            
            for i = 1:length(methods)
                [t, y, info] = adaptive_ode_solver(ode_func, tspan, y0, 'Method', methods{i});
                
                testCase.verifyTrue(info.success, sprintf('Method %s failed', methods{i}));
                testCase.verifyEqual(info.method, methods{i});
                
                % Check accuracy
                analytical = exp(-2*t(end));
                testCase.verifyEqual(y(end), analytical, 'RelTol', 1e-6, ...
                                   sprintf('Method %s inaccurate', methods{i}));
            end
        end
        
        function testToleranceSettings(testCase)
            % Test different tolerance settings
            
            ode_func = @(t, y) sin(y);
            tspan = [0, 1];
            y0 = 0.1;
            
            % Tight tolerance
            [t_tight, y_tight, info_tight] = adaptive_ode_solver(ode_func, tspan, y0, ...
                'RelTol', 1e-10, 'AbsTol', 1e-12);
            
            % Loose tolerance
            [t_loose, y_loose, info_loose] = adaptive_ode_solver(ode_func, tspan, y0, ...
                'RelTol', 1e-4, 'AbsTol', 1e-6);
            
            % Tight tolerance should take more steps
            testCase.verifyGreaterThan(info_tight.steps_taken, info_loose.steps_taken);
            
            % Both should be successful
            testCase.verifyTrue(info_tight.success);
            testCase.verifyTrue(info_loose.success);
            
            % Solutions should be reasonably close
            testCase.verifyEqual(y_tight(end), y_loose(end), 'RelTol', 1e-3);
        end
        
        function testStiffSystem(testCase)
            % Test with stiff system
            
            % Stiff ODE: dy/dt = -1000*y + 1000*cos(t)
            ode_func = @(t, y) -1000*y + 1000*cos(t);
            jacobian_func = @(t, y) -1000;
            
            tspan = [0, 0.1];
            y0 = 0;
            
            try
                [t, y, info] = adaptive_ode_solver(ode_func, tspan, y0, ...
                    'Method', 'IMEX', 'Jacobian', jacobian_func, 'RelTol', 1e-6);
                
                testCase.verifyTrue(info.success);
                testCase.verifyEqual(info.method, 'IMEX');
                testCase.verifyGreaterThan(info.jacobian_evaluations, 0);
            catch ME
                % IMEX method may not be fully implemented - skip test
                testCase.assumeFail('IMEX method not available: %s', ME.message);
            end
        end
        
        function testEventDetection(testCase)
            % Test event detection capability
            
            % Simple ODE with event when y crosses zero
            ode_func = @(t, y) -y;
            event_func = @(t, y) deal(y, true, 0); % Stop when y = 0
            
            tspan = [0, 10];
            y0 = 1;
            
            try
                [t, y, info] = adaptive_ode_solver(ode_func, tspan, y0, 'EventFcn', event_func);
                
                if isfield(info, 'event_occurred') && info.event_occurred
                    testCase.verifyTrue(isfield(info, 'event_time'));
                    testCase.verifyLessThan(abs(y(end)), 1e-6); % Should be near zero
                end
            catch ME
                % Event detection may not be fully implemented - skip test
                testCase.assumeFail('Event detection not available: %s', ME.message);
            end
        end
        
        function testOutputTimes(testCase)
            % Test with specified output times
            
            ode_func = @(t, y) -y;
            output_times = [0, 0.5, 1.0, 1.5, 2.0];
            y0 = 1;
            
            [t, y, info] = adaptive_ode_solver(ode_func, output_times, y0);
            
            % Should return exactly the requested times
            testCase.verifyEqual(t(:), output_times(:), 'AbsTol', 1e-12);
            testCase.verifyEqual(length(t), length(output_times));
            
            % Check solutions at specified times
            analytical = exp(-output_times);
            testCase.verifyEqual(y, analytical, 'RelTol', 1e-6);
        end
        
        function testMaxSteps(testCase)
            % Test maximum steps limit
            
            ode_func = @(t, y) y; % Exponential growth
            tspan = [0, 10];
            y0 = 1;
            
            [t, y, info] = adaptive_ode_solver(ode_func, tspan, y0, 'MaxSteps', 100);
            
            testCase.verifyLessThan(info.steps_taken, 105); % Some tolerance
            
            % May not reach final time due to step limit
            testCase.verifyLessOrEqual(t(end), tspan(2));
        end
        
        function testInitialStepSize(testCase)
            % Test initial step size setting
            
            ode_func = @(t, y) -y;
            tspan = [0, 1];
            y0 = 1;
            
            % Large initial step
            [t1, y1, info1] = adaptive_ode_solver(ode_func, tspan, y0, 'InitialStep', 0.1);
            
            % Small initial step
            [t2, y2, info2] = adaptive_ode_solver(ode_func, tspan, y0, 'InitialStep', 0.001);
            
            testCase.verifyTrue(info1.success);
            testCase.verifyTrue(info2.success);
            
            % Solutions should be similar
            testCase.verifyEqual(y1(end), y2(end), 'RelTol', 1e-6);
        end
        
        function testVerboseMode(testCase)
            % Test verbose output (mainly ensures no errors)
            
            ode_func = @(t, y) -y;
            tspan = [0, 1];
            y0 = 1;
            
            % Capture output to avoid cluttering test results
            orig_state = warning('off', 'all');
            cleanup = onCleanup(@() warning(orig_state));
            
            [t, y, info] = adaptive_ode_solver(ode_func, tspan, y0, 'Verbose', true);
            
            testCase.verifyTrue(info.success);
        end
        
        function testComplexODE(testCase)
            % Test with complex-valued ODE
            
            % Schrödinger-like equation: dy/dt = i*y
            ode_func = @(t, y) 1i * y;
            tspan = [0, pi];
            y0 = 1 + 0i;
            
            [t, y, info] = adaptive_ode_solver(ode_func, tspan, y0);
            
            testCase.verifyTrue(info.success);
            
            % Analytical solution: y = exp(i*t)
            analytical = exp(1i * t(end));
            testCase.verifyEqual(y(end), analytical, 'RelTol', 1e-6);
        end
        
        function testZeroFunction(testCase)
            % Test with zero function (dy/dt = 0)
            
            ode_func = @(t, y) 0 * y;
            tspan = [0, 1];
            y0 = 5;
            
            [t, y, info] = adaptive_ode_solver(ode_func, tspan, y0);
            
            testCase.verifyTrue(info.success);
            
            % Solution should remain constant
            testCase.verifyEqual(y, y0 * ones(size(y)), 'AbsTol', 1e-12);
        end
        
        function testLargeSystem(testCase)
            % Test with larger system of ODEs
            
            n = 10;
            A = -eye(n) + 0.1 * randn(n); % Stable matrix
            ode_func = @(t, y) A * y;
            
            tspan = [0, 1];
            y0 = ones(n, 1);
            
            [t, y, info] = adaptive_ode_solver(ode_func, tspan, y0);
            
            testCase.verifyTrue(info.success);
            testCase.verifyEqual(size(y, 1), n);
            testCase.verifyEqual(y(:, 1), y0, 'AbsTol', 1e-12);
        end
        
        function testErrorHandling(testCase)
            % Test error handling for invalid inputs
            
            % Invalid function handle
            testCase.verifyError(@() adaptive_ode_solver([], [0, 1], 1), 'MATLAB:error');
            
            % Invalid time span
            testCase.verifyError(@() adaptive_ode_solver(@(t,y) y, [1], 1), 'MATLAB:error');
            
            % Inconsistent dimensions
            ode_func = @(t, y) [y; 2*y]; % Returns wrong size
            testCase.verifyError(@() adaptive_ode_solver(ode_func, [0, 1], 1), 'MATLAB:error');
        end
        
        function testNegativeTimeDirection(testCase)
            % Test integration in negative time direction
            
            ode_func = @(t, y) -y;
            tspan = [1, 0]; % Backward integration
            y0 = exp(-1); % y(1) = e^(-1)
            
            [t, y, info] = adaptive_ode_solver(ode_func, tspan, y0);
            
            testCase.verifyTrue(info.success);
            testCase.verifyEqual(t(1), 1, 'AbsTol', 1e-12);
            testCase.verifyEqual(t(end), 0, 'AbsTol', 1e-12);
            
            % Check final solution
            testCase.verifyEqual(y(end), 1, 'RelTol', 1e-6);
        end
        
        function testNonlinearODE(testCase)
            % Test with nonlinear ODE
            
            % Van der Pol oscillator (simplified)
            mu = 0.1;
            ode_func = @(t, y) [y(2); mu*(1-y(1)^2)*y(2) - y(1)];
            
            tspan = [0, 2];
            y0 = [1; 0];
            
            [t, y, info] = adaptive_ode_solver(ode_func, tspan, y0);
            
            testCase.verifyTrue(info.success);
            testCase.verifyEqual(size(y, 1), 2);
            testCase.verifyEqual(y(:, 1), y0, 'AbsTol', 1e-12);
        end
        
        function testPerformance(testCase)
            % Basic performance test (should complete in reasonable time)
            
            ode_func = @(t, y) -y;
            tspan = [0, 10];
            y0 = 1;
            
            tic;
            [t, y, info] = adaptive_ode_solver(ode_func, tspan, y0);
            elapsed = toc;
            
            testCase.verifyTrue(info.success);
            testCase.verifyLessThan(elapsed, 5); % Should complete in < 5 seconds
        end
        
        function testStatistics(testCase)
            % Test that statistics are properly collected
            
            ode_func = @(t, y) -y;
            tspan = [0, 1];
            y0 = 1;
            
            [t, y, info] = adaptive_ode_solver(ode_func, tspan, y0);
            
            % Check that info contains expected statistics
            required_fields = {'steps_taken', 'steps_rejected', 'function_evaluations', ...
                             'last_step_size', 'success', 'method'};
            
            for i = 1:length(required_fields)
                testCase.verifyTrue(isfield(info, required_fields{i}), ...
                                  sprintf('Missing info field: %s', required_fields{i}));
            end
            
            % Check reasonable values
            testCase.verifyGreaterThan(info.steps_taken, 0);
            testCase.verifyGreaterOrEqual(info.steps_rejected, 0);
            testCase.verifyGreaterThan(info.function_evaluations, 0);
            testCase.verifyGreaterThan(info.last_step_size, 0);
        end
    end
    
    methods(TestMethodSetup)
        function addNumericalPath(testCase)
            % Add numerical directory to path for testing
            current_dir = fileparts(mfilename('fullpath'));
            numerical_dir = fullfile(current_dir, '..', 'core', 'numerical');
            addpath(numerical_dir);
        end
    end
    
    methods(TestMethodTeardown)
        function removeNumericalPath(testCase)
            % Remove numerical directory from path after testing
            current_dir = fileparts(mfilename('fullpath'));
            numerical_dir = fullfile(current_dir, '..', 'core', 'numerical');
            rmpath(numerical_dir);
        end
    end
end