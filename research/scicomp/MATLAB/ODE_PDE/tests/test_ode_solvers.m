function test_ode_solvers()
%TEST_ODE_SOLVERS Test suite for ODE solvers
%   Comprehensive tests for all ODE solving functionality
%
%   Author: Berkeley SciComp Team
%   Date: 2024
fprintf('Running ODE Solver Tests\n');
fprintf('========================\n\n');
% Add path to core modules
addpath('../core');
% Test results storage
test_results = struct();
all_passed = true;
try
    %% Test 1: Explicit Euler Method
    fprintf('Test 1: Explicit Euler Method\n');
    fprintf('------------------------------\n');
    % Simple exponential decay: dy/dt = -2y, y(0) = 1
    % Analytical solution: y(t) = exp(-2t)
    lambda = 2;
    dydt = @(t, y) -lambda * y;
    analytical = @(t) exp(-lambda * t);
    solver_euler = ExplicitEuler();
    y0 = 1;
    t_span = [0, 1];
    dt = 0.01;
    result_euler = solver_euler.solve(dydt, y0, t_span, dt);
    % Check basic properties
    assert(result_euler.success, 'Euler solve should succeed');
    assert(length(result_euler.t) == length(result_euler.y), 'Time and solution vectors should have same length');
    assert(abs(result_euler.y(1) - y0) < 1e-12, 'Initial condition should be preserved');
    % Check accuracy (Euler is first-order, so error should be reasonable)
    y_exact = analytical(result_euler.t(end));
    error_euler = abs(result_euler.y(end) - y_exact);
    assert(error_euler < 0.1, 'Euler error should be reasonable for this step size');
    test_results.euler = struct('passed', true, 'error', error_euler);
    fprintf('✓ Explicit Euler tests passed (error: %.6f)\n\n', error_euler);
    %% Test 2: Runge-Kutta 4 Method
    fprintf('Test 2: Runge-Kutta 4 Method\n');
    fprintf('----------------------------\n');
    solver_rk4 = RungeKutta4();
    result_rk4 = solver_rk4.solve(dydt, y0, t_span, dt);
    % Check basic properties
    assert(result_rk4.success, 'RK4 solve should succeed');
    assert(length(result_rk4.t) == length(result_rk4.y), 'Time and solution vectors should have same length');
    % Check accuracy (RK4 is fourth-order, should be much more accurate)
    error_rk4 = abs(result_rk4.y(end) - y_exact);
    assert(error_rk4 < 1e-6, 'RK4 error should be very small');
    assert(error_rk4 < error_euler, 'RK4 should be more accurate than Euler');
    test_results.rk4 = struct('passed', true, 'error', error_rk4);
    fprintf('✓ RK4 tests passed (error: %.6e)\n\n', error_rk4);
    %% Test 3: System of ODEs (Harmonic Oscillator)
    fprintf('Test 3: System of ODEs - Harmonic Oscillator\n');
    fprintf('--------------------------------------------\n');
    % Harmonic oscillator: d²y/dt² + ω²y = 0
    % Convert to system: dy1/dt = y2, dy2/dt = -ω²y1
    omega = 2;
    harmonic_ode = @(t, y) [y(2); -omega^2 * y(1)];
    y0_harmonic = [1; 0];  % Initial position and velocity
    % Analytical solution: y(t) = cos(ωt), dy/dt(t) = -ω sin(ωt)
    analytical_harmonic = @(t) [cos(omega * t); -omega * sin(omega * t)];
    result_harmonic = solver_rk4.solve(harmonic_ode, y0_harmonic, [0, pi], 0.01);
    assert(result_harmonic.success, 'Harmonic oscillator solve should succeed');
    assert(size(result_harmonic.y, 2) == 2, 'Should have 2 solution components');
    % Check accuracy at final time
    y_exact_harmonic = analytical_harmonic(result_harmonic.t(end));
    error_harmonic = norm(result_harmonic.y(end, :)' - y_exact_harmonic);
    assert(error_harmonic < 1e-6, 'Harmonic oscillator error should be small');
    % Check energy conservation
    kinetic = 0.5 * result_harmonic.y(:, 2).^2;
    potential = 0.5 * omega^2 * result_harmonic.y(:, 1).^2;
    total_energy = kinetic + potential;
    energy_variation = std(total_energy) / mean(total_energy);
    assert(energy_variation < 1e-8, 'Energy should be conserved');
    test_results.harmonic = struct('passed', true, 'error', error_harmonic, ...
                                  'energy_variation', energy_variation);
    fprintf('✓ Harmonic oscillator tests passed (error: %.6e, energy variation: %.6e)\n\n', ...
            error_harmonic, energy_variation);
    %% Test 4: Convergence Order
    fprintf('Test 4: Convergence Order Analysis\n');
    fprintf('----------------------------------\n');
    % Test convergence rate for both Euler and RK4
    dt_values = [0.1, 0.05, 0.025, 0.0125];
    errors_euler = zeros(size(dt_values));
    errors_rk4 = zeros(size(dt_values));
    t_final = 1;
    y_exact_final = analytical(t_final);
    for i = 1:length(dt_values)
        dt_test = dt_values(i);
        % Euler
        result_e = solver_euler.solve(dydt, y0, [0, t_final], dt_test);
        if result_e.success
            errors_euler(i) = abs(result_e.y(end) - y_exact_final);
        else
            errors_euler(i) = NaN;
        end
        % RK4
        result_r = solver_rk4.solve(dydt, y0, [0, t_final], dt_test);
        if result_r.success
            errors_rk4(i) = abs(result_r.y(end) - y_exact_final);
        else
            errors_rk4(i) = NaN;
        end
    end
    % Check convergence rates
    % Euler should be approximately first-order
    rate_euler = compute_convergence_rate(dt_values, errors_euler);
    assert(abs(rate_euler - 1.0) < 0.5, 'Euler convergence rate should be approximately 1');
    % RK4 should be approximately fourth-order
    rate_rk4 = compute_convergence_rate(dt_values, errors_rk4);
    assert(abs(rate_rk4 - 4.0) < 1.0, 'RK4 convergence rate should be approximately 4');
    test_results.convergence = struct('passed', true, 'euler_rate', rate_euler, ...
                                     'rk4_rate', rate_rk4);
    fprintf('✓ Convergence tests passed (Euler rate: %.2f, RK4 rate: %.2f)\n\n', ...
            rate_euler, rate_rk4);
    %% Test 5: Error Handling
    fprintf('Test 5: Error Handling\n');
    fprintf('---------------------\n');
    % Test with invalid inputs
    try
        % Invalid function handle
        solver_rk4.solve('not_a_function', y0, t_span, dt);
        assert(false, 'Should throw error for invalid function');
    catch ME
        assert(contains(ME.message, 'function_handle'), 'Should catch invalid function handle');
    end
    try
        % Invalid time span
        solver_rk4.solve(dydt, y0, [1, 0], dt);  % t_final < t_initial
        assert(false, 'Should throw error for invalid time span');
    catch ME
        assert(contains(ME.message, 'time'), 'Should catch invalid time span');
    end
    try
        % Zero time step
        solver_rk4.solve(dydt, y0, t_span, 0);
        assert(false, 'Should throw error for zero time step');
    catch ME
        assert(contains(ME.message, 'zero'), 'Should catch zero time step');
    end
    test_results.error_handling = struct('passed', true);
    fprintf('✓ Error handling tests passed\n\n');
    %% Test 6: Solver Properties
    fprintf('Test 6: Solver Properties\n');
    fprintf('-------------------------\n');
    % Test solver info
    info_euler = solver_euler.get_solver_info();
    assert(strcmp(info_euler.name, 'Explicit Euler'), 'Euler name should be correct');
    assert(info_euler.order == 1, 'Euler order should be 1');
    info_rk4 = solver_rk4.get_solver_info();
    assert(strcmp(info_rk4.name, 'Runge-Kutta 4'), 'RK4 name should be correct');
    assert(info_rk4.order == 4, 'RK4 order should be 4');
    % Test reset functionality
    solver_rk4.reset();
    assert(solver_rk4.step_count == 0, 'Step count should reset to 0');
    test_results.properties = struct('passed', true);
    fprintf('✓ Solver properties tests passed\n\n');
    %% Test 7: Special Cases
    fprintf('Test 7: Special Cases\n');
    fprintf('--------------------\n');
    % Zero time span
    result_zero = solver_rk4.solve(dydt, y0, [0, 0], dt);
    assert(result_zero.success, 'Zero time span should succeed');
    assert(length(result_zero.y) == 1, 'Should return only initial condition');
    assert(result_zero.y(1) == y0, 'Should preserve initial condition');
    % Very small time step
    result_small = solver_rk4.solve(dydt, y0, [0, 0.001], 1e-6);
    assert(result_small.success, 'Very small time step should succeed');
    assert(length(result_small.y) > 100, 'Should take many small steps');
    test_results.special_cases = struct('passed', true);
    fprintf('✓ Special cases tests passed\n\n');
catch ME
    all_passed = false;
    fprintf('❌ Test failed: %s\n', ME.message);
    fprintf('Stack trace:\n');
    for i = 1:length(ME.stack)
        fprintf('  %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
end
%% Summary
fprintf('=== Test Summary ===\n');
if all_passed
    fprintf('✓ All ODE solver tests PASSED\n');
    % Display test results
    test_fields = fieldnames(test_results);
    for i = 1:length(test_fields)
        result = test_results.(test_fields{i});
        if result.passed
            fprintf('  %s: PASSED\n', test_fields{i});
        end
    end
else
    fprintf('❌ Some tests FAILED\n');
end
fprintf('\nTest completed.\n');
end
function rate = compute_convergence_rate(dt_values, errors)
%COMPUTE_CONVERGENCE_RATE Compute convergence rate from errors
    valid_idx = ~isnan(errors) & errors > 0;
    if sum(valid_idx) < 2
        rate = NaN;
        return;
    end
    dt_valid = dt_values(valid_idx);
    errors_valid = errors(valid_idx);
    % Compute rates between consecutive points
    rates = [];
    for i = 2:length(errors_valid)
        rate_i = log(errors_valid(i-1) / errors_valid(i)) / log(dt_valid(i-1) / dt_valid(i));
        if isfinite(rate_i)
            rates = [rates, rate_i];
        end
    end
    if isempty(rates)
        rate = NaN;
    else
        rate = mean(rates);
    end
end