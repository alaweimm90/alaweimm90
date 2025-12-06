function test_pde_solvers()
%TEST_PDE_SOLVERS Test suite for PDE solvers
%   Comprehensive tests for all PDE solving functionality
%
%   Author: Berkeley SciComp Team
%   Date: 2024
fprintf('Running PDE Solver Tests\n');
fprintf('========================\n\n');
% Add path to core modules
addpath('../core');
% Test results storage
test_results = struct();
all_passed = true;
try
    %% Test 1: Heat Equation Solver - Steady State
    fprintf('Test 1: Heat Equation - Steady State\n');
    fprintf('------------------------------------\n');
    % Setup domain
    x = linspace(0, 1, 51);
    domain.x = x;
    % Boundary conditions: u(0) = 0, u(1) = 0
    bc.dirichlet = containers.Map({0, 50}, {0, 0});
    % Thermal diffusivity
    alpha = 0.1;
    % Create solver
    heat_solver = HeatEquationSolver(domain, bc, alpha);
    % Test steady heat equation: -α∇²u = f(x)
    % Source: f(x) = π²sin(πx)
    % Analytical solution: u(x) = sin(πx)/α
    source_term = @(x) pi^2 * sin(pi * x);
    u_analytical = sin(pi * x) / alpha;
    result_steady = heat_solver.solve_steady(source_term);
    assert(result_steady.success, 'Steady heat equation should solve successfully');
    assert(length(result_steady.u) == length(x), 'Solution size should match domain');
    assert(strcmp(result_steady.pde_type, 'parabolic'), 'PDE type should be parabolic');
    % Check boundary conditions
    assert(abs(result_steady.u(1)) < 1e-10, 'Left boundary condition should be satisfied');
    assert(abs(result_steady.u(end)) < 1e-10, 'Right boundary condition should be satisfied');
    % Check accuracy
    error_steady = max(abs(result_steady.u' - u_analytical));
    assert(error_steady < 1e-6, 'Steady solution should be accurate');
    test_results.heat_steady = struct('passed', true, 'error', error_steady);
    fprintf('✓ Steady heat equation tests passed (error: %.6e)\n\n', error_steady);
    %% Test 2: Heat Equation Solver - Transient
    fprintf('Test 2: Heat Equation - Transient\n');
    fprintf('---------------------------------\n');
    % Test transient heat equation: ∂u/∂t = α∇²u
    % Initial condition: u(x,0) = sin(πx)
    % Analytical solution: u(x,t) = sin(πx)exp(-π²αt)
    initial_condition = @(x) sin(pi * x);
    T_final = 0.1;
    dt = 0.001;
    result_transient = heat_solver.solve_transient(initial_condition, ...
        'time_span', [0, T_final], 'dt', dt);
    assert(result_transient.success, 'Transient heat equation should solve successfully');
    assert(size(result_transient.u, 2) == length(x), 'Spatial dimension should match domain');
    assert(length(result_transient.t) > 1, 'Should have multiple time steps');
    % Check initial condition
    initial_error = max(abs(result_transient.u(1, :) - sin(pi * x)));
    assert(initial_error < 1e-12, 'Initial condition should be preserved');
    % Check analytical solution at final time
    u_analytical_final = sin(pi * x) * exp(-pi^2 * alpha * T_final);
    final_error = max(abs(result_transient.u(end, :) - u_analytical_final));
    assert(final_error < 0.01, 'Final solution should be reasonably accurate');
    % Check monotonic energy decay
    energy = zeros(size(result_transient.t));
    for i = 1:length(result_transient.t)
        energy(i) = trapz(x, result_transient.u(i, :).^2);
    end
    assert(all(diff(energy) <= 0), 'Energy should decrease monotonically');
    test_results.heat_transient = struct('passed', true, 'error', final_error);
    fprintf('✓ Transient heat equation tests passed (error: %.6e)\n\n', final_error);
    %% Test 3: Boundary Condition Types
    fprintf('Test 3: Different Boundary Condition Types\n');
    fprintf('------------------------------------------\n');
    % Test Neumann boundary conditions
    bc_neumann.dirichlet = containers.Map({0}, {0});  % u(0) = 0
    bc_neumann.neumann = containers.Map({50}, {0});    % du/dx(1) = 0
    heat_solver_neumann = HeatEquationSolver(domain, bc_neumann, alpha);
    result_neumann = heat_solver_neumann.solve_steady(source_term);
    assert(result_neumann.success, 'Neumann BC problem should solve successfully');
    assert(abs(result_neumann.u(1)) < 1e-10, 'Dirichlet BC should be satisfied');
    % Test non-zero Dirichlet conditions
    bc_nonzero.dirichlet = containers.Map({0, 50}, {1, 2});
    heat_solver_nonzero = HeatEquationSolver(domain, bc_nonzero, alpha);
    % Use constant source term for this test
    source_constant = @(x) ones(size(x));
    result_nonzero = heat_solver_nonzero.solve_steady(source_constant);
    assert(result_nonzero.success, 'Non-zero Dirichlet BC should solve successfully');
    assert(abs(result_nonzero.u(1) - 1) < 1e-10, 'Left Dirichlet BC should be satisfied');
    assert(abs(result_nonzero.u(end) - 2) < 1e-10, 'Right Dirichlet BC should be satisfied');
    test_results.boundary_conditions = struct('passed', true);
    fprintf('✓ Boundary condition tests passed\n\n');
    %% Test 4: Grid Convergence
    fprintf('Test 4: Grid Convergence Study\n');
    fprintf('------------------------------\n');
    % Test convergence with different grid sizes
    grid_sizes = [21, 41, 81];
    errors_convergence = zeros(size(grid_sizes));
    for i = 1:length(grid_sizes)
        n = grid_sizes(i);
        x_test = linspace(0, 1, n);
        domain_test.x = x_test;
        bc_test.dirichlet = containers.Map({0, n-1}, {0, 0});
        solver_test = HeatEquationSolver(domain_test, bc_test, alpha);
        result_test = solver_test.solve_steady(source_term);
        if result_test.success
            u_analytical_test = sin(pi * x_test) / alpha;
            errors_convergence(i) = max(abs(result_test.u' - u_analytical_test));
        else
            errors_convergence(i) = NaN;
        end
    end
    % Check that error decreases with grid refinement
    valid_errors = errors_convergence(~isnan(errors_convergence));
    assert(length(valid_errors) >= 2, 'Should have at least 2 valid solutions');
    assert(all(diff(valid_errors) < 0), 'Error should decrease with grid refinement');
    % Estimate convergence rate (should be approximately 2 for finite differences)
    if length(valid_errors) >= 2
        h1 = 1 / (grid_sizes(1) - 1);
        h2 = 1 / (grid_sizes(2) - 1);
        conv_rate = log(valid_errors(1) / valid_errors(2)) / log(h1 / h2);
        assert(conv_rate > 1.5 && conv_rate < 3, 'Convergence rate should be approximately 2');
        test_results.convergence = struct('passed', true, 'rate', conv_rate, ...
                                         'errors', errors_convergence);
        fprintf('✓ Grid convergence tests passed (rate: %.2f)\n\n', conv_rate);
    else
        test_results.convergence = struct('passed', true, 'rate', NaN);
        fprintf('✓ Grid convergence tests passed (insufficient data for rate)\n\n');
    end
    %% Test 5: Solver Properties and Methods
    fprintf('Test 5: Solver Properties and Methods\n');
    fprintf('-------------------------------------\n');
    % Test solver info
    info = heat_solver.get_solver_info();
    assert(strcmp(info.name, 'Heat Equation Solver'), 'Solver name should be correct');
    assert(strcmp(info.pde_type, 'parabolic'), 'PDE type should be parabolic');
    assert(info.dimension == 1, 'Dimension should be 1');
    assert(info.domain_size == length(x), 'Domain size should match');
    % Test domain validation
    try
        invalid_domain.x = [1, 0.5, 0];  % Not sorted
        HeatEquationSolver(invalid_domain, bc, alpha);
        assert(false, 'Should reject unsorted domain');
    catch ME
        assert(contains(ME.message, 'sorted'), 'Should catch unsorted domain');
    end
    % Test parameter validation
    try
        HeatEquationSolver(domain, bc, -1);  % Negative diffusivity
        assert(false, 'Should reject negative diffusivity');
    catch ME
        assert(contains(ME.message, 'diffusivity') || contains(ME.message, 'positive'), ...
               'Should catch negative diffusivity');
    end
    test_results.properties = struct('passed', true);
    fprintf('✓ Solver properties tests passed\n\n');
    %% Test 6: Time Stepping Schemes
    fprintf('Test 6: Different Time Stepping Schemes\n');
    fprintf('---------------------------------------\n');
    % Test different schemes if implemented
    schemes = {'implicit'};  % Add others if implemented: 'explicit', 'crank_nicolson'
    for i = 1:length(schemes)
        scheme = schemes{i};
        % Create solver with specific scheme
        if isprop(heat_solver, 'scheme') || isfield(heat_solver, 'scheme')
            heat_solver.scheme = scheme;
        end
        % Test with small time step for stability
        dt_test = 0.0001;
        result_scheme = heat_solver.solve_transient(initial_condition, ...
            'time_span', [0, 0.01], 'dt', dt_test);
        assert(result_scheme.success, sprintf('%s scheme should work', scheme));
        % Check that solution remains bounded
        assert(all(isfinite(result_scheme.u(:))), ...
               sprintf('%s scheme should produce finite values', scheme));
        fprintf('✓ %s scheme test passed\n', scheme);
    end
    test_results.time_schemes = struct('passed', true);
    fprintf('✓ Time stepping scheme tests passed\n\n');
    %% Test 7: Error Handling
    fprintf('Test 7: Error Handling\n');
    fprintf('---------------------\n');
    % Test empty domain
    try
        empty_domain.x = [];
        HeatEquationSolver(empty_domain, bc, alpha);
        assert(false, 'Should reject empty domain');
    catch ME
        assert(contains(ME.message, 'points') || contains(ME.message, 'size'), ...
               'Should catch empty domain');
    end
    % Test invalid boundary conditions
    try
        invalid_bc = struct();  % Empty BC
        solver_invalid = HeatEquationSolver(domain, invalid_bc, alpha);
        % This might not fail in constructor, but should fail in solve
        solver_invalid.solve_steady(source_term);
    catch ME
        % This is expected - invalid BC should cause problems
    end
    % Test negative time step
    try
        heat_solver.solve_transient(initial_condition, ...
            'time_span', [0, 1], 'dt', -0.01);
        assert(false, 'Should reject negative time step');
    catch ME
        assert(contains(ME.message, 'positive') || contains(ME.message, 'time'), ...
               'Should catch negative time step');
    end
    test_results.error_handling = struct('passed', true);
    fprintf('✓ Error handling tests passed\n\n');
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
    fprintf('✓ All PDE solver tests PASSED\n');
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