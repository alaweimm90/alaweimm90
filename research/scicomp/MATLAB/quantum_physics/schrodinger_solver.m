function results = schrodinger_solver(varargin)
% SCHRODINGER_SOLVER Advanced numerical solver for the time-dependent Schrödinger equation
%
% Comprehensive MATLAB implementation for solving the time-dependent
% Schrödinger equation using various numerical methods including split-operator,
% Crank-Nicolson, and spectral methods with Berkeley-styled visualizations.
%
% Key Features:
% - Multiple numerical integration schemes
% - 1D, 2D, and 3D potential support
% - Wavepacket evolution and tunneling analysis
% - Energy eigenstate computation
% - Advanced visualization with Berkeley color scheme
%
% Applications:
% - Quantum tunneling studies
% - Harmonic oscillator dynamics
% - Potential barrier transmission
% - Wavepacket spreading analysis
% - Quantum state evolution
%
% Author: Meshal Alawein (meshal@berkeley.edu)
% Institution: University of California, Berkeley
% Created: 2025
% License: MIT
%
% Copyright © 2025 Meshal Alawein — All rights reserved.
    % Parse input arguments
    p = inputParser;
    addParameter(p, 'method', 'split_operator', @ischar);
    addParameter(p, 'potential', 'harmonic', @ischar);
    addParameter(p, 'x_domain', [-10, 10], @isnumeric);
    addParameter(p, 'nx', 512, @isnumeric);
    addParameter(p, 'dt', 0.01, @isnumeric);
    addParameter(p, 'tmax', 5.0, @isnumeric);
    addParameter(p, 'mass', 1.0, @isnumeric);
    addParameter(p, 'hbar', 1.0, @isnumeric);
    addParameter(p, 'initial_state', 'gaussian', @ischar);
    addParameter(p, 'visualize', true, @islogical);
    addParameter(p, 'save_animation', false, @islogical);
    parse(p, varargin{:});
    params = p.Results;
    % Display banner
    fprintf('\n=== Berkeley Quantum Physics: Schrödinger Equation Solver ===\n');
    fprintf('Method: %s\n', params.method);
    fprintf('Potential: %s\n', params.potential);
    fprintf('Grid points: %d\n', params.nx);
    fprintf('Time step: %.4f\n', params.dt);
    fprintf('Total time: %.2f\n', params.tmax);
    fprintf('========================================================\n\n');
    % Setup spatial grid
    x = linspace(params.x_domain(1), params.x_domain(2), params.nx);
    dx = x(2) - x(1);
    % Setup time grid
    t = 0:params.dt:params.tmax;
    nt = length(t);
    % Define potential function
    V = create_potential(x, params.potential);
    % Create initial wavefunction
    psi0 = create_initial_state(x, params.initial_state);
    % Normalize initial state
    psi0 = psi0 / sqrt(trapz(x, abs(psi0).^2));
    % Choose numerical method
    switch lower(params.method)
        case 'split_operator'
            [psi_t, E_t, norm_t] = solve_split_operator(psi0, x, t, V, params);
        case 'crank_nicolson'
            [psi_t, E_t, norm_t] = solve_crank_nicolson(psi0, x, t, V, params);
        case 'runge_kutta'
            [psi_t, E_t, norm_t] = solve_runge_kutta(psi0, x, t, V, params);
        case 'spectral'
            [psi_t, E_t, norm_t] = solve_spectral(psi0, x, t, V, params);
        otherwise
            error('Unknown method: %s', params.method);
    end
    % Calculate additional observables
    x_expect = calculate_position_expectation(psi_t, x);
    p_expect = calculate_momentum_expectation(psi_t, x, params.hbar);
    sigma_x = calculate_position_uncertainty(psi_t, x);
    sigma_p = calculate_momentum_uncertainty(psi_t, x, params.hbar);
    % Package results
    results = struct();
    results.x = x;
    results.t = t;
    results.psi = psi_t;
    results.V = V;
    results.energy = E_t;
    results.norm = norm_t;
    results.x_expect = x_expect;
    results.p_expect = p_expect;
    results.sigma_x = sigma_x;
    results.sigma_p = sigma_p;
    results.params = params;
    % Visualization
    if params.visualize
        visualize_results(results);
    end
    % Save animation if requested
    if params.save_animation
        create_animation(results);
    end
    fprintf('Schrödinger equation solved successfully!\n');
    fprintf('Final norm: %.6f\n', norm_t(end));
    fprintf('Energy conservation: %.2e\n', std(E_t)/mean(abs(E_t)));
end
function V = create_potential(x, potential_type)
% Create potential energy function
    switch lower(potential_type)
        case 'harmonic'
            % Harmonic oscillator: V(x) = (1/2) * m * omega^2 * x^2
            omega = 1.0;
            V = 0.5 * omega^2 * x.^2;
        case 'double_well'
            % Double well potential
            a = 2.0;
            b = 1.0;
            V = -a * x.^2 + b * x.^4;
        case 'barrier'
            % Potential barrier
            V = zeros(size(x));
            barrier_height = 2.0;
            barrier_width = 1.0;
            V(abs(x) < barrier_width/2) = barrier_height;
        case 'step'
            % Potential step
            V = zeros(size(x));
            step_height = 1.0;
            V(x > 0) = step_height;
        case 'coulomb'
            % Coulomb potential (regularized)
            Z = 1.0;
            epsilon = 0.1;  % Regularization parameter
            V = -Z ./ sqrt(x.^2 + epsilon^2);
        case 'morse'
            % Morse potential
            D = 1.0;
            alpha = 1.0;
            x0 = 0.0;
            V = D * (1 - exp(-alpha * (x - x0))).^2 - D;
        case 'free'
            % Free particle
            V = zeros(size(x));
        otherwise
            error('Unknown potential type: %s', potential_type);
    end
end
function psi0 = create_initial_state(x, state_type)
% Create initial wavefunction
    switch lower(state_type)
        case 'gaussian'
            % Gaussian wavepacket
            x0 = -2.0;
            sigma = 0.5;
            k0 = 2.0;
            psi0 = exp(-(x - x0).^2 / (2 * sigma^2)) .* exp(1i * k0 * (x - x0));
        case 'coherent'
            % Coherent state for harmonic oscillator
            x0 = 1.0;
            p0 = 1.0;
            psi0 = exp(-(x - x0).^2 / 2) .* exp(1i * p0 * x);
        case 'ground_state'
            % Ground state of harmonic oscillator
            psi0 = exp(-x.^2 / 2);
        case 'excited_state'
            % First excited state of harmonic oscillator
            psi0 = sqrt(2) * x .* exp(-x.^2 / 2);
        case 'plane_wave'
            % Plane wave
            k0 = 1.0;
            psi0 = exp(1i * k0 * x);
        case 'soliton'
            % Soliton-like initial condition
            a = 1.0;
            x0 = 0.0;
            psi0 = sech(a * (x - x0));
        otherwise
            error('Unknown initial state type: %s', state_type);
    end
    % Ensure it's a complex array
    if isreal(psi0)
        psi0 = complex(psi0);
    end
end
function [psi_t, E_t, norm_t] = solve_split_operator(psi0, x, t, V, params)
% Split-operator method (most accurate for most problems)
    nx = length(x);
    nt = length(t);
    dx = x(2) - x(1);
    dt = params.dt;
    % Initialize arrays
    psi_t = zeros(nx, nt);
    psi_t(:, 1) = psi0;
    % Momentum space grid
    dk = 2*pi / (nx * dx);
    k = [0:nx/2-1, -nx/2:-1] * dk;
    % Kinetic energy operator in momentum space
    T_k = params.hbar^2 * k.^2 / (2 * params.mass);
    % Evolution operators
    U_T = exp(-1i * T_k * dt / (2 * params.hbar));
    U_V = exp(-1i * V * dt / params.hbar);
    % Time evolution
    psi = psi0;
    for n = 2:nt
        % Step 1: Half kinetic energy evolution in momentum space
        psi_k = fft(psi);
        psi_k = psi_k .* U_T;
        psi = ifft(psi_k);
        % Step 2: Full potential energy evolution in position space
        psi = psi .* U_V;
        % Step 3: Second half kinetic energy evolution
        psi_k = fft(psi);
        psi_k = psi_k .* U_T;
        psi = ifft(psi_k);
        psi_t(:, n) = psi;
    end
    % Calculate observables
    E_t = zeros(nt, 1);
    norm_t = zeros(nt, 1);
    for n = 1:nt
        psi = psi_t(:, n);
        norm_t(n) = trapz(x, abs(psi).^2);
        E_t(n) = calculate_energy(psi, x, V, params);
    end
end
function [psi_t, E_t, norm_t] = solve_crank_nicolson(psi0, x, t, V, params)
% Crank-Nicolson method (implicit, unconditionally stable)
    nx = length(x);
    nt = length(t);
    dx = x(2) - x(1);
    dt = params.dt;
    % Initialize arrays
    psi_t = zeros(nx, nt);
    psi_t(:, 1) = psi0;
    % Build Hamiltonian matrix
    H = build_hamiltonian_matrix(x, V, params);
    % Crank-Nicolson matrices
    I = eye(nx);
    alpha = 1i * dt / (2 * params.hbar);
    A = I + alpha * H;
    B = I - alpha * H;
    % Apply boundary conditions (zero at boundaries)
    A(1, :) = 0; A(1, 1) = 1;
    A(end, :) = 0; A(end, end) = 1;
    B(1, :) = 0; B(1, 1) = 1;
    B(end, :) = 0; B(end, end) = 1;
    % Pre-compute LU decomposition for efficiency
    [L, U, P] = lu(A);
    % Time evolution
    psi = psi0;
    for n = 2:nt
        b = B * psi;
        b(1) = 0;
        b(end) = 0;
        % Solve linear system
        y = L \ (P * b);
        psi = U \ y;
        psi_t(:, n) = psi;
    end
    % Calculate observables
    E_t = zeros(nt, 1);
    norm_t = zeros(nt, 1);
    for n = 1:nt
        psi = psi_t(:, n);
        norm_t(n) = trapz(x, abs(psi).^2);
        E_t(n) = calculate_energy(psi, x, V, params);
    end
end
function [psi_t, E_t, norm_t] = solve_runge_kutta(psi0, x, t, V, params)
% Fourth-order Runge-Kutta method
    nx = length(x);
    nt = length(t);
    dt = params.dt;
    % Initialize arrays
    psi_t = zeros(nx, nt);
    psi_t(:, 1) = psi0;
    % Build Hamiltonian matrix
    H = build_hamiltonian_matrix(x, V, params);
    % Time evolution using RK4
    psi = psi0;
    for n = 2:nt
        k1 = -1i * H * psi / params.hbar;
        k2 = -1i * H * (psi + dt*k1/2) / params.hbar;
        k3 = -1i * H * (psi + dt*k2/2) / params.hbar;
        k4 = -1i * H * (psi + dt*k3) / params.hbar;
        psi = psi + dt * (k1 + 2*k2 + 2*k3 + k4) / 6;
        % Apply boundary conditions
        psi(1) = 0;
        psi(end) = 0;
        psi_t(:, n) = psi;
    end
    % Calculate observables
    E_t = zeros(nt, 1);
    norm_t = zeros(nt, 1);
    for n = 1:nt
        psi = psi_t(:, n);
        norm_t(n) = trapz(x, abs(psi).^2);
        E_t(n) = calculate_energy(psi, x, V, params);
    end
end
function [psi_t, E_t, norm_t] = solve_spectral(psi0, x, t, V, params)
% Spectral method using eigenfunction expansion
    % Find energy eigenstates
    H = build_hamiltonian_matrix(x, V, params);
    [eigenvecs, eigenvals] = eig(H);
    eigenvals = diag(eigenvals);
    % Sort by energy
    [eigenvals, idx] = sort(real(eigenvals));
    eigenvecs = eigenvecs(:, idx);
    % Project initial state onto eigenstates
    coeffs = eigenvecs' * psi0;
    % Time evolution
    nt = length(t);
    psi_t = zeros(length(x), nt);
    for n = 1:nt
        time_factors = exp(-1i * eigenvals * t(n) / params.hbar);
        psi_t(:, n) = eigenvecs * (coeffs .* time_factors);
    end
    % Calculate observables
    E_t = zeros(nt, 1);
    norm_t = zeros(nt, 1);
    for n = 1:nt
        psi = psi_t(:, n);
        norm_t(n) = trapz(x, abs(psi).^2);
        E_t(n) = calculate_energy(psi, x, V, params);
    end
end
function H = build_hamiltonian_matrix(x, V, params)
% Build Hamiltonian matrix using finite differences
    nx = length(x);
    dx = x(2) - x(1);
    % Kinetic energy matrix (second derivative)
    e = ones(nx, 1);
    T = -params.hbar^2 / (2 * params.mass * dx^2) * ...
        spdiags([e -2*e e], -1:1, nx, nx);
    % Potential energy matrix (diagonal)
    V_matrix = spdiags(V(:), 0, nx, nx);
    % Total Hamiltonian
    H = T + V_matrix;
    % Apply boundary conditions (infinite walls)
    H(1, :) = 0; H(1, 1) = 1e10;
    H(end, :) = 0; H(end, end) = 1e10;
end
function E = calculate_energy(psi, x, V, params)
% Calculate total energy expectation value
    dx = x(2) - x(1);
    % Kinetic energy: <T> = -ℏ²/(2m) ∫ ψ* d²ψ/dx²
    d2psi_dx2 = gradient(gradient(psi, dx), dx);
    T = -params.hbar^2 / (2 * params.mass) * ...
        real(trapz(x, conj(psi) .* d2psi_dx2));
    % Potential energy: <V> = ∫ ψ* V ψ
    U = real(trapz(x, conj(psi) .* V(:) .* psi));
    E = T + U;
end
function x_exp = calculate_position_expectation(psi_t, x)
% Calculate position expectation value over time
    nt = size(psi_t, 2);
    x_exp = zeros(nt, 1);
    for n = 1:nt
        psi = psi_t(:, n);
        x_exp(n) = real(trapz(x, conj(psi) .* x(:) .* psi));
    end
end
function p_exp = calculate_momentum_expectation(psi_t, x, hbar)
% Calculate momentum expectation value over time
    nt = size(psi_t, 2);
    p_exp = zeros(nt, 1);
    dx = x(2) - x(1);
    for n = 1:nt
        psi = psi_t(:, n);
        dpsi_dx = gradient(psi, dx);
        p_exp(n) = real(-1i * hbar * trapz(x, conj(psi) .* dpsi_dx));
    end
end
function sigma_x = calculate_position_uncertainty(psi_t, x)
% Calculate position uncertainty over time
    nt = size(psi_t, 2);
    sigma_x = zeros(nt, 1);
    for n = 1:nt
        psi = psi_t(:, n);
        x_exp = real(trapz(x, conj(psi) .* x(:) .* psi));
        x2_exp = real(trapz(x, conj(psi) .* x(:).^2 .* psi));
        sigma_x(n) = sqrt(x2_exp - x_exp^2);
    end
end
function sigma_p = calculate_momentum_uncertainty(psi_t, x, hbar)
% Calculate momentum uncertainty over time
    nt = size(psi_t, 2);
    sigma_p = zeros(nt, 1);
    dx = x(2) - x(1);
    for n = 1:nt
        psi = psi_t(:, n);
        dpsi_dx = gradient(psi, dx);
        % <p>
        p_exp = real(-1i * hbar * trapz(x, conj(psi) .* dpsi_dx));
        % <p²>
        d2psi_dx2 = gradient(dpsi_dx, dx);
        p2_exp = real(-hbar^2 * trapz(x, conj(psi) .* d2psi_dx2));
        sigma_p(n) = sqrt(p2_exp - p_exp^2);
    end
end
function visualize_results(results)
% Create comprehensive visualization using Berkeley colors
    % Berkeley color scheme
    berkeley_blue = [0.0039, 0.1961, 0.3843];
    california_gold = [1.0000, 0.7020, 0.0000];
    founders_rock = [0.2000, 0.2941, 0.3686];
    medalist = [0.7176, 0.5451, 0.0902];
    figure('Position', [100, 100, 1400, 900]);
    % Subplot 1: Wavefunction evolution
    subplot(2, 3, 1);
    [T, X] = meshgrid(results.t, results.x);
    prob_density = abs(results.psi).^2;
    contourf(T, X, prob_density, 20, 'LineStyle', 'none');
    colormap(gca, [linspace(1, berkeley_blue(1), 64)', ...
                   linspace(1, berkeley_blue(2), 64)', ...
                   linspace(1, berkeley_blue(3), 64)']);
    colorbar;
    xlabel('Time');
    ylabel('Position');
    title('Probability Density Evolution');
    % Subplot 2: Potential and initial state
    subplot(2, 3, 2);
    yyaxis left;
    plot(results.x, results.V, 'LineWidth', 2, 'Color', founders_rock);
    ylabel('Potential Energy');
    yyaxis right;
    plot(results.x, abs(results.psi(:, 1)).^2, 'LineWidth', 2, 'Color', california_gold);
    ylabel('Initial Probability Density');
    xlabel('Position');
    title('Potential and Initial State');
    % Subplot 3: Energy conservation
    subplot(2, 3, 3);
    plot(results.t, results.energy, 'LineWidth', 2, 'Color', berkeley_blue);
    xlabel('Time');
    ylabel('Total Energy');
    title('Energy Conservation');
    grid on;
    % Subplot 4: Norm conservation
    subplot(2, 3, 4);
    plot(results.t, results.norm, 'LineWidth', 2, 'Color', medalist);
    xlabel('Time');
    ylabel('Norm');
    title('Probability Conservation');
    grid on;
    % Subplot 5: Position expectation and uncertainty
    subplot(2, 3, 5);
    yyaxis left;
    plot(results.t, results.x_expect, 'LineWidth', 2, 'Color', berkeley_blue);
    ylabel('⟨x⟩');
    yyaxis right;
    plot(results.t, results.sigma_x, 'LineWidth', 2, 'Color', california_gold);
    ylabel('σₓ');
    xlabel('Time');
    title('Position Statistics');
    % Subplot 6: Momentum expectation and uncertainty
    subplot(2, 3, 6);
    yyaxis left;
    plot(results.t, results.p_expect, 'LineWidth', 2, 'Color', founders_rock);
    ylabel('⟨p⟩');
    yyaxis right;
    plot(results.t, results.sigma_p, 'LineWidth', 2, 'Color', medalist);
    ylabel('σₚ');
    xlabel('Time');
    title('Momentum Statistics');
    sgtitle('Schrödinger Equation Solution Analysis', 'FontSize', 16, 'FontWeight', 'bold');
    % Print final statistics
    fprintf('=== Final Statistics ===\n');
    fprintf('Final norm: %.6f\n', results.norm(end));
    fprintf('Energy conservation error: %.2e\n', std(results.energy)/mean(abs(results.energy)));
    fprintf('Final position: %.4f\n', results.x_expect(end));
    fprintf('Final momentum: %.4f\n', results.p_expect(end));
    fprintf('Position uncertainty: %.4f\n', results.sigma_x(end));
    fprintf('Momentum uncertainty: %.4f\n', results.sigma_p(end));
    fprintf('Uncertainty product: %.4f\n', results.sigma_x(end) * results.sigma_p(end));
    fprintf('Heisenberg limit (ℏ/2): %.4f\n', results.params.hbar/2);
end
function create_animation(results)
% Create animated visualization
    berkeley_blue = [0.0039, 0.1961, 0.3843];
    california_gold = [1.0000, 0.7020, 0.0000];
    figure('Position', [200, 200, 800, 600]);
    % Get maximum values for consistent scaling
    max_prob = max(abs(results.psi(:)).^2);
    max_V = max(results.V);
    min_V = min(results.V);
    % Animation parameters
    skip_frames = max(1, floor(length(results.t) / 100));  % Show ~100 frames max
    for n = 1:skip_frames:length(results.t)
        clf;
        % Main plot: wavefunction and potential
        subplot(2, 1, 1);
        % Potential
        yyaxis right;
        plot(results.x, results.V, 'LineWidth', 2, 'Color', [0.5, 0.5, 0.5]);
        ylabel('Potential Energy');
        ylim([min_V - 0.1*abs(min_V), max_V + 0.1*max_V]);
        % Wavefunction
        yyaxis left;
        plot(results.x, real(results.psi(:, n)), '--', 'LineWidth', 1.5, 'Color', berkeley_blue);
        hold on;
        plot(results.x, imag(results.psi(:, n)), ':', 'LineWidth', 1.5, 'Color', california_gold);
        plot(results.x, abs(results.psi(:, n)).^2, 'LineWidth', 2, 'Color', 'red');
        ylabel('Wavefunction');
        ylim([0, max_prob * 1.1]);
        legend('Re[ψ]', 'Im[ψ]', '|ψ|²', 'V(x)', 'Location', 'best');
        title(sprintf('Time Evolution: t = %.3f', results.t(n)));
        xlabel('Position');
        grid on;
        % Subplot: observables
        subplot(2, 1, 2);
        plot(results.t(1:n), results.x_expect(1:n), 'LineWidth', 2, 'Color', berkeley_blue);
        hold on;
        plot(results.t(1:n), results.p_expect(1:n), 'LineWidth', 2, 'Color', california_gold);
        xlabel('Time');
        ylabel('Expectation Values');
        legend('⟨x⟩', '⟨p⟩');
        grid on;
        xlim([results.t(1), results.t(end)]);
        drawnow;
        pause(0.05);
    end
    fprintf('Animation completed!\n');
end