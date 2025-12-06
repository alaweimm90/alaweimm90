classdef QuantumHarmonic < handle
    %QUANTUMHARMONIC Quantum harmonic oscillator implementation
    %
    % Comprehensive MATLAB implementation of the quantum harmonic oscillator
    % with analytical eigenstates, time evolution, coherent states, and
    % Wigner function calculations.
    %
    % Features:
    %   - Analytical eigenstate calculations using Hermite polynomials
    %   - Time evolution of arbitrary initial states
    %   - Coherent state generation and dynamics
    %   - Phase space representations (Wigner functions)
    %   - Expectation value calculations
    %   - Berkeley-styled visualization
    %
    % Usage:
    %   qho = QuantumHarmonic(omega, 'mass', mass, 'xMax', xMax, 'nPoints', nPoints);
    %   psi0 = qho.eigenstate(0);              % Ground state
    %   psiCoh = qho.coherentState(1.5);       % Coherent state
    %   psiT = qho.timeEvolution(psi0, t);     % Time evolution
    %   qho.plot();                            % Visualize
    %
    % Properties:
    %   omega     - Angular frequency (rad/s)
    %   mass      - Particle mass (kg)
    %   x         - Position grid
    %   x0        - Characteristic length scale
    %   E0        - Zero-point energy
    %
    % Methods:
    %   eigenstate      - Calculate energy eigenstate
    %   coherentState   - Generate coherent state
    %   timeEvolution   - Evolve wavefunction in time
    %   expectationValue - Calculate expectation values
    %   wignerFunction  - Compute Wigner function
    %   plot            - Berkeley-styled visualization
    %
    % Author: Meshal Alawein (meshal@berkeley.edu)
    % Institution: University of California, Berkeley
    % License: MIT
    % Copyright © 2025 Meshal Alawein — All rights reserved.
    properties (SetAccess = private)
        omega       % Angular frequency (rad/s)
        mass        % Particle mass (kg)
        x           % Position grid
        dx          % Grid spacing
        x0          % Characteristic length scale sqrt(hbar/(m*omega))
        E0          % Zero-point energy hbar*omega/2
        nMax        % Maximum quantum number for basis
        xMax        % Maximum position (in units of x0)
        nPoints     % Number of grid points
    end
    properties (Constant, Hidden)
        HBAR = 1.0545718176461565e-34;  % Reduced Planck constant (J⋅s)
        ME = 9.1093837015e-31;          % Electron mass (kg)
    end
    methods
        function obj = QuantumHarmonic(omega, varargin)
            %QUANTUMHARMONIC Constructor for quantum harmonic oscillator
            %
            % Syntax:
            %   qho = QuantumHarmonic(omega)
            %   qho = QuantumHarmonic(omega, 'mass', mass)
            %   qho = QuantumHarmonic(omega, 'mass', mass, 'xMax', xMax, 'nPoints', nPoints)
            %
            % Inputs:
            %   omega    - Angular frequency (rad/s)
            %   mass     - Particle mass (kg), default: electron mass
            %   xMax     - Maximum position (in units of x0), default: 10.0
            %   nPoints  - Number of grid points, default: 1000
            %   nMax     - Maximum quantum number, default: 50
            %
            % Example:
            %   qho = QuantumHarmonic(1.0, 'mass', 9.109e-31, 'xMax', 8.0);
            % Parse inputs
            p = inputParser;
            addRequired(p, 'omega', @(x) isscalar(x) && x > 0);
            addParameter(p, 'mass', obj.ME, @(x) isscalar(x) && x > 0);
            addParameter(p, 'xMax', 10.0, @(x) isscalar(x) && x > 0);
            addParameter(p, 'nPoints', 1000, @(x) isscalar(x) && x > 0);
            addParameter(p, 'nMax', 50, @(x) isscalar(x) && x > 0);
            parse(p, omega, varargin{:});
            % Set properties
            obj.omega = p.Results.omega;
            obj.mass = p.Results.mass;
            obj.xMax = p.Results.xMax;
            obj.nPoints = p.Results.nPoints;
            obj.nMax = p.Results.nMax;
            % Calculate characteristic scales
            obj.x0 = sqrt(obj.HBAR / (obj.mass * obj.omega));
            obj.E0 = 0.5 * obj.HBAR * obj.omega;
            % Create position grid
            obj.x = linspace(-obj.xMax * obj.x0, obj.xMax * obj.x0, obj.nPoints);
            obj.dx = obj.x(2) - obj.x(1);
        end
        function E = energy(obj, n)
            %ENERGY Calculate energy eigenvalue for quantum number n
            %
            % Syntax:
            %   E = qho.energy(n)
            %
            % Inputs:
            %   n - Quantum number (non-negative integer)
            %
            % Outputs:
            %   E - Energy eigenvalue in Joules
            %
            % Formula: E_n = ℏω(n + 1/2)
            validateattributes(n, {'numeric'}, {'nonnegative', 'integer'});
            E = obj.HBAR * obj.omega * (n + 0.5);
        end
        function psi = eigenstate(obj, n, x)
            %EIGENSTATE Calculate eigenstate wavefunction using Hermite polynomials
            %
            % Syntax:
            %   psi = qho.eigenstate(n)
            %   psi = qho.eigenstate(n, x)
            %
            % Inputs:
            %   n - Quantum number (non-negative integer)
            %   x - Position grid (optional, uses internal grid if not provided)
            %
            % Outputs:
            %   psi - Wavefunction values (normalized)
            %
            % Formula:
            %   ψₙ(x) = (mω/πℏ)^(1/4) * (1/√(2ⁿn!)) * exp(-mωx²/2ℏ) * Hₙ(√(mω/ℏ)x)
            validateattributes(n, {'numeric'}, {'nonnegative', 'integer', 'scalar'});
            if nargin < 3
                x = obj.x;
            end
            % Dimensionless coordinate
            xi = x / obj.x0;
            % Normalization constant
            N = (1 / (pi^0.25 * sqrt(2^n * factorial(n))));
            % Hermite polynomial (using MATLAB's built-in hermiteH)
            H_n = hermiteH(n, xi);
            % Wavefunction
            psi = N * exp(-xi.^2 / 2) .* H_n;
        end
        function psi = coherentState(obj, alpha, x, nTerms)
            %COHERENTSTATE Generate coherent state |α⟩
            %
            % Syntax:
            %   psi = qho.coherentState(alpha)
            %   psi = qho.coherentState(alpha, x)
            %   psi = qho.coherentState(alpha, x, nTerms)
            %
            % Inputs:
            %   alpha  - Coherent state parameter (complex)
            %   x      - Position grid (optional)
            %   nTerms - Number of terms in expansion (optional)
            %
            % Outputs:
            %   psi - Coherent state wavefunction
            %
            % Formula: |α⟩ = exp(-|α|²/2) * Σₙ (αⁿ/√n!) |n⟩
            if nargin < 3
                x = obj.x;
            end
            if nargin < 4
                % Adaptive cutoff based on |α|
                nTerms = max(20, round(3 * abs(alpha)^2) + 20);
                nTerms = min(nTerms, obj.nMax);
            end
            % Initialize wavefunction
            psi = zeros(size(x));
            % Coherent state expansion
            normalization = exp(-0.5 * abs(alpha)^2);
            for n = 0:nTerms-1
                coefficient = normalization * (alpha^n) / sqrt(factorial(n));
                psi = psi + coefficient * obj.eigenstate(n, x);
            end
        end
        function psiT = timeEvolution(obj, psiInitial, t, x)
            %TIMEEVOLUTION Evolve wavefunction in time analytically
            %
            % Syntax:
            %   psiT = qho.timeEvolution(psiInitial, t)
            %   psiT = qho.timeEvolution(psiInitial, t, x)
            %
            % Inputs:
            %   psiInitial - Initial wavefunction
            %   t          - Time(s) for evolution (scalar or array)
            %   x          - Position grid (optional)
            %
            % Outputs:
            %   psiT - Time-evolved wavefunction(s)
            %
            % Uses eigenstate expansion: ψ(t) = Σₙ cₙ exp(-iEₙt/ℏ) |n⟩
            if nargin < 4
                x = obj.x;
            end
            % Expand initial state in energy eigenbasis
            coefficients = obj.expandInEigenbasis(psiInitial, x);
            if isscalar(t)
                % Single time
                psiT = zeros(size(x));
                for n = 1:length(coefficients)
                    if abs(coefficients(n)) > 1e-12  % Skip negligible terms
                        phase = exp(-1i * obj.energy(n-1) * t / obj.HBAR);
                        psiT = psiT + coefficients(n) * phase * obj.eigenstate(n-1, x);
                    end
                end
            else
                % Multiple times
                psiT = zeros(length(t), length(x));
                for i = 1:length(t)
                    for n = 1:length(coefficients)
                        if abs(coefficients(n)) > 1e-12
                            phase = exp(-1i * obj.energy(n-1) * t(i) / obj.HBAR);
                            psiT(i, :) = psiT(i, :) + coefficients(n) * phase * obj.eigenstate(n-1, x);
                        end
                    end
                end
            end
        end
        function coefficients = expandInEigenbasis(obj, psi, x, nTerms)
            %EXPANDINEIGENBASIS Expand wavefunction in energy eigenbasis
            %
            % Syntax:
            %   coefficients = qho.expandInEigenbasis(psi)
            %   coefficients = qho.expandInEigenbasis(psi, x, nTerms)
            %
            % Inputs:
            %   psi    - Wavefunction to expand
            %   x      - Position grid (optional)
            %   nTerms - Number of terms in expansion (optional)
            %
            % Outputs:
            %   coefficients - Expansion coefficients cₙ = ⟨n|ψ⟩
            if nargin < 3
                x = obj.x;
            end
            if nargin < 4
                nTerms = min(50, obj.nMax);
            end
            coefficients = zeros(nTerms, 1);
            for n = 1:nTerms
                eigenstate_n = obj.eigenstate(n-1, x);
                coefficients(n) = trapz(x, conj(eigenstate_n) .* psi);
            end
        end
        function expVal = expectationValue(obj, psi, observable, x)
            %EXPECTATIONVALUE Calculate expectation value of observable
            %
            % Syntax:
            %   expVal = qho.expectationValue(psi, observable)
            %   expVal = qho.expectationValue(psi, observable, x)
            %
            % Inputs:
            %   psi        - Quantum state wavefunction
            %   observable - Observable name ('x', 'p', 'x2', 'p2', 'H')
            %   x          - Position grid (optional)
            %
            % Outputs:
            %   expVal - Expectation value
            if nargin < 4
                x = obj.x;
            end
            switch lower(observable)
                case 'x'
                    expVal = obj.expectationPosition(psi, x);
                case 'p'
                    expVal = obj.expectationMomentum(psi, x);
                case 'x2'
                    expVal = obj.expectationPositionSquared(psi, x);
                case 'p2'
                    expVal = obj.expectationMomentumSquared(psi, x);
                case 'h'
                    expVal = obj.expectationHamiltonian(psi, x);
                otherwise
                    error('Unknown observable: %s', observable);
            end
        end
        function [X, P, W] = wignerFunction(obj, psi, x, xRange, pRange, nPoints)
            %WIGNERFUNCTION Calculate Wigner function for phase space representation
            %
            % Syntax:
            %   [X, P, W] = qho.wignerFunction(psi)
            %   [X, P, W] = qho.wignerFunction(psi, x, xRange, pRange, nPoints)
            %
            % Inputs:
            %   psi     - Quantum state wavefunction
            %   x       - Position grid (optional)
            %   xRange  - Position range [xMin, xMax] (optional)
            %   pRange  - Momentum range [pMin, pMax] (optional)
            %   nPoints - Grid points [nX, nP] (optional)
            %
            % Outputs:
            %   X, P, W - Position, momentum meshgrids and Wigner function
            %
            % Formula: W(x,p) = (1/πℏ) ∫ dy ψ*(x+y/2) ψ(x-y/2) exp(ipy/ℏ)
            if nargin < 3
                x = obj.x;
            end
            if nargin < 4
                xMax = 3 * obj.x0;
                xRange = [-xMax, xMax];
            end
            if nargin < 5
                pMax = 3 * obj.HBAR / obj.x0;
                pRange = [-pMax, pMax];
            end
            if nargin < 6
                nPoints = [100, 100];
            end
            % Phase space grids
            xW = linspace(xRange(1), xRange(2), nPoints(1));
            pW = linspace(pRange(1), pRange(2), nPoints(2));
            [X, P] = meshgrid(xW, pW);
            W = zeros(size(X));
            % Calculate Wigner function
            for i = 1:length(xW)
                for j = 1:length(pW)
                    W(j, i) = obj.wignerPoint(psi, x, xW(i), pW(j));
                end
            end
        end
        function fig = plot(obj, varargin)
            %PLOT Berkeley-styled visualization of quantum harmonic oscillator
            %
            % Syntax:
            %   fig = qho.plot()
            %   fig = qho.plot('states', [0, 1, 2])
            %   fig = qho.plot('coherent', 1.5)
            %   fig = qho.plot('wigner', psi)
            %
            % Options:
            %   'states'   - Plot energy eigenstates (default: [0, 1, 2, 3])
            %   'coherent' - Plot coherent state with given alpha
            %   'wigner'   - Show Wigner function of given state
            %   'savefig'  - Save figure to file
            % Parse options
            p = inputParser;
            addParameter(p, 'states', 0:3, @isnumeric);
            addParameter(p, 'coherent', [], @isnumeric);
            addParameter(p, 'wigner', [], @isnumeric);
            addParameter(p, 'savefig', '', @ischar);
            parse(p, varargin{:});
            % Create figure with Berkeley styling
            fig = figure('Name', 'Quantum Harmonic Oscillator', 'NumberTitle', 'off');
            if ~isempty(p.Results.coherent)
                % Plot coherent state
                obj.plotCoherentState(p.Results.coherent);
            elseif ~isempty(p.Results.wigner)
                % Plot Wigner function
                obj.plotWignerFunction(p.Results.wigner);
            else
                % Plot energy eigenstates (default)
                obj.plotEigenstates(p.Results.states);
            end
            % Save figure if requested
            if ~isempty(p.Results.savefig)
                saveas(fig, p.Results.savefig, 'png');
                fprintf('Figure saved to: %s\n', p.Results.savefig);
            end
        end
        function plotEigenstates(obj, states)
            %PLOTEIGENSTATES Plot energy eigenstates with Berkeley styling
            % Get Berkeley colors
            colors = getBerkeleyColors();
            subplot(1, 2, 1);
            hold on;
            % Plot wavefunctions
            for i = 1:length(states)
                n = states(i);
                psi_n = obj.eigenstate(n);
                color = colors{mod(i-1, length(colors)) + 1};
                plot(obj.x / obj.x0, real(psi_n), 'Color', color, ...
                     'LineWidth', 2, 'DisplayName', sprintf('\\psi_{%d}(x)', n));
            end
            xlabel('Position (x/x_0)');
            ylabel('Wavefunction');
            title('Harmonic Oscillator Eigenstates', 'FontWeight', 'bold');
            legend('show', 'Location', 'best');
            grid on;
            hold off;
            subplot(1, 2, 2);
            hold on;
            % Plot probability densities
            for i = 1:length(states)
                n = states(i);
                psi_n = obj.eigenstate(n);
                prob_density = abs(psi_n).^2;
                color = colors{mod(i-1, length(colors)) + 1};
                plot(obj.x / obj.x0, prob_density, 'Color', color, ...
                     'LineWidth', 2, 'DisplayName', sprintf('|\\psi_{%d}(x)|^2', n));
                % Classical turning points
                energy = obj.energy(n);
                x_classical = sqrt(2 * energy / (obj.mass * obj.omega^2)) / obj.x0;
                xline(x_classical, '--', 'Color', color, 'Alpha', 0.5);
                xline(-x_classical, '--', 'Color', color, 'Alpha', 0.5);
            end
            xlabel('Position (x/x_0)');
            ylabel('Probability Density');
            title('Probability Densities', 'FontWeight', 'bold');
            legend('show', 'Location', 'best');
            grid on;
            hold off;
            % Add Berkeley branding
            sgtitle('🐻💙💛 UC Berkeley - Quantum Harmonic Oscillator', ...
                    'FontSize', 14, 'FontWeight', 'bold');
        end
        function plotCoherentState(obj, alpha)
            %PLOTCOHERENTSTATE Plot coherent state
            psi_coh = obj.coherentState(alpha);
            subplot(1, 2, 1);
            plot(obj.x / obj.x0, real(psi_coh), 'b-', 'LineWidth', 2, 'DisplayName', 'Re[\\psi(x)]');
            hold on;
            plot(obj.x / obj.x0, imag(psi_coh), 'r--', 'LineWidth', 2, 'DisplayName', 'Im[\\psi(x)]');
            fill(obj.x / obj.x0, abs(psi_coh).^2, 'g', 'Alpha', 0.3, 'DisplayName', '|\\psi(x)|^2');
            hold off;
            xlabel('Position (x/x_0)');
            ylabel('Wavefunction');
            title(sprintf('Coherent State \\alpha = %.1f', alpha), 'FontWeight', 'bold');
            legend('show');
            grid on;
            % Calculate expectation values
            x_exp = obj.expectationValue(psi_coh, 'x');
            p_exp = obj.expectationValue(psi_coh, 'p');
            subplot(1, 2, 2);
            % Time evolution animation would go here
            text(0.5, 0.5, sprintf('⟨x⟩ = %.2f x_0\n⟨p⟩ = %.2e', ...
                                   real(x_exp)/obj.x0, real(p_exp)), ...
                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                 'FontSize', 12, 'Units', 'normalized');
            title('Expectation Values', 'FontWeight', 'bold');
        end
        function plotWignerFunction(obj, psi)
            %PLOTWIGNERFUNCTION Plot Wigner function
            [X, P, W] = obj.wignerFunction(psi);
            contourf(X / obj.x0, P, W, 20);
            colorbar;
            xlabel('Position (x_0)');
            ylabel('Momentum');
            title('Wigner Function', 'FontWeight', 'bold');
            axis equal;
        end
    end
    methods (Access = private)
        % Private helper methods for expectation values
        function expVal = expectationPosition(obj, psi, x)
            integrand = conj(psi) .* x .* psi;
            expVal = trapz(x, integrand);
        end
        function expVal = expectationMomentum(obj, psi, x)
            dpsi_dx = gradient(psi, obj.dx);
            integrand = conj(psi) .* (-1i * obj.HBAR * dpsi_dx);
            expVal = trapz(x, integrand);
        end
        function expVal = expectationPositionSquared(obj, psi, x)
            integrand = conj(psi) .* x.^2 .* psi;
            expVal = trapz(x, integrand);
        end
        function expVal = expectationMomentumSquared(obj, psi, x)
            d2psi_dx2 = gradient(gradient(psi, obj.dx), obj.dx);
            integrand = conj(psi) .* (-obj.HBAR^2 * d2psi_dx2);
            expVal = trapz(x, integrand);
        end
        function expVal = expectationHamiltonian(obj, psi, x)
            T = obj.expectationMomentumSquared(psi, x) / (2 * obj.mass);
            V = 0.5 * obj.mass * obj.omega^2 * obj.expectationPositionSquared(psi, x);
            expVal = T + V;
        end
        function W_point = wignerPoint(obj, psi, x, x_val, p_val)
            % Calculate Wigner function at a single point
            y_max = min(2 * max(abs(x)), 10 * obj.x0);
            y = linspace(-y_max, y_max, 200);
            dy = y(2) - y(1);
            % Interpolate wavefunction
            psi_plus = interp1(x, psi, x_val + y/2, 'linear', 0);
            psi_minus = interp1(x, psi, x_val - y/2, 'linear', 0);
            % Calculate integrand
            integrand = conj(psi_plus) .* psi_minus .* exp(1i * p_val * y / obj.HBAR);
            % Integrate
            W_point = real(trapz(y, integrand)) / (pi * obj.HBAR);
        end
    end
end
function colors = getBerkeleyColors()
    %GETBERKELEYCOLORS Get Berkeley color palette
    colors = {
        [0, 50, 98] / 255,          % Berkeley Blue
        [253, 181, 21] / 255,       % California Gold
        [0, 85, 58] / 255,          % Green Dark
        [119, 7, 71] / 255,         % Rose Dark
        [67, 17, 112] / 255,        % Purple Dark
        [140, 21, 21] / 255,        % Red Dark
        [210, 105, 30] / 255,       % Orange Dark
        [0, 76, 90] / 255           % Teal Dark
    };
end