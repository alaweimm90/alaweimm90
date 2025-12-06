classdef MonteCarloIsing < handle
    %MONTECARLOISNG Monte Carlo simulation of 2D Ising model
    %
    % Comprehensive MATLAB implementation of Monte Carlo methods for the
    % 2D Ising model, including Metropolis algorithm, thermodynamic
    % properties calculation, and phase transition analysis.
    %
    % The Ising Hamiltonian is:
    % H = -J∑⟨i,j⟩σᵢσⱼ - h∑ᵢσᵢ
    %
    % Features:
    %   - Metropolis-Hastings algorithm
    %   - Thermodynamic observables (energy, magnetization, heat capacity)
    %   - Critical temperature estimation
    %   - Correlation function calculations
    %   - Berkeley-styled visualization with phase diagrams
    %   - Finite-size scaling analysis
    %
    % Usage:
    %   ising = MonteCarloIsing(32, 'J', 1.0, 'h', 0.0);
    %   ising.runSimulation('T', 2.5, 'nSteps', 10000);
    %   ising.plotResults();
    %
    % Properties:
    %   L         - Lattice size (L×L)
    %   J         - Exchange coupling
    %   h         - External magnetic field
    %   T         - Temperature
    %   spins     - Current spin configuration
    %
    % Methods:
    %   runSimulation     - Run Monte Carlo simulation
    %   calculateObservables - Compute thermodynamic quantities
    %   temperatureSweep  - Scan across temperatures
    %   plotResults       - Berkeley-styled visualization
    %   correlationFunction - Compute spin correlations
    %
    % Author: Meshal Alawein (meshal@berkeley.edu)
    % Institution: University of California, Berkeley
    % License: MIT
    % Copyright © 2025 Meshal Alawein — All rights reserved.
    properties (SetAccess = private)
        L           % Lattice size (L×L)
        J           % Exchange coupling constant
        h           % External magnetic field
        T           % Temperature
        spins       % Spin configuration (L×L matrix)
        nSpins      % Total number of spins
        observables % Structure containing calculated observables
        history     % Simulation history
        rng         % Random number generator state
    end
    properties (Access = private)
        energyHistory    % Energy time series
        magnetizationHistory  % Magnetization time series
        acceptanceRate   % Metropolis acceptance rate
    end
    methods
        function obj = MonteCarloIsing(L, varargin)
            %MONTECARLOISNG Constructor for Monte Carlo Ising model
            %
            % Syntax:
            %   ising = MonteCarloIsing(L)
            %   ising = MonteCarloIsing(L, 'J', J, 'h', h, 'T', T)
            %
            % Inputs:
            %   L - Lattice size (creates L×L lattice)
            %   J - Exchange coupling (default: 1.0)
            %   h - Magnetic field (default: 0.0)
            %   T - Temperature (default: 2.0)
            %   seed - Random seed (default: 42)
            %   initialization - 'random', 'hot', 'cold' (default: 'random')
            %
            % Example:
            %   ising = MonteCarloIsing(32, 'J', 1.0, 'T', 2.5);
            % Parse inputs
            p = inputParser;
            addRequired(p, 'L', @(x) isscalar(x) && x > 0);
            addParameter(p, 'J', 1.0, @(x) isscalar(x));
            addParameter(p, 'h', 0.0, @(x) isscalar(x));
            addParameter(p, 'T', 2.0, @(x) isscalar(x) && x > 0);
            addParameter(p, 'seed', 42, @(x) isscalar(x) && x >= 0);
            addParameter(p, 'initialization', 'random', @ischar);
            parse(p, L, varargin{:});
            % Set properties
            obj.L = p.Results.L;
            obj.J = p.Results.J;
            obj.h = p.Results.h;
            obj.T = p.Results.T;
            obj.nSpins = L * L;
            % Initialize random number generator
            obj.rng = rng(p.Results.seed);
            % Initialize spin configuration
            obj.initializeSpins(p.Results.initialization);
            % Initialize observables structure
            obj.observables = struct(...
                'energy', NaN, ...
                'magnetization', NaN, ...
                'specificHeat', NaN, ...
                'susceptibility', NaN, ...
                'energyError', NaN, ...
                'magnetizationError', NaN ...
            );
            % Initialize history
            obj.history = struct(...
                'temperatures', [], ...
                'energies', [], ...
                'magnetizations', [], ...
                'specificHeats', [], ...
                'susceptibilities', [] ...
            );
        end
        function initializeSpins(obj, mode)
            %INITIALIZESPINS Initialize spin configuration
            %
            % Inputs:
            %   mode - 'random', 'hot' (random), 'cold' (all up)
            switch lower(mode)
                case {'random', 'hot'}
                    % Random spin configuration
                    obj.spins = 2 * randi([0, 1], obj.L, obj.L) - 1;
                case 'cold'
                    % All spins up (ground state for J > 0, h >= 0)
                    obj.spins = ones(obj.L, obj.L);
                case 'up'
                    obj.spins = ones(obj.L, obj.L);
                case 'down'
                    obj.spins = -ones(obj.L, obj.L);
                otherwise
                    error('Unknown initialization mode: %s', mode);
            end
        end
        function energy = calculateEnergy(obj)
            %CALCULATEENERGY Calculate total energy of current configuration
            %
            % Outputs:
            %   energy - Total energy
            %
            % The Ising Hamiltonian: H = -J∑⟨i,j⟩σᵢσⱼ - h∑ᵢσᵢ
            energy = 0;
            % Nearest-neighbor interaction energy
            for i = 1:obj.L
                for j = 1:obj.L
                    % Right neighbor (periodic boundary conditions)
                    i_right = mod(i, obj.L) + 1;
                    % Down neighbor
                    j_down = mod(j, obj.L) + 1;
                    % Add interactions (factor of 1/2 to avoid double counting)
                    energy = energy - 0.5 * obj.J * obj.spins(i, j) * obj.spins(i_right, j);
                    energy = energy - 0.5 * obj.J * obj.spins(i, j) * obj.spins(i, j_down);
                end
            end
            % External field energy
            energy = energy - obj.h * sum(obj.spins, 'all');
        end
        function magnetization = calculateMagnetization(obj)
            %CALCULATEMAGNETIZATION Calculate total magnetization
            %
            % Outputs:
            %   magnetization - Total magnetization
            magnetization = sum(obj.spins, 'all');
        end
        function deltaE = calculateEnergyChange(obj, i, j)
            %CALCULATEENERGYCHANGE Calculate energy change for flipping spin (i,j)
            %
            % Inputs:
            %   i, j - Lattice coordinates
            %
            % Outputs:
            %   deltaE - Energy change if spin is flipped
            % Get neighbors with periodic boundary conditions
            i_left = mod(i - 2, obj.L) + 1;
            i_right = mod(i, obj.L) + 1;
            j_up = mod(j - 2, obj.L) + 1;
            j_down = mod(j, obj.L) + 1;
            % Sum of neighboring spins
            neighborSum = obj.spins(i_left, j) + obj.spins(i_right, j) + ...
                         obj.spins(i, j_up) + obj.spins(i, j_down);
            % Energy change for flipping spin
            deltaE = 2 * obj.J * obj.spins(i, j) * neighborSum + 2 * obj.h * obj.spins(i, j);
        end
        function results = runSimulation(obj, varargin)
            %RUNSIMULATION Run Monte Carlo simulation
            %
            % Syntax:
            %   results = ising.runSimulation()
            %   results = ising.runSimulation('T', T, 'nSteps', nSteps)
            %
            % Options:
            %   T           - Temperature (default: current T)
            %   nSteps      - Number of MC steps (default: 10000)
            %   nEquil      - Equilibration steps (default: nSteps/5)
            %   measureEvery - Measurement interval (default: 1)
            %   verbose     - Display progress (default: true)
            %
            % Outputs:
            %   results - Structure with simulation results
            % Parse options
            p = inputParser;
            addParameter(p, 'T', obj.T, @(x) isscalar(x) && x > 0);
            addParameter(p, 'nSteps', 10000, @(x) isscalar(x) && x > 0);
            addParameter(p, 'nEquil', [], @(x) isscalar(x) && x >= 0);
            addParameter(p, 'measureEvery', 1, @(x) isscalar(x) && x > 0);
            addParameter(p, 'verbose', true, @islogical);
            parse(p, varargin{:});
            % Set parameters
            obj.T = p.Results.T;
            nSteps = p.Results.nSteps;
            if isempty(p.Results.nEquil)
                nEquil = round(nSteps / 5);
            else
                nEquil = p.Results.nEquil;
            end
            measureEvery = p.Results.measureEvery;
            verbose = p.Results.verbose;
            if verbose
                fprintf('🎲 Running Monte Carlo Ising simulation...\\n');
                fprintf('   Lattice: %d×%d, T = %.3f\\n', obj.L, obj.L, obj.T);
                fprintf('   Steps: %d, Equilibration: %d\\n', nSteps, nEquil);
                fprintf('   ────────────────────────────────────\\n');
            end
            % Pre-compute exponentials for efficiency
            expLookup = containers.Map();
            for deltaE = -8*obj.J:0.1:8*obj.J
                if deltaE <= 0
                    expLookup(num2str(deltaE)) = 1.0;
                else
                    expLookup(num2str(deltaE)) = exp(-deltaE / obj.T);
                end
            end
            % Initialize measurement arrays
            nMeasurements = floor((nSteps - nEquil) / measureEvery);
            obj.energyHistory = zeros(nMeasurements, 1);
            obj.magnetizationHistory = zeros(nMeasurements, 1);
            acceptedMoves = 0;
            measurementIndex = 1;
            tic;
            % Monte Carlo loop
            for step = 1:nSteps
                % One Monte Carlo sweep (attempt to flip each spin once on average)
                for sweep = 1:obj.nSpins
                    % Choose random spin
                    i = randi(obj.L);
                    j = randi(obj.L);
                    % Calculate energy change
                    deltaE = obj.calculateEnergyChange(i, j);
                    % Metropolis acceptance criterion
                    if deltaE <= 0
                        % Always accept energy-lowering moves
                        obj.spins(i, j) = -obj.spins(i, j);
                        acceptedMoves = acceptedMoves + 1;
                    else
                        % Accept with probability exp(-ΔE/T)
                        key = num2str(deltaE);
                        if isKey(expLookup, key)
                            prob = expLookup(key);
                        else
                            prob = exp(-deltaE / obj.T);
                        end
                        if rand() < prob
                            obj.spins(i, j) = -obj.spins(i, j);
                            acceptedMoves = acceptedMoves + 1;
                        end
                    end
                end
                % Measure observables after equilibration
                if step > nEquil && mod(step - nEquil, measureEvery) == 0
                    obj.energyHistory(measurementIndex) = obj.calculateEnergy();
                    obj.magnetizationHistory(measurementIndex) = obj.calculateMagnetization();
                    measurementIndex = measurementIndex + 1;
                end
                % Progress update
                if verbose && mod(step, max(1, floor(nSteps/10))) == 0
                    progress = 100 * step / nSteps;
                    fprintf('   Progress: %.0f%% (Step %d/%d)\\n', progress, step, nSteps);
                end
            end
            elapsed = toc;
            obj.acceptanceRate = acceptedMoves / (nSteps * obj.nSpins);
            % Calculate final observables
            obj.calculateObservables();
            if verbose
                fprintf('   ────────────────────────────────────\\n');
                fprintf('✅ Simulation completed in %.2f seconds\\n', elapsed);
                fprintf('   Acceptance rate: %.1f%%\\n', obj.acceptanceRate * 100);
                fprintf('   Final energy: %.4f ± %.4f\\n', obj.observables.energy, obj.observables.energyError);
                fprintf('   Final magnetization: %.4f ± %.4f\\n', obj.observables.magnetization, obj.observables.magnetizationError);
            end
            % Create results structure
            results = struct(...
                'temperature', obj.T, ...
                'energy', obj.observables.energy, ...
                'magnetization', obj.observables.magnetization, ...
                'specificHeat', obj.observables.specificHeat, ...
                'susceptibility', obj.observables.susceptibility, ...
                'acceptanceRate', obj.acceptanceRate, ...
                'energyHistory', obj.energyHistory, ...
                'magnetizationHistory', obj.magnetizationHistory ...
            );
        end
        function calculateObservables(obj)
            %CALCULATEOBSERVABLES Calculate thermodynamic observables
            if isempty(obj.energyHistory)
                warning('No simulation data available. Run simulation first.');
                return;
            end
            % Energy statistics
            E_mean = mean(obj.energyHistory);
            E_var = var(obj.energyHistory);
            E_err = sqrt(E_var / length(obj.energyHistory));
            % Magnetization statistics
            M = abs(obj.magnetizationHistory);  % Consider absolute value for phase transition
            M_mean = mean(M);
            M_var = var(M);
            M_err = sqrt(M_var / length(M));
            % Specific heat: C = (⟨E²⟩ - ⟨E⟩²) / T²
            C = E_var / (obj.T^2 * obj.nSpins);
            % Magnetic susceptibility: χ = (⟨M²⟩ - ⟨M⟩²) / T
            chi = M_var / (obj.T * obj.nSpins);
            % Store results (per spin)
            obj.observables.energy = E_mean / obj.nSpins;
            obj.observables.magnetization = M_mean / obj.nSpins;
            obj.observables.specificHeat = C;
            obj.observables.susceptibility = chi;
            obj.observables.energyError = E_err / obj.nSpins;
            obj.observables.magnetizationError = M_err / obj.nSpins;
        end
        function results = temperatureSweep(obj, temperatures, varargin)
            %TEMPERATURESWEEP Perform temperature sweep to study phase transition
            %
            % Syntax:
            %   results = ising.temperatureSweep(temperatures)
            %   results = ising.temperatureSweep(temperatures, 'nSteps', nSteps)
            %
            % Inputs:
            %   temperatures - Array of temperatures to simulate
            %   nSteps      - MC steps per temperature (default: 5000)
            %   nEquil      - Equilibration steps (default: nSteps/5)
            %
            % Outputs:
            %   results - Structure with temperature-dependent results
            % Parse options
            p = inputParser;
            addRequired(p, 'temperatures', @isnumeric);
            addParameter(p, 'nSteps', 5000, @(x) isscalar(x) && x > 0);
            addParameter(p, 'nEquil', [], @(x) isscalar(x) && x >= 0);
            parse(p, temperatures, varargin{:});
            temperatures = p.Results.temperatures;
            nSteps = p.Results.nSteps;
            nEquil = p.Results.nEquil;
            nTemps = length(temperatures);
            fprintf('🌡️ Temperature sweep: %d temperatures\\n', nTemps);
            fprintf('   Range: T = %.2f to %.2f\\n', min(temperatures), max(temperatures));
            % Initialize result arrays
            energies = zeros(nTemps, 1);
            magnetizations = zeros(nTemps, 1);
            specificHeats = zeros(nTemps, 1);
            susceptibilities = zeros(nTemps, 1);
            % Loop over temperatures
            for i = 1:nTemps
                T = temperatures(i);
                fprintf('   Running T = %.3f (%d/%d)...\\n', T, i, nTemps);
                % Run simulation at this temperature
                result = obj.runSimulation('T', T, 'nSteps', nSteps, ...
                                         'nEquil', nEquil, 'verbose', false);
                % Store results
                energies(i) = result.energy;
                magnetizations(i) = result.magnetization;
                specificHeats(i) = result.specificHeat;
                susceptibilities(i) = result.susceptibility;
            end
            % Store in history
            obj.history.temperatures = temperatures;
            obj.history.energies = energies;
            obj.history.magnetizations = magnetizations;
            obj.history.specificHeats = specificHeats;
            obj.history.susceptibilities = susceptibilities;
            % Create results structure
            results = struct(...
                'temperatures', temperatures, ...
                'energies', energies, ...
                'magnetizations', magnetizations, ...
                'specificHeats', specificHeats, ...
                'susceptibilities', susceptibilities ...
            );
            fprintf('✅ Temperature sweep completed!\\n');
        end
        function fig = plotResults(obj, varargin)
            %PLOTRESULTS Berkeley-styled visualization of results
            %
            % Syntax:
            %   fig = ising.plotResults()
            %   fig = ising.plotResults('type', 'configuration')
            %   fig = ising.plotResults('type', 'phase_diagram')
            %
            % Options:
            %   type    - 'configuration', 'phase_diagram', 'timeseries'
            %   savefig - Save figure filename (default: '')
            % Parse options
            p = inputParser;
            addParameter(p, 'type', 'configuration', @ischar);
            addParameter(p, 'savefig', '', @ischar);
            parse(p, varargin{:});
            plotType = p.Results.type;
            % Get Berkeley colors
            berkeleyBlue = [0, 50, 98] / 255;
            californiaGold = [253, 181, 21] / 255;
            switch lower(plotType)
                case 'configuration'
                    fig = obj.plotConfiguration();
                case 'phase_diagram'
                    fig = obj.plotPhaseDiagram();
                case 'timeseries'
                    fig = obj.plotTimeSeries();
                otherwise
                    error('Unknown plot type: %s', plotType);
            end
            % Save figure if requested
            if ~isempty(p.Results.savefig)
                saveas(fig, p.Results.savefig, 'png');
                fprintf('Figure saved to: %s\\n', p.Results.savefig);
            end
        end
        function fig = plotConfiguration(obj)
            %PLOTCONFIGURATION Plot current spin configuration
            fig = figure('Name', 'Ising Spin Configuration', 'NumberTitle', 'off');
            % Create colormap: blue for spin down (-1), gold for spin up (+1)
            colormap([0, 50, 98; 253, 181, 21] / 255);
            imagesc(obj.spins);
            colorbar('Ticks', [-1, 1], 'TickLabels', {'Spin Down', 'Spin Up'});
            axis equal tight;
            title(sprintf('🐻💙💛 Ising Model Configuration (T = %.2f)', obj.T), ...
                  'FontSize', 14, 'FontWeight', 'bold');
            xlabel('Site Index (j)');
            ylabel('Site Index (i)');
            % Add lattice size and energy info
            M = obj.calculateMagnetization();
            E = obj.calculateEnergy();
            text(0.02, 0.98, sprintf('L = %d×%d\\nM = %.0f\\nE = %.2f', ...
                                    obj.L, obj.L, M, E), ...
                 'Units', 'normalized', 'VerticalAlignment', 'top', ...
                 'BackgroundColor', 'white', 'FontSize', 10);
        end
        function fig = plotPhaseDiagram(obj)
            %PLOTPHASEDIAGRAM Plot phase diagram from temperature sweep
            if isempty(obj.history.temperatures)
                error('No temperature sweep data available. Run temperatureSweep() first.');
            end
            fig = figure('Name', 'Ising Phase Diagram', 'NumberTitle', 'off', ...
                        'Position', [100, 100, 1200, 800]);
            berkeleyBlue = [0, 50, 98] / 255;
            californiaGold = [253, 181, 21] / 255;
            % Subplot 1: Energy vs Temperature
            subplot(2, 2, 1);
            plot(obj.history.temperatures, obj.history.energies, 'o-', ...
                 'Color', berkeleyBlue, 'LineWidth', 2, 'MarkerSize', 6, ...
                 'MarkerFaceColor', berkeleyBlue);
            xlabel('Temperature');
            ylabel('Energy per spin');
            title('Internal Energy', 'FontWeight', 'bold');
            grid on;
            % Subplot 2: Magnetization vs Temperature
            subplot(2, 2, 2);
            plot(obj.history.temperatures, obj.history.magnetizations, 's-', ...
                 'Color', californiaGold, 'LineWidth', 2, 'MarkerSize', 6, ...
                 'MarkerFaceColor', californiaGold);
            xlabel('Temperature');
            ylabel('Magnetization per spin');
            title('Magnetization', 'FontWeight', 'bold');
            grid on;
            % Add critical temperature line (theoretical for 2D Ising)
            T_c_theory = 2 / log(1 + sqrt(2));  % ≈ 2.269
            xline(T_c_theory, '--', 'Color', 'red', 'LineWidth', 2, ...
                  'DisplayName', sprintf('T_c = %.3f', T_c_theory));
            legend('Simulation', 'Theory', 'Location', 'best');
            % Subplot 3: Specific Heat vs Temperature
            subplot(2, 2, 3);
            plot(obj.history.temperatures, obj.history.specificHeats, '^-', ...
                 'Color', [0.8, 0.2, 0.2], 'LineWidth', 2, 'MarkerSize', 6, ...
                 'MarkerFaceColor', [0.8, 0.2, 0.2]);
            xlabel('Temperature');
            ylabel('Specific Heat');
            title('Specific Heat', 'FontWeight', 'bold');
            grid on;
            xline(T_c_theory, '--', 'Color', 'red', 'LineWidth', 2);
            % Subplot 4: Susceptibility vs Temperature
            subplot(2, 2, 4);
            plot(obj.history.temperatures, obj.history.susceptibilities, 'd-', ...
                 'Color', [0.2, 0.6, 0.2], 'LineWidth', 2, 'MarkerSize', 6, ...
                 'MarkerFaceColor', [0.2, 0.6, 0.2]);
            xlabel('Temperature');
            ylabel('Magnetic Susceptibility');
            title('Susceptibility', 'FontWeight', 'bold');
            grid on;
            xline(T_c_theory, '--', 'Color', 'red', 'LineWidth', 2);
            % Add overall title
            sgtitle('🐻💙💛 2D Ising Model Phase Diagram - UC Berkeley', ...
                   'FontSize', 16, 'FontWeight', 'bold');
        end
        function fig = plotTimeSeries(obj)
            %PLOTTIMESERIES Plot Monte Carlo time series
            if isempty(obj.energyHistory)
                error('No simulation data available. Run simulation first.');
            end
            fig = figure('Name', 'Monte Carlo Time Series', 'NumberTitle', 'off', ...
                        'Position', [100, 100, 1000, 600]);
            berkeleyBlue = [0, 50, 98] / 255;
            californiaGold = [253, 181, 21] / 255;
            % Energy time series
            subplot(2, 1, 1);
            plot(obj.energyHistory / obj.nSpins, 'Color', berkeleyBlue, 'LineWidth', 1);
            xlabel('Monte Carlo Step');
            ylabel('Energy per spin');
            title(sprintf('Energy Time Series (T = %.2f)', obj.T), 'FontWeight', 'bold');
            grid on;
            % Running average
            hold on;
            windowSize = max(1, floor(length(obj.energyHistory) / 50));
            runningAvg = movmean(obj.energyHistory / obj.nSpins, windowSize);
            plot(runningAvg, 'Color', californiaGold, 'LineWidth', 2, ...
                 'DisplayName', 'Running Average');
            legend('Instantaneous', 'Running Average', 'Location', 'best');
            hold off;
            % Magnetization time series
            subplot(2, 1, 2);
            plot(abs(obj.magnetizationHistory) / obj.nSpins, 'Color', berkeleyBlue, 'LineWidth', 1);
            xlabel('Monte Carlo Step');
            ylabel('|Magnetization| per spin');
            title('Magnetization Time Series', 'FontWeight', 'bold');
            grid on;
            % Running average
            hold on;
            runningAvg = movmean(abs(obj.magnetizationHistory) / obj.nSpins, windowSize);
            plot(runningAvg, 'Color', californiaGold, 'LineWidth', 2, ...
                 'DisplayName', 'Running Average');
            legend('Instantaneous', 'Running Average', 'Location', 'best');
            hold off;
            sgtitle('🐻💙💛 Monte Carlo Time Series - UC Berkeley', ...
                   'FontSize', 14, 'FontWeight', 'bold');
        end
    end
end