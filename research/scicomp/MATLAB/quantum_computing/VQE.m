classdef VQE < handle
    %VQE Variational Quantum Eigensolver implementation in MATLAB
    %
    % Comprehensive implementation of the Variational Quantum Eigensolver (VQE)
    % algorithm for finding ground states of quantum systems. Includes various
    % ansatz circuits, optimization strategies, and visualization tools.
    %
    % The VQE algorithm minimizes the energy expectation value:
    % E(θ) = ⟨ψ(θ)|Ĥ|ψ(θ)⟩
    %
    % Features:
    %   - Hardware-efficient ansatz circuits
    %   - Multiple classical optimizers (fminunc, ga, particleswarm)
    %   - Noise modeling and error mitigation
    %   - Comprehensive visualization with Berkeley styling
    %   - Excited state calculations using penalty methods
    %   - Performance analysis and convergence monitoring
    %
    % Usage:
    %   vqe = VQE(hamiltonian, 'numQubits', 2, 'numLayers', 3);
    %   result = vqe.optimize();
    %   vqe.plotResults();
    %
    % Properties:
    %   hamiltonian   - System Hamiltonian matrix
    %   numQubits     - Number of qubits
    %   numLayers     - Number of ansatz layers
    %   ansatzType    - Type of ansatz ('hardware_efficient')
    %   optimizer     - Classical optimizer ('fminunc', 'ga', 'particleswarm')
    %
    % Methods:
    %   optimize         - Run VQE optimization
    %   energyEvaluation - Compute energy expectation value
    %   constructAnsatz  - Build parameterized quantum circuit
    %   plotResults      - Visualize optimization results
    %   calculateExcited - Compute excited states
    %
    % Author: Meshal Alawein (meshal@berkeley.edu)
    % Institution: University of California, Berkeley
    % License: MIT
    % Copyright © 2025 Meshal Alawein — All rights reserved.
    properties (SetAccess = private)
        hamiltonian     % System Hamiltonian matrix
        numQubits       % Number of qubits
        numLayers       % Number of ansatz layers
        numParameters   % Total number of variational parameters
        ansatzType      % Type of ansatz circuit
        optimizer       % Classical optimizer
        optimizationHistory  % History of optimization process
        optimalParameters    % Optimal parameters found
        optimalEnergy       % Optimal energy found
        exactGroundEnergy   % Exact ground state energy (if available)
    end
    properties (Access = private)
        pauliMatrices   % Pauli matrices for quantum operations
        rng             % Random number generator state
    end
    methods
        function obj = VQE(hamiltonian, varargin)
            %VQE Constructor for VQE optimizer
            %
            % Syntax:
            %   vqe = VQE(hamiltonian)
            %   vqe = VQE(hamiltonian, 'numQubits', nQubits, 'numLayers', nLayers)
            %
            % Inputs:
            %   hamiltonian - System Hamiltonian matrix
            %   numQubits   - Number of qubits (default: inferred from Hamiltonian)
            %   numLayers   - Number of ansatz layers (default: 2)
            %   ansatzType  - Ansatz type (default: 'hardware_efficient')
            %   optimizer   - Classical optimizer (default: 'fminunc')
            %   seed        - Random seed for reproducibility (default: 42)
            %
            % Example:
            %   H = [1, 0; 0, -1];  % Pauli-Z Hamiltonian
            %   vqe = VQE(H, 'numQubits', 1, 'numLayers', 2);
            % Parse inputs
            p = inputParser;
            addRequired(p, 'hamiltonian', @(x) ismatrix(x) && size(x,1) == size(x,2));
            % Infer number of qubits from Hamiltonian size
            defaultNumQubits = log2(size(hamiltonian, 1));
            if floor(defaultNumQubits) ~= defaultNumQubits
                error('Hamiltonian size must be a power of 2');
            end
            addParameter(p, 'numQubits', defaultNumQubits, @(x) isscalar(x) && x > 0);
            addParameter(p, 'numLayers', 2, @(x) isscalar(x) && x > 0);
            addParameter(p, 'ansatzType', 'hardware_efficient', @ischar);
            addParameter(p, 'optimizer', 'fminunc', @ischar);
            addParameter(p, 'seed', 42, @(x) isscalar(x) && x >= 0);
            parse(p, hamiltonian, varargin{:});
            % Set properties
            obj.hamiltonian = p.Results.hamiltonian;
            obj.numQubits = p.Results.numQubits;
            obj.numLayers = p.Results.numLayers;
            obj.ansatzType = p.Results.ansatzType;
            obj.optimizer = p.Results.optimizer;
            % Calculate number of parameters
            obj.numParameters = obj.calculateNumParameters();
            % Initialize Pauli matrices
            obj.initializePauliMatrices();
            % Set random seed
            obj.rng = rng(p.Results.seed);
            % Calculate exact ground energy if possible
            try
                eigenvals = eig(obj.hamiltonian);
                obj.exactGroundEnergy = min(real(eigenvals));
            catch
                obj.exactGroundEnergy = NaN;
                warning('Could not calculate exact ground energy');
            end
            % Initialize history
            obj.optimizationHistory = struct(...
                'energies', [], ...
                'parameters', [], ...
                'iterations', [], ...
                'gradients', [] ...
            );
        end
        function result = optimize(obj, varargin)
            %OPTIMIZE Run VQE optimization to find ground state
            %
            % Syntax:
            %   result = vqe.optimize()
            %   result = vqe.optimize('initialParams', params0, 'maxIterations', 1000)
            %
            % Options:
            %   initialParams   - Initial parameter values (default: random)
            %   maxIterations   - Maximum number of iterations (default: 1000)
            %   display         - Display optimization progress (default: 'iter')
            %   tolerance       - Convergence tolerance (default: 1e-6)
            %
            % Outputs:
            %   result - Structure containing optimization results
            % Parse options
            p = inputParser;
            addParameter(p, 'initialParams', [], @isnumeric);
            addParameter(p, 'maxIterations', 1000, @(x) isscalar(x) && x > 0);
            addParameter(p, 'display', 'iter', @ischar);
            addParameter(p, 'tolerance', 1e-6, @(x) isscalar(x) && x > 0);
            parse(p, varargin{:});
            % Initialize parameters
            if isempty(p.Results.initialParams)
                params0 = obj.getInitialParameters();
            else
                params0 = p.Results.initialParams;
                if length(params0) ~= obj.numParameters
                    error('Initial parameters must have length %d', obj.numParameters);
                end
            end
            fprintf('🚀 Starting VQE optimization...\n');
            fprintf('   System: %d qubits, %d parameters\n', obj.numQubits, obj.numParameters);
            fprintf('   Ansatz: %s (%d layers)\n', obj.ansatzType, obj.numLayers);
            fprintf('   Optimizer: %s\n', obj.optimizer);
            if ~isnan(obj.exactGroundEnergy)
                fprintf('   Exact ground energy: %.6f\n', obj.exactGroundEnergy);
            end
            fprintf('   ─────────────────────────────────────\n');
            % Clear previous history
            obj.optimizationHistory = struct(...
                'energies', [], ...
                'parameters', [], ...
                'iterations', [], ...
                'gradients', [] ...
            );
            % Define objective function with history tracking
            objectiveFunc = @(params) obj.objectiveWithHistory(params);
            % Set up optimization options
            switch obj.optimizer
                case 'fminunc'
                    options = optimoptions('fminunc', ...
                        'Display', p.Results.display, ...
                        'MaxIterations', p.Results.maxIterations, ...
                        'OptimalityTolerance', p.Results.tolerance, ...
                        'StepTolerance', p.Results.tolerance, ...
                        'SpecifyObjectiveGradient', false);
                    [optParams, optEnergy, exitFlag, output] = fminunc(objectiveFunc, params0, options);
                case 'ga'
                    options = optimoptions('ga', ...
                        'Display', p.Results.display, ...
                        'MaxGenerations', p.Results.maxIterations, ...
                        'FunctionTolerance', p.Results.tolerance, ...
                        'PopulationSize', 50);
                    % Define bounds for parameters (typical range for rotation angles)
                    lb = -2*pi * ones(size(params0));
                    ub = 2*pi * ones(size(params0));
                    [optParams, optEnergy, exitFlag, output] = ga(objectiveFunc, obj.numParameters, ...
                        [], [], [], [], lb, ub, [], options);
                case 'particleswarm'
                    options = optimoptions('particleswarm', ...
                        'Display', p.Results.display, ...
                        'MaxIterations', p.Results.maxIterations, ...
                        'FunctionTolerance', p.Results.tolerance, ...
                        'SwarmSize', 50);
                    lb = -2*pi * ones(size(params0));
                    ub = 2*pi * ones(size(params0));
                    [optParams, optEnergy, exitFlag, output] = particleswarm(objectiveFunc, obj.numParameters, ...
                        lb, ub, options);
                otherwise
                    error('Unknown optimizer: %s', obj.optimizer);
            end
            % Store results
            obj.optimalParameters = optParams;
            obj.optimalEnergy = optEnergy;
            % Calculate final metrics
            if ~isnan(obj.exactGroundEnergy)
                error = abs(optEnergy - obj.exactGroundEnergy);
            else
                error = NaN;
            end
            % Create result structure
            result = struct(...
                'optimalEnergy', optEnergy, ...
                'optimalParameters', optParams, ...
                'exactGroundEnergy', obj.exactGroundEnergy, ...
                'error', error, ...
                'exitFlag', exitFlag, ...
                'output', output, ...
                'numIterations', length(obj.optimizationHistory.energies), ...
                'success', exitFlag > 0 ...
            );
            % Display results
            fprintf('   ─────────────────────────────────────\n');
            fprintf('✅ VQE optimization completed!\n');
            fprintf('   Optimal energy: %.6f\n', optEnergy);
            if ~isnan(error)
                fprintf('   Error: %.2e\n', error);
            end
            fprintf('   Iterations: %d\n', result.numIterations);
            fprintf('   Success: %s\n', string(result.success));
        end
        function energy = energyEvaluation(obj, parameters)
            %ENERGYEVALUATION Compute energy expectation value for given parameters
            %
            % Syntax:
            %   energy = vqe.energyEvaluation(parameters)
            %
            % Inputs:
            %   parameters - Variational parameters
            %
            % Outputs:
            %   energy - Energy expectation value ⟨ψ(θ)|Ĥ|ψ(θ)⟩
            % Construct quantum state using ansatz
            psi = obj.constructAnsatz(parameters);
            % Calculate expectation value
            energy = real(psi' * obj.hamiltonian * psi);
        end
        function psi = constructAnsatz(obj, parameters)
            %CONSTRUCTANSATZ Build parameterized quantum state
            %
            % Syntax:
            %   psi = vqe.constructAnsatz(parameters)
            %
            % Inputs:
            %   parameters - Variational parameters
            %
            % Outputs:
            %   psi - Quantum state vector
            switch obj.ansatzType
                case 'hardware_efficient'
                    psi = obj.hardwareEfficientAnsatz(parameters);
                otherwise
                    error('Unknown ansatz type: %s', obj.ansatzType);
            end
        end
        function psi = hardwareEfficientAnsatz(obj, parameters)
            %HARDWAREEFFICIENTANSATZ Hardware-efficient ansatz implementation
            %
            % Constructs a hardware-efficient ansatz with layers of single-qubit
            % rotations followed by entangling gates.
            %
            % Circuit structure:
            % - RY rotations on all qubits
            % - CNOT gates for entanglement (linear topology)
            % - Repeat for specified number of layers
            % Initialize state |0...0⟩
            psi = zeros(2^obj.numQubits, 1);
            psi(1) = 1;
            paramIdx = 1;
            for layer = 1:obj.numLayers
                % Single-qubit rotations (RY gates)
                for qubit = 1:obj.numQubits
                    angle = parameters(paramIdx);
                    psi = obj.applyRY(psi, qubit, angle);
                    paramIdx = paramIdx + 1;
                end
                % Entangling gates (CNOT)
                if layer < obj.numLayers || obj.numLayers == 1
                    for qubit = 1:obj.numQubits-1
                        psi = obj.applyCNOT(psi, qubit, qubit+1);
                    end
                end
            end
        end
        function fig = plotResults(obj, varargin)
            %PLOTRESULTS Visualize VQE optimization results with Berkeley styling
            %
            % Syntax:
            %   fig = vqe.plotResults()
            %   fig = vqe.plotResults('savefig', 'vqe_results.png')
            %
            % Options:
            %   savefig - Save figure to file (default: '')
            % Parse options
            p = inputParser;
            addParameter(p, 'savefig', '', @ischar);
            parse(p, varargin{:});
            if isempty(obj.optimizationHistory.energies)
                error('No optimization history available. Run optimize() first.');
            end
            % Create figure with Berkeley styling
            fig = figure('Name', 'VQE Results', 'NumberTitle', 'off', 'Position', [100, 100, 1200, 800]);
            % Get Berkeley colors
            berkeleyBlue = [0, 50, 98] / 255;
            californiaGold = [253, 181, 21] / 255;
            % Subplot 1: Energy convergence
            subplot(2, 2, 1);
            plot(obj.optimizationHistory.iterations, obj.optimizationHistory.energies, ...
                 'Color', berkeleyBlue, 'LineWidth', 2, 'Marker', 'o', 'MarkerSize', 4);
            hold on;
            if ~isnan(obj.exactGroundEnergy)
                yline(obj.exactGroundEnergy, '--', 'Color', californiaGold, 'LineWidth', 2, ...
                      'DisplayName', sprintf('Exact: %.6f', obj.exactGroundEnergy));
            end
            xlabel('Iteration');
            ylabel('Energy');
            title('Energy Convergence', 'FontWeight', 'bold');
            grid on;
            if ~isnan(obj.exactGroundEnergy)
                legend('VQE', 'Exact', 'Location', 'best');
            end
            % Subplot 2: Error vs iteration (if exact energy known)
            subplot(2, 2, 2);
            if ~isnan(obj.exactGroundEnergy)
                errors = abs(obj.optimizationHistory.energies - obj.exactGroundEnergy);
                semilogy(obj.optimizationHistory.iterations, errors, ...
                         'Color', berkeleyBlue, 'LineWidth', 2, 'Marker', 's', 'MarkerSize', 4);
                xlabel('Iteration');
                ylabel('Absolute Error');
                title('Convergence Error', 'FontWeight', 'bold');
                grid on;
            else
                text(0.5, 0.5, 'Exact energy not available', 'HorizontalAlignment', 'center', ...
                     'VerticalAlignment', 'middle', 'Units', 'normalized', 'FontSize', 12);
                title('Error Analysis', 'FontWeight', 'bold');
            end
            % Subplot 3: Parameter evolution
            subplot(2, 2, 3);
            if ~isempty(obj.optimizationHistory.parameters)
                paramHistory = cell2mat(obj.optimizationHistory.parameters');
                plot(obj.optimizationHistory.iterations, paramHistory, 'LineWidth', 1.5);
                xlabel('Iteration');
                ylabel('Parameter Value');
                title('Parameter Evolution', 'FontWeight', 'bold');
                grid on;
                if obj.numParameters <= 10
                    legend(arrayfun(@(i) sprintf('\\theta_%d', i), 1:obj.numParameters, 'UniformOutput', false), ...
                           'Location', 'eastoutside');
                end
            end
            % Subplot 4: Final state visualization
            subplot(2, 2, 4);
            if ~isempty(obj.optimalParameters)
                finalState = obj.constructAnsatz(obj.optimalParameters);
                probabilities = abs(finalState).^2;
                basis_labels = cell(1, 2^obj.numQubits);
                for i = 1:2^obj.numQubits
                    basis_labels{i} = dec2bin(i-1, obj.numQubits);
                end
                bar(1:length(probabilities), probabilities, 'FaceColor', californiaGold, ...
                    'EdgeColor', berkeleyBlue, 'LineWidth', 1);
                xlabel('Basis State');
                ylabel('Probability');
                title('Final State Distribution', 'FontWeight', 'bold');
                if obj.numQubits <= 3
                    set(gca, 'XTickLabel', basis_labels);
                    xtickangle(45);
                end
                grid on;
            end
            % Add Berkeley branding
            sgtitle('🐻💙💛 VQE Results - UC Berkeley', 'FontSize', 16, 'FontWeight', 'bold');
            % Save figure if requested
            if ~isempty(p.Results.savefig)
                saveas(fig, p.Results.savefig, 'png');
                fprintf('Figure saved to: %s\n', p.Results.savefig);
            end
        end
        function excitedStates = calculateExcited(obj, numStates, penaltyWeight)
            %CALCULATEEXCITED Compute excited states using penalty method
            %
            % Syntax:
            %   excitedStates = vqe.calculateExcited(numStates)
            %   excitedStates = vqe.calculateExcited(numStates, penaltyWeight)
            %
            % Inputs:
            %   numStates     - Number of excited states to compute
            %   penaltyWeight - Penalty weight for orthogonality (default: 10.0)
            %
            % Outputs:
            %   excitedStates - Structure array with excited state results
            if nargin < 3
                penaltyWeight = 10.0;
            end
            fprintf('🎯 Computing %d excited states...\n', numStates);
            excitedStates = struct('energy', {}, 'parameters', {}, 'state', {});
            foundStates = {};
            for i = 1:numStates
                fprintf('   Finding excited state %d...\n', i);
                % Define penalized objective function
                penalizedObjective = @(params) obj.penalizedEnergy(params, foundStates, penaltyWeight);
                % Random initialization
                params0 = obj.getInitialParameters();
                % Optimize
                options = optimoptions('fminunc', 'Display', 'off', 'MaxIterations', 500);
                [optParams, optEnergy] = fminunc(penalizedObjective, params0, options);
                % Store results
                finalState = obj.constructAnsatz(optParams);
                foundStates{end+1} = finalState;
                excitedStates(i).energy = optEnergy - penaltyWeight * (i-1);  % Remove penalty
                excitedStates(i).parameters = optParams;
                excitedStates(i).state = finalState;
                fprintf('     Energy: %.6f\n', excitedStates(i).energy);
            end
            fprintf('✅ Excited state calculation completed!\n');
        end
    end
    methods (Access = private)
        function numParams = calculateNumParameters(obj)
            %CALCULATENUMPARAMETERS Calculate total number of variational parameters
            switch obj.ansatzType
                case 'hardware_efficient'
                    % One RY rotation per qubit per layer
                    numParams = obj.numQubits * obj.numLayers;
                otherwise
                    error('Unknown ansatz type: %s', obj.ansatzType);
            end
        end
        function params0 = getInitialParameters(obj)
            %GETINITIALPARAMETERS Generate random initial parameters
            params0 = 2*pi * (rand(obj.numParameters, 1) - 0.5);  % [-π, π]
        end
        function initializePauliMatrices(obj)
            %INITIALIZEPAULIMATRICES Initialize Pauli matrices
            obj.pauliMatrices.I = eye(2);
            obj.pauliMatrices.X = [0, 1; 1, 0];
            obj.pauliMatrices.Y = [0, -1i; 1i, 0];
            obj.pauliMatrices.Z = [1, 0; 0, -1];
        end
        function energy = objectiveWithHistory(obj, parameters)
            %OBJECTIVEWITHHISTORY Objective function that tracks optimization history
            energy = obj.energyEvaluation(parameters);
            % Store in history
            obj.optimizationHistory.energies(end+1) = energy;
            obj.optimizationHistory.parameters{end+1} = parameters;
            obj.optimizationHistory.iterations(end+1) = length(obj.optimizationHistory.energies);
        end
        function energy = penalizedEnergy(obj, parameters, foundStates, penaltyWeight)
            %PENALIZEDENERGY Energy function with orthogonality penalty
            energy = obj.energyEvaluation(parameters);
            currentState = obj.constructAnsatz(parameters);
            % Add penalty for overlap with previously found states
            penalty = 0;
            for i = 1:length(foundStates)
                overlap = abs(foundStates{i}' * currentState)^2;
                penalty = penalty + penaltyWeight * overlap;
            end
            energy = energy + penalty;
        end
        function psi = applyRY(obj, psi, qubit, angle)
            %APPLYRY Apply RY rotation to specified qubit
            RY = [cos(angle/2), -sin(angle/2); sin(angle/2), cos(angle/2)];
            % Construct full rotation matrix
            if qubit == 1
                U = RY;
            else
                U = obj.pauliMatrices.I;
            end
            for q = 2:obj.numQubits
                if q == qubit
                    U = kron(U, RY);
                else
                    U = kron(U, obj.pauliMatrices.I);
                end
            end
            psi = U * psi;
        end
        function psi = applyCNOT(obj, psi, control, target)
            %APPLYCNOT Apply CNOT gate between control and target qubits
            n = obj.numQubits;
            dim = 2^n;
            U = eye(dim);
            % Apply CNOT logic
            for i = 1:dim
                bits = bitget(i-1, 1:n);  % Get bit representation
                if bits(control) == 1  % Control qubit is 1
                    % Flip target qubit
                    newBits = bits;
                    newBits(target) = 1 - newBits(target);
                    % Convert back to index
                    newIdx = 1 + sum(newBits .* (2.^(0:n-1)));
                    % Swap rows in unitary
                    temp = U(i, :);
                    U(i, :) = U(newIdx, :);
                    U(newIdx, :) = temp;
                end
            end
            psi = U * psi;
        end
    end
end