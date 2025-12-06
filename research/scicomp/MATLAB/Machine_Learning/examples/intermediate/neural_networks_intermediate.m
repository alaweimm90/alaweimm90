function neural_networks_intermediate()
% NEURAL_NETWORKS_INTERMEDIATE - Intermediate Neural Networks Example
%
% This example demonstrates intermediate neural network concepts using
% the Berkeley SciComp Machine Learning toolbox. We'll build and train
% a multi-layer perceptron for a scientific regression problem.
%
% Learning Objectives:
%   - Understand neural network architecture design
%   - Learn activation functions and their effects
%   - Implement proper training procedures
%   - Analyze training dynamics and optimization
%   - Apply to quantum harmonic oscillator problem
%
% Author: Berkeley SciComp Team
% Date: 2024
fprintf('🧠 Berkeley SciComp: Intermediate Neural Networks\n');
fprintf('================================================\n');
fprintf('Building neural networks for quantum physics problems\n');
fprintf('We''ll predict energy eigenvalues of quantum harmonic oscillator.\n\n');
try
    % Create quantum physics dataset
    fprintf('🔬 Creating quantum harmonic oscillator dataset...\n');
    [X, y, problem_info] = create_quantum_dataset();
    % Demonstrate network architecture design
    demonstrate_architecture_design(X, y);
    % Compare activation functions
    compare_activation_functions(X, y);
    % Analyze training dynamics
    analyze_training_dynamics(X, y);
    % Physics interpretation
    physics_interpretation(problem_info);
    fprintf('\n✨ Intermediate example completed successfully!\n');
    fprintf('\nKey Takeaways:\n');
    fprintf('• Network depth affects learning capacity\n');
    fprintf('• Activation functions determine network expressivity\n');
    fprintf('• Proper training monitoring prevents overfitting\n');
    fprintf('• Neural networks can learn complex physics relationships\n');
    fprintf('\nNext Steps:\n');
    fprintf('• Try the advanced physics-informed networks example\n');
    fprintf('• Experiment with different optimizers\n');
    fprintf('• Explore regularization techniques\n');
catch ME
    fprintf('❌ Error occurred: %s\n', ME.message);
    fprintf('Please check your installation and try again.\n');
end
end
function [X, y, problem_info] = create_quantum_dataset()
% Create quantum harmonic oscillator dataset
%
% We'll predict energy eigenvalues E_n = ħω(n + 1/2)
% as a function of quantum number n and oscillator parameters
% Physical parameters
hbar = 1.054571817e-34;  % Reduced Planck constant (J⋅s)
m = 9.1093837015e-31;    % Electron mass (kg)
omega_range = [1e13, 1e15]; % Angular frequency range (rad/s)
n_max = 20;              % Maximum quantum number
% Generate parameter combinations
n_samples = 1000;
rng(42); % For reproducibility
% Quantum numbers (discrete)
n_values = randi([0, n_max], n_samples, 1);
% Angular frequencies (continuous)
omega_values = omega_range(1) + (omega_range(2) - omega_range(1)) * rand(n_samples, 1);
% Spring constants (derived from omega and mass)
k_values = m * omega_values.^2;
% Create feature matrix
X = [n_values, omega_values, k_values, m * ones(n_samples, 1)];
% Calculate exact energy eigenvalues
y = hbar * omega_values .* (n_values + 0.5);
% Convert to more manageable units (eV)
eV = 1.602176634e-19;  % Elementary charge (C)
y = y / eV;  % Convert J to eV
% Normalize features for better training
X(:, 2) = X(:, 2) / 1e14;  % Normalize omega
X(:, 3) = X(:, 3) / 1e-16; % Normalize k
X(:, 4) = X(:, 4) / 1e-30; % Normalize m
% Store problem information
problem_info = struct();
problem_info.hbar = hbar;
problem_info.m = m;
problem_info.omega_range = omega_range;
problem_info.n_max = n_max;
problem_info.eV = eV;
problem_info.feature_names = {'Quantum Number n', 'Angular Frequency ω (×10¹⁴)', ...
                              'Spring Constant k (×10⁻¹⁶)', 'Mass m (×10⁻³⁰)'};
problem_info.target_name = 'Energy (eV)';
fprintf('Dataset created: %d samples\n', n_samples);
fprintf('Features: %s\n', strjoin(problem_info.feature_names, ', '));
fprintf('Target: %s\n', problem_info.target_name);
fprintf('Energy range: %.6f to %.6f eV\n', min(y), max(y));
end
function demonstrate_architecture_design(X, y)
% Demonstrate different network architectures
fprintf('\n🏗️ Network Architecture Design\n');
fprintf('=====================================\n');
% Split data
split_idx = floor(0.8 * length(y));
X_train = X(1:split_idx, :);
X_test = X(split_idx+1:end, :);
y_train = y(1:split_idx);
y_test = y(split_idx+1:end);
% Test different architectures
architectures = {
    [4, 10, 1],      % Shallow network
    [4, 20, 10, 1],  % Medium network
    [4, 50, 25, 10, 1], % Deep network
    [4, 100, 50, 20, 1] % Very deep network
};
architecture_names = {'Shallow (4-10-1)', 'Medium (4-20-10-1)', ...
                     'Deep (4-50-25-10-1)', 'Very Deep (4-100-50-20-1)'};
results = struct();
for i = 1:length(architectures)
    fprintf('\nTesting %s architecture...\n', architecture_names{i});
    % Create and train network
    network = neural_networks.MLP(architectures{i}, ...
                                 'Activations', 'relu', ...
                                 'OutputActivation', 'linear', ...
                                 'LearningRate', 0.001);
    tic;
    network.fit(X_train, y_train, 'Epochs', 200, 'Verbose', false);
    training_time = toc;
    % Evaluate
    y_pred = network.predict(X_test);
    r2 = network.score(X_test, y_test);
    rmse = sqrt(mean((y_test - y_pred).^2));
    % Store results
    results(i).name = architecture_names{i};
    results(i).r2 = r2;
    results(i).rmse = rmse;
    results(i).training_time = training_time;
    results(i).n_parameters = network.getNParameters();
    fprintf('  R² Score: %.4f\n', r2);
    fprintf('  RMSE: %.6f eV\n', rmse);
    fprintf('  Training time: %.2f seconds\n', training_time);
    fprintf('  Parameters: %d\n', network.getNParameters());
end
% Visualize architecture comparison
figure('Position', [100, 100, 1200, 800]);
% Performance vs complexity
subplot(2, 2, 1);
n_params = [results.n_parameters];
r2_scores = [results.r2];
scatter(n_params, r2_scores, 100, 'filled');
for i = 1:length(results)
    text(n_params(i), r2_scores(i), sprintf('  %d', i), 'VerticalAlignment', 'middle');
end
xlabel('Number of Parameters');
ylabel('R² Score');
title('Performance vs Model Complexity');
grid on;
% Training time vs complexity
subplot(2, 2, 2);
training_times = [results.training_time];
scatter(n_params, training_times, 100, 'filled');
for i = 1:length(results)
    text(n_params(i), training_times(i), sprintf('  %d', i), 'VerticalAlignment', 'middle');
end
xlabel('Number of Parameters');
ylabel('Training Time (seconds)');
title('Training Time vs Model Complexity');
grid on;
% RMSE comparison
subplot(2, 2, 3);
rmse_values = [results.rmse];
bar(1:length(results), rmse_values, 'FaceColor', [0, 50, 98]/255);
set(gca, 'XTickLabel', {results.name}, 'XTickLabelRotation', 45);
ylabel('RMSE (eV)');
title('RMSE by Architecture');
grid on;
% Architecture summary table
subplot(2, 2, 4);
axis off;
header = {'Architecture', 'R²', 'RMSE (eV)', 'Time (s)', 'Params'};
data = cell(length(results), 5);
for i = 1:length(results)
    data{i, 1} = results(i).name;
    data{i, 2} = sprintf('%.4f', results(i).r2);
    data{i, 3} = sprintf('%.6f', results(i).rmse);
    data{i, 4} = sprintf('%.2f', results(i).training_time);
    data{i, 5} = sprintf('%d', results(i).n_parameters);
end
% Create table
table_data = [header; data];
y_pos = linspace(0.9, 0.1, length(results) + 1);
for i = 1:length(results) + 1
    for j = 1:5
        if i == 1
            text(0.1 + (j-1)*0.18, y_pos(i), table_data{i, j}, 'FontWeight', 'bold');
        else
            text(0.1 + (j-1)*0.18, y_pos(i), table_data{i, j});
        end
    end
end
title('Architecture Comparison Summary');
sgtitle('Neural Network Architecture Analysis', 'FontSize', 14, 'FontWeight', 'bold');
end
function compare_activation_functions(X, y)
% Compare different activation functions
fprintf('\n⚡ Activation Function Comparison\n');
fprintf('==================================\n');
% Split data
split_idx = floor(0.8 * length(y));
X_train = X(1:split_idx, :);
X_test = X(split_idx+1:end, :);
y_train = y(1:split_idx);
y_test = y(split_idx+1:end);
% Test different activation functions
activations = {'relu', 'tanh', 'sigmoid'};
activation_names = {'ReLU', 'Tanh', 'Sigmoid'};
% Fixed architecture for fair comparison
architecture = [4, 50, 25, 1];
activation_results = struct();
for i = 1:length(activations)
    fprintf('\nTesting %s activation...\n', activation_names{i});
    % Create and train network
    network = neural_networks.MLP(architecture, ...
                                 'Activations', activations{i}, ...
                                 'OutputActivation', 'linear', ...
                                 'LearningRate', 0.001);
    network.fit(X_train, y_train, 'Epochs', 300, 'Verbose', false);
    % Evaluate
    y_pred = network.predict(X_test);
    r2 = network.score(X_test, y_test);
    rmse = sqrt(mean((y_test - y_pred).^2));
    % Get training history
    history = network.getTrainingHistory();
    % Store results
    activation_results(i).name = activation_names{i};
    activation_results(i).activation = activations{i};
    activation_results(i).r2 = r2;
    activation_results(i).rmse = rmse;
    activation_results(i).history = history;
    activation_results(i).predictions = y_pred;
    fprintf('  R² Score: %.4f\n', r2);
    fprintf('  RMSE: %.6f eV\n', rmse);
end
% Visualize activation function comparison
figure('Position', [200, 100, 1400, 1000]);
% Training curves
subplot(2, 3, 1);
for i = 1:length(activation_results)
    if ~isempty(activation_results(i).history)
        plot(activation_results(i).history.loss, 'LineWidth', 2, ...
             'DisplayName', activation_results(i).name);
        hold on;
    end
end
xlabel('Epoch');
ylabel('Training Loss');
title('Training Loss by Activation Function');
legend('Location', 'best');
set(gca, 'YScale', 'log');
grid on;
% Performance comparison
subplot(2, 3, 2);
r2_values = [activation_results.r2];
bar(1:length(activation_results), r2_values, 'FaceColor', [253, 181, 21]/255);
set(gca, 'XTickLabel', {activation_results.name});
ylabel('R² Score');
title('R² Score by Activation Function');
grid on;
% RMSE comparison
subplot(2, 3, 3);
rmse_values = [activation_results.rmse];
bar(1:length(activation_results), rmse_values, 'FaceColor', [0, 50, 98]/255);
set(gca, 'XTickLabel', {activation_results.name});
ylabel('RMSE (eV)');
title('RMSE by Activation Function');
grid on;
% Prediction scatter plots
for i = 1:length(activation_results)
    subplot(2, 3, 3 + i);
    scatter(y_test, activation_results(i).predictions, 30, 'filled', 'Alpha', 0.6);
    hold on;
    % Perfect prediction line
    min_val = min([min(y_test), min(activation_results(i).predictions)]);
    max_val = max([max(y_test), max(activation_results(i).predictions)]);
    plot([min_val, max_val], [min_val, max_val], '--', 'Color', [0.5, 0.5, 0.5], 'LineWidth', 2);
    xlabel('Actual Energy (eV)');
    ylabel('Predicted Energy (eV)');
    title(sprintf('%s: R² = %.4f', activation_results(i).name, activation_results(i).r2));
    grid on;
    axis equal;
end
sgtitle('Activation Function Comparison', 'FontSize', 14, 'FontWeight', 'bold');
end
function analyze_training_dynamics(X, y)
% Analyze training dynamics and optimization
fprintf('\n📈 Training Dynamics Analysis\n');
fprintf('==============================\n');
% Split data with validation set
n_samples = length(y);
train_idx = 1:floor(0.6 * n_samples);
val_idx = (floor(0.6 * n_samples) + 1):floor(0.8 * n_samples);
test_idx = (floor(0.8 * n_samples) + 1):n_samples;
X_train = X(train_idx, :);
X_val = X(val_idx, :);
X_test = X(test_idx, :);
y_train = y(train_idx);
y_val = y(val_idx);
y_test = y(test_idx);
fprintf('Data split: %d train, %d validation, %d test\n', ...
        length(train_idx), length(val_idx), length(test_idx));
% Create network with monitoring
network = neural_networks.MLP([4, 64, 32, 1], ...
                             'Activations', 'relu', ...
                             'OutputActivation', 'linear', ...
                             'LearningRate', 0.001);
% Train with validation monitoring
fprintf('\nTraining with validation monitoring...\n');
network.fit(X_train, y_train, ...
           'Epochs', 500, ...
           'ValidationData', {X_val, y_val}, ...
           'Verbose', true, ...
           'EarlyStopping', true, ...
           'Patience', 50);
% Get comprehensive training history
history = network.getTrainingHistory();
% Analyze final performance
y_pred_test = network.predict(X_test);
r2_test = network.score(X_test, y_test);
rmse_test = sqrt(mean((y_test - y_pred_test).^2));
fprintf('\nFinal Performance:\n');
fprintf('  Test R² Score: %.4f\n', r2_test);
fprintf('  Test RMSE: %.6f eV\n', rmse_test);
% Visualize training dynamics
figure('Position', [300, 100, 1400, 1000]);
% Loss curves
subplot(2, 3, 1);
epochs = 1:length(history.loss);
semilogy(epochs, history.loss, 'LineWidth', 2, 'DisplayName', 'Training Loss');
hold on;
if isfield(history, 'val_loss')
    semilogy(epochs, history.val_loss, 'LineWidth', 2, 'DisplayName', 'Validation Loss');
end
xlabel('Epoch');
ylabel('Loss');
title('Training and Validation Loss');
legend('Location', 'best');
grid on;
% Learning rate schedule
subplot(2, 3, 2);
if isfield(history, 'learning_rate')
    plot(epochs, history.learning_rate, 'LineWidth', 2);
    xlabel('Epoch');
    ylabel('Learning Rate');
    title('Learning Rate Schedule');
    grid on;
else
    text(0.5, 0.5, 'Learning Rate\nData Not Available', ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
    title('Learning Rate Schedule');
end
% Gradient norms
subplot(2, 3, 3);
if isfield(history, 'grad_norm')
    semilogy(epochs, history.grad_norm, 'LineWidth', 2);
    xlabel('Epoch');
    ylabel('Gradient Norm');
    title('Gradient Norm Evolution');
    grid on;
else
    text(0.5, 0.5, 'Gradient Norm\nData Not Available', ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
    title('Gradient Norm Evolution');
end
% Final predictions
subplot(2, 3, 4);
scatter(y_test, y_pred_test, 50, 'filled', 'Alpha', 0.7);
hold on;
min_val = min([min(y_test), min(y_pred_test)]);
max_val = max([max(y_test), max(y_pred_test)]);
plot([min_val, max_val], [min_val, max_val], '--', 'Color', [0.5, 0.5, 0.5], 'LineWidth', 2);
xlabel('Actual Energy (eV)');
ylabel('Predicted Energy (eV)');
title(sprintf('Final Predictions (R² = %.4f)', r2_test));
grid on;
axis equal;
% Residual analysis
subplot(2, 3, 5);
residuals = y_test - y_pred_test;
scatter(y_pred_test, residuals, 50, 'filled', 'Alpha', 0.7);
hold on;
yline(0, '--', 'Color', [0.5, 0.5, 0.5], 'LineWidth', 2);
xlabel('Predicted Energy (eV)');
ylabel('Residuals (eV)');
title('Residual Analysis');
grid on;
% Error distribution
subplot(2, 3, 6);
histogram(residuals, 20, 'FaceColor', [0, 50, 98]/255, 'EdgeColor', 'none');
xlabel('Residuals (eV)');
ylabel('Frequency');
title('Residual Distribution');
grid on;
sgtitle('Training Dynamics and Model Analysis', 'FontSize', 14, 'FontWeight', 'bold');
end
function physics_interpretation(problem_info)
% Provide physics interpretation of results
fprintf('\n🔬 Physics Interpretation\n');
fprintf('==========================\n');
fprintf('Quantum Harmonic Oscillator Energy Levels:\n');
fprintf('\n');
fprintf('The quantum harmonic oscillator is fundamental in quantum mechanics.\n');
fprintf('Energy eigenvalues: E_n = ħω(n + 1/2)\n');
fprintf('\n');
fprintf('Key Physics Concepts:\n');
fprintf('• Quantized energy levels (discrete n values)\n');
fprintf('• Zero-point energy (1/2)ħω even at n=0\n');
fprintf('• Linear relationship with frequency ω\n');
fprintf('• Fundamental in molecular vibrations\n');
fprintf('\n');
fprintf('Neural Network Learning:\n');
fprintf('• Network learned the linear relationship E ∝ ω\n');
fprintf('• Discrete quantum numbers handled effectively\n');
fprintf('• Physical constants implicitly learned\n');
fprintf('• Generalizes to new parameter combinations\n');
fprintf('\n');
fprintf('Applications:\n');
fprintf('• Molecular vibrational spectroscopy\n');
fprintf('• Quantum field theory foundations\n');
fprintf('• Solid state physics (phonons)\n');
fprintf('• Quantum optics and cavity QED\n');
end