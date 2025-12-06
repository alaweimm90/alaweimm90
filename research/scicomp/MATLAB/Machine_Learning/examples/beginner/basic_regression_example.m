function basic_regression_example()
% BASIC_REGRESSION_EXAMPLE - Beginner Machine Learning with Linear Regression
%
% This example demonstrates fundamental linear regression concepts using
% the Berkeley SciComp Machine Learning toolbox. We'll analyze a simple
% physics dataset and build our first predictive model.
%
% Learning Objectives:
%   - Understand linear regression basics
%   - Learn data preprocessing steps
%   - Visualize model performance
%   - Interpret regression results
%
% Author: Berkeley SciComp Team
% Date: 2024
fprintf('🎯 Berkeley SciComp: Basic Linear Regression\n');
fprintf('====================================================\n');
fprintf('Welcome to your first machine learning example!\n');
fprintf('We''ll analyze projectile motion using linear regression.\n\n');
try
    % Create physics dataset
    fprintf('📊 Creating projectile motion dataset...\n');
    [X, y_noisy, y_true] = create_physics_dataset();
    % Demonstrate basic regression
    model = demonstrate_basic_regression(X, y_noisy, y_true);
    % Provide physics insights
    physics_insights();
    % Interactive exploration
    interactive_exploration(model);
    fprintf('\n✨ Example completed successfully!\n');
    fprintf('\nKey Takeaways:\n');
    fprintf('• Linear regression finds the best linear relationship\n');
    fprintf('• R² measures how well the model explains the data\n');
    fprintf('• Residual analysis helps identify model limitations\n');
    fprintf('• Uncertainty quantification provides confidence estimates\n');
    fprintf('\nNext Steps:\n');
    fprintf('• Try the polynomial regression example\n');
    fprintf('• Explore non-linear relationships\n');
    fprintf('• Learn about regularization techniques\n');
catch ME
    fprintf('❌ Error occurred: %s\n', ME.message);
    fprintf('Please check your installation and try again.\n');
end
end
function [X, y_noisy, y_true] = create_physics_dataset()
% Create a synthetic physics dataset: projectile motion
%
% We'll model the relationship between launch angle and range
% for projectile motion with air resistance.
% Parameters
v0 = 50;  % Initial velocity (m/s)
g = 9.81; % Gravity (m/s²)
k = 0.01; % Air resistance coefficient
% Launch angles (degrees)
angles_deg = linspace(15, 75, 100)';
angles_rad = deg2rad(angles_deg);
% Calculate range with air resistance (simplified model)
% R ≈ (v₀²/g) * sin(2θ) * (1 - k*v₀*sin(θ))
ranges = (v0^2 / g) * sin(2 * angles_rad) .* (1 - k * v0 * sin(angles_rad));
% Add realistic measurement noise
rng(42); % For reproducibility
noise = normrnd(0, 2, size(ranges));
y_noisy = ranges + noise;
X = angles_deg;
y_true = ranges;
fprintf('Dataset created: %d data points\n', length(X));
fprintf('Feature: Launch angle (degrees)\n');
fprintf('Target: Projectile range (meters)\n');
end
function model = demonstrate_basic_regression(X, y_noisy, y_true)
% Demonstrate basic linear regression workflow
fprintf('\n🚀 Basic Linear Regression Example\n');
fprintf('==================================================\n');
% Split data
split_idx = floor(0.8 * length(X));
X_train = X(1:split_idx);
X_test = X(split_idx+1:end);
y_train = y_noisy(1:split_idx);
y_test = y_noisy(split_idx+1:end);
fprintf('Data split: %d training, %d testing samples\n', ...
        length(X_train), length(X_test));
% Create and train model
fprintf('\n🔧 Training linear regression model...\n');
model = supervised.LinearRegression('Solver', 'svd', ...
                                   'FitIntercept', true, ...
                                   'UncertaintyEstimation', true);
model.fit(X_train, y_train);
fprintf('✅ Model training complete!\n');
% Make predictions
y_pred = model.predict(X_test);
fprintf('📈 Predictions generated\n');
% Evaluate model
r2 = model.score(X_test, y_test);
rmse = sqrt(mean((y_test - y_pred).^2));
mae = mean(abs(y_test - y_pred));
fprintf('\n📊 Model Performance:\n');
fprintf('  R² Score: %.4f\n', r2);
fprintf('  RMSE: %.2f meters\n', rmse);
fprintf('  MAE: %.2f meters\n', mae);
% Display model parameters
summary = model.getSummary();
fprintf('\n🔍 Model Parameters:\n');
fprintf('  Intercept: %.4f\n', summary.intercept);
fprintf('  Coefficient: %.4f\n', summary.coefficients);
if isfield(summary, 'coefficientStdErrors')
    fprintf('  Coefficient Std Error: %.4f\n', summary.coefficientStdErrors);
end
% Visualize results
fprintf('\n📈 Creating visualizations...\n');
visualize_results(X, y_noisy, y_true, X_train, y_train, X_test, y_test, model);
end
function visualize_results(X, y_noisy, y_true, X_train, y_train, X_test, y_test, model)
% Create comprehensive visualizations
% Berkeley colors
berkeleyBlue = [0, 50, 98]/255;
californiaGold = [253, 181, 21]/255;
figure('Position', [100, 100, 1400, 1000]);
% Plot 1: Data and model fit
subplot(2, 2, 1);
% Plot true function
plot(X, y_true, '--', 'Color', [0.5, 0.5, 0.5], 'LineWidth', 2, ...
     'DisplayName', 'True Function');
hold on;
% Plot noisy data
scatter(X_train, y_train, 30, berkeleyBlue, 'filled', 'MarkerFaceAlpha', 0.6, ...
        'DisplayName', 'Training Data');
scatter(X_test, y_test, 40, californiaGold, 'filled', 's', 'MarkerFaceAlpha', 0.8, ...
        'DisplayName', 'Test Data');
% Plot model predictions
X_plot = linspace(min(X), max(X), 200)';
y_plot = model.predict(X_plot);
plot(X_plot, y_plot, 'Color', 'red', 'LineWidth', 2, 'DisplayName', 'Model Prediction');
xlabel('Launch Angle (degrees)');
ylabel('Range (meters)');
title('Projectile Motion: Model Fit');
legend('Location', 'best');
grid on;
grid minor;
% Plot 2: Predictions vs Actual
subplot(2, 2, 2);
y_pred = model.predict(X_test);
scatter(y_test, y_pred, 50, berkeleyBlue, 'filled', 'MarkerFaceAlpha', 0.7);
hold on;
% Perfect prediction line
min_val = min([min(y_test), min(y_pred)]);
max_val = max([max(y_test), max(y_pred)]);
plot([min_val, max_val], [min_val, max_val], '--', 'Color', californiaGold, ...
     'LineWidth', 2, 'DisplayName', 'Perfect Prediction');
xlabel('Actual Range (meters)');
ylabel('Predicted Range (meters)');
title('Predictions vs Actual Values');
legend();
grid on;
grid minor;
% Plot 3: Residuals
subplot(2, 2, 3);
residuals = y_test - y_pred;
scatter(y_pred, residuals, 50, berkeleyBlue, 'filled', 'MarkerFaceAlpha', 0.7);
hold on;
yline(0, '--', 'Color', californiaGold, 'LineWidth', 2);
xlabel('Predicted Range (meters)');
ylabel('Residuals (meters)');
title('Residual Analysis');
grid on;
grid minor;
% Plot 4: Model interpretation
subplot(2, 2, 4);
% Show effect of angle on range
angles = linspace(15, 75, 100)';
ranges_pred = model.predict(angles);
plot(angles, ranges_pred, 'Color', berkeleyBlue, 'LineWidth', 3, ...
     'DisplayName', 'Model Prediction');
xlabel('Launch Angle (degrees)');
ylabel('Predicted Range (meters)');
title('Model Interpretation: Angle vs Range');
grid on;
grid minor;
% Find optimal angle
[optimal_range, optimal_idx] = max(ranges_pred);
optimal_angle = angles(optimal_idx);
hold on;
plot(optimal_angle, optimal_range, 'ro', 'MarkerSize', 10, 'LineWidth', 2, ...
     'DisplayName', sprintf('Optimal: %.1f° → %.1fm', optimal_angle, optimal_range));
legend();
sgtitle('Linear Regression Analysis: Projectile Motion', ...
        'FontSize', 16, 'FontWeight', 'bold');
end
function physics_insights()
% Provide physics insights from the model
fprintf('\n🔬 Physics Insights\n');
fprintf('==============================\n');
fprintf('In ideal projectile motion (no air resistance):\n');
fprintf('• Optimal launch angle is 45°\n');
fprintf('• Range ∝ sin(2θ)\n');
fprintf('• Maximum range = v₀²/g\n');
fprintf('\n');
fprintf('With air resistance:\n');
fprintf('• Optimal angle is slightly less than 45°\n');
fprintf('• Range decreases due to drag\n');
fprintf('• Our model captures this linear approximation\n');
end
function interactive_exploration(model)
% Interactive exploration of the model
fprintf('\n🎮 Interactive Exploration\n');
fprintf('===================================\n');
fprintf('Try different launch angles:\n');
test_angles = [30, 37, 45, 52, 60];
for i = 1:length(test_angles)
    angle = test_angles(i);
    predicted_range = model.predict(angle);
    fprintf('  %2d° → %6.2f meters\n', angle, predicted_range);
end
fprintf('\nTry your own angles! (modify the test_angles array above)\n');
end