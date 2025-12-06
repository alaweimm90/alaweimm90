function test_machine_learning()
% TEST_MACHINE_LEARNING - Comprehensive test suite for Machine Learning package
%
% This function runs comprehensive tests for all Machine Learning components
% in the Berkeley SciComp MATLAB toolbox.
%
% Test Coverage:
%   - Supervised learning algorithms
%   - Unsupervised learning algorithms
%   - Neural networks
%   - Physics-informed ML
%   - Optimization algorithms
%   - Utility functions
%
% Author: Berkeley SciComp Team
% Date: 2024
fprintf('🧪 Berkeley SciComp: Machine Learning Test Suite\n');
fprintf('===============================================\n');
fprintf('Running comprehensive tests for Machine Learning package...\n\n');
% Initialize test results
testResults = struct();
testResults.passed = 0;
testResults.failed = 0;
testResults.total = 0;
try
    % Test supervised learning
    fprintf('📊 Testing Supervised Learning...\n');
    testResults = run_supervised_tests(testResults);
    % Test unsupervised learning
    fprintf('\n🔍 Testing Unsupervised Learning...\n');
    testResults = run_unsupervised_tests(testResults);
    % Test neural networks
    fprintf('\n🧠 Testing Neural Networks...\n');
    testResults = run_neural_network_tests(testResults);
    % Test physics-informed ML
    fprintf('\n🌊 Testing Physics-Informed ML...\n');
    testResults = run_physics_informed_tests(testResults);
    % Test optimization
    fprintf('\n⚡ Testing Optimization...\n');
    testResults = run_optimization_tests(testResults);
    % Test utilities
    fprintf('\n🛠️ Testing Utilities...\n');
    testResults = run_utility_tests(testResults);
    % Print final results
    print_test_summary(testResults);
catch ME
    fprintf('❌ Critical error in test suite: %s\n', ME.message);
    fprintf('Stack trace:\n');
    for i = 1:length(ME.stack)
        fprintf('  %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
end
end
function testResults = run_supervised_tests(testResults)
% Test supervised learning algorithms
% Generate test data
rng(42);
n_samples = 100;
n_features = 3;
X = randn(n_samples, n_features);
y_reg = X * [2; -1; 3] + 0.1 * randn(n_samples, 1);
y_class = double(y_reg > median(y_reg));
% Test LinearRegression
testResults = run_test(testResults, 'LinearRegression Basic', ...
    @() test_linear_regression_basic(X, y_reg));
testResults = run_test(testResults, 'LinearRegression Solvers', ...
    @() test_linear_regression_solvers(X, y_reg));
% Test PolynomialRegression
testResults = run_test(testResults, 'PolynomialRegression', ...
    @() test_polynomial_regression(X, y_reg));
% Test LogisticRegression
testResults = run_test(testResults, 'LogisticRegression', ...
    @() test_logistic_regression(X, y_class));
end
function testResults = run_unsupervised_tests(testResults)
% Test unsupervised learning algorithms
% Generate test data
rng(42);
n_samples = 200;
n_features = 4;
X = [randn(100, n_features); randn(100, n_features) + 3];
% Test KMeans
testResults = run_test(testResults, 'KMeans Basic', ...
    @() test_kmeans_basic(X));
testResults = run_test(testResults, 'KMeans Initialization', ...
    @() test_kmeans_initialization(X));
% Test PCA
testResults = run_test(testResults, 'PCA Basic', ...
    @() test_pca_basic(X));
testResults = run_test(testResults, 'PCA Algorithms', ...
    @() test_pca_algorithms(X));
end
function testResults = run_neural_network_tests(testResults)
% Test neural network implementations
% Generate test data
rng(42);
n_samples = 150;
n_features = 5;
X = randn(n_samples, n_features);
y = sin(sum(X, 2)) + 0.1 * randn(n_samples, 1);
% Test MLP
testResults = run_test(testResults, 'MLP Basic', ...
    @() test_mlp_basic(X, y));
testResults = run_test(testResults, 'MLP Activations', ...
    @() test_mlp_activations(X, y));
% Test Autoencoder
testResults = run_test(testResults, 'Autoencoder', ...
    @() test_autoencoder(X));
end
function testResults = run_physics_informed_tests(testResults)
% Test physics-informed ML methods
% Test PINN
testResults = run_test(testResults, 'PINN Creation', ...
    @() test_pinn_creation());
testResults = run_test(testResults, 'PINN Heat Equation', ...
    @() test_pinn_heat_equation());
end
function testResults = run_optimization_tests(testResults)
% Test optimization algorithms
% Simple quadratic objective
obj_func = @(x) sum((x - [1; 2]).^2);
x0 = [0; 0];
% Test SGD
testResults = run_test(testResults, 'SGD Optimizer', ...
    @() test_sgd_optimizer(obj_func, x0));
% Test Adam
testResults = run_test(testResults, 'Adam Optimizer', ...
    @() test_adam_optimizer(obj_func, x0));
end
function testResults = run_utility_tests(testResults)
% Test utility functions
% Generate test data
rng(42);
n_samples = 100;
n_features = 4;
X = randn(n_samples, n_features);
y = randn(n_samples, 1);
% Add some missing values
X(randi(n_samples, 5, 1), randi(n_features, 5, 1)) = NaN;
% Test DataProcessor
testResults = run_test(testResults, 'DataProcessor Cleaning', ...
    @() test_data_processor_cleaning(X, y));
testResults = run_test(testResults, 'DataProcessor Scaling', ...
    @() test_data_processor_scaling(X));
testResults = run_test(testResults, 'DataProcessor Splitting', ...
    @() test_data_processor_splitting(X, y));
end
% Individual test functions
function test_linear_regression_basic(X, y)
% Test basic linear regression functionality
model = supervised.LinearRegression('Solver', 'svd');
model.fit(X, y);
% Check model is fitted
assert(model.isFitted(), 'Model should be fitted');
% Test predictions
y_pred = model.predict(X);
assert(length(y_pred) == length(y), 'Prediction length mismatch');
% Test scoring
r2 = model.score(X, y);
assert(r2 > 0.8, 'R² score should be high for good fit');
end
function test_linear_regression_solvers(X, y)
% Test different solvers
solvers = {'svd', 'normal'};
for i = 1:length(solvers)
    model = supervised.LinearRegression('Solver', solvers{i});
    model.fit(X, y);
    r2 = model.score(X, y);
    assert(r2 > 0.5, sprintf('Solver %s failed', solvers{i}));
end
end
function test_polynomial_regression(X, y)
% Test polynomial regression
% Use only first feature for simplicity
X_1d = X(:, 1);
model = supervised.PolynomialRegression(2);
model.fit(X_1d, y);
y_pred = model.predict(X_1d);
assert(length(y_pred) == length(y), 'Prediction length mismatch');
end
function test_logistic_regression(X, y)
% Test logistic regression
model = supervised.LogisticRegression();
model.fit(X, y);
y_pred = model.predict(X);
assert(all(ismember(y_pred, [0, 1])), 'Predictions should be binary');
accuracy = mean(y_pred == y);
assert(accuracy > 0.6, 'Accuracy should be reasonable');
end
function test_kmeans_basic(X)
% Test basic K-means functionality
kmeans = unsupervised.KMeans(2);
labels = kmeans.fit(X);
assert(length(unique(labels)) <= 2, 'Should have at most 2 clusters');
assert(all(ismember(labels, [1, 2])), 'Labels should be 1 or 2');
centroids = kmeans.getCentroids();
assert(size(centroids, 1) == 2, 'Should have 2 centroids');
end
function test_kmeans_initialization(X)
% Test K-means initialization methods
init_methods = {'k-means++', 'random'};
for i = 1:length(init_methods)
    kmeans = unsupervised.KMeans(2, 'Init', init_methods{i});
    labels = kmeans.fit(X);
    assert(length(unique(labels)) <= 2, sprintf('Init %s failed', init_methods{i}));
end
end
function test_pca_basic(X)
% Test basic PCA functionality
pca = unsupervised.PCA(2);
X_transformed = pca.fitTransform(X);
assert(size(X_transformed, 2) == 2, 'Should have 2 components');
assert(size(X_transformed, 1) == size(X, 1), 'Sample count should match');
varianceRatio = pca.getExplainedVarianceRatio();
assert(length(varianceRatio) == 2, 'Should have 2 variance ratios');
assert(all(varianceRatio >= 0) && all(varianceRatio <= 1), 'Variance ratios should be in [0,1]');
end
function test_pca_algorithms(X)
% Test different PCA algorithms
algorithms = {'svd', 'eigen'};
for i = 1:length(algorithms)
    pca = unsupervised.PCA(2, 'Algorithm', algorithms{i});
    X_transformed = pca.fitTransform(X);
    assert(size(X_transformed, 2) == 2, sprintf('Algorithm %s failed', algorithms{i}));
end
end
function test_mlp_basic(X, y)
% Test basic MLP functionality
mlp = neural_networks.MLP([size(X, 2), 10, 1]);
mlp.fit(X, y, 'Epochs', 50, 'Verbose', false);
y_pred = mlp.predict(X);
assert(length(y_pred) == length(y), 'Prediction length mismatch');
r2 = mlp.score(X, y);
assert(r2 > 0, 'R² should be positive');
end
function test_mlp_activations(X, y)
% Test different activation functions
activations = {'relu', 'tanh', 'sigmoid'};
for i = 1:length(activations)
    mlp = neural_networks.MLP([size(X, 2), 5, 1], 'Activations', activations{i});
    mlp.fit(X, y, 'Epochs', 20, 'Verbose', false);
    y_pred = mlp.predict(X);
    assert(length(y_pred) == length(y), sprintf('Activation %s failed', activations{i}));
end
end
function test_autoencoder(X)
% Test autoencoder functionality
encoder = neural_networks.Autoencoder(size(X, 2), [8, 4], 2);
encoder.fit(X, 'Epochs', 30, 'Verbose', false);
X_encoded = encoder.encode(X);
assert(size(X_encoded, 2) == 2, 'Encoded size should be 2');
X_decoded = encoder.decode(X_encoded);
assert(size(X_decoded, 2) == size(X, 2), 'Decoded size should match original');
end
function test_pinn_creation()
% Test PINN creation
layers = [2, 20, 20, 1];
pinn = physics_informed.PINN(layers);
assert(isa(pinn, 'physics_informed.PINN'), 'Should create PINN object');
end
function test_pinn_heat_equation()
% Test PINN for heat equation (simplified)
layers = [2, 10, 1];  % Small network for fast testing
pinn = physics_informed.PINN(layers);
% Quick training
results = pinn.train([0, 1], [0, 0.1], 'EquationType', 'heat', ...
                    'Epochs', 10, 'Verbose', false);
assert(isfield(results, 'lossHistory'), 'Should return training results');
end
function test_sgd_optimizer(obj_func, x0)
% Test SGD optimizer
sgd = optimization.SGD(0.1);
[x_opt, f_opt, history] = sgd.minimize(obj_func, x0, 'MaxIter', 100, 'Verbose', false);
assert(f_opt < obj_func(x0), 'Should improve objective function');
assert(isfield(history, 'f'), 'Should return optimization history');
end
function test_adam_optimizer(obj_func, x0)
% Test Adam optimizer
adam = optimization.Adam(0.1);
[x_opt, f_opt, history] = adam.minimize(obj_func, x0, 'MaxIter', 100, 'Verbose', false);
assert(f_opt < obj_func(x0), 'Should improve objective function');
assert(isfield(history, 'f'), 'Should return optimization history');
end
function test_data_processor_cleaning(X, y)
% Test data cleaning functionality
processor = utils.DataProcessor();
[X_clean, y_clean] = processor.cleanData(X, y, 'Verbose', false);
assert(size(X_clean, 1) <= size(X, 1), 'Cleaned data should have <= samples');
assert(size(X_clean, 2) == size(X, 2), 'Feature count should match');
assert(~any(isnan(X_clean(:))), 'Cleaned data should have no NaN values');
end
function test_data_processor_scaling(X)
% Test data scaling functionality
% Remove NaN values for scaling test
X_clean = X(~any(isnan(X), 2), :);
processor = utils.DataProcessor();
X_scaled = processor.standardScale(X_clean);
assert(size(X_scaled) == size(X_clean), 'Scaled data size should match');
assert(abs(mean(X_scaled(:))) < 1e-10, 'Scaled data should have zero mean');
end
function test_data_processor_splitting(X, y)
% Test data splitting functionality
% Remove NaN values for splitting test
valid_idx = ~any(isnan(X), 2);
X_clean = X(valid_idx, :);
y_clean = y(valid_idx);
processor = utils.DataProcessor();
[X_train, X_test, y_train, y_test] = processor.trainTestSplit(X_clean, y_clean, 0.8);
assert(size(X_train, 1) + size(X_test, 1) == size(X_clean, 1), 'Split sizes should sum to total');
assert(length(y_train) == size(X_train, 1), 'X and y train sizes should match');
assert(length(y_test) == size(X_test, 1), 'X and y test sizes should match');
end
% Utility functions
function testResults = run_test(testResults, testName, testFunc)
% Run a single test and update results
testResults.total = testResults.total + 1;
try
    testFunc();
    fprintf('  ✅ %s\n', testName);
    testResults.passed = testResults.passed + 1;
catch ME
    fprintf('  ❌ %s: %s\n', testName, ME.message);
    testResults.failed = testResults.failed + 1;
end
end
function print_test_summary(testResults)
% Print final test summary
fprintf('\n' + string(repmat('=', 1, 50)) + '\n');
fprintf('📋 Test Summary\n');
fprintf(string(repmat('=', 1, 50)) + '\n');
fprintf('Total tests: %d\n', testResults.total);
fprintf('Passed: %d\n', testResults.passed);
fprintf('Failed: %d\n', testResults.failed);
fprintf('Success rate: %.1f%%\n', (testResults.passed / testResults.total) * 100);
if testResults.failed == 0
    fprintf('\n🎉 All tests passed! Machine Learning package is working correctly.\n');
else
    fprintf('\n⚠️  Some tests failed. Please check the implementation.\n');
end
end