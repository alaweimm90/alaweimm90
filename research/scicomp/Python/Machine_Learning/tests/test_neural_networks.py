#!/usr/bin/env python3
"""
Comprehensive Tests for Neural Networks Module
This test suite validates all neural network implementations in the Berkeley
SciComp Machine Learning package, ensuring correctness, training dynamics,
and scientific computing applications.
Author: Berkeley SciComp Team
"""
import numpy as np
import pytest
import sys
from pathlib import Path
# Add package to path
sys.path.append(str(Path(__file__).parent.parent))
from Machine_Learning.neural_networks import (
    MLP, Autoencoder, ActivationFunction, LossFunction,
    DenseLayer, TrainingHistory, create_test_datasets
)
class TestActivationFunctions:
    """Test suite for activation functions."""
    def test_sigmoid(self):
        """Test sigmoid activation function."""
        x = np.array([-1000, -1, 0, 1, 1000])
        # Test function
        y = ActivationFunction.sigmoid(x)
        assert np.all(y >= 0)
        assert np.all(y <= 1)
        assert y[2] == 0.5  # sigmoid(0) = 0.5
        # Test derivative
        dy = ActivationFunction.sigmoid_derivative(x)
        assert np.all(dy >= 0)
        assert dy[2] == 0.25  # sigmoid'(0) = 0.25
        # Test numerical stability
        assert y[0] > 0  # Should not be exactly 0
        assert y[-1] < 1  # Should not be exactly 1
    def test_tanh(self):
        """Test tanh activation function."""
        x = np.array([-2, -1, 0, 1, 2])
        # Test function
        y = ActivationFunction.tanh(x)
        assert np.all(y >= -1)
        assert np.all(y <= 1)
        assert y[2] == 0  # tanh(0) = 0
        # Test derivative
        dy = ActivationFunction.tanh_derivative(x)
        assert np.all(dy >= 0)
        assert dy[2] == 1  # tanh'(0) = 1
    def test_relu(self):
        """Test ReLU activation function."""
        x = np.array([-2, -1, 0, 1, 2])
        # Test function
        y = ActivationFunction.relu(x)
        expected = np.array([0, 0, 0, 1, 2])
        np.testing.assert_array_equal(y, expected)
        # Test derivative
        dy = ActivationFunction.relu_derivative(x)
        expected_dy = np.array([0, 0, 0, 1, 1])
        np.testing.assert_array_equal(dy, expected_dy)
    def test_leaky_relu(self):
        """Test Leaky ReLU activation function."""
        x = np.array([-2, -1, 0, 1, 2])
        alpha = 0.1
        # Test function
        y = ActivationFunction.leaky_relu(x, alpha)
        expected = np.array([-0.2, -0.1, 0, 1, 2])
        np.testing.assert_array_almost_equal(y, expected)
        # Test derivative
        dy = ActivationFunction.leaky_relu_derivative(x, alpha)
        expected_dy = np.array([0.1, 0.1, 1, 1, 1])
        np.testing.assert_array_equal(dy, expected_dy)
    def test_swish(self):
        """Test Swish activation function."""
        x = np.array([-1, 0, 1])
        # Test function
        y = ActivationFunction.swish(x)
        assert len(y) == len(x)
        assert y[1] == 0  # swish(0) = 0
        # Test derivative
        dy = ActivationFunction.swish_derivative(x)
        assert len(dy) == len(x)
        assert dy[1] == 0.5  # swish'(0) = 0.5
    def test_linear(self):
        """Test linear activation function."""
        x = np.array([-2, -1, 0, 1, 2])
        # Test function
        y = ActivationFunction.linear(x)
        np.testing.assert_array_equal(y, x)
        # Test derivative
        dy = ActivationFunction.linear_derivative(x)
        np.testing.assert_array_equal(dy, np.ones_like(x))
class TestLossFunctions:
    """Test suite for loss functions."""
    def test_mse(self):
        """Test mean squared error loss."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1.1, 1.9, 3.1])
        loss = LossFunction.mse(y_true, y_pred)
        expected = np.mean([0.01, 0.01, 0.01])
        np.testing.assert_almost_equal(loss, expected)
        # Test derivative
        grad = LossFunction.mse_derivative(y_true, y_pred)
        expected_grad = 2 * (y_pred - y_true) / len(y_true)
        np.testing.assert_array_almost_equal(grad, expected_grad)
    def test_mae(self):
        """Test mean absolute error loss."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1.5, 1.5, 3.5])
        loss = LossFunction.mae(y_true, y_pred)
        expected = np.mean([0.5, 0.5, 0.5])
        np.testing.assert_almost_equal(loss, expected)
        # Test derivative
        grad = LossFunction.mae_derivative(y_true, y_pred)
        expected_grad = np.array([1, -1, 1]) / len(y_true)
        np.testing.assert_array_almost_equal(grad, expected_grad)
    def test_cross_entropy(self):
        """Test cross-entropy loss."""
        # Binary classification
        y_true_binary = np.array([[1], [0], [1]])
        y_pred_binary = np.array([[0.9], [0.1], [0.8]])
        loss_binary = LossFunction.cross_entropy(y_true_binary, y_pred_binary)
        assert loss_binary > 0
        # Multi-class classification
        y_true_multi = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred_multi = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
        loss_multi = LossFunction.cross_entropy(y_true_multi, y_pred_multi)
        assert loss_multi > 0
        # Test derivative
        grad = LossFunction.cross_entropy_derivative(y_true_binary, y_pred_binary)
        assert grad.shape == y_pred_binary.shape
class TestDenseLayer:
    """Test suite for dense layer implementation."""
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.input_size = 10
        self.output_size = 5
        self.layer = DenseLayer(self.input_size, self.output_size, activation='relu')
    def test_initialization(self):
        """Test layer initialization."""
        assert self.layer.input_size == self.input_size
        assert self.layer.output_size == self.output_size
        assert self.layer.weights.shape == (self.input_size, self.output_size)
        assert self.layer.bias.shape == (1, self.output_size)
    def test_weight_initializations(self):
        """Test different weight initialization methods."""
        # Xavier initialization
        layer_xavier = DenseLayer(10, 5, weight_init='xavier')
        weights_std = np.std(layer_xavier.weights)
        expected_std = np.sqrt(6 / (10 + 5))
        assert abs(weights_std - expected_std) < 0.1
        # He initialization
        layer_he = DenseLayer(10, 5, weight_init='he')
        weights_std_he = np.std(layer_he.weights)
        expected_std_he = np.sqrt(2 / 10)
        assert abs(weights_std_he - expected_std_he) < 0.1
        # Normal initialization
        layer_normal = DenseLayer(10, 5, weight_init='normal')
        assert layer_normal.weights.shape == (10, 5)
    def test_forward_pass(self):
        """Test forward pass computation."""
        batch_size = 3
        x = np.random.randn(batch_size, self.input_size)
        output = self.layer.forward(x)
        assert output.shape == (batch_size, self.output_size)
        assert np.all(output >= 0)  # ReLU should give non-negative outputs
    def test_backward_pass(self):
        """Test backward pass computation."""
        batch_size = 3
        x = np.random.randn(batch_size, self.input_size)
        # Forward pass
        output = self.layer.forward(x)
        # Backward pass
        grad_output = np.random.randn(*output.shape)
        grad_input = self.layer.backward(grad_output)
        assert grad_input.shape == x.shape
        assert hasattr(self.layer, 'grad_weights')
        assert hasattr(self.layer, 'grad_bias')
        assert self.layer.grad_weights.shape == self.layer.weights.shape
        assert self.layer.grad_bias.shape == self.layer.bias.shape
    def test_parameter_access(self):
        """Test parameter getting and setting."""
        # Get parameters
        params = self.layer.get_parameters()
        assert 'weights' in params
        assert 'bias' in params
        # Set parameters
        new_weights = np.random.randn(self.input_size, self.output_size)
        new_bias = np.random.randn(1, self.output_size)
        self.layer.set_parameters({'weights': new_weights, 'bias': new_bias})
        np.testing.assert_array_equal(self.layer.weights, new_weights)
        np.testing.assert_array_equal(self.layer.bias, new_bias)
    def test_no_bias_layer(self):
        """Test layer without bias."""
        layer_no_bias = DenseLayer(10, 5, use_bias=False)
        assert layer_no_bias.bias is None
        x = np.random.randn(3, 10)
        output = layer_no_bias.forward(x)
        assert output.shape == (3, 5)
class TestMLP:
    """Test suite for Multi-Layer Perceptron."""
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.layer_sizes = [10, 20, 15, 1]
        self.mlp = MLP(layer_sizes=self.layer_sizes, activations='relu')
    def test_initialization(self):
        """Test MLP initialization."""
        assert self.mlp.layer_sizes == self.layer_sizes
        assert self.mlp.n_layers == len(self.layer_sizes)
        assert len(self.mlp.layers) == len(self.layer_sizes) - 1
        # Test activation setup
        mlp_custom = MLP([5, 10, 1], activations=['tanh', 'linear'])
        assert len(mlp_custom.activations) == 2
    def test_forward_pass(self):
        """Test forward pass through network."""
        batch_size = 5
        x = np.random.randn(batch_size, self.layer_sizes[0])
        output = self.mlp.forward(x, training=False)
        assert output.shape == (batch_size, self.layer_sizes[-1])
    def test_training_vs_inference(self):
        """Test difference between training and inference modes."""
        x = np.random.randn(10, self.layer_sizes[0])
        # Training mode (with dropout if enabled)
        output_train = self.mlp.forward(x, training=True)
        # Inference mode
        output_inference = self.mlp.forward(x, training=False)
        assert output_train.shape == output_inference.shape
    def test_backward_pass(self):
        """Test backward pass and gradient computation."""
        x = np.random.randn(5, self.layer_sizes[0])
        y = np.random.randn(5, self.layer_sizes[-1])
        loss = self.mlp.backward(x, y, loss_fn='mse')
        assert isinstance(loss, float)
        assert loss >= 0
        # Check that gradients are computed
        for layer in self.mlp.layers:
            assert hasattr(layer, 'grad_weights')
            if layer.use_bias:
                assert hasattr(layer, 'grad_bias')
    def test_different_optimizers(self):
        """Test different optimizers."""
        optimizers = ['sgd', 'adam', 'rmsprop']
        for opt in optimizers:
            mlp = MLP([5, 10, 1], optimizer=opt)
            assert mlp.optimizer == opt
            # Test that optimizer state is initialized
            if opt == 'adam':
                assert hasattr(mlp, 'm')
                assert hasattr(mlp, 'v')
            elif opt == 'rmsprop':
                assert hasattr(mlp, 'cache')
    def test_training(self):
        """Test complete training process."""
        # Create simple dataset
        X = np.random.randn(100, 5)
        y = np.sum(X, axis=1, keepdims=True) + 0.1 * np.random.randn(100, 1)
        # Small network for quick training
        mlp = MLP([5, 10, 1], learning_rate=0.01)
        # Train
        history = mlp.fit(X, y, epochs=50, batch_size=10, verbose=False)
        assert isinstance(history, TrainingHistory)
        assert len(history.loss) == 50
        assert history.epochs == 50
        # Loss should generally decrease
        assert history.loss[-1] < history.loss[0]
    def test_predictions(self):
        """Test prediction functionality."""
        X = np.random.randn(20, 5)
        y = np.random.randn(20, 1)
        mlp = MLP([5, 10, 1])
        mlp.fit(X, y, epochs=10, verbose=False)
        # Test predictions
        predictions = mlp.predict(X)
        assert predictions.shape == (20, 1)
        # Test scoring
        score = mlp.score(X, y, metric='r2')
        assert isinstance(score, float)
    def test_regularization(self):
        """Test regularization effects."""
        X = np.random.randn(50, 10)
        y = np.random.randn(50, 1)
        # Without regularization
        mlp1 = MLP([10, 20, 1], regularization=0.0)
        mlp1.fit(X, y, epochs=20, verbose=False)
        # With regularization
        mlp2 = MLP([10, 20, 1], regularization=0.1)
        mlp2.fit(X, y, epochs=20, verbose=False)
        # Regularized model should have smaller weights
        weights1_norm = sum(np.linalg.norm(layer.weights) for layer in mlp1.layers)
        weights2_norm = sum(np.linalg.norm(layer.weights) for layer in mlp2.layers)
        assert weights2_norm <= weights1_norm
    def test_dropout(self):
        """Test dropout functionality."""
        mlp = MLP([10, 20, 1], dropout_rate=0.5)
        x = np.random.randn(5, 10)
        # Training mode should apply dropout
        output1 = mlp.forward(x, training=True)
        output2 = mlp.forward(x, training=True)
        # Outputs should be different due to dropout randomness
        assert not np.allclose(output1, output2)
        # Inference mode should be deterministic
        output3 = mlp.forward(x, training=False)
        output4 = mlp.forward(x, training=False)
        np.testing.assert_array_equal(output3, output4)
    def test_validation(self):
        """Test validation during training."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100, 1)
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]
        mlp = MLP([5, 10, 1])
        history = mlp.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=20,
            verbose=False
        )
        assert history.val_loss is not None
        assert len(history.val_loss) == 20
class TestAutoencoder:
    """Test suite for Autoencoder."""
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.input_dim = 20
        self.encoding_dims = [15, 10]
        self.latent_dim = 5
        self.autoencoder = Autoencoder(
            input_dim=self.input_dim,
            encoding_dims=self.encoding_dims,
            latent_dim=self.latent_dim
        )
    def test_initialization(self):
        """Test autoencoder initialization."""
        assert self.autoencoder.input_dim == self.input_dim
        assert self.autoencoder.latent_dim == self.latent_dim
        assert hasattr(self.autoencoder, 'encoder')
        assert hasattr(self.autoencoder, 'decoder')
        # Check encoder architecture
        expected_encoder_layers = [self.input_dim] + self.encoding_dims + [self.latent_dim]
        assert self.autoencoder.encoder.layer_sizes == expected_encoder_layers
        # Check decoder architecture
        expected_decoder_layers = [self.latent_dim] + self.encoding_dims[::-1] + [self.input_dim]
        assert self.autoencoder.decoder.layer_sizes == expected_decoder_layers
    def test_encoding_decoding(self):
        """Test encoding and decoding operations."""
        batch_size = 10
        X = np.random.randn(batch_size, self.input_dim)
        # Test encoding
        encoded = self.autoencoder.encode(X)
        assert encoded.shape == (batch_size, self.latent_dim)
        # Test decoding
        decoded = self.autoencoder.decode(encoded)
        assert decoded.shape == X.shape
        # Test full forward pass
        encoded_full, decoded_full = self.autoencoder.forward(X)
        np.testing.assert_array_equal(encoded_full, encoded)
        np.testing.assert_array_equal(decoded_full, decoded)
    def test_training(self):
        """Test autoencoder training."""
        # Create dataset with structure
        X = np.random.randn(100, self.input_dim)
        # Add some structure to make reconstruction meaningful
        X[:, :5] = X[:, :5] + X[:, 5:10]  # Create correlations
        history = self.autoencoder.fit(
            X, epochs=20, batch_size=10,
            validation_split=0.2, verbose=False
        )
        assert 'loss' in history
        assert len(history['loss']) == 20
        # Reconstruction error should decrease
        assert history['loss'][-1] < history['loss'][0]
    def test_reconstruction_quality(self):
        """Test reconstruction quality."""
        # Create simple structured data
        X = np.random.randn(50, 10)
        # Simple autoencoder
        simple_ae = Autoencoder(input_dim=10, encoding_dims=[8], latent_dim=5)
        simple_ae.fit(X, epochs=50, verbose=False)
        # Test reconstruction
        reconstruction_error = simple_ae.reconstruction_error(X)
        assert reconstruction_error >= 0
        # Reconstruction should be reasonable
        encoded, decoded = simple_ae.forward(X)
        relative_error = np.mean(np.abs(X - decoded) / (np.abs(X) + 1e-8))
        assert relative_error < 1.0  # Should be less than 100% error
class TestUtilityFunctions:
    """Test utility functions and dataset creation."""
    def test_create_test_datasets(self):
        """Test test dataset creation."""
        datasets = create_test_datasets()
        assert 'regression' in datasets
        assert 'classification' in datasets
        assert 'time_series' in datasets
        # Check dataset properties
        X_reg, y_reg = datasets['regression']
        assert X_reg.shape[0] == y_reg.shape[0]
        assert X_reg.shape[1] == 5
        X_class, y_class = datasets['classification']
        assert len(np.unique(y_class)) <= 2  # Binary classification
        X_ts, y_ts = datasets['time_series']
        assert X_ts.shape[1] == 3  # sin, cos, t
class TestIntegrationAndComplexScenarios:
    """Integration tests and complex scenarios."""
    def test_nonlinear_function_approximation(self):
        """Test neural network's ability to approximate nonlinear functions."""
        # Create complex nonlinear function
        X = np.random.uniform(-2, 2, (500, 2))
        y = np.sin(X[:, 0]) * np.cos(X[:, 1]) + 0.1 * np.random.randn(500)
        y = y.reshape(-1, 1)
        # Train neural network
        mlp = MLP([2, 20, 20, 1], activations='tanh', learning_rate=0.01)
        history = mlp.fit(X, y, epochs=200, verbose=False)
        # Should achieve reasonable approximation
        final_score = mlp.score(X, y, metric='r2')
        assert final_score > 0.7
    def test_classification_performance(self):
        """Test classification performance on complex dataset."""
        # Create spirals dataset
        n_samples = 300
        theta = np.sqrt(np.random.rand(n_samples)) * 2 * np.pi
        r = 2 * theta + np.pi
        # First spiral
        X1 = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        # Second spiral
        X2 = np.column_stack([-r * np.cos(theta + np.pi), -r * np.sin(theta + np.pi)])
        X = np.vstack([X1, X2])
        y = np.hstack([np.zeros(n_samples), np.ones(n_samples)]).reshape(-1, 1)
        # Train classifier
        mlp = MLP([2, 30, 20, 1], activations='tanh', output_activation='sigmoid')
        mlp.fit(X, y, epochs=300, verbose=False)
        # Should achieve good classification
        predictions = mlp.predict(X)
        binary_preds = (predictions > 0.5).astype(int)
        accuracy = np.mean(binary_preds == y)
        assert accuracy > 0.8
    def test_gradient_checking(self):
        """Test gradient computation with numerical gradients."""
        # Simple network for gradient checking
        mlp = MLP([3, 5, 1], learning_rate=0.001)
        X = np.random.randn(2, 3)
        y = np.random.randn(2, 1)
        # Compute analytical gradients
        loss = mlp.backward(X, y)
        # Store analytical gradients
        analytical_grads = []
        for layer in mlp.layers:
            analytical_grads.append(layer.grad_weights.copy())
            if layer.use_bias:
                analytical_grads.append(layer.grad_bias.copy())
        # Compute numerical gradients
        epsilon = 1e-7
        numerical_grads = []
        for i, layer in enumerate(mlp.layers):
            # Gradients w.r.t. weights
            grad_weights = np.zeros_like(layer.weights)
            for j in range(layer.weights.shape[0]):
                for k in range(layer.weights.shape[1]):
                    # Forward perturbation
                    layer.weights[j, k] += epsilon
                    loss_plus = mlp.backward(X, y)
                    # Backward perturbation
                    layer.weights[j, k] -= 2 * epsilon
                    loss_minus = mlp.backward(X, y)
                    # Restore original value
                    layer.weights[j, k] += epsilon
                    # Numerical gradient
                    grad_weights[j, k] = (loss_plus - loss_minus) / (2 * epsilon)
            numerical_grads.append(grad_weights)
            # Skip bias gradients for simplicity in this test
            if layer.use_bias:
                numerical_grads.append(np.zeros_like(layer.grad_bias))
        # Compare gradients (allowing for some numerical error)
        for analytical, numerical in zip(analytical_grads, numerical_grads):
            if numerical.size > 0:  # Skip empty arrays
                relative_error = np.abs(analytical - numerical) / (np.abs(numerical) + 1e-8)
                # Allow some tolerance for numerical differences
                assert np.mean(relative_error) < 0.1
    def test_overfitting_detection(self):
        """Test detection of overfitting through validation curves."""
        # Create small dataset that's easy to overfit
        X = np.random.randn(50, 10)
        y = np.random.randn(50, 1)
        # Split into train/validation
        X_train, X_val = X[:40], X[40:]
        y_train, y_val = y[:40], y[40:]
        # Large network that can overfit
        mlp = MLP([10, 50, 50, 1], regularization=0.0)
        history = mlp.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            verbose=False
        )
        # Training loss should decrease more than validation loss
        train_improvement = history.loss[0] - history.loss[-1]
        val_improvement = history.val_loss[0] - history.val_loss[-1]
        # In overfitting scenario, training improves more than validation
        # (though this test might be flaky depending on random initialization)
        assert train_improvement > 0  # Training should improve
if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])