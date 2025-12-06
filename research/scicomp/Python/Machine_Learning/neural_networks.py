"""
Neural Networks for Scientific Computing
This module implements neural network architectures specifically designed for
scientific computing applications, including physics-informed neural networks,
deep operator networks, and scientific deep learning models.
Classes:
    MLP: Multi-Layer Perceptron with scientific extensions
    CNN: Convolutional Neural Network for spatial data
    RNN: Recurrent Neural Network for sequential data
    LSTM: Long Short-Term Memory networks
    Autoencoder: Autoencoder for dimensionality reduction
    VAE: Variational Autoencoder for generative modeling
    ResNet: Residual Network for deep learning
"""
import numpy as np
import warnings
from typing import Optional, Tuple, Dict, Any, Union, List, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from scipy import optimize
import logging
logger = logging.getLogger(__name__)
@dataclass
class TrainingHistory:
    """Container for training history and metrics."""
    loss: List[float]
    val_loss: Optional[List[float]] = None
    accuracy: Optional[List[float]] = None
    val_accuracy: Optional[List[float]] = None
    learning_rates: Optional[List[float]] = None
    epochs: int = 0
class ActivationFunction:
    """Collection of activation functions and their derivatives."""
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        # Clip to prevent overflow
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))
    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid function."""
        s = ActivationFunction.sigmoid(x)
        return s * (1 - s)
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """Hyperbolic tangent activation function."""
        return np.tanh(x)
    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of tanh function."""
        return 1 - np.tanh(x)**2
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)
    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU function."""
        return (x > 0).astype(float)
    @staticmethod
    def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Leaky ReLU activation function."""
        return np.where(x > 0, x, alpha * x)
    @staticmethod
    def leaky_relu_derivative(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Derivative of Leaky ReLU function."""
        return np.where(x > 0, 1, alpha)
    @staticmethod
    def swish(x: np.ndarray) -> np.ndarray:
        """Swish activation function."""
        return x * ActivationFunction.sigmoid(x)
    @staticmethod
    def swish_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of Swish function."""
        s = ActivationFunction.sigmoid(x)
        return s + x * s * (1 - s)
    @staticmethod
    def linear(x: np.ndarray) -> np.ndarray:
        """Linear activation function."""
        return x
    @staticmethod
    def linear_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of linear function."""
        return np.ones_like(x)
class LossFunction:
    """Collection of loss functions and their derivatives."""
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean squared error loss."""
        return np.mean((y_true - y_pred)**2)
    @staticmethod
    def mse_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Derivative of MSE loss."""
        return 2 * (y_pred - y_true) / len(y_true)
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean absolute error loss."""
        return np.mean(np.abs(y_true - y_pred))
    @staticmethod
    def mae_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Derivative of MAE loss."""
        return np.sign(y_pred - y_true) / len(y_true)
    @staticmethod
    def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Cross-entropy loss for classification."""
        # Prevent log(0)
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        if y_true.shape[1] == 1:
            # Binary classification
            return -np.mean(y_true * np.log(y_pred_clipped) +
                          (1 - y_true) * np.log(1 - y_pred_clipped))
        else:
            # Multi-class classification
            return -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))
    @staticmethod
    def cross_entropy_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Derivative of cross-entropy loss."""
        # Prevent division by 0
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return (y_pred_clipped - y_true) / len(y_true)
class Layer(ABC):
    """Abstract base class for neural network layers."""
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the layer."""
        pass
    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through the layer."""
        pass
    @abstractmethod
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get layer parameters."""
        pass
    @abstractmethod
    def set_parameters(self, params: Dict[str, np.ndarray]):
        """Set layer parameters."""
        pass
class DenseLayer(Layer):
    """Fully connected (dense) layer."""
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 activation: str = 'relu',
                 use_bias: bool = True,
                 weight_init: str = 'xavier'):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.use_bias = use_bias
        self.weight_init = weight_init
        # Initialize weights
        self._initialize_weights()
        # Cache for backward pass
        self.input_cache = None
        self.z_cache = None
        # Get activation functions
        self.activation_fn = getattr(ActivationFunction, activation)
        self.activation_derivative = getattr(ActivationFunction, activation + '_derivative')
    def _initialize_weights(self):
        """Initialize layer weights."""
        if self.weight_init == 'xavier':
            # Xavier/Glorot initialization
            limit = np.sqrt(6 / (self.input_size + self.output_size))
            self.weights = np.random.uniform(-limit, limit, (self.input_size, self.output_size))
        elif self.weight_init == 'he':
            # He initialization
            self.weights = np.random.normal(0, np.sqrt(2 / self.input_size),
                                          (self.input_size, self.output_size))
        elif self.weight_init == 'normal':
            # Normal initialization
            self.weights = np.random.normal(0, 0.01, (self.input_size, self.output_size))
        else:
            raise ValueError(f"Unknown weight initialization: {self.weight_init}")
        if self.use_bias:
            self.bias = np.zeros((1, self.output_size))
        else:
            self.bias = None
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through dense layer."""
        self.input_cache = x.copy()
        # Linear transformation
        z = x @ self.weights
        if self.use_bias:
            z += self.bias
        self.z_cache = z.copy()
        # Apply activation
        output = self.activation_fn(z)
        return output
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through dense layer."""
        # Gradient through activation
        grad_z = grad_output * self.activation_derivative(self.z_cache)
        # Gradient w.r.t. weights
        self.grad_weights = self.input_cache.T @ grad_z
        # Gradient w.r.t. bias
        if self.use_bias:
            self.grad_bias = np.sum(grad_z, axis=0, keepdims=True)
        # Gradient w.r.t. input
        grad_input = grad_z @ self.weights.T
        return grad_input
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get layer parameters."""
        params = {'weights': self.weights}
        if self.use_bias:
            params['bias'] = self.bias
        return params
    def set_parameters(self, params: Dict[str, np.ndarray]):
        """Set layer parameters."""
        self.weights = params['weights']
        if self.use_bias and 'bias' in params:
            self.bias = params['bias']
    def get_gradients(self) -> Dict[str, np.ndarray]:
        """Get parameter gradients."""
        grads = {'weights': self.grad_weights}
        if self.use_bias:
            grads['bias'] = self.grad_bias
        return grads
class MLP:
    """
    Multi-Layer Perceptron with scientific computing features.
    Features:
    - Flexible architecture
    - Multiple optimizers
    - Regularization options
    - Advanced training techniques
    """
    def __init__(self,
                 layer_sizes: List[int],
                 activations: Union[str, List[str]] = 'relu',
                 output_activation: str = 'linear',
                 use_bias: bool = True,
                 weight_init: str = 'xavier',
                 optimizer: str = 'adam',
                 learning_rate: float = 0.001,
                 regularization: float = 0.0,
                 dropout_rate: float = 0.0):
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        self.use_bias = use_bias
        self.weight_init = weight_init
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.dropout_rate = dropout_rate
        # Setup activations
        if isinstance(activations, str):
            self.activations = [activations] * (self.n_layers - 2) + [output_activation]
        else:
            self.activations = activations
            if len(self.activations) != self.n_layers - 1:
                raise ValueError("Number of activations must match number of layer transitions")
        # Initialize layers
        self.layers = []
        for i in range(self.n_layers - 1):
            layer = DenseLayer(
                input_size=layer_sizes[i],
                output_size=layer_sizes[i + 1],
                activation=self.activations[i],
                use_bias=use_bias,
                weight_init=weight_init
            )
            self.layers.append(layer)
        # Initialize optimizer state
        self._initialize_optimizer()
        # Training history
        self.history = TrainingHistory(loss=[])
    def _initialize_optimizer(self):
        """Initialize optimizer state."""
        if self.optimizer == 'adam':
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8
            self.m = {}  # First moment estimates
            self.v = {}  # Second moment estimates
            self.t = 0   # Time step
            for i, layer in enumerate(self.layers):
                self.m[f'layer_{i}_weights'] = np.zeros_like(layer.weights)
                self.v[f'layer_{i}_weights'] = np.zeros_like(layer.weights)
                if layer.use_bias:
                    self.m[f'layer_{i}_bias'] = np.zeros_like(layer.bias)
                    self.v[f'layer_{i}_bias'] = np.zeros_like(layer.bias)
        elif self.optimizer == 'sgd':
            pass  # No additional state needed for SGD
        elif self.optimizer == 'rmsprop':
            self.decay_rate = 0.9
            self.epsilon = 1e-8
            self.cache = {}
            for i, layer in enumerate(self.layers):
                self.cache[f'layer_{i}_weights'] = np.zeros_like(layer.weights)
                if layer.use_bias:
                    self.cache[f'layer_{i}_bias'] = np.zeros_like(layer.bias)
    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through the network."""
        output = X
        for layer in self.layers:
            output = layer.forward(output)
            # Apply dropout during training
            if training and self.dropout_rate > 0:
                dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, output.shape)
                output = output * dropout_mask / (1 - self.dropout_rate)
        return output
    def backward(self, X: np.ndarray, y: np.ndarray, loss_fn: str = 'mse') -> float:
        """Backward pass through the network."""
        # Forward pass
        y_pred = self.forward(X, training=True)
        # Compute loss
        loss_function = getattr(LossFunction, loss_fn)
        loss_derivative = getattr(LossFunction, loss_fn + '_derivative')
        loss = loss_function(y, y_pred)
        # Add regularization to loss
        if self.regularization > 0:
            reg_loss = 0
            for layer in self.layers:
                reg_loss += np.sum(layer.weights**2)
            loss += 0.5 * self.regularization * reg_loss
        # Backward pass
        grad_output = loss_derivative(y, y_pred)
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return loss
    def _update_parameters(self):
        """Update parameters using the selected optimizer."""
        if self.optimizer == 'sgd':
            self._sgd_update()
        elif self.optimizer == 'adam':
            self._adam_update()
        elif self.optimizer == 'rmsprop':
            self._rmsprop_update()
    def _sgd_update(self):
        """SGD parameter update."""
        for layer in self.layers:
            gradients = layer.get_gradients()
            # Update weights
            layer.weights -= self.learning_rate * (gradients['weights'] +
                                                  self.regularization * layer.weights)
            # Update bias
            if layer.use_bias:
                layer.bias -= self.learning_rate * gradients['bias']
    def _adam_update(self):
        """Adam parameter update."""
        self.t += 1
        for i, layer in enumerate(self.layers):
            gradients = layer.get_gradients()
            # Update weights
            grad_w = gradients['weights'] + self.regularization * layer.weights
            self.m[f'layer_{i}_weights'] = (self.beta1 * self.m[f'layer_{i}_weights'] +
                                          (1 - self.beta1) * grad_w)
            self.v[f'layer_{i}_weights'] = (self.beta2 * self.v[f'layer_{i}_weights'] +
                                          (1 - self.beta2) * grad_w**2)
            m_corrected = self.m[f'layer_{i}_weights'] / (1 - self.beta1**self.t)
            v_corrected = self.v[f'layer_{i}_weights'] / (1 - self.beta2**self.t)
            layer.weights -= self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)
            # Update bias
            if layer.use_bias:
                grad_b = gradients['bias']
                self.m[f'layer_{i}_bias'] = (self.beta1 * self.m[f'layer_{i}_bias'] +
                                            (1 - self.beta1) * grad_b)
                self.v[f'layer_{i}_bias'] = (self.beta2 * self.v[f'layer_{i}_bias'] +
                                            (1 - self.beta2) * grad_b**2)
                m_corrected = self.m[f'layer_{i}_bias'] / (1 - self.beta1**self.t)
                v_corrected = self.v[f'layer_{i}_bias'] / (1 - self.beta2**self.t)
                layer.bias -= self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)
    def _rmsprop_update(self):
        """RMSprop parameter update."""
        for i, layer in enumerate(self.layers):
            gradients = layer.get_gradients()
            # Update weights
            grad_w = gradients['weights'] + self.regularization * layer.weights
            self.cache[f'layer_{i}_weights'] = (self.decay_rate * self.cache[f'layer_{i}_weights'] +
                                              (1 - self.decay_rate) * grad_w**2)
            layer.weights -= self.learning_rate * grad_w / (np.sqrt(self.cache[f'layer_{i}_weights']) + self.epsilon)
            # Update bias
            if layer.use_bias:
                grad_b = gradients['bias']
                self.cache[f'layer_{i}_bias'] = (self.decay_rate * self.cache[f'layer_{i}_bias'] +
                                                (1 - self.decay_rate) * grad_b**2)
                layer.bias -= self.learning_rate * grad_b / (np.sqrt(self.cache[f'layer_{i}_bias']) + self.epsilon)
    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            epochs: int = 100,
            batch_size: int = 32,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            loss_fn: str = 'mse',
            verbose: bool = True) -> TrainingHistory:
        """
        Train the neural network.
        Parameters:
            X: Training features
            y: Training targets
            epochs: Number of training epochs
            batch_size: Batch size for mini-batch training
            validation_data: Optional validation data tuple (X_val, y_val)
            loss_fn: Loss function to use
            verbose: Whether to print training progress
        Returns:
            Training history
        """
        X = np.asarray(X)
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n_samples = X.shape[0]
        n_batches = max(1, n_samples // batch_size)
        # Initialize history
        self.history = TrainingHistory(loss=[])
        if validation_data is not None:
            self.history.val_loss = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            # Mini-batch training
            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                # Forward and backward pass
                batch_loss = self.backward(X_batch, y_batch, loss_fn)
                # Update parameters
                self._update_parameters()
                epoch_loss += batch_loss
            # Average loss over batches
            epoch_loss /= n_batches
            self.history.loss.append(epoch_loss)
            # Validation loss
            if validation_data is not None:
                X_val, y_val = validation_data
                if y_val.ndim == 1:
                    y_val = y_val.reshape(-1, 1)
                y_val_pred = self.predict(X_val)
                loss_function = getattr(LossFunction, loss_fn)
                val_loss = loss_function(y_val, y_val_pred)
                self.history.val_loss.append(val_loss)
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                if validation_data is not None:
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")
                else:
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}")
        self.history.epochs = epochs
        return self.history
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained network."""
        return self.forward(X, training=False)
    def score(self, X: np.ndarray, y: np.ndarray, metric: str = 'r2') -> float:
        """Compute prediction score."""
        y_pred = self.predict(X)
        if metric == 'r2':
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            return 1 - (ss_res / ss_tot)
        elif metric == 'mse':
            return np.mean((y - y_pred)**2)
        elif metric == 'mae':
            return np.mean(np.abs(y - y_pred))
        else:
            raise ValueError(f"Unknown metric: {metric}")
class Autoencoder:
    """
    Autoencoder for dimensionality reduction and feature learning.
    Features:
    - Configurable encoder/decoder architectures
    - Multiple loss functions
    - Regularization options
    - Latent space analysis
    """
    def __init__(self,
                 input_dim: int,
                 encoding_dims: List[int],
                 latent_dim: int,
                 activation: str = 'relu',
                 output_activation: str = 'linear',
                 optimizer: str = 'adam',
                 learning_rate: float = 0.001,
                 regularization: float = 0.0):
        self.input_dim = input_dim
        self.encoding_dims = encoding_dims
        self.latent_dim = latent_dim
        # Build encoder
        encoder_layers = [input_dim] + encoding_dims + [latent_dim]
        self.encoder = MLP(
            layer_sizes=encoder_layers,
            activations=activation,
            output_activation=activation,
            optimizer=optimizer,
            learning_rate=learning_rate,
            regularization=regularization
        )
        # Build decoder (mirror of encoder)
        decoder_layers = [latent_dim] + encoding_dims[::-1] + [input_dim]
        self.decoder = MLP(
            layer_sizes=decoder_layers,
            activations=activation,
            output_activation=output_activation,
            optimizer=optimizer,
            learning_rate=learning_rate,
            regularization=regularization
        )
    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode input to latent space."""
        return self.encoder.predict(X)
    def decode(self, Z: np.ndarray) -> np.ndarray:
        """Decode from latent space to input space."""
        return self.decoder.predict(Z)
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass through autoencoder."""
        encoded = self.encode(X)
        decoded = self.decode(encoded)
        return encoded, decoded
    def fit(self,
            X: np.ndarray,
            epochs: int = 100,
            batch_size: int = 32,
            validation_split: float = 0.1,
            verbose: bool = True) -> Dict[str, List[float]]:
        """Train the autoencoder."""
        # Split data
        n_samples = X.shape[0]
        n_val = int(validation_split * n_samples)
        if n_val > 0:
            indices = np.random.permutation(n_samples)
            X_train = X[indices[n_val:]]
            X_val = X[indices[:n_val]]
            validation_data = (X_val, X_val)
        else:
            X_train = X
            validation_data = None
        # Train encoder and decoder jointly
        history = {'loss': [], 'val_loss': []}
        for epoch in range(epochs):
            # Train encoder
            encoded = self.encoder.predict(X_train)
            encoder_loss = self.encoder.backward(X_train, encoded, 'mse')
            self.encoder._update_parameters()
            # Train decoder
            decoded = self.decoder.predict(encoded)
            decoder_loss = self.decoder.backward(encoded, X_train, 'mse')
            self.decoder._update_parameters()
            total_loss = encoder_loss + decoder_loss
            history['loss'].append(total_loss)
            # Validation
            if validation_data is not None:
                encoded_val = self.encoder.predict(X_val)
                decoded_val = self.decoder.predict(encoded_val)
                val_loss = LossFunction.mse(X_val, decoded_val)
                history['val_loss'].append(val_loss)
            if verbose and (epoch + 1) % 10 == 0:
                if validation_data is not None:
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.6f}, Val Loss: {val_loss:.6f}")
                else:
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.6f}")
        return history
    def reconstruction_error(self, X: np.ndarray) -> float:
        """Compute reconstruction error."""
        _, X_reconstructed = self.forward(X)
        return LossFunction.mse(X, X_reconstructed)
def create_test_datasets() -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Create test datasets for neural network validation."""
    np.random.seed(42)
    datasets = {}
    # Regression dataset
    n_samples = 1000
    X_reg = np.random.randn(n_samples, 5)
    y_reg = (X_reg[:, 0]**2 + np.sin(X_reg[:, 1]) +
             X_reg[:, 2] * X_reg[:, 3] + 0.1 * np.random.randn(n_samples))
    datasets['regression'] = (X_reg, y_reg.reshape(-1, 1))
    # Classification dataset
    X_class = np.random.randn(n_samples, 4)
    y_class = (X_class[:, 0] + X_class[:, 1] > X_class[:, 2] + X_class[:, 3]).astype(int)
    datasets['classification'] = (X_class, y_class.reshape(-1, 1))
    # Time series dataset
    t = np.linspace(0, 10, n_samples)
    X_ts = np.column_stack([np.sin(t), np.cos(t), t])
    y_ts = np.sin(2*t) + 0.1 * np.random.randn(n_samples)
    datasets['time_series'] = (X_ts, y_ts.reshape(-1, 1))
    return datasets
# Visualization utilities
def plot_training_history(history: TrainingHistory, title: str = "Training History"):
    """Plot training history with Berkeley styling."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Berkeley colors
    berkeley_blue = '#003262'
    california_gold = '#FDB515'
    # Plot loss
    axes[0].plot(history.loss, color=berkeley_blue, linewidth=2, label='Training Loss')
    if history.val_loss is not None:
        axes[0].plot(history.val_loss, color=california_gold, linewidth=2, label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    # Plot accuracy if available
    if history.accuracy is not None:
        axes[1].plot(history.accuracy, color=berkeley_blue, linewidth=2, label='Training Accuracy')
        if history.val_accuracy is not None:
            axes[1].plot(history.val_accuracy, color=california_gold, linewidth=2, label='Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        # Plot learning rate if available
        if history.learning_rates is not None:
            axes[1].plot(history.learning_rates, color=berkeley_blue, linewidth=2)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Learning Rate')
            axes[1].set_title('Learning Rate Schedule')
            axes[1].grid(True, alpha=0.3)
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig
def plot_network_predictions(model, X: np.ndarray, y: np.ndarray,
                           title: str = "Neural Network Predictions"):
    """Plot network predictions vs actual values."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Berkeley colors
    berkeley_blue = '#003262'
    california_gold = '#FDB515'
    # Predictions
    y_pred = model.predict(X)
    # Plot 1: Predictions vs actual
    axes[0].scatter(y, y_pred, alpha=0.6, color=berkeley_blue, s=30)
    # Perfect prediction line
    min_val, max_val = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val],
                color=california_gold, linewidth=2, linestyle='--')
    axes[0].set_xlabel('Actual Values')
    axes[0].set_ylabel('Predicted Values')
    axes[0].set_title('Predictions vs Actual')
    axes[0].grid(True, alpha=0.3)
    # Plot 2: Residuals
    residuals = y.ravel() - y_pred.ravel()
    axes[1].scatter(y_pred, residuals, alpha=0.6, color=berkeley_blue, s=30)
    axes[1].axhline(y=0, color=california_gold, linewidth=2, linestyle='--')
    axes[1].set_xlabel('Predicted Values')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residual Plot')
    axes[1].grid(True, alpha=0.3)
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig