# neural_networks
**Module:** `Python/Machine_Learning/neural_networks.py`
## Overview
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
## Functions
### `create_test_datasets()`
Create test datasets for neural network validation.
**Source:** [Line 707](Python/Machine_Learning/neural_networks.py#L707)
### `plot_training_history(history, title)`
Plot training history with Berkeley styling.
**Source:** [Line 734](Python/Machine_Learning/neural_networks.py#L734)
### `plot_network_predictions(model, X, y, title)`
Plot network predictions vs actual values.
**Source:** [Line 778](Python/Machine_Learning/neural_networks.py#L778)
## Classes
### `TrainingHistory`
Container for training history and metrics.
**Class Source:** [Line 31](Python/Machine_Learning/neural_networks.py#L31)
### `ActivationFunction`
Collection of activation functions and their derivatives.
#### Methods
##### `sigmoid(x)`
Sigmoid activation function.
**Source:** [Line 45](Python/Machine_Learning/neural_networks.py#L45)
##### `sigmoid_derivative(x)`
Derivative of sigmoid function.
**Source:** [Line 52](Python/Machine_Learning/neural_networks.py#L52)
##### `tanh(x)`
Hyperbolic tangent activation function.
**Source:** [Line 58](Python/Machine_Learning/neural_networks.py#L58)
##### `tanh_derivative(x)`
Derivative of tanh function.
**Source:** [Line 63](Python/Machine_Learning/neural_networks.py#L63)
##### `relu(x)`
ReLU activation function.
**Source:** [Line 68](Python/Machine_Learning/neural_networks.py#L68)
##### `relu_derivative(x)`
Derivative of ReLU function.
**Source:** [Line 73](Python/Machine_Learning/neural_networks.py#L73)
##### `leaky_relu(x, alpha)`
Leaky ReLU activation function.
**Source:** [Line 78](Python/Machine_Learning/neural_networks.py#L78)
##### `leaky_relu_derivative(x, alpha)`
Derivative of Leaky ReLU function.
**Source:** [Line 83](Python/Machine_Learning/neural_networks.py#L83)
##### `swish(x)`
Swish activation function.
**Source:** [Line 88](Python/Machine_Learning/neural_networks.py#L88)
##### `swish_derivative(x)`
Derivative of Swish function.
**Source:** [Line 93](Python/Machine_Learning/neural_networks.py#L93)
##### `linear(x)`
Linear activation function.
**Source:** [Line 99](Python/Machine_Learning/neural_networks.py#L99)
##### `linear_derivative(x)`
Derivative of linear function.
**Source:** [Line 104](Python/Machine_Learning/neural_networks.py#L104)
**Class Source:** [Line 41](Python/Machine_Learning/neural_networks.py#L41)
### `LossFunction`
Collection of loss functions and their derivatives.
#### Methods
##### `mse(y_true, y_pred)`
Mean squared error loss.
**Source:** [Line 113](Python/Machine_Learning/neural_networks.py#L113)
##### `mse_derivative(y_true, y_pred)`
Derivative of MSE loss.
**Source:** [Line 118](Python/Machine_Learning/neural_networks.py#L118)
##### `mae(y_true, y_pred)`
Mean absolute error loss.
**Source:** [Line 123](Python/Machine_Learning/neural_networks.py#L123)
##### `mae_derivative(y_true, y_pred)`
Derivative of MAE loss.
**Source:** [Line 128](Python/Machine_Learning/neural_networks.py#L128)
##### `cross_entropy(y_true, y_pred)`
Cross-entropy loss for classification.
**Source:** [Line 133](Python/Machine_Learning/neural_networks.py#L133)
##### `cross_entropy_derivative(y_true, y_pred)`
Derivative of cross-entropy loss.
**Source:** [Line 146](Python/Machine_Learning/neural_networks.py#L146)
**Class Source:** [Line 109](Python/Machine_Learning/neural_networks.py#L109)
### `Layer`
Abstract base class for neural network layers.
#### Methods
##### `forward(self, x)`
Forward pass through the layer.
**Source:** [Line 157](Python/Machine_Learning/neural_networks.py#L157)
##### `backward(self, grad_output)`
Backward pass through the layer.
**Source:** [Line 162](Python/Machine_Learning/neural_networks.py#L162)
##### `get_parameters(self)`
Get layer parameters.
**Source:** [Line 167](Python/Machine_Learning/neural_networks.py#L167)
##### `set_parameters(self, params)`
Set layer parameters.
**Source:** [Line 172](Python/Machine_Learning/neural_networks.py#L172)
**Class Source:** [Line 153](Python/Machine_Learning/neural_networks.py#L153)
### `DenseLayer`
Fully connected (dense) layer.
#### Methods
##### `__init__(self, input_size, output_size, activation, use_bias, weight_init)`
*No documentation available.*
**Source:** [Line 180](Python/Machine_Learning/neural_networks.py#L180)
##### `_initialize_weights(self)`
Initialize layer weights.
**Source:** [Line 203](Python/Machine_Learning/neural_networks.py#L203)
##### `forward(self, x)`
Forward pass through dense layer.
**Source:** [Line 224](Python/Machine_Learning/neural_networks.py#L224)
##### `backward(self, grad_output)`
Backward pass through dense layer.
**Source:** [Line 239](Python/Machine_Learning/neural_networks.py#L239)
##### `get_parameters(self)`
Get layer parameters.
**Source:** [Line 256](Python/Machine_Learning/neural_networks.py#L256)
##### `set_parameters(self, params)`
Set layer parameters.
**Source:** [Line 263](Python/Machine_Learning/neural_networks.py#L263)
##### `get_gradients(self)`
Get parameter gradients.
**Source:** [Line 269](Python/Machine_Learning/neural_networks.py#L269)
**Class Source:** [Line 177](Python/Machine_Learning/neural_networks.py#L177)
### `MLP`
Multi-Layer Perceptron with scientific computing features.
Features:
- Flexible architecture
- Multiple optimizers
- Regularization options
- Advanced training techniques
#### Methods
##### `__init__(self, layer_sizes, activations, output_activation, use_bias, weight_init, optimizer, learning_rate, regularization, dropout_rate)`
*No documentation available.*
**Source:** [Line 288](Python/Machine_Learning/neural_networks.py#L288)
##### `_initialize_optimizer(self)`
Initialize optimizer state.
**Source:** [Line 334](Python/Machine_Learning/neural_networks.py#L334)
##### `forward(self, X, training)`
Forward pass through the network.
**Source:** [Line 364](Python/Machine_Learning/neural_networks.py#L364)
##### `backward(self, X, y, loss_fn)`
Backward pass through the network.
**Source:** [Line 377](Python/Machine_Learning/neural_networks.py#L377)
##### `_update_parameters(self)`
Update parameters using the selected optimizer.
**Source:** [Line 403](Python/Machine_Learning/neural_networks.py#L403)
##### `_sgd_update(self)`
SGD parameter update.
**Source:** [Line 412](Python/Machine_Learning/neural_networks.py#L412)
##### `_adam_update(self)`
Adam parameter update.
**Source:** [Line 425](Python/Machine_Learning/neural_networks.py#L425)
##### `_rmsprop_update(self)`
RMSprop parameter update.
**Source:** [Line 459](Python/Machine_Learning/neural_networks.py#L459)
##### `fit(self, X, y, epochs, batch_size, validation_data, loss_fn, verbose)`
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
**Source:** [Line 481](Python/Machine_Learning/neural_networks.py#L481)
##### `predict(self, X)`
Make predictions using the trained network.
**Source:** [Line 567](Python/Machine_Learning/neural_networks.py#L567)
##### `score(self, X, y, metric)`
Compute prediction score.
**Source:** [Line 571](Python/Machine_Learning/neural_networks.py#L571)
**Class Source:** [Line 277](Python/Machine_Learning/neural_networks.py#L277)
### `Autoencoder`
Autoencoder for dimensionality reduction and feature learning.
Features:
- Configurable encoder/decoder architectures
- Multiple loss functions
- Regularization options
- Latent space analysis
#### Methods
##### `__init__(self, input_dim, encoding_dims, latent_dim, activation, output_activation, optimizer, learning_rate, regularization)`
*No documentation available.*
**Source:** [Line 598](Python/Machine_Learning/neural_networks.py#L598)
##### `encode(self, X)`
Encode input to latent space.
**Source:** [Line 634](Python/Machine_Learning/neural_networks.py#L634)
##### `decode(self, Z)`
Decode from latent space to input space.
**Source:** [Line 638](Python/Machine_Learning/neural_networks.py#L638)
##### `forward(self, X)`
Forward pass through autoencoder.
**Source:** [Line 642](Python/Machine_Learning/neural_networks.py#L642)
##### `fit(self, X, epochs, batch_size, validation_split, verbose)`
Train the autoencoder.
**Source:** [Line 648](Python/Machine_Learning/neural_networks.py#L648)
##### `reconstruction_error(self, X)`
Compute reconstruction error.
**Source:** [Line 701](Python/Machine_Learning/neural_networks.py#L701)
**Class Source:** [Line 587](Python/Machine_Learning/neural_networks.py#L587)