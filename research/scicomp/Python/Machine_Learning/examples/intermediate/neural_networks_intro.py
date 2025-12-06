#!/usr/bin/env python3
"""
Intermediate Example: Neural Networks for Scientific Computing
This example demonstrates neural network fundamentals using the Berkeley
SciComp package for solving scientific problems. We'll build and train
a neural network to learn complex nonlinear relationships.
Learning Objectives:
- Understand neural network architecture
- Learn training dynamics and optimization
- Apply to nonlinear scientific problems
- Analyze network behavior and performance
Author: Berkeley SciComp Team
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
# Add package to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from Machine_Learning.neural_networks import MLP, create_test_datasets
from Machine_Learning.utils import DataProcessor, ModelEvaluator, Visualizer
# Berkeley styling
berkeley_blue = '#003262'
california_gold = '#FDB515'
plt.style.use('seaborn-v0_8')
def create_nonlinear_physics_dataset():
    """
    Create a complex physics dataset: quantum harmonic oscillator energies.
    We'll model the relationship between quantum numbers and energy levels
    with anharmonic corrections.
    """
    print("🔬 Creating quantum harmonic oscillator dataset...")
    # Quantum numbers
    n = np.arange(0, 50)  # Principal quantum number
    # Parameters
    hbar_omega = 1.0  # Fundamental frequency
    alpha = 0.1  # Anharmonicity parameter
    # Energy levels with anharmonic correction
    # E_n = ℏω(n + 1/2) + α ℏω (n + 1/2)²
    E_harmonic = hbar_omega * (n + 0.5)
    E_anharmonic = alpha * hbar_omega * (n + 0.5)**2
    E_total = E_harmonic + E_anharmonic
    # Add quantum fluctuations (noise)
    noise = np.random.normal(0, 0.02, len(E_total))
    E_measured = E_total + noise
    # Create feature matrix with multiple quantum numbers
    # Include n, n², and n³ terms for complexity
    X = np.column_stack([
        n,
        n**2,
        n**3,
        np.sin(0.1 * n),  # Periodic corrections
        np.exp(-0.05 * n)  # Damping term
    ])
    return X, E_measured, E_total
def demonstrate_neural_network_basics():
    """Demonstrate basic neural network concepts."""
    print("\n🧠 Neural Network Fundamentals")
    print("=" * 45)
    # 1. Create dataset
    X, y_measured, y_true = create_nonlinear_physics_dataset()
    print(f"Dataset created: {len(X)} quantum states")
    print(f"Features: {X.shape[1]} quantum mechanical parameters")
    print(f"Target: Energy levels (ℏω units)")
    # 2. Preprocess data
    processor = DataProcessor()
    X_scaled = processor.scale_features(X, method='standard')
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y_measured[:split_idx], y_measured[split_idx:]
    print(f"\nData preprocessed and split: {len(X_train)} training samples")
    # 3. Create neural network architectures
    networks = {
        'shallow': MLP(
            layer_sizes=[X.shape[1], 10, 1],
            activations='tanh',
            learning_rate=0.01,
            optimizer='adam'
        ),
        'deep': MLP(
            layer_sizes=[X.shape[1], 20, 15, 10, 1],
            activations='tanh',
            learning_rate=0.01,
            optimizer='adam'
        ),
        'wide': MLP(
            layer_sizes=[X.shape[1], 50, 1],
            activations='relu',
            learning_rate=0.01,
            optimizer='adam'
        )
    }
    # 4. Train networks and compare
    results = {}
    for name, network in networks.items():
        print(f"\n🔧 Training {name} network...")
        # Train with validation
        validation_data = (X_test, y_test)
        history = network.fit(
            X_train, y_train,
            epochs=200,
            batch_size=16,
            validation_data=validation_data,
            verbose=False
        )
        # Evaluate
        evaluator = ModelEvaluator()
        eval_results = evaluator.evaluate_regression(network, X_test, y_test)
        results[name] = {
            'network': network,
            'history': history,
            'evaluation': eval_results
        }
        print(f"  R² Score: {eval_results.r2:.4f}")
        print(f"  RMSE: {eval_results.rmse:.4f}")
    return results, X, y_measured, y_true, X_test, y_test
def analyze_network_behavior(results, X, y_measured, y_true, X_test, y_test):
    """Analyze and visualize network behavior."""
    print("\n📊 Analyzing Network Behavior")
    print("=" * 40)
    # Create comprehensive visualization
    fig = plt.figure(figsize=(16, 12))
    # Plot 1: Training histories
    ax1 = plt.subplot(3, 3, 1)
    for name, result in results.items():
        history = result['history']
        epochs = range(1, len(history.loss) + 1)
        ax1.plot(epochs, history.loss, label=f'{name} (train)', linewidth=2)
        if history.val_loss:
            ax1.plot(epochs, history.val_loss, '--', label=f'{name} (val)', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    # Plot 2: Model comparison
    ax2 = plt.subplot(3, 3, 2)
    r2_scores = [results[name]['evaluation'].r2 for name in results.keys()]
    rmse_scores = [results[name]['evaluation'].rmse for name in results.keys()]
    x_pos = np.arange(len(results))
    ax2.bar(x_pos, r2_scores, color=[berkeley_blue, california_gold, '#859438'])
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(results.keys())
    ax2.set_ylabel('R² Score')
    ax2.set_title('Model Comparison')
    ax2.grid(True, alpha=0.3)
    # Plot 3: Best model predictions
    best_name = max(results.keys(), key=lambda k: results[k]['evaluation'].r2)
    best_network = results[best_name]['network']
    ax3 = plt.subplot(3, 3, 3)
    # Full dataset predictions
    X_full_scaled = DataProcessor().scale_features(X, method='standard')
    y_pred_full = best_network.predict(X_full_scaled)
    # Plot quantum numbers vs energies
    n_values = X[:, 0]  # First column is quantum number n
    ax3.plot(n_values, y_true, '--', color='gray', linewidth=2, alpha=0.7, label='True Function')
    ax3.scatter(n_values, y_measured, color=berkeley_blue, alpha=0.6, s=20, label='Measurements')
    ax3.plot(n_values, y_pred_full, color='red', linewidth=2, label=f'{best_name} Network')
    ax3.set_xlabel('Quantum Number n')
    ax3.set_ylabel('Energy (ℏω)')
    ax3.set_title('Quantum Energy Levels')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    # Plot 4: Residual analysis
    ax4 = plt.subplot(3, 3, 4)
    y_pred_test = best_network.predict(X_test)
    residuals = y_test - y_pred_test
    ax4.scatter(y_pred_test, residuals, color=berkeley_blue, alpha=0.7, s=30)
    ax4.axhline(y=0, color=california_gold, linestyle='--', linewidth=2)
    ax4.set_xlabel('Predicted Energy')
    ax4.set_ylabel('Residuals')
    ax4.set_title('Residual Analysis')
    ax4.grid(True, alpha=0.3)
    # Plot 5: Network architecture visualization
    ax5 = plt.subplot(3, 3, 5)
    # Visualize the best network architecture
    layer_sizes = best_network.layer_sizes
    max_neurons = max(layer_sizes)
    for i, size in enumerate(layer_sizes):
        y_positions = np.linspace(-max_neurons/2, max_neurons/2, size)
        x_position = i
        ax5.scatter([x_position] * size, y_positions, s=100,
                   color=berkeley_blue if i == 0 else (california_gold if i == len(layer_sizes)-1 else '#859438'))
        # Draw connections
        if i < len(layer_sizes) - 1:
            next_size = layer_sizes[i + 1]
            next_y_positions = np.linspace(-max_neurons/2, max_neurons/2, next_size)
            for y1 in y_positions:
                for y2 in next_y_positions:
                    ax5.plot([x_position, x_position + 1], [y1, y2],
                           'k-', alpha=0.1, linewidth=0.5)
    ax5.set_xlim(-0.5, len(layer_sizes) - 0.5)
    ax5.set_ylim(-max_neurons/2 - 1, max_neurons/2 + 1)
    ax5.set_xlabel('Layer')
    ax5.set_ylabel('Neurons')
    ax5.set_title(f'{best_name.title()} Network Architecture')
    ax5.grid(True, alpha=0.3)
    # Plot 6: Feature importance analysis
    ax6 = plt.subplot(3, 3, 6)
    # Compute feature importance using permutation
    feature_names = ['n', 'n²', 'n³', 'sin(0.1n)', 'exp(-0.05n)']
    importances = compute_feature_importance(best_network, X_test, y_test)
    bars = ax6.barh(range(len(importances)), importances, color=berkeley_blue)
    ax6.set_yticks(range(len(importances)))
    ax6.set_yticklabels(feature_names)
    ax6.set_xlabel('Importance Score')
    ax6.set_title('Feature Importance')
    ax6.grid(True, alpha=0.3)
    # Plot 7: Activation analysis
    ax7 = plt.subplot(3, 3, 7)
    # Show activation patterns for different quantum numbers
    sample_inputs = X_full_scaled[::5]  # Every 5th sample
    activations = analyze_activations(best_network, sample_inputs)
    im = ax7.imshow(activations.T, cmap='viridis', aspect='auto')
    ax7.set_xlabel('Sample')
    ax7.set_ylabel('Neuron')
    ax7.set_title('Hidden Layer Activations')
    plt.colorbar(im, ax=ax7)
    # Plot 8: Prediction uncertainty
    ax8 = plt.subplot(3, 3, 8)
    # Bootstrap for uncertainty estimation
    uncertainties = bootstrap_uncertainty(best_network, X_test, y_test)
    ax8.scatter(y_test, y_pred_test, c=uncertainties, cmap='plasma', s=40)
    ax8.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
            '--', color='red', linewidth=2)
    ax8.set_xlabel('Actual Energy')
    ax8.set_ylabel('Predicted Energy')
    ax8.set_title('Prediction Uncertainty')
    plt.colorbar(ax8.collections[0], ax=ax8, label='Uncertainty')
    # Plot 9: Learning dynamics
    ax9 = plt.subplot(3, 3, 9)
    # Show how different optimizers perform
    optimizers = ['sgd', 'adam', 'rmsprop']
    final_losses = []
    for opt in optimizers:
        temp_network = MLP(
            layer_sizes=[X.shape[1], 20, 1],
            optimizer=opt,
            learning_rate=0.01
        )
        # Quick training
        history = temp_network.fit(X_train, y_train, epochs=50, verbose=False)
        final_losses.append(history.loss[-1])
    ax9.bar(optimizers, final_losses, color=[berkeley_blue, california_gold, '#859438'])
    ax9.set_ylabel('Final Loss')
    ax9.set_title('Optimizer Comparison')
    ax9.grid(True, alpha=0.3)
    plt.suptitle('Neural Network Analysis: Quantum Harmonic Oscillator',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    return best_network
def compute_feature_importance(network, X_test, y_test):
    """Compute feature importance using permutation method."""
    baseline_loss = np.mean((y_test - network.predict(X_test))**2)
    importances = []
    for i in range(X_test.shape[1]):
        # Permute feature i
        X_permuted = X_test.copy()
        X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
        # Compute loss with permuted feature
        permuted_loss = np.mean((y_test - network.predict(X_permuted))**2)
        # Importance is the increase in loss
        importance = permuted_loss - baseline_loss
        importances.append(importance)
    return np.array(importances)
def analyze_activations(network, inputs):
    """Analyze hidden layer activations."""
    # Get activations from first hidden layer
    activations = []
    for x in inputs:
        # Forward pass through first layer only
        output = network.layers[0].forward(x.reshape(1, -1))
        activations.append(output.flatten())
    return np.array(activations)
def bootstrap_uncertainty(network, X_test, y_test, n_bootstrap=50):
    """Estimate prediction uncertainty using bootstrap."""
    predictions = []
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(len(X_test), len(X_test), replace=True)
        X_boot = X_test[indices]
        # Predict on bootstrap sample
        y_pred_boot = network.predict(X_boot)
        predictions.append(y_pred_boot)
    # Compute uncertainty as standard deviation
    predictions = np.array(predictions)
    uncertainties = np.std(predictions, axis=0)
    return uncertainties
def physics_interpretation():
    """Provide physics interpretation of results."""
    print("\n🔬 Physics Interpretation")
    print("=" * 35)
    print("Quantum Harmonic Oscillator with Anharmonicity:")
    print("• E_n = ℏω(n + 1/2) + α(n + 1/2)² + corrections")
    print("• Neural networks can capture complex nonlinear corrections")
    print("• Feature importance reveals which quantum terms matter most")
    print("• Network learns quantum selection rules implicitly")
    print()
    print("Real-world applications:")
    print("• Molecular vibrational spectroscopy")
    print("• Quantum dot energy levels")
    print("• Solid-state physics calculations")
def advanced_techniques():
    """Demonstrate advanced neural network techniques."""
    print("\n🚀 Advanced Techniques")
    print("=" * 30)
    # Regularization example
    print("1. Regularization Effects:")
    X, y_measured, _ = create_nonlinear_physics_dataset()
    processor = DataProcessor()
    X_scaled = processor.scale_features(X, method='standard')
    split_idx = int(0.8 * len(X))
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y_measured[:split_idx], y_measured[split_idx:]
    regularizations = [0.0, 0.01, 0.1, 1.0]
    for reg in regularizations:
        network = MLP(
            layer_sizes=[X.shape[1], 30, 1],
            regularization=reg,
            learning_rate=0.01
        )
        network.fit(X_train, y_train, epochs=100, verbose=False)
        score = network.score(X_test, y_test)
        print(f"   λ = {reg:4.2f} → R² = {score:.4f}")
    print("\n2. Activation Function Comparison:")
    activations = ['tanh', 'relu', 'sigmoid']
    for activation in activations:
        network = MLP(
            layer_sizes=[X.shape[1], 20, 1],
            activations=activation,
            learning_rate=0.01
        )
        network.fit(X_train, y_train, epochs=100, verbose=False)
        score = network.score(X_test, y_test)
        print(f"   {activation:7s} → R² = {score:.4f}")
def main():
    """Main function to run the complete example."""
    print("🧠 Berkeley SciComp: Neural Networks for Science")
    print("=" * 60)
    print("Learn how neural networks solve complex scientific problems!")
    print("We'll model quantum harmonic oscillator energies.\n")
    try:
        # Set random seed for reproducibility
        np.random.seed(42)
        # Main demonstration
        results, X, y_measured, y_true, X_test, y_test = demonstrate_neural_network_basics()
        # Detailed analysis
        best_network = analyze_network_behavior(results, X, y_measured, y_true, X_test, y_test)
        # Physics interpretation
        physics_interpretation()
        # Advanced techniques
        advanced_techniques()
        print("\n✨ Example completed successfully!")
        print("\nKey Takeaways:")
        print("• Neural networks excel at learning nonlinear relationships")
        print("• Architecture choice affects performance significantly")
        print("• Feature importance reveals physical insights")
        print("• Regularization prevents overfitting")
        print("• Visualization helps understand network behavior")
        print("\nNext Steps:")
        print("• Try physics-informed neural networks (PINNs)")
        print("• Explore convolutional networks for spatial data")
        print("• Learn about uncertainty quantification")
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("Please check your installation and try again.")
if __name__ == "__main__":
    main()