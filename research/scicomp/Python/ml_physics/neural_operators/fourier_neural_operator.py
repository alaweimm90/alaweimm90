#!/usr/bin/env python3
"""
Fourier Neural Operator (FNO)
Advanced implementation of Fourier Neural Operators for learning solution
operators of partial differential equations. FNOs can learn mappings between
infinite-dimensional function spaces and generalize across different PDE
parameters and domain discretizations.
Key Features:
- 1D, 2D, and 3D Fourier Neural Operators
- Spectral convolution in Fourier domain
- Resolution-invariant learning
- Multiple PDE applications (Navier-Stokes, Darcy flow, etc.)
- Efficient implementation with FFT operations
Applications:
- Fluid dynamics simulation acceleration
- Climate modeling and weather prediction
- Material property prediction
- Electromagnetic field computation
- Seismic wave propagation
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
Created: 2025
License: MIT
Copyright © 2025 Meshal Alawein — All rights reserved.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union, Callable, Dict, List, Any
from dataclasses import dataclass
import warnings
# Constants and Berkeley styling
pi = np.pi
BERKELEY_BLUE = '#003262'
CALIFORNIA_GOLD = '#FDB515'
@dataclass
class FNOConfig:
    """Configuration for Fourier Neural Operator."""
    # Network architecture
    n_modes_x: int = 16  # Number of Fourier modes in x direction
    n_modes_y: Optional[int] = None  # For 2D problems
    n_modes_z: Optional[int] = None  # For 3D problems
    hidden_channels: int = 64
    n_layers: int = 4
    # Input/output dimensions
    input_channels: int = 1
    output_channels: int = 1
    # Training parameters
    batch_size: int = 32
    n_epochs: int = 500
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    # Data parameters
    grid_size: Union[int, Tuple[int, ...]] = 64
    # Regularization
    dropout: float = 0.0
    spectral_norm: bool = False
    def __post_init__(self):
        if isinstance(self.grid_size, int):
            self.grid_size = (self.grid_size,)
class SpectralConv1d(nn.Module):
    """1D Spectral Convolution Layer."""
    def __init__(self, in_channels: int, out_channels: int, n_modes: int):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        # Fourier weights (complex-valued)
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            torch.randn(in_channels, out_channels, n_modes, dtype=torch.cfloat) * scale
        )
    def forward(self, x):
        """
        Forward pass of spectral convolution.
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, channels, grid_x)
        Returns
        -------
        torch.Tensor
            Output tensor after spectral convolution
        """
        batch_size, in_channels, grid_x = x.shape
        # FFT along spatial dimension
        x_ft = torch.fft.rfft(x, dim=-1)
        # Truncate to n_modes
        x_ft = x_ft[:, :, :self.n_modes]
        # Spectral convolution
        out_ft = torch.zeros(batch_size, self.out_channels, self.n_modes,
                           dtype=torch.cfloat, device=x.device)
        for i in range(self.in_channels):
            for j in range(self.out_channels):
                out_ft[:, j, :] += x_ft[:, i, :] * self.weights[i, j, :]
        # Pad back to original size
        x_size = x.shape[-1]
        out_ft_padded = torch.zeros(batch_size, self.out_channels,
                                   x_size//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft_padded[:, :, :self.n_modes] = out_ft
        # Inverse FFT
        x_out = torch.fft.irfft(out_ft_padded, n=x_size, dim=-1)
        return x_out
class SpectralConv2d(nn.Module):
    """2D Spectral Convolution Layer."""
    def __init__(self, in_channels: int, out_channels: int,
                 n_modes_x: int, n_modes_y: int):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes_x = n_modes_x
        self.n_modes_y = n_modes_y
        # Fourier weights
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            torch.randn(in_channels, out_channels, n_modes_x, n_modes_y,
                       dtype=torch.cfloat) * scale
        )
        self.weights2 = nn.Parameter(
            torch.randn(in_channels, out_channels, n_modes_x, n_modes_y,
                       dtype=torch.cfloat) * scale
        )
    def forward(self, x):
        """Forward pass of 2D spectral convolution."""
        batch_size, in_channels, grid_x, grid_y = x.shape
        # 2D FFT
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))
        # Initialize output
        out_ft = torch.zeros(batch_size, self.out_channels,
                           grid_x, grid_y//2 + 1,
                           dtype=torch.cfloat, device=x.device)
        # Spectral convolution (low frequency modes)
        for i in range(self.in_channels):
            for j in range(self.out_channels):
                # Upper left quadrant
                out_ft[:, j, :self.n_modes_x, :self.n_modes_y] += \
                    x_ft[:, i, :self.n_modes_x, :self.n_modes_y] * self.weights1[i, j, :, :]
                # Lower left quadrant
                out_ft[:, j, -self.n_modes_x:, :self.n_modes_y] += \
                    x_ft[:, i, -self.n_modes_x:, :self.n_modes_y] * self.weights2[i, j, :, :]
        # Inverse 2D FFT
        x_out = torch.fft.irfft2(out_ft, s=(grid_x, grid_y), dim=(-2, -1))
        return x_out
class SpectralConv3d(nn.Module):
    """3D Spectral Convolution Layer."""
    def __init__(self, in_channels: int, out_channels: int,
                 n_modes_x: int, n_modes_y: int, n_modes_z: int):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes_x = n_modes_x
        self.n_modes_y = n_modes_y
        self.n_modes_z = n_modes_z
        # Fourier weights for different quadrants
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            torch.randn(in_channels, out_channels, n_modes_x, n_modes_y, n_modes_z,
                       dtype=torch.cfloat) * scale
        )
        self.weights2 = nn.Parameter(
            torch.randn(in_channels, out_channels, n_modes_x, n_modes_y, n_modes_z,
                       dtype=torch.cfloat) * scale
        )
        self.weights3 = nn.Parameter(
            torch.randn(in_channels, out_channels, n_modes_x, n_modes_y, n_modes_z,
                       dtype=torch.cfloat) * scale
        )
        self.weights4 = nn.Parameter(
            torch.randn(in_channels, out_channels, n_modes_x, n_modes_y, n_modes_z,
                       dtype=torch.cfloat) * scale
        )
    def forward(self, x):
        """Forward pass of 3D spectral convolution."""
        batch_size, in_channels, grid_x, grid_y, grid_z = x.shape
        # 3D FFT
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))
        # Initialize output
        out_ft = torch.zeros(batch_size, self.out_channels,
                           grid_x, grid_y, grid_z//2 + 1,
                           dtype=torch.cfloat, device=x.device)
        # Spectral convolution (8 octants)
        for i in range(self.in_channels):
            for j in range(self.out_channels):
                # Octant 1: +++
                out_ft[:, j, :self.n_modes_x, :self.n_modes_y, :self.n_modes_z] += \
                    x_ft[:, i, :self.n_modes_x, :self.n_modes_y, :self.n_modes_z] * \
                    self.weights1[i, j, :, :, :]
                # Octant 2: ++-
                out_ft[:, j, :self.n_modes_x, :self.n_modes_y, -self.n_modes_z:] += \
                    x_ft[:, i, :self.n_modes_x, :self.n_modes_y, -self.n_modes_z:] * \
                    self.weights2[i, j, :, :, :]
                # Octant 3: +-+
                out_ft[:, j, :self.n_modes_x, -self.n_modes_y:, :self.n_modes_z] += \
                    x_ft[:, i, :self.n_modes_x, -self.n_modes_y:, :self.n_modes_z] * \
                    self.weights3[i, j, :, :, :]
                # Octant 4: +--
                out_ft[:, j, :self.n_modes_x, -self.n_modes_y:, -self.n_modes_z:] += \
                    x_ft[:, i, :self.n_modes_x, -self.n_modes_y:, -self.n_modes_z:] * \
                    self.weights4[i, j, :, :, :]
        # Inverse 3D FFT
        x_out = torch.fft.irfftn(out_ft, s=(grid_x, grid_y, grid_z), dim=(-3, -2, -1))
        return x_out
class FNOLayer(nn.Module):
    """Single FNO Layer combining spectral and local convolutions."""
    def __init__(self, channels: int, n_modes: Union[int, Tuple[int, ...]],
                 dimensions: int = 1, dropout: float = 0.0):
        super(FNOLayer, self).__init__()
        self.channels = channels
        self.dimensions = dimensions
        # Spectral convolution
        if dimensions == 1:
            self.spectral_conv = SpectralConv1d(channels, channels, n_modes)
            self.local_conv = nn.Conv1d(channels, channels, 1)
        elif dimensions == 2:
            n_modes_x, n_modes_y = n_modes if isinstance(n_modes, tuple) else (n_modes, n_modes)
            self.spectral_conv = SpectralConv2d(channels, channels, n_modes_x, n_modes_y)
            self.local_conv = nn.Conv2d(channels, channels, 1)
        elif dimensions == 3:
            n_modes_x, n_modes_y, n_modes_z = n_modes if isinstance(n_modes, tuple) else (n_modes, n_modes, n_modes)
            self.spectral_conv = SpectralConv3d(channels, channels, n_modes_x, n_modes_y, n_modes_z)
            self.local_conv = nn.Conv3d(channels, channels, 1)
        else:
            raise ValueError(f"Unsupported dimensions: {dimensions}")
        # Activation and normalization
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Layer normalization
        if dimensions == 1:
            self.norm = nn.LayerNorm([channels])
        elif dimensions == 2:
            self.norm = nn.GroupNorm(min(32, channels), channels)
        else:
            self.norm = nn.GroupNorm(min(32, channels), channels)
    def forward(self, x):
        """Forward pass through FNO layer."""
        # Spectral convolution
        x_spectral = self.spectral_conv(x)
        # Local convolution
        x_local = self.local_conv(x)
        # Combine and apply activation
        x = self.activation(x_spectral + x_local + x)
        x = self.dropout(x)
        # Normalization (reshape for layer norm if needed)
        if self.dimensions == 1 and isinstance(self.norm, nn.LayerNorm):
            original_shape = x.shape
            x = x.transpose(-1, -2)  # (batch, grid, channels)
            x = self.norm(x)
            x = x.transpose(-1, -2)  # (batch, channels, grid)
        else:
            x = self.norm(x)
        return x
class FourierNeuralOperator(nn.Module):
    """
    Fourier Neural Operator for learning solution operators of PDEs.
    The FNO learns mappings between function spaces by parameterizing
    the integral kernel in Fourier space, enabling resolution-invariant
    learning and efficient computation.
    Parameters
    ----------
    config : FNOConfig
        Configuration parameters for the FNO
    """
    def __init__(self, config: FNOConfig):
        super(FourierNeuralOperator, self).__init__()
        self.config = config
        self.dimensions = len(config.grid_size)
        # Input projection
        self.input_proj = self._create_input_projection()
        # FNO layers
        self.fno_layers = nn.ModuleList()
        n_modes = self._get_n_modes()
        for _ in range(config.n_layers):
            layer = FNOLayer(
                config.hidden_channels,
                n_modes,
                self.dimensions,
                config.dropout
            )
            self.fno_layers.append(layer)
        # Output projection
        self.output_proj = self._create_output_projection()
        # Initialize weights
        self.apply(self._init_weights)
    def _create_input_projection(self):
        """Create input projection layers."""
        if self.dimensions == 1:
            return nn.Sequential(
                nn.Conv1d(self.config.input_channels, self.config.hidden_channels, 1),
                nn.GELU()
            )
        elif self.dimensions == 2:
            return nn.Sequential(
                nn.Conv2d(self.config.input_channels, self.config.hidden_channels, 1),
                nn.GELU()
            )
        elif self.dimensions == 3:
            return nn.Sequential(
                nn.Conv3d(self.config.input_channels, self.config.hidden_channels, 1),
                nn.GELU()
            )
    def _create_output_projection(self):
        """Create output projection layers."""
        if self.dimensions == 1:
            return nn.Sequential(
                nn.Conv1d(self.config.hidden_channels, self.config.hidden_channels, 1),
                nn.GELU(),
                nn.Conv1d(self.config.hidden_channels, self.config.output_channels, 1)
            )
        elif self.dimensions == 2:
            return nn.Sequential(
                nn.Conv2d(self.config.hidden_channels, self.config.hidden_channels, 1),
                nn.GELU(),
                nn.Conv2d(self.config.hidden_channels, self.config.output_channels, 1)
            )
        elif self.dimensions == 3:
            return nn.Sequential(
                nn.Conv3d(self.config.hidden_channels, self.config.hidden_channels, 1),
                nn.GELU(),
                nn.Conv3d(self.config.hidden_channels, self.config.output_channels, 1)
            )
    def _get_n_modes(self):
        """Get number of Fourier modes for each dimension."""
        if self.dimensions == 1:
            return self.config.n_modes_x
        elif self.dimensions == 2:
            n_modes_y = self.config.n_modes_y or self.config.n_modes_x
            return (self.config.n_modes_x, n_modes_y)
        elif self.dimensions == 3:
            n_modes_y = self.config.n_modes_y or self.config.n_modes_x
            n_modes_z = self.config.n_modes_z or self.config.n_modes_x
            return (self.config.n_modes_x, n_modes_y, n_modes_z)
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    def forward(self, x):
        """
        Forward pass through FNO.
        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape:
            - 1D: (batch, input_channels, grid_x)
            - 2D: (batch, input_channels, grid_x, grid_y)
            - 3D: (batch, input_channels, grid_x, grid_y, grid_z)
        Returns
        -------
        torch.Tensor
            Output tensor with same spatial dimensions but output_channels
        """
        # Input projection
        x = self.input_proj(x)
        # FNO layers
        for layer in self.fno_layers:
            x = layer(x)
        # Output projection
        x = self.output_proj(x)
        return x
class FNOTrainer:
    """Trainer class for Fourier Neural Operators."""
    def __init__(self, model: FourierNeuralOperator, config: FNOConfig):
        self.model = model
        self.config = config
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        # Optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=100, gamma=0.5
        )
        # Loss function
        self.criterion = nn.MSELoss()
        # Training history
        self.train_losses = []
        self.val_losses = []
    def train_step(self, x_batch, y_batch):
        """Single training step."""
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        self.optimizer.zero_grad()
        # Forward pass
        y_pred = self.model(x_batch)
        # Compute loss
        loss = self.criterion(y_pred, y_batch)
        # Backward pass
        loss.backward()
        self.optimizer.step()
        return loss.item()
    def validate(self, val_loader):
        """Validation step."""
        self.model.eval()
        val_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                y_pred = self.model(x_batch)
                loss = self.criterion(y_pred, y_batch)
                val_loss += loss.item()
                n_batches += 1
        self.model.train()
        return val_loss / n_batches if n_batches > 0 else 0.0
    def train(self, train_loader, val_loader=None):
        """Train the FNO model."""
        print(f"Training FNO on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self.model.train()
        for epoch in range(self.config.n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for x_batch, y_batch in train_loader:
                loss = self.train_step(x_batch, y_batch)
                epoch_loss += loss
                n_batches += 1
            # Average training loss
            avg_train_loss = epoch_loss / n_batches
            self.train_losses.append(avg_train_loss)
            # Validation
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
            # Learning rate scheduling
            self.scheduler.step()
            # Print progress
            if epoch % 50 == 0:
                print(f"Epoch {epoch}/{self.config.n_epochs}")
                print(f"  Train Loss: {avg_train_loss:.6f}")
                if val_loader is not None:
                    print(f"  Val Loss: {val_loss:.6f}")
                print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        print("Training completed!")
    def plot_training_history(self):
        """Plot training and validation loss."""
        berkeley_plot = BerkeleyPlot()
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(len(self.train_losses))
        ax.semilogy(epochs, self.train_losses,
                   color=berkeley_plot.colors['berkeley_blue'],
                   linewidth=2, label='Training Loss')
        if self.val_losses:
            ax.semilogy(epochs, self.val_losses,
                       color=berkeley_plot.colors['california_gold'],
                       linewidth=2, label='Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('FNO Training History')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.show()
# Utility functions for generating synthetic data
def generate_burgers_data(n_samples: int = 1000, grid_size: int = 64,
                         viscosity: float = 0.01, T: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic Burgers equation data.
    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    grid_size : int
        Spatial grid resolution
    viscosity : float
        Viscosity parameter
    T : float
        Final time
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Initial conditions and final solutions
    """
    x = torch.linspace(0, 2*pi, grid_size, dtype=torch.float32)
    # Random initial conditions
    initial_conditions = []
    final_solutions = []
    for _ in range(n_samples):
        # Random Fourier coefficients for initial condition
        n_modes = 5
        coeffs = torch.randn(n_modes) * 0.5
        # Generate initial condition
        u0 = torch.zeros_like(x)
        for k in range(1, n_modes + 1):
            u0 += coeffs[k-1] * torch.sin(k * x)
        # Solve Burgers equation (simplified - in practice use proper solver)
        # This is a placeholder - use actual numerical solver
        u_final = u0 * torch.exp(-viscosity * T * (torch.arange(grid_size, dtype=torch.float32)**2))
        initial_conditions.append(u0)
        final_solutions.append(u_final)
    initial_conditions = torch.stack(initial_conditions).unsqueeze(1)  # Add channel dim
    final_solutions = torch.stack(final_solutions).unsqueeze(1)
    return initial_conditions, final_solutions
if __name__ == "__main__":
    # Example: 1D Burgers equation
    print("=== FNO for 1D Burgers Equation ===")
    # Configuration
    config = FNOConfig(
        n_modes_x=16,
        hidden_channels=32,
        n_layers=4,
        input_channels=1,
        output_channels=1,
        grid_size=(64,),
        n_epochs=200,
        learning_rate=1e-3
    )
    # Create model
    model = FourierNeuralOperator(config)
    trainer = FNOTrainer(model, config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    # Generate synthetic data
    print("Generating synthetic Burgers equation data...")
    X_train, y_train = generate_burgers_data(n_samples=800, grid_size=64)
    X_val, y_val = generate_burgers_data(n_samples=200, grid_size=64)
    print(f"Training data shape: {X_train.shape} -> {y_train.shape}")
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False
    )
    # Train model
    trainer.train(train_loader, val_loader)
    # Plot training history
    trainer.plot_training_history()
    # Test prediction
    model.eval()
    with torch.no_grad():
        x_test = X_val[:5].to(trainer.device)
        y_pred = model(x_test).cpu()
        y_true = y_val[:5]
    # Plot comparison
    berkeley_plot = BerkeleyPlot()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    x_grid = np.linspace(0, 2*pi, 64)
    for i in range(5):
        ax = axes[i]
        ax.plot(x_grid, x_test[i, 0].cpu(), 'k--', linewidth=2, label='Initial')
        ax.plot(x_grid, y_true[i, 0], color=berkeley_plot.colors['berkeley_blue'],
               linewidth=2, label='True')
        ax.plot(x_grid, y_pred[i, 0], color=berkeley_plot.colors['california_gold'],
               linewidth=2, label='FNO')
        ax.set_title(f'Test Sample {i+1}')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
    # Remove empty subplot
    axes[5].remove()
    plt.tight_layout()
    plt.show()
    print("FNO demonstration completed!")