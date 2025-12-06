#!/usr/bin/env python3
"""
Physics-Informed Neural Networks for spin transport and LLG dynamics.

Embeds spin diffusion and magnetization physics as loss function constraints.
Solves coupled transport-dynamics problems without explicit discretization.

Author: Meshal Alawein <meshal@berkeley.edu>
Copyright © 2025 Meshal Alawein — All rights reserved.
License: MIT
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class PINNConfig:
    """PINN training configuration"""
    hidden_layers: List[int] = None
    activation: str = 'tanh'
    learning_rate: float = 1e-3
    max_epochs: int = 10000
    physics_weight: float = 1.0
    data_weight: float = 1.0
    boundary_weight: float = 1.0
    patience: int = 500
    device: str = 'auto'
    seed: int = 42
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [50, 50, 50]
        
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PhysicsInformedNN(nn.Module, ABC):
    """Base class for Physics-Informed Neural Networks"""
    
    def __init__(self, config: PINNConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        
        # Set random seeds for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Build network architecture
        self.network = self._build_network()
        self.network.to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=config.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=config.patience//4, factor=0.8, verbose=True
        )
        
        # Training history
        self.loss_history = []
        self.physics_loss_history = []
        self.data_loss_history = []
        self.boundary_loss_history = []
        
    def _build_network(self) -> nn.Module:
        """Build network with hidden layers and activation functions"""
        layers = []
        
        layers.append(nn.Linear(self.input_dim, self.config.hidden_layers[0]))
        layers.append(self._get_activation())
        
        for i in range(len(self.config.hidden_layers) - 1):
            layers.append(nn.Linear(self.config.hidden_layers[i], self.config.hidden_layers[i+1]))
            layers.append(self._get_activation())
        
        layers.append(nn.Linear(self.config.hidden_layers[-1], self.output_dim))
        
        return nn.Sequential(*layers)
    
    def _get_activation(self) -> nn.Module:
        """Select activation function"""
        activations = {
            'tanh': nn.Tanh,
            'relu': nn.ReLU,
            'sigmoid': nn.Sigmoid,
            'swish': nn.SiLU
        }
        if self.config.activation not in activations:
            raise ValueError(f"Unknown activation: {self.config.activation}")
        return activations[self.config.activation]()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network"""
        return self.network(x)
    
    @property
    @abstractmethod
    def input_dim(self) -> int:
        """Network input dimension"""
        pass
    
    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Network output dimension"""
        pass
    
    @abstractmethod
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Physics-based loss from PDE residuals"""
        pass
    
    def data_loss(self, x: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """MSE loss against training data"""
        y_pred = self.forward(x)
        return nn.MSELoss()(y_pred, y_true)
    
    def boundary_loss(self, x_boundary: torch.Tensor, 
                     boundary_values: torch.Tensor) -> torch.Tensor:
        """MSE loss for boundary conditions"""
        y_boundary = self.forward(x_boundary)
        return nn.MSELoss()(y_boundary, boundary_values)
    
    def total_loss(self, x_physics: torch.Tensor, 
                   x_data: Optional[torch.Tensor] = None,
                   y_data: Optional[torch.Tensor] = None,
                   x_boundary: Optional[torch.Tensor] = None,
                   y_boundary: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Weighted sum of physics, data, and boundary losses"""
        
        loss_physics = self.physics_loss(x_physics)
        
        loss_data = torch.tensor(0.0, device=self.device)
        if x_data is not None and y_data is not None:
            loss_data = self.data_loss(x_data, y_data)
        
        loss_boundary = torch.tensor(0.0, device=self.device)
        if x_boundary is not None and y_boundary is not None:
            loss_boundary = self.boundary_loss(x_boundary, y_boundary)
        
        total_loss = (self.config.physics_weight * loss_physics +
                     self.config.data_weight * loss_data +
                     self.config.boundary_weight * loss_boundary)
        
        loss_dict = {
            'total': total_loss.item(),
            'physics': loss_physics.item(),
            'data': loss_data.item(),
            'boundary': loss_boundary.item()
        }
        
        return total_loss, loss_dict
    
    def train_model(self, x_physics: torch.Tensor,
                   x_data: Optional[torch.Tensor] = None,
                   y_data: Optional[torch.Tensor] = None,
                   x_boundary: Optional[torch.Tensor] = None,
                   y_boundary: Optional[torch.Tensor] = None) -> Dict[str, List[float]]:
        """Train PINN with Adam optimizer and early stopping"""
        
        logger.info(f"Training PINN on {self.device}")
        logger.info(f"Physics points: {len(x_physics)}")
        if x_data is not None:
            logger.info(f"Data points: {len(x_data)}")
        if x_boundary is not None:
            logger.info(f"Boundary points: {len(x_boundary)}")
        
        self.train()
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.max_epochs):
            self.optimizer.zero_grad()
            
            loss, loss_dict = self.total_loss(x_physics, x_data, y_data, 
                                            x_boundary, y_boundary)
            
            loss.backward()
            self.optimizer.step()
            
            self.loss_history.append(loss_dict['total'])
            self.physics_loss_history.append(loss_dict['physics'])
            self.data_loss_history.append(loss_dict['data'])
            self.boundary_loss_history.append(loss_dict['boundary'])
            
            self.scheduler.step(loss)
            
            if loss_dict['total'] < best_loss:
                best_loss = loss_dict['total']
                patience_counter = 0
                torch.save(self.state_dict(), 'best_pinn_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 1000 == 0 or epoch == self.config.max_epochs - 1:
                logger.info(f"Epoch {epoch:5d}: Loss = {loss_dict['total']:.2e} "
                          f"[Physics: {loss_dict['physics']:.2e}, "
                          f"Data: {loss_dict['data']:.2e}, "
                          f"Boundary: {loss_dict['boundary']:.2e}]")
        
        self.load_state_dict(torch.load('best_pinn_model.pth'))
        
        return {
            'loss': self.loss_history,
            'physics_loss': self.physics_loss_history,
            'data_loss': self.data_loss_history,
            'boundary_loss': self.boundary_loss_history
        }
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict with trained model in eval mode"""
        self.eval()
        with torch.no_grad():
            return self.forward(x)
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training loss history"""
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.semilogy(self.loss_history)
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.semilogy(self.physics_loss_history)
        plt.title('Physics Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.semilogy(self.data_loss_history)
        plt.title('Data Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.semilogy(self.boundary_loss_history)
        plt.title('Boundary Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class SpinTransportPINN(PhysicsInformedNN):
    """PINN for 3D spin diffusion equations in F/N heterostructures"""
    
    def __init__(self, config: PINNConfig, material_params: Dict[str, float]):
        self.material_params = material_params
        super().__init__(config)
    
    @property
    def input_dim(self) -> int:
        return 3  # (x, y, z)
    
    @property
    def output_dim(self) -> int:
        return 4  # (V_c, V_sx, V_sy, V_sz)
    
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Residual loss from spin diffusion and charge continuity PDEs"""
        x.requires_grad_(True)
        
        u = self.forward(x)
        
        V_c = u[:, 0:1]
        V_sx = u[:, 1:2] 
        V_sy = u[:, 2:3]
        V_sz = u[:, 3:4]
        
        lambda_sf = self.material_params['lambda_sf']
        
        def calculate_laplacian(V, x):
            """∇²V using automatic differentiation"""
            grad_outputs = torch.ones_like(V)
            grad_V = grad(V, x, grad_outputs=grad_outputs, create_graph=True)[0]
            
            laplacian = 0
            for i in range(3):
                grad2_V = grad(grad_V[:, i:i+1], x, grad_outputs=grad_outputs, 
                             create_graph=True)[0][:, i:i+1]
                laplacian = laplacian + grad2_V
            
            return laplacian
        
        # Spin diffusion: ∇²V_s - V_s/λ_sf² = 0
        laplacian_sx = calculate_laplacian(V_sx, x)
        laplacian_sy = calculate_laplacian(V_sy, x)
        laplacian_sz = calculate_laplacian(V_sz, x)
        
        pde_sx = laplacian_sx - V_sx / (lambda_sf**2)
        pde_sy = laplacian_sy - V_sy / (lambda_sf**2)
        pde_sz = laplacian_sz - V_sz / (lambda_sf**2)
        
        # Charge continuity: ∇²V_c = 0
        pde_c = calculate_laplacian(V_c, x)
        
        physics_loss = torch.mean(pde_c**2 + pde_sx**2 + pde_sy**2 + pde_sz**2)
        
        return physics_loss

class LLGDynamicsPINN(PhysicsInformedNN):
    """PINN for solving LLG magnetization dynamics"""
    
    def __init__(self, config: PINNConfig, llg_params: Dict[str, float]):
        self.llg_params = llg_params
        super().__init__(config)
    
    @property
    def input_dim(self) -> int:
        return 1  # time t
    
    @property
    def output_dim(self) -> int:
        return 3  # (mx, my, mz) magnetization components
    
    def physics_loss(self, t: torch.Tensor) -> torch.Tensor:
        """Physics loss for LLG equation"""
        t.requires_grad_(True)
        
        # Forward pass
        m = self.forward(t)
        mx, my, mz = m[:, 0:1], m[:, 1:2], m[:, 2:3]
        
        # Calculate time derivatives
        grad_outputs = torch.ones_like(mx)
        dmx_dt = grad(mx, t, grad_outputs=grad_outputs, create_graph=True)[0]
        dmy_dt = grad(my, t, grad_outputs=grad_outputs, create_graph=True)[0]
        dmz_dt = grad(mz, t, grad_outputs=grad_outputs, create_graph=True)[0]
        
        # LLG parameters
        alpha = self.llg_params['alpha']
        gamma = self.llg_params['gamma']
        
        # Effective field (simplified - constant field)
        H_eff = torch.tensor([0.0, 0.0, 0.1], device=self.device)  # 0.1 T along z
        Hx, Hy, Hz = H_eff[0], H_eff[1], H_eff[2]
        
        # Cross products: m × H_eff
        m_cross_H_x = my * Hz - mz * Hy
        m_cross_H_y = mz * Hx - mx * Hz  
        m_cross_H_z = mx * Hy - my * Hx
        
        # m × (m × H_eff)
        m_cross_mH_x = my * m_cross_H_z - mz * m_cross_H_y
        m_cross_mH_y = mz * m_cross_H_x - mx * m_cross_H_z
        m_cross_mH_z = mx * m_cross_H_y - my * m_cross_H_x
        
        # LLG equation: dm/dt = -γ/(1+α²) [m × H + α m × (m × H)]
        prefactor = -gamma / (1 + alpha**2)
        
        # RHS of LLG equation
        dmx_dt_llg = prefactor * (m_cross_H_x + alpha * m_cross_mH_x)
        dmy_dt_llg = prefactor * (m_cross_H_y + alpha * m_cross_mH_y)
        dmz_dt_llg = prefactor * (m_cross_H_z + alpha * m_cross_mH_z)
        
        # PDE residuals
        pde_x = dmx_dt - dmx_dt_llg
        pde_y = dmy_dt - dmy_dt_llg
        pde_z = dmz_dt - dmz_dt_llg
        
        # Magnetization magnitude constraint: |m| = 1
        magnitude_constraint = (mx**2 + my**2 + mz**2 - 1)**2
        
        physics_loss = torch.mean(pde_x**2 + pde_y**2 + pde_z**2) + \
                      10.0 * torch.mean(magnitude_constraint)  # Higher weight for constraint
        
        return physics_loss

class SpinTorquePINN(PhysicsInformedNN):
    """PINN for coupled spin transport and LLG dynamics with spin-transfer torque"""
    
    def __init__(self, config: PINNConfig, device_params: Dict[str, float]):
        self.device_params = device_params
        super().__init__(config)
    
    @property
    def input_dim(self) -> int:
        return 4  # (x, y, z, t) - space and time
    
    @property
    def output_dim(self) -> int:
        return 7  # (V_c, V_sx, V_sy, V_sz, mx, my, mz)
    
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Combined physics loss for transport + dynamics"""
        x.requires_grad_(True)
        
        # Split input coordinates
        spatial_coords = x[:, :3]  # (x, y, z)
        time_coord = x[:, 3:4]     # t
        
        # Forward pass
        u = self.forward(x)
        
        # Split outputs
        V_c = u[:, 0:1]
        V_sx, V_sy, V_sz = u[:, 1:2], u[:, 2:3], u[:, 3:4]
        mx, my, mz = u[:, 4:5], u[:, 5:6], u[:, 6:7]
        
        # Physics loss components
        loss_transport = self._transport_physics_loss(u[:, :4], spatial_coords)
        loss_llg = self._llg_physics_loss(u[:, 4:], time_coord)
        loss_coupling = self._coupling_loss(u, spatial_coords, time_coord)
        
        return loss_transport + loss_llg + 0.1 * loss_coupling
    
    def _transport_physics_loss(self, V, spatial_coords):
        """Transport physics (similar to SpinTransportPINN)"""
        # Implementation similar to SpinTransportPINN.physics_loss
        # but working with the voltage components from the combined output
        return torch.tensor(0.0, device=self.device)  # Placeholder
    
    def _llg_physics_loss(self, m, time_coord):
        """LLG physics (similar to LLGDynamicsPINN)"""
        # Implementation similar to LLGDynamicsPINN.physics_loss
        # but working with magnetization from combined output
        return torch.tensor(0.0, device=self.device)  # Placeholder
    
    def _coupling_loss(self, u, spatial_coords, time_coord):
        """Coupling between transport and dynamics (spin-transfer torque)"""
        # This would implement the coupling terms between spin currents and torques
        return torch.tensor(0.0, device=self.device)  # Placeholder

def create_training_data(domain_bounds: Dict[str, Tuple[float, float]], 
                        n_points: int) -> torch.Tensor:
    """Create training points for PINN"""
    
    dims = []
    bounds_list = []
    
    for dim_name, (min_val, max_val) in domain_bounds.items():
        bounds_list.append((min_val, max_val))
        dims.append(dim_name)
    
    # Generate random points in the domain
    points = []
    for min_val, max_val in bounds_list:
        points.append(np.random.uniform(min_val, max_val, n_points))
    
    training_points = np.column_stack(points)
    
    return torch.FloatTensor(training_points)

def solve_spin_transport_pinn(material_params: Dict[str, float],
                             domain_bounds: Dict[str, Tuple[float, float]],
                             boundary_conditions: Optional[Dict] = None,
                             config: Optional[PINNConfig] = None) -> Tuple[SpinTransportPINN, Dict]:
    """Solve spin transport using PINN"""
    
    config = config or PINNConfig()
    
    # Create model
    model = SpinTransportPINN(config, material_params)
    
    # Create training data
    x_physics = create_training_data(domain_bounds, 10000)
    
    # Add boundary conditions if provided
    x_boundary, y_boundary = None, None
    if boundary_conditions:
        # This would set up boundary condition data
        pass
    
    # Train model
    history = model.train_model(x_physics, x_boundary=x_boundary, y_boundary=y_boundary)
    
    return model, history

def solve_llg_dynamics_pinn(llg_params: Dict[str, float],
                           time_domain: Tuple[float, float],
                           initial_conditions: Optional[torch.Tensor] = None,
                           config: Optional[PINNConfig] = None) -> Tuple[LLGDynamicsPINN, Dict]:
    """Solve LLG dynamics using PINN"""
    
    config = config or PINNConfig()
    
    # Create model
    model = LLGDynamicsPINN(config, llg_params)
    
    # Create training data
    domain_bounds = {'t': time_domain}
    t_physics = create_training_data(domain_bounds, 1000)
    
    # Initial condition as boundary condition
    x_boundary, y_boundary = None, None
    if initial_conditions is not None:
        t_init = torch.zeros(1, 1)
        x_boundary = t_init
        y_boundary = initial_conditions.reshape(1, -1)
    
    # Train model
    history = model.train_model(t_physics, x_boundary=x_boundary, y_boundary=y_boundary)
    
    return model, history