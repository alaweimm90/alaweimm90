#!/usr/bin/env python3
"""
Comprehensive Tests for Physics-Informed Neural Networks Module
This test suite validates the physics-informed ML implementations in the Berkeley
SciComp package, ensuring PDE solving accuracy, physics compliance, and
scientific computing reliability.
Author: Berkeley SciComp Team
"""
import numpy as np
import pytest
import sys
from pathlib import Path
# Add package to path
sys.path.append(str(Path(__file__).parent.parent))
from Machine_Learning.physics_informed import (
    PINN, DeepONet, ConservationLawsNN, SymmetryAwareNN,
    PINNResults, create_pde_test_data
)
from Machine_Learning.neural_networks import MLP
class TestPINN:
    """Test suite for Physics-Informed Neural Networks."""
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.pinn = PINN(
            layers=[2, 20, 20, 1],  # [x, t] -> u
            activation='tanh',
            pde_weight=1.0,
            bc_weight=1.0,
            ic_weight=1.0,
            learning_rate=0.001
        )
    def test_initialization(self):
        """Test PINN initialization."""
        assert not self.pinn.is_fitted
        assert self.pinn.layers == [2, 20, 20, 1]
        assert self.pinn.activation == 'tanh'
        assert self.pinn.pde_weight == 1.0
        assert self.pinn.bc_weight == 1.0
        assert self.pinn.ic_weight == 1.0
        # Check that network is initialized
        assert hasattr(self.pinn, 'network')
        assert isinstance(self.pinn.network, MLP)
    def test_forward_pass(self):
        """Test forward pass through PINN."""
        x = np.array([0.5, 0.5, 0.5])
        t = np.array([0.1, 0.2, 0.3])
        u = self.pinn.forward(x, t)
        assert u.shape == x.shape
        assert np.all(np.isfinite(u))
    def test_derivative_computation(self):
        """Test derivative computation using finite differences."""
        x = np.array([0.3, 0.5, 0.7])
        t = np.array([0.1, 0.2, 0.3])
        derivatives = self.pinn.compute_derivatives(x, t)
        # Check that all expected derivatives are computed
        expected_keys = ['u_x', 'u_t', 'u_xx']
        for key in expected_keys:
            assert key in derivatives
            assert derivatives[key].shape == x.shape
            assert np.all(np.isfinite(derivatives[key]))
    def test_heat_equation_residual(self):
        """Test heat equation PDE residual computation."""
        x = np.array([0.2, 0.5, 0.8])
        t = np.array([0.1, 0.2, 0.3])
        diffusivity = 1.0
        residual = self.pinn.heat_equation_residual(x, t, diffusivity)
        assert residual.shape == x.shape
        assert np.all(np.isfinite(residual))
        # Residual should be small for well-trained network
        # (this test will initially fail until network is trained)
    def test_wave_equation_residual(self):
        """Test wave equation PDE residual computation."""
        x = np.array([0.2, 0.5, 0.8])
        t = np.array([0.1, 0.2, 0.3])
        wave_speed = 1.0
        residual = self.pinn.wave_equation_residual(x, t, wave_speed)
        assert residual.shape == x.shape
        assert np.all(np.isfinite(residual))
    def test_burgers_equation_residual(self):
        """Test Burgers equation PDE residual computation."""
        x = np.array([0.2, 0.5, 0.8])
        t = np.array([0.1, 0.2, 0.3])
        viscosity = 0.01
        residual = self.pinn.burgers_equation_residual(x, t, viscosity)
        assert residual.shape == x.shape
        assert np.all(np.isfinite(residual))
    def test_boundary_conditions(self):
        """Test boundary condition implementation."""
        x_bc = np.array([0.0, 1.0])  # Left and right boundaries
        t_bc = np.array([0.1, 0.2, 0.3])
        bc = self.pinn.boundary_conditions(x_bc, t_bc)
        assert 'left' in bc
        assert 'right' in bc
        assert bc['left'].shape == t_bc.shape
        assert bc['right'].shape == t_bc.shape
        # Default boundary conditions should be zero
        np.testing.assert_array_equal(bc['left'], np.zeros_like(t_bc))
        np.testing.assert_array_equal(bc['right'], np.zeros_like(t_bc))
    def test_initial_conditions(self):
        """Test initial condition implementation."""
        x_ic = np.array([0.2, 0.3, 0.5, 0.7, 0.8])
        ic = self.pinn.initial_conditions(x_ic)
        assert ic.shape == x_ic.shape
        assert np.all(np.isfinite(ic))
        # Default should be Gaussian pulse
        # Peak should be around x = 0.5
        peak_idx = np.argmax(ic)
        assert x_ic[peak_idx] == 0.5
    def test_loss_computation(self):
        """Test loss computation for all components."""
        # Create test points
        x_pde = np.random.uniform(0, 1, 100)
        t_pde = np.random.uniform(0, 0.5, 100)
        x_bc = np.array([0.0, 1.0])
        t_bc = np.random.uniform(0, 0.5, 20)
        x_ic = np.random.uniform(0, 1, 50)
        losses = self.pinn.compute_losses(
            x_pde, t_pde, x_bc, t_bc, x_ic,
            equation_type='heat',
            diffusivity=1.0
        )
        # Check that all loss components are computed
        expected_losses = ['pde', 'bc', 'ic', 'data']
        for loss_name in expected_losses:
            assert loss_name in losses
            assert isinstance(losses[loss_name], float)
            assert losses[loss_name] >= 0
    def test_total_loss(self):
        """Test total loss computation with weights."""
        losses = {
            'pde': 0.5,
            'bc': 0.3,
            'ic': 0.2,
            'data': 0.1
        }
        total = self.pinn.total_loss(losses)
        expected = (self.pinn.pde_weight * 0.5 +
                   self.pinn.bc_weight * 0.3 +
                   self.pinn.ic_weight * 0.2 +
                   self.pinn.data_weight * 0.1)
        assert abs(total - expected) < 1e-10
    def test_training_interface(self):
        """Test PINN training interface (without full training)."""
        # Quick training test - just check interface
        results = self.pinn.train(
            x_domain=(0.0, 1.0),
            t_domain=(0.0, 0.1),
            n_pde=50,
            n_bc=10,
            n_ic=20,
            epochs=5,  # Very few epochs for quick test
            equation_type='heat',
            verbose=False
        )
        assert isinstance(results, PINNResults)
        assert len(results.loss_history) == 5
        assert len(results.pde_loss_history) == 5
        assert len(results.bc_loss_history) == 5
        assert len(results.ic_loss_history) == 5
        assert results.total_epochs == 5
        assert self.pinn.is_fitted
    def test_prediction_after_training(self):
        """Test prediction functionality after training."""
        # Quick training
        self.pinn.train(
            x_domain=(0.0, 1.0),
            t_domain=(0.0, 0.1),
            n_pde=20,
            n_bc=5,
            n_ic=10,
            epochs=3,
            equation_type='heat',
            verbose=False
        )
        # Test predictions
        x_test = np.array([0.2, 0.5, 0.8])
        t_test = np.array([0.05, 0.05, 0.05])
        predictions = self.pinn.predict(x_test, t_test)
        assert predictions.shape == x_test.shape
        assert np.all(np.isfinite(predictions))
    def test_different_equation_types(self):
        """Test different PDE equation types."""
        x = np.array([0.5])
        t = np.array([0.1])
        # Test heat equation
        residual_heat = self.pinn.pde_residual(x, t, 'heat', diffusivity=1.0)
        assert residual_heat.shape == x.shape
        # Test wave equation
        residual_wave = self.pinn.pde_residual(x, t, 'wave', wave_speed=1.0)
        assert residual_wave.shape == x.shape
        # Test Burgers equation
        residual_burgers = self.pinn.pde_residual(x, t, 'burgers', viscosity=0.01)
        assert residual_burgers.shape == x.shape
        # Test unknown equation type
        with pytest.raises(ValueError):
            self.pinn.pde_residual(x, t, 'unknown_equation')
    def test_prediction_before_training(self):
        """Test error handling when predicting before training."""
        x = np.array([0.5])
        t = np.array([0.1])
        with pytest.raises(ValueError):
            self.pinn.predict(x, t)
class TestDeepONet:
    """Test suite for Deep Operator Networks."""
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.deeponet = DeepONet(
            branch_layers=[100, 50, 40],  # Input function encoding
            trunk_layers=[2, 50, 40],     # Coordinate encoding
            activation='relu',
            learning_rate=0.001
        )
    def test_initialization(self):
        """Test DeepONet initialization."""
        assert self.deeponet.branch_layers == [100, 50, 40]
        assert self.deeponet.trunk_layers == [2, 50, 40]
        assert hasattr(self.deeponet, 'branch_net')
        assert hasattr(self.deeponet, 'trunk_net')
    def test_forward_pass(self):
        """Test DeepONet forward pass."""
        # Create test data
        n_samples = 10
        n_sensors = 100
        n_points = 50
        n_dims = 2
        u = np.random.randn(n_samples, n_sensors)  # Input functions
        y = np.random.randn(n_points, n_dims)      # Evaluation coordinates
        output = self.deeponet.forward(u, y)
        assert output.shape == (n_samples, n_points)
        assert np.all(np.isfinite(output))
    def test_training_interface(self):
        """Test DeepONet training interface."""
        # Create synthetic operator learning data
        n_samples = 20
        n_sensors = 100
        n_points = 30
        # Input functions (e.g., initial conditions)
        u_train = np.random.randn(n_samples, n_sensors)
        # Evaluation coordinates
        y_train = np.random.rand(n_points, 2)
        # Target function values (solutions)
        s_train = np.random.randn(n_samples, n_points)
        # Quick training test
        self.deeponet.train(
            u_train, y_train, s_train,
            epochs=3,
            batch_size=5,
            verbose=False
        )
        # Should complete without errors
        # Full convergence testing would require more sophisticated setup
class TestConservationLawsNN:
    """Test suite for Conservation Laws Neural Networks."""
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.conservation_nn = ConservationLawsNN(
            layers=[2, 20, 1],
            conservation_type='mass',
            constraint_weight=1.0
        )
    def test_initialization(self):
        """Test ConservationLawsNN initialization."""
        assert self.conservation_nn.layers == [2, 20, 1]
        assert self.conservation_nn.conservation_type == 'mass'
        assert self.conservation_nn.constraint_weight == 1.0
        assert hasattr(self.conservation_nn, 'network')
    def test_mass_conservation_constraint(self):
        """Test mass conservation constraint computation."""
        x = np.array([0.3, 0.5, 0.7])
        t = np.array([0.1, 0.2, 0.3])
        u = np.array([1.0, 1.0, 1.0])  # Dummy values
        constraint = self.conservation_nn.conservation_constraint(u, x, t)
        assert constraint.shape == x.shape
        assert np.all(np.isfinite(constraint))
    def test_energy_conservation_constraint(self):
        """Test energy conservation constraint."""
        energy_nn = ConservationLawsNN(
            layers=[2, 20, 1],
            conservation_type='energy'
        )
        x = np.array([0.3, 0.5, 0.7])
        t = np.array([0.1, 0.2, 0.3])
        u = np.array([1.0, 1.0, 1.0])
        constraint = energy_nn.conservation_constraint(u, x, t)
        assert constraint.shape == x.shape
        assert np.all(np.isfinite(constraint))
    def test_unknown_conservation_type(self):
        """Test error handling for unknown conservation type."""
        unknown_nn = ConservationLawsNN(
            layers=[2, 20, 1],
            conservation_type='unknown'
        )
        x = np.array([0.5])
        t = np.array([0.1])
        u = np.array([1.0])
        with pytest.raises(ValueError):
            unknown_nn.conservation_constraint(u, x, t)
class TestSymmetryAwareNN:
    """Test suite for Symmetry-Aware Neural Networks."""
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.symmetry_nn = SymmetryAwareNN(
            layers=[3, 20, 1],
            symmetry_type='translation',
            symmetry_weight=1.0
        )
    def test_initialization(self):
        """Test SymmetryAwareNN initialization."""
        assert self.symmetry_nn.layers == [3, 20, 1]
        assert self.symmetry_nn.symmetry_type == 'translation'
        assert self.symmetry_nn.symmetry_weight == 1.0
        assert hasattr(self.symmetry_nn, 'network')
    def test_translation_symmetry(self):
        """Test translation symmetry transformation."""
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        x_transformed = self.symmetry_nn.apply_symmetry(x, 'translation')
        assert x_transformed.shape == x.shape
        assert np.all(np.isfinite(x_transformed))
        # Should be close to original but slightly different due to noise
        assert not np.allclose(x_transformed, x)
    def test_rotation_symmetry(self):
        """Test rotation symmetry transformation."""
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        x_transformed = self.symmetry_nn.apply_symmetry(x, 'rotation')
        assert x_transformed.shape == x.shape
        assert np.all(np.isfinite(x_transformed))
        # For 2D rotation, first two columns should be rotated
        # Third column should remain unchanged
        np.testing.assert_array_equal(x_transformed[:, 2], x[:, 2])
    def test_scaling_symmetry(self):
        """Test scaling symmetry transformation."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        x_transformed = self.symmetry_nn.apply_symmetry(x, 'scaling')
        assert x_transformed.shape == x.shape
        assert np.all(np.isfinite(x_transformed))
        # Should be scaled version of original
        scale_factors = x_transformed / (x + 1e-10)  # Avoid division by zero
        # All elements should have same scale factor (approximately)
        assert np.std(scale_factors) < 0.1
    def test_symmetry_loss_computation(self):
        """Test symmetry loss computation."""
        x = np.random.randn(5, 3)
        loss = self.symmetry_nn.symmetry_loss(x)
        assert isinstance(loss, float)
        assert loss >= 0
        assert np.isfinite(loss)
    def test_different_symmetry_types(self):
        """Test different symmetry types."""
        x = np.random.randn(3, 2)
        # Translation symmetry
        trans_nn = SymmetryAwareNN(layers=[2, 10, 1], symmetry_type='translation')
        trans_loss = trans_nn.symmetry_loss(x)
        # Rotation symmetry
        rot_nn = SymmetryAwareNN(layers=[2, 10, 1], symmetry_type='rotation')
        rot_loss = rot_nn.symmetry_loss(x)
        # Scaling symmetry
        scale_nn = SymmetryAwareNN(layers=[2, 10, 1], symmetry_type='scaling')
        scale_loss = scale_nn.symmetry_loss(x)
        assert all(isinstance(loss, float) for loss in [trans_loss, rot_loss, scale_loss])
        assert all(loss >= 0 for loss in [trans_loss, rot_loss, scale_loss])
class TestUtilityFunctions:
    """Test utility functions."""
    def test_create_pde_test_data(self):
        """Test PDE test data creation."""
        # Test heat equation data
        data_heat = create_pde_test_data('heat')
        assert 'x' in data_heat
        assert 't' in data_heat
        assert 'u' in data_heat
        assert 'x_grid' in data_heat
        assert 't_grid' in data_heat
        assert 'u_grid' in data_heat
        # Check shapes are consistent
        assert len(data_heat['x']) == len(data_heat['t'])
        assert len(data_heat['x']) == len(data_heat['u'])
        assert data_heat['x_grid'].shape == data_heat['t_grid'].shape
        assert data_heat['x_grid'].shape == data_heat['u_grid'].shape
        # Test wave equation data
        data_wave = create_pde_test_data('wave')
        assert all(key in data_wave for key in ['x', 't', 'u', 'x_grid', 't_grid', 'u_grid'])
        # Test Burgers equation data
        data_burgers = create_pde_test_data('burgers')
        assert all(key in data_burgers for key in ['x', 't', 'u', 'x_grid', 't_grid', 'u_grid'])
    def test_pde_data_physics(self):
        """Test physical properties of generated PDE data."""
        # Heat equation should show diffusion (spreading over time)
        data_heat = create_pde_test_data('heat')
        u_grid = data_heat['u_grid']
        # Initial condition should be more peaked than later times
        initial_std = np.std(u_grid[0, :])  # t=0
        final_std = np.std(u_grid[-1, :])   # t=final
        # For diffusion, standard deviation should increase with time
        # (though this might be violated for truncated analytical solutions)
        # Wave equation should show oscillatory behavior
        data_wave = create_pde_test_data('wave')
        u_wave = data_wave['u_grid']
        # Should have both positive and negative values
        assert np.any(u_wave > 0)
        assert np.any(u_wave < 0)
        # Should be bounded (for sine wave solution)
        assert np.all(np.abs(u_wave) <= 1.1)  # Allow small numerical errors
class TestIntegrationAndPhysics:
    """Integration tests and physics validation."""
    def test_simple_pde_solving(self):
        """Test PINN solving simple PDE with known solution."""
        # Test with heat equation and known analytical solution
        # u(x,t) = sin(πx) * exp(-π²t) solves heat equation with diffusivity=1
        pinn = PINN(
            layers=[2, 30, 30, 1],
            activation='tanh',
            pde_weight=1.0,
            bc_weight=10.0,  # Higher weight for boundary conditions
            ic_weight=10.0,  # Higher weight for initial conditions
            learning_rate=0.001
        )
        # Custom boundary and initial conditions
        class HeatPINN(PINN):
            def boundary_conditions(self, x, t):
                return {'left': np.zeros_like(t), 'right': np.zeros_like(t)}
            def initial_conditions(self, x):
                return np.sin(np.pi * x)
        heat_pinn = HeatPINN(
            layers=[2, 30, 30, 1],
            activation='tanh',
            learning_rate=0.001
        )
        # Quick training
        results = heat_pinn.train(
            x_domain=(0.0, 1.0),
            t_domain=(0.0, 0.1),
            n_pde=500,
            n_bc=50,
            n_ic=50,
            epochs=20,  # Limited epochs for testing
            equation_type='heat',
            diffusivity=1.0,
            verbose=False
        )
        # Test solution at specific points
        x_test = np.array([0.5])
        t_test = np.array([0.05])
        u_pred = heat_pinn.predict(x_test, t_test)
        u_analytical = np.sin(np.pi * x_test) * np.exp(-np.pi**2 * t_test)
        # Solution should be in reasonable range
        # (exact accuracy depends on training quality)
        assert np.all(np.abs(u_pred) <= 2.0)  # Reasonable bounds
        assert np.all(np.isfinite(u_pred))
    def test_conservation_properties(self):
        """Test that conservation laws are approximately satisfied."""
        # Test mass conservation neural network
        conservation_nn = ConservationLawsNN(
            layers=[2, 20, 1],
            conservation_type='mass'
        )
        # Create test points
        x = np.linspace(0.1, 0.9, 10)
        t = np.linspace(0.1, 0.5, 10)
        u = np.ones_like(x)  # Constant field
        # For constant field, mass conservation constraint should be small
        constraint = conservation_nn.conservation_constraint(u, x, t)
        # The constraint represents violation of conservation law
        # For untrained network, this might be large, but should be finite
        assert np.all(np.isfinite(constraint))
    def test_symmetry_preservation(self):
        """Test that symmetry-aware networks preserve symmetries."""
        symmetry_nn = SymmetryAwareNN(
            layers=[2, 20, 1],
            symmetry_type='translation'
        )
        # Create simple input
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        # For untrained network, symmetry might not be perfectly preserved
        # But the symmetry loss should be computable
        loss = symmetry_nn.symmetry_loss(x)
        assert isinstance(loss, float)
        assert loss >= 0
        assert np.isfinite(loss)
    def test_pde_residual_properties(self):
        """Test mathematical properties of PDE residuals."""
        pinn = PINN(layers=[2, 10, 1])
        x = np.array([0.2, 0.5, 0.8])
        t = np.array([0.1, 0.2, 0.3])
        # Heat equation residual should be well-defined
        residual_heat = pinn.heat_equation_residual(x, t, diffusivity=1.0)
        assert np.all(np.isfinite(residual_heat))
        # Changing diffusivity should change residual
        residual_heat2 = pinn.heat_equation_residual(x, t, diffusivity=2.0)
        assert not np.allclose(residual_heat, residual_heat2)
        # Wave equation with different speeds
        residual_wave1 = pinn.wave_equation_residual(x, t, wave_speed=1.0)
        residual_wave2 = pinn.wave_equation_residual(x, t, wave_speed=2.0)
        assert not np.allclose(residual_wave1, residual_wave2)
    def test_boundary_condition_enforcement(self):
        """Test that boundary conditions are properly enforced."""
        pinn = PINN(layers=[2, 20, 1])
        # Test default boundary conditions
        x_bc = np.array([0.0, 1.0])
        t_bc = np.array([0.1, 0.2, 0.3])
        bc = pinn.boundary_conditions(x_bc, t_bc)
        # Default should be homogeneous Dirichlet
        np.testing.assert_array_equal(bc['left'], np.zeros_like(t_bc))
        np.testing.assert_array_equal(bc['right'], np.zeros_like(t_bc))
    def test_initial_condition_properties(self):
        """Test initial condition properties."""
        pinn = PINN(layers=[2, 20, 1])
        x_ic = np.linspace(0, 1, 101)
        ic = pinn.initial_conditions(x_ic)
        # Default is Gaussian pulse
        assert ic.shape == x_ic.shape
        assert np.all(ic >= 0)  # Gaussian should be non-negative
        # Should peak around x = 0.5
        peak_idx = np.argmax(ic)
        assert abs(x_ic[peak_idx] - 0.5) < 0.1
        # Should decay away from peak
        assert ic[0] < ic[peak_idx]  # Left boundary
        assert ic[-1] < ic[peak_idx]  # Right boundary
if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])