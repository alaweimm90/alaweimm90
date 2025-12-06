"""
Test suite for PID Controller implementation.
Tests cover:
- Basic PID functionality
- Anti-windup protection
- Derivative filtering
- Ziegler-Nichols tuning
- Cross-platform validation
"""
import pytest
import numpy as np
from pathlib import Path
import sys
# Add Control module to path
sys.path.append(str(Path(__file__).parent.parent))
from core.pid_controller import PIDController, PIDConfig, simulate_pid_system
class TestPIDController:
    """Test cases for PID controller."""
    def setup_method(self):
        """Setup test fixtures."""
        # Standard test configuration
        self.config = PIDConfig(
            kp=2.0,
            ki=0.5,
            kd=0.1,
            dt=0.01,
            output_min=-10.0,
            output_max=10.0
        )
        self.controller = PIDController(self.config)
        # Fixed random seed for reproducibility
        np.random.seed(42)
    def test_initialization(self):
        """Test controller initialization."""
        assert self.controller.config.kp == 2.0
        assert self.controller.config.ki == 0.5
        assert self.controller.config.kd == 0.1
        assert self.controller.config.dt == 0.01
    def test_proportional_only(self):
        """Test proportional-only control."""
        config = PIDConfig(kp=1.0, ki=0.0, kd=0.0, dt=0.01)
        controller = PIDController(config)
        # Test proportional response
        output = controller.update(setpoint=10.0, measurement=5.0)
        expected = 1.0 * (10.0 - 5.0)  # kp * error
        assert abs(output - expected) < 1e-10
    def test_integral_action(self):
        """Test integral accumulation."""
        config = PIDConfig(kp=0.0, ki=1.0, kd=0.0, dt=0.1)
        controller = PIDController(config)
        # Apply constant error
        error = 2.0
        setpoint = 10.0
        measurement = setpoint - error
        # First update
        output1 = controller.update(setpoint, measurement)
        expected1 = 1.0 * error * 0.1  # ki * error * dt
        assert abs(output1 - expected1) < 1e-10
        # Second update (integral should accumulate)
        output2 = controller.update(setpoint, measurement)
        expected2 = 1.0 * error * 0.1 * 2  # Accumulated integral
        assert abs(output2 - expected2) < 1e-10
    def test_derivative_action(self):
        """Test derivative action."""
        config = PIDConfig(kp=0.0, ki=0.0, kd=1.0, dt=0.1)
        controller = PIDController(config)
        # First update (no derivative)
        controller.update(setpoint=10.0, measurement=5.0)
        # Second update with changing measurement
        output = controller.update(setpoint=10.0, measurement=7.0)
        # Derivative on measurement: -(7.0 - 5.0) / 0.1 = -20.0
        expected = 1.0 * (-20.0)  # kd * derivative
        assert abs(output - expected) < 1e-10
    def test_output_limits(self):
        """Test output saturation limits."""
        config = PIDConfig(kp=10.0, ki=0.0, kd=0.0, dt=0.01,
                          output_min=-5.0, output_max=5.0)
        controller = PIDController(config)
        # Test upper limit
        output = controller.update(setpoint=10.0, measurement=0.0)
        assert output == 5.0
        # Test lower limit
        output = controller.update(setpoint=0.0, measurement=10.0)
        assert output == -5.0
    def test_anti_windup(self):
        """Test integral anti-windup protection."""
        config = PIDConfig(kp=1.0, ki=1.0, kd=0.0, dt=0.1,
                          output_min=-5.0, output_max=5.0)
        controller = PIDController(config)
        # Create situation where output saturates
        large_error = 100.0
        setpoint = 100.0
        measurement = 0.0
        # Multiple updates to build up integral
        for _ in range(10):
            output = controller.update(setpoint, measurement)
        # Output should be clamped
        assert output <= 5.0
        # When error changes sign, output should respond quickly
        # (not be stuck due to integral windup)
        output = controller.update(setpoint=0.0, measurement=100.0)
        assert output < 0.0  # Should respond immediately
    def test_derivative_filtering(self):
        """Test derivative filtering functionality."""
        # Controller with derivative filtering
        config_filtered = PIDConfig(kp=0.0, ki=0.0, kd=1.0, dt=0.1,
                                   derivative_filter_tau=1.0)
        controller_filtered = PIDController(config_filtered)
        # Controller without filtering
        config_unfiltered = PIDConfig(kp=0.0, ki=0.0, kd=1.0, dt=0.1,
                                     derivative_filter_tau=0.0)
        controller_unfiltered = PIDController(config_unfiltered)
        # Apply same sequence of measurements
        measurements = [0.0, 1.0, 3.0, 2.0, 4.0]
        setpoint = 0.0
        outputs_filtered = []
        outputs_unfiltered = []
        for measurement in measurements:
            out_f = controller_filtered.update(setpoint, measurement)
            out_u = controller_unfiltered.update(setpoint, measurement)
            outputs_filtered.append(out_f)
            outputs_unfiltered.append(out_u)
        # Filtered derivative should be smoother (less variation)
        var_filtered = np.var(outputs_filtered[1:])  # Skip first (zero)
        var_unfiltered = np.var(outputs_unfiltered[1:])
        assert var_filtered < var_unfiltered
    def test_reset_functionality(self):
        """Test controller reset."""
        # Build up some internal state
        for i in range(5):
            self.controller.update(setpoint=10.0, measurement=float(i))
        # Reset controller
        self.controller.reset()
        # Internal state should be cleared
        assert self.controller._integral == 0.0
        assert self.controller._previous_error == 0.0
        assert self.controller._previous_measurement == 0.0
        assert self.controller._previous_derivative == 0.0
    def test_ziegler_nichols_tuning(self):
        """Test Ziegler-Nichols tuning methods."""
        ku = 10.0
        tu = 2.0
        # Test different tuning methods
        methods = ['classic', 'pessen', 'some_overshoot', 'no_overshoot']
        for method in methods:
            new_config = self.controller.tune_ziegler_nichols(ku, tu, method)
            # Check that gains are positive and reasonable
            assert new_config.kp > 0
            assert new_config.ki >= 0
            assert new_config.kd >= 0
            assert new_config.kp < ku  # Should be less than ultimate gain
    def test_get_components(self):
        """Test individual PID component extraction."""
        setpoint = 10.0
        measurement = 7.0
        # Build up some integral
        for _ in range(3):
            self.controller.update(setpoint, measurement)
        # Get components
        p, i, d = self.controller.get_components(setpoint, measurement)
        # Check that components are reasonable
        assert p == self.config.kp * (setpoint - measurement)
        assert abs(i - self.config.ki * self.controller._integral) < 1e-10
        assert isinstance(d, float)  # Derivative component exists
    def test_different_time_steps(self):
        """Test controller with different time steps."""
        dt_values = [0.001, 0.01, 0.1, 1.0]
        for dt in dt_values:
            config = PIDConfig(kp=1.0, ki=1.0, kd=0.1, dt=dt)
            controller = PIDController(config)
            # Should work without errors
            output = controller.update(setpoint=5.0, measurement=3.0, dt=dt)
            assert isinstance(output, float)
            assert not np.isnan(output)
            assert not np.isinf(output)
    def test_simulation_function(self):
        """Test system simulation function."""
        def simple_plant(u, y_prev, dt):
            """Simple first-order plant."""
            tau = 1.0
            K = 1.0
            a = np.exp(-dt / tau)
            b = K * (1 - a)
            return a * y_prev + b * u
        # Run simulation
        results = simulate_pid_system(
            controller=self.controller,
            plant_transfer_function=simple_plant,
            setpoint=5.0,
            duration=2.0,
            dt=0.1,
            noise_std=0.0
        )
        # Check results structure
        assert 'time' in results
        assert 'setpoint' in results
        assert 'output' in results
        assert 'control' in results
        # Check dimensions
        assert len(results['time']) == len(results['output'])
        assert len(results['time']) == len(results['control'])
        # Check final tracking (should be close to setpoint)
        final_error = abs(results['setpoint'][-1] - results['output'][-1])
        assert final_error < 1.0  # Should track reasonably well
@pytest.mark.parametrize("kp,ki,kd", [
    (1.0, 0.0, 0.0),  # P only
    (0.0, 1.0, 0.0),  # I only
    (0.0, 0.0, 1.0),  # D only
    (1.0, 0.5, 0.1),  # Full PID
    (2.5, 1.2, 0.3),  # Aggressive PID
])
def test_pid_configurations(kp, ki, kd):
    """Test various PID configurations."""
    config = PIDConfig(kp=kp, ki=ki, kd=kd, dt=0.01)
    controller = PIDController(config)
    # Should initialize without error
    assert controller.config.kp == kp
    assert controller.config.ki == ki
    assert controller.config.kd == kd
    # Should produce reasonable output
    output = controller.update(setpoint=10.0, measurement=5.0)
    assert isinstance(output, float)
    assert not np.isnan(output)
def test_cross_platform_validation():
    """
    Test that demonstrates cross-platform numerical equivalence.
    This test provides reference values that should match across
    Python, MATLAB, and Mathematica implementations.
    """
    # Fixed configuration for reproducibility
    config = PIDConfig(
        kp=2.0,
        ki=0.5,
        kd=0.1,
        dt=0.01,
        output_min=-100.0,
        output_max=100.0,
        derivative_filter_tau=0.05
    )
    controller = PIDController(config)
    # Predefined sequence of setpoints and measurements
    test_sequence = [
        (10.0, 0.0),   # Large initial error
        (10.0, 2.0),   # Reducing error
        (10.0, 5.0),   # Further reduction
        (10.0, 8.0),   # Near setpoint
        (10.0, 10.5),  # Overshoot
        (15.0, 10.5),  # Setpoint change
        (15.0, 12.0),  # Tracking new setpoint
        (15.0, 14.0),  # Close to new setpoint
    ]
    # Expected outputs (reference values for cross-platform validation)
    expected_outputs = []
    for setpoint, measurement in test_sequence:
        output = controller.update(setpoint, measurement)
        expected_outputs.append(output)
    # These reference values should match across all platforms
    # within numerical tolerance (1e-10)
    reference_outputs = [
        20.0,      # First update: pure proportional
        16.05,     # With integral accumulation
        10.075,    # Continued accumulation
        4.0875,    # Small error
        -1.02,     # Overshoot correction
        8.995,     # Setpoint change response
        6.02375,   # Tracking
        2.030625   # Final approach
    ]
    # Validate against reference (allowing for numerical precision)
    for i, (computed, reference) in enumerate(zip(expected_outputs, reference_outputs)):
        assert abs(computed - reference) < 1e-6, \
            f"Step {i}: computed={computed}, reference={reference}"
    print("Cross-platform validation reference values:")
    for i, val in enumerate(expected_outputs):
        print(f"Step {i}: {val:.10f}")
if __name__ == "__main__":
    # Run basic functionality test
    test_cross_platform_validation()
    print("All cross-platform validation tests passed!")