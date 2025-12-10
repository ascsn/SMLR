"""Tests for smlr.optimization module."""
import numpy as np
import pytest

from smlr.optimization import (
    OptimizerBackend,
    OptimizerConfig,
    ScipyOptimizer,
    create_optimizer,
    get_available_backends,
)


def rosenbrock(x):
    """Rosenbrock function for testing."""
    return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


def quadratic(x):
    """Simple quadratic function."""
    return np.sum((x - 2.0) ** 2)


class TestOptimizerConfig:
    """Tests for OptimizerConfig dataclass."""
    
    def test_default_config(self):
        config = OptimizerConfig()
        assert config.backend == OptimizerBackend.SCIPY
        assert config.max_iterations == 1000
        assert config.learning_rate == 0.01
    
    def test_custom_config(self):
        config = OptimizerConfig(
            backend=OptimizerBackend.SCIPY,
            method="TNC",
            max_iterations=500,
        )
        assert config.method == "TNC"
        assert config.max_iterations == 500


class TestScipyOptimizer:
    """Tests for ScipyOptimizer."""
    
    def test_minimize_quadratic(self):
        config = OptimizerConfig(
            backend=OptimizerBackend.SCIPY,
            method="L-BFGS-B",
            max_iterations=100,
        )
        opt = ScipyOptimizer(config)
        
        x0 = np.array([0.0, 0.0, 0.0])
        result = opt.minimize(quadratic, x0)
        
        assert result.converged
        assert result.cost < 1e-6
        np.testing.assert_array_almost_equal(result.params, [2.0, 2.0, 2.0], decimal=3)
    
    def test_minimize_with_bounds(self):
        config = OptimizerConfig(method="L-BFGS-B", max_iterations=100)
        opt = ScipyOptimizer(config)
        
        x0 = np.array([0.0, 0.0])
        bounds = [(0.0, 1.0), (0.0, 1.0)]  # Constrain to [0, 1]^2
        
        result = opt.minimize(quadratic, x0, bounds=bounds)
        
        # Solution should be at bounds (closest to [2, 2])
        np.testing.assert_array_almost_equal(result.params, [1.0, 1.0], decimal=3)


class TestCreateOptimizer:
    """Tests for create_optimizer factory."""
    
    def test_create_scipy_optimizer(self):
        opt = create_optimizer("scipy", method="L-BFGS-B")
        assert isinstance(opt, ScipyOptimizer)
    
    def test_create_with_kwargs(self):
        opt = create_optimizer(
            "scipy",
            method="TNC",
            max_iterations=500,
            tolerance=1e-8,
        )
        assert opt.config.max_iterations == 500
        assert opt.config.tolerance == 1e-8
    
    def test_unknown_backend_raises(self):
        with pytest.raises((ValueError, KeyError)):
            create_optimizer("unknown_backend")


class TestGetAvailableBackends:
    """Tests for get_available_backends."""
    
    def test_scipy_always_available(self):
        backends = get_available_backends()
        assert "scipy" in backends
    
    def test_returns_list(self):
        backends = get_available_backends()
        assert isinstance(backends, list)
        assert len(backends) >= 1


class TestOptimizationResult:
    """Test the optimization result container."""
    
    def test_scipy_result_attributes(self):
        config = OptimizerConfig(max_iterations=50)
        opt = ScipyOptimizer(config)
        
        result = opt.minimize(quadratic, np.zeros(2))
        
        assert hasattr(result, 'params')
        assert hasattr(result, 'cost')
        assert hasattr(result, 'n_iterations')
        assert hasattr(result, 'converged')
        assert hasattr(result, 'message')


class TestMultiDimensionalOptimization:
    """Test optimization in high dimensions."""
    
    def test_high_dim_quadratic(self):
        """Verify optimization works for 10+ dimensions."""
        n_dims = 15
        
        def high_dim_quadratic(x):
            return np.sum((x - np.arange(n_dims)) ** 2)
        
        config = OptimizerConfig(method="L-BFGS-B", max_iterations=200)
        opt = ScipyOptimizer(config)
        
        x0 = np.zeros(n_dims)
        result = opt.minimize(high_dim_quadratic, x0)
        
        assert result.converged
        np.testing.assert_array_almost_equal(
            result.params, np.arange(n_dims), decimal=3
        )
