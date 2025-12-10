"""Tests for smlr.pmm (Parametric Matrix Model)."""
import numpy as np
import pytest

from smlr.pmm import ParametricMatrixModel, PMMConfig, PMMResult
from smlr.backends import get_emulator, list_backends
from smlr.data import StrengthDataset, StrengthSample
from smlr.lorentz import lorentzian_sum


def generate_simple_spectrum(alpha, beta, energy):
    """Simple 2-parameter spectrum for testing."""
    center = 10 + 3 * alpha - 2 * beta
    strength = 1.0 + 0.2 * alpha
    width = 1.5
    return lorentzian_sum(energy, [center], [strength], [width])


def build_test_dataset(n_samples=20, seed=42):
    """Build a small test dataset."""
    rng = np.random.default_rng(seed)
    energy = np.linspace(0, 30, 100)
    samples = []
    for i in range(n_samples):
        alpha = rng.uniform(0, 1)
        beta = rng.uniform(0, 1)
        spectrum = generate_simple_spectrum(alpha, beta, energy)
        samples.append(StrengthSample(
            params=np.array([alpha, beta]),
            energy=energy,
            strength=spectrum,
            label=f"sample_{i}"
        ))
    return StrengthDataset(samples)


class TestPMMConfig:
    """Test PMMConfig dataclass."""
    
    def test_default_values(self):
        config = PMMConfig()
        assert config.n_poles == 10
        assert config.retain == 0.9  # New default
        assert config.width_mode == "parametric"  # Changed default
        assert config.optimizer == "scipy"
        assert config.max_iterations == 5000  # Changed default
    
    def test_custom_values(self):
        config = PMMConfig(n_poles=5, width_mode="global", verbose=True, retain=0.8)
        assert config.n_poles == 5
        assert config.width_mode == "global"
        assert config.verbose is True
        assert config.retain == 0.8


class TestParametricMatrixModel:
    """Test ParametricMatrixModel class."""
    
    def test_initialization(self):
        pmm = ParametricMatrixModel(n_poles=5)
        assert pmm.n_poles == 5
        assert pmm.param_dim is None
        assert pmm.D is None
    
    def test_initialization_with_config(self):
        config = PMMConfig(n_poles=8, verbose=False)
        pmm = ParametricMatrixModel(config=config)
        assert pmm.n_poles == 8
    
    def test_fit_and_predict(self):
        """Test basic fit and predict workflow."""
        dataset = build_test_dataset(n_samples=15)
        energy = np.linspace(0, 30, 100)
        
        pmm = ParametricMatrixModel(n_poles=3, max_iterations=100, verbose=False)
        pmm.fit(dataset, reference_point=np.array([0.5, 0.5]))
        
        # Check that parameters are set
        assert pmm.param_dim == 2
        assert pmm.D is not None
        assert len(pmm.S) == 2
        assert pmm.v0 is not None
        
        # Test prediction
        result = pmm.predict(np.array([0.3, 0.7]), energy)
        assert isinstance(result, PMMResult)
        assert len(result.eigenvalues) == 3
        assert len(result.strengths) == 3
        assert len(result.spectrum) == len(energy)
    
    def test_get_eigenvalues(self):
        """Test getting eigenvalues without full spectrum."""
        dataset = build_test_dataset(n_samples=10)
        
        pmm = ParametricMatrixModel(n_poles=3, max_iterations=50)
        pmm.fit(dataset)
        
        eigenvalues, strengths = pmm.get_eigenvalues(np.array([0.5, 0.5]))
        assert len(eigenvalues) == 3
        assert len(strengths) == 3
        assert np.all(strengths >= 0)  # Squared projections are positive
    
    def test_model_diagnostics(self):
        """Test diagnostic information."""
        dataset = build_test_dataset(n_samples=10)
        
        pmm = ParametricMatrixModel(n_poles=3, max_iterations=50)
        
        # Before fitting
        diag = pmm.model_diagnostics()
        assert diag["fitted"] is False
        
        # After fitting
        pmm.fit(dataset)
        diag = pmm.model_diagnostics()
        assert diag["fitted"] is True
        assert diag["param_dim"] == 2
        assert diag["n_poles"] == 3
    
    def test_predict_batch(self):
        """Test batch prediction."""
        dataset = build_test_dataset(n_samples=10)
        energy = np.linspace(0, 30, 100)
        
        pmm = ParametricMatrixModel(n_poles=3, max_iterations=50)
        pmm.fit(dataset)
        
        params = np.array([
            [0.3, 0.3],
            [0.5, 0.5],
            [0.7, 0.7],
        ])
        results = pmm.predict_batch(params, energy)
        
        assert len(results) == 3
        for r in results:
            assert isinstance(r, PMMResult)


class TestBackends:
    """Test backend factory functions."""
    
    def test_list_backends(self):
        backends = list_backends()
        assert "regression" in backends
        assert "pmm" in backends
        assert "lorentzian" in backends
        assert "matrix" in backends
    
    def test_get_emulator_regression(self):
        emu = get_emulator("regression", n_components=4)
        assert hasattr(emu, 'fit')
        assert hasattr(emu, 'predict_mixture')
    
    def test_get_emulator_lorentzian_alias(self):
        emu = get_emulator("lorentzian", n_components=4)
        assert hasattr(emu, 'fit')
    
    def test_get_emulator_pmm(self):
        emu = get_emulator("pmm", n_poles=5)
        assert isinstance(emu, ParametricMatrixModel)
        assert emu.n_poles == 5
    
    def test_get_emulator_matrix_alias(self):
        emu = get_emulator("matrix", n_poles=5)
        assert isinstance(emu, ParametricMatrixModel)
    
    def test_get_emulator_invalid_backend(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_emulator("invalid_backend")
    
    def test_n_components_to_n_poles(self):
        """n_components should work for PMM too."""
        emu = get_emulator("pmm", n_components=7)
        assert emu.n_poles == 7


class TestPMMPhysics:
    """Test physics properties of PMM."""
    
    def test_eigenvalues_sorted(self):
        """Eigenvalues from eigh should be sorted."""
        dataset = build_test_dataset(n_samples=10)
        
        pmm = ParametricMatrixModel(n_poles=5, max_iterations=50)
        pmm.fit(dataset)
        
        eigenvalues, _ = pmm.get_eigenvalues(np.array([0.5, 0.5]))
        assert np.all(np.diff(eigenvalues) >= 0), "Eigenvalues should be sorted"
    
    def test_strengths_positive(self):
        """Transition strengths are squared, so must be positive."""
        dataset = build_test_dataset(n_samples=10)
        
        pmm = ParametricMatrixModel(n_poles=5, max_iterations=50)
        pmm.fit(dataset)
        
        _, strengths = pmm.get_eigenvalues(np.array([0.5, 0.5]))
        assert np.all(strengths >= 0), "Strengths should be non-negative"
    
    def test_spectrum_positive(self):
        """Lorentzian spectrum should be positive."""
        dataset = build_test_dataset(n_samples=10)
        energy = np.linspace(0, 30, 100)
        
        pmm = ParametricMatrixModel(n_poles=5, max_iterations=50)
        pmm.fit(dataset)
        
        result = pmm.predict(np.array([0.5, 0.5]), energy)
        assert np.all(result.spectrum >= 0), "Spectrum should be non-negative"
