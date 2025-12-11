"""Tests for smlr.base (Unified emulator interface)."""
import numpy as np
import pytest

from smlr import Surrogate, StrengthDataset, StrengthSample
from smlr.base import BaseEmulator, EmulatorResult
from smlr.emulator import StrengthEmulator
from smlr.pmm import ParametricMatrixModel
from smlr.lorentz import lorentzian_sum


def _make_simple_dataset(n_samples=10, seed=42):
    """Create a simple test dataset."""
    rng = np.random.default_rng(seed)
    energy = np.linspace(0, 20, 100)
    samples = []
    
    for _ in range(n_samples):
        alpha = rng.uniform(0, 1)
        center = 10 + 2 * alpha
        strength = 1.0 + 0.5 * alpha
        width = 1.5
        spectrum = lorentzian_sum(energy, [center], [strength], [width])
        samples.append(StrengthSample(
            params=np.array([alpha]),
            energy=energy,
            strength=spectrum
        ))
    
    return StrengthDataset(samples)


class TestEmulatorResult:
    """Test EmulatorResult dataclass."""
    
    def test_creation(self):
        result = EmulatorResult(
            spectrum=np.array([1.0, 2.0, 3.0]),
            energy=np.array([0.0, 1.0, 2.0]),
            poles=np.array([1.0]),
            strengths=np.array([0.5]),
            widths=np.array([0.3])
        )
        assert len(result.spectrum) == 3
        assert len(result.energy) == 3
        assert len(result.poles) == 1
    
    def test_optional_metadata(self):
        result = EmulatorResult(
            spectrum=np.array([1.0]),
            energy=np.array([0.0]),
            poles=np.array([1.0]),
            strengths=np.array([0.5]),
            widths=np.array([0.3]),
            metadata={"key": "value"}
        )
        assert result.metadata["key"] == "value"
    
    def test_default_metadata(self):
        result = EmulatorResult(
            spectrum=np.array([1.0]),
            energy=np.array([0.0]),
            poles=np.array([1.0]),
            strengths=np.array([0.5]),
            widths=np.array([0.3])
        )
        assert result.metadata == {}


class TestBaseEmulator:
    """Test that backends properly implement BaseEmulator."""
    
    def test_strength_emulator_is_base_emulator(self):
        emu = StrengthEmulator(n_components=2)
        assert isinstance(emu, BaseEmulator)
    
    def test_pmm_is_base_emulator(self):
        pmm = ParametricMatrixModel(n_poles=3)
        assert isinstance(pmm, BaseEmulator)
    
    def test_strength_emulator_returns_emulator_result(self):
        dataset = _make_simple_dataset()
        energy = np.linspace(0, 20, 100)
        
        emu = StrengthEmulator(n_components=1, random_state=42)
        emu.fit(dataset)
        
        result = emu.predict(np.array([0.5]), energy)
        assert isinstance(result, EmulatorResult)
    
    def test_pmm_returns_emulator_result(self):
        dataset = _make_simple_dataset()
        energy = np.linspace(0, 20, 100)
        
        pmm = ParametricMatrixModel(n_poles=2, max_iterations=50)
        pmm.fit(dataset)
        
        result = pmm.predict(np.array([0.5]), energy)
        assert isinstance(result, EmulatorResult)


class TestSurrogate:
    """Test Surrogate unified wrapper class."""
    
    def test_create_with_regression_backend(self):
        model = Surrogate("regression", n_components=2)
        assert model.backend_name == "regression"
        assert isinstance(model.backend, StrengthEmulator)
    
    def test_create_with_pmm_backend(self):
        model = Surrogate("pmm", n_poles=3)
        assert model.backend_name == "pmm"
        assert isinstance(model.backend, ParametricMatrixModel)
    
    def test_aliases(self):
        # Test lorentzian alias for regression
        model = Surrogate("lorentzian", n_components=2)
        assert isinstance(model.backend, StrengthEmulator)
        
        # Test matrix alias for pmm
        model = Surrogate("matrix", n_poles=3)
        assert isinstance(model.backend, ParametricMatrixModel)
    
    def test_fit_and_predict(self):
        dataset = _make_simple_dataset()
        energy = np.linspace(0, 20, 100)
        
        model = Surrogate("regression", n_components=1, random_state=42)
        model.fit(dataset)
        
        result = model.predict(np.array([0.5]), energy)
        assert isinstance(result, EmulatorResult)
        assert len(result.spectrum) == len(energy)
    
    def test_predict_batch(self):
        dataset = _make_simple_dataset()
        energy = np.linspace(0, 20, 100)
        
        model = Surrogate("pmm", n_poles=2, max_iterations=50)
        model.fit(dataset)
        
        params = np.array([[0.3], [0.5], [0.7]])
        results = model.predict_batch(params, energy)
        
        assert len(results) == 3
        for r in results:
            assert isinstance(r, EmulatorResult)
    
    def test_score(self):
        dataset = _make_simple_dataset()
        
        model = Surrogate("regression", n_components=1, random_state=42)
        model.fit(dataset)
        
        score = model.score(dataset)
        assert isinstance(score, float)
        assert score >= 0
    
    def test_get_info(self):
        model = Surrogate("regression", n_components=2)
        info = model.get_info()
        
        assert "surrogate_backend" in info
        assert info["surrogate_backend"] == "regression"
        assert info["n_poles"] == 2
    
    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            Surrogate("invalid_backend")
    
    def test_repr(self):
        model = Surrogate("regression", n_components=2)
        repr_str = repr(model)
        assert "Surrogate" in repr_str
        assert "regression" in repr_str
