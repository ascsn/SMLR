"""Tests for smlr.observables module."""
import numpy as np
import pytest

from smlr.observables import (
    SumRule,
    DipolePolarizability,
    CustomObservable,
    ObservableSet,
    create_trk_sum_rule,
    create_polarizability,
)
from smlr.lorentz import lorentzian_sum


def _synthetic_spectrum(energy):
    """Create a simple test spectrum."""
    centers = np.array([5.0, 15.0, 25.0])
    strengths = np.array([1.0, 2.0, 0.5])
    widths = np.array([1.0, 2.0, 1.5])
    return lorentzian_sum(energy, centers, strengths, widths)


class TestSumRule:
    """Tests for SumRule observable."""
    
    def test_m0_is_positive(self):
        energy = np.linspace(0, 40, 200)
        strength = _synthetic_spectrum(energy)
        
        m0 = SumRule(k=0)
        value = m0.compute(energy, strength)
        assert value > 0
    
    def test_m1_larger_than_m0(self):
        """Higher moments should be larger for positive energies."""
        energy = np.linspace(1, 40, 200)
        strength = _synthetic_spectrum(energy)
        
        m0 = SumRule(k=0).compute(energy, strength)
        m1 = SumRule(k=1).compute(energy, strength)
        
        # For E > 1, m_1 > m_0
        assert m1 > m0
    
    def test_energy_range_filter(self):
        energy = np.linspace(0, 40, 200)
        strength = _synthetic_spectrum(energy)
        
        # Full range
        m0_full = SumRule(k=0).compute(energy, strength)
        
        # Restricted range
        m0_partial = SumRule(k=0, energy_min=10, energy_max=30).compute(energy, strength)
        
        # Partial should be less than full
        assert m0_partial < m0_full
    
    def test_loss_with_target(self):
        energy = np.linspace(0, 40, 200)
        strength = _synthetic_spectrum(energy)
        
        m0_value = SumRule(k=0).compute(energy, strength)
        
        # Loss should be 0 when target equals actual value
        m0_exact = SumRule(k=0, target=m0_value)
        assert m0_exact.loss(energy, strength) == pytest.approx(0.0, abs=1e-10)
        
        # Loss should be positive when target differs
        m0_wrong = SumRule(k=0, target=m0_value * 2, weight=1.0)
        assert m0_wrong.loss(energy, strength) > 0


class TestDipolePolarizability:
    """Tests for DipolePolarizability observable."""
    
    def test_positive_value(self):
        energy = np.linspace(2, 40, 200)  # Avoid E=0
        strength = _synthetic_spectrum(energy)
        
        alpha_d = DipolePolarizability(energy_min=1.0)
        value = alpha_d.compute(energy, strength)
        assert value > 0
    
    def test_from_poles_matches_integral(self):
        """Pole-based computation should approximately match integration."""
        pole_energies = np.array([5.0, 15.0, 25.0])
        pole_strengths = np.array([1.0, 2.0, 0.5])
        
        alpha_d = DipolePolarizability(energy_min=1.0)
        
        # Compute from poles directly
        from_poles = alpha_d.compute_from_poles(pole_energies, pole_strengths)
        
        # Compute from integrated spectrum
        energy = np.linspace(2, 40, 500)
        strength = lorentzian_sum(energy, pole_energies, pole_strengths, np.full(3, 1.0))
        from_integral = alpha_d.compute(energy, strength)
        
        # Should be close (not exact due to Lorentzian tails)
        assert from_poles == pytest.approx(from_integral, rel=0.3)


class TestCustomObservable:
    """Tests for CustomObservable."""
    
    def test_custom_function(self):
        def max_value(energy, strength, **kw):
            return float(np.max(strength))
        
        obs = CustomObservable(max_value, name="peak")
        
        energy = np.linspace(0, 40, 200)
        strength = _synthetic_spectrum(energy)
        
        value = obs.compute(energy, strength)
        assert value == pytest.approx(np.max(strength))
    
    def test_custom_with_target(self):
        def sum_strength(energy, strength, **kw):
            return float(np.sum(strength))
        
        obs = CustomObservable(sum_strength, name="sum", target=100.0, weight=2.0)
        
        energy = np.linspace(0, 40, 200)
        strength = np.ones(200) * 0.5
        
        # Sum should be 100
        value = obs.compute(energy, strength)
        assert value == pytest.approx(100.0)
        
        # Loss should be ~0
        assert obs.loss(energy, strength) == pytest.approx(0.0, abs=1e-6)


class TestObservableSet:
    """Tests for ObservableSet collection."""
    
    def test_compute_all(self):
        energy = np.linspace(1, 40, 200)
        strength = _synthetic_spectrum(energy)
        
        obs_set = ObservableSet([
            SumRule(k=0, name="m0"),
            SumRule(k=1, name="m1"),
        ])
        
        results = obs_set.compute_all(energy, strength)
        assert len(results) == 2
        assert results[0].name == "m0"
        assert results[1].name == "m1"
    
    def test_to_dict(self):
        energy = np.linspace(1, 40, 200)
        strength = _synthetic_spectrum(energy)
        
        obs_set = ObservableSet([
            SumRule(k=0, name="m0"),
            SumRule(k=1, name="m1"),
        ])
        
        result_dict = obs_set.to_dict(energy, strength)
        assert "m0" in result_dict
        assert "m1" in result_dict
    
    def test_total_loss(self):
        energy = np.linspace(1, 40, 200)
        strength = _synthetic_spectrum(energy)
        
        m0 = SumRule(k=0).compute(energy, strength)
        m1 = SumRule(k=1).compute(energy, strength)
        
        # With correct targets, total loss should be ~0
        obs_set = ObservableSet([
            SumRule(k=0, target=m0),
            SumRule(k=1, target=m1),
        ])
        
        assert obs_set.total_loss(energy, strength) == pytest.approx(0.0, abs=1e-8)
