from __future__ import annotations

import csv
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from smlr.core import RetainedModePolicy, numerics
from smlr.core import split_by_parameter_ranges
from smlr.domains import PaperBetaDecayAdapter
from smlr.metrics import integrated_strength_error, observable_error_summary, relative_error, weighted_spectral_loss
from smlr.serialization import load_emulator, package_existing_emulator, save_emulator
from smlr.specs import StrengthGridSpec, h2_2d_strength_spec, paper_beta_em2_spec, paper_dipole_em1_spec
from smlr.validation import GENERIC_2D_PARAMETER_NAMES, validate_strength_grid


class CoreNumericsRegressionTest(unittest.TestCase):
    def test_lorentzian_known_values_float64(self):
        energy = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        poles = tf.constant([0.5], dtype=tf.float64)
        strengths = tf.constant([2.0], dtype=tf.float64)
        width = tf.constant(0.4, dtype=tf.float64)
        actual = numerics.give_me_lorentzian(energy, poles, strengths, width, dtype=tf.float64).numpy()
        expected = 2.0 * (0.4 / (2.0 * np.pi)) / ((energy - 0.5) ** 2 + 0.4**2 / 4.0)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-7)

    def test_centered_spectrum_initialization(self):
        energies = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        strengths = np.array([1.0, 4.0, 9.0, 16.0, 25.0])
        diag, v0, info = numerics.centered_spectrum_initialization(energies, strengths, n=6, retain=0.5, dtype=np.float64)
        self.assertEqual(info, (1, 4, 3))
        np.testing.assert_allclose(diag, [-3.0, -1.0, 0.0, 1.0, 3.0, 5.0])
        np.testing.assert_allclose(v0, [0.0, 2.0, 3.0, 4.0, 0.0, 0.0])

    def test_retained_mode_policy(self):
        policy = RetainedModePolicy(kind="centered", retain=0.5)
        self.assertEqual(policy.indices(6), (1, 4, 3))
        np.testing.assert_array_equal(policy.apply(np.arange(6)), [1, 2, 3])
        self.assertEqual(RetainedModePolicy.from_dict(policy.to_dict()).indices(6), (1, 4, 3))

    def test_parameter_range_split(self):
        points = np.array([[0.0, 0.0], [0.0, 2.0], [1.0, 0.0], [1.0, 2.0]])
        split = split_by_parameter_ranges(points, ("alpha", "beta"), {"alpha": [0.0, 0.5]})
        np.testing.assert_array_equal(split.train_mask, [True, True, False, False])
        np.testing.assert_array_equal(split.test_indices, [2, 3])


class MetricsRegressionTest(unittest.TestCase):
    def test_observable_relative_error_summary(self):
        pred = np.array([1.0, 2.2, 2.7])
        true = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(relative_error(pred, true), [0.0, 0.1, 0.1])
        summary = observable_error_summary(pred, true)
        self.assertAlmostEqual(summary["mean_relative_error"], 2.0 / 30.0)
        self.assertAlmostEqual(summary["max_absolute_error"], 0.3)

    def test_integrated_strength_and_weighted_loss(self):
        energy = np.array([0.0, 1.0, 2.0])
        true = np.array([[1.0, 2.0, 1.0], [2.0, 2.0, 2.0]])
        pred = np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 3.0]])
        errors = integrated_strength_error(pred, true, energy)
        self.assertEqual(errors.shape, (2,))
        self.assertGreater(errors[0], 0.0)
        self.assertAlmostEqual(weighted_spectral_loss(pred, true, energy, weights=[1.0, 3.0]), np.average(errors, weights=[1.0, 3.0]))


class UserDataValidationTest(unittest.TestCase):
    def test_valid_strength_grid_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for p1 in ("0.0", "1.0"):
                for p2 in ("0.0", "2.0"):
                    (root / f"strength_{p1}_{p2}.out").write_text("0.0 1.0\n1.0 2.0\n")
            report = validate_strength_grid(root)
            self.assertTrue(report.ok)
            self.assertEqual(report.files_checked, 4)
            self.assertEqual(len(report.points), 4)
            self.assertEqual(report.parameter_names, GENERIC_2D_PARAMETER_NAMES)

    def test_nonrectangular_strength_grid_fails_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "strength_0.0_0.0.out").write_text("0.0 1.0\n1.0 2.0\n")
            (root / "strength_2.0_0.0.out").write_text("0.0 1.0\n1.0 2.0\n")
            (root / "strength_0.0_1.0.out").write_text("0.0 1.0\n1.0 2.0\n")
            report = validate_strength_grid(
                root,
            )
            self.assertFalse(report.ok)
            self.assertIn("not rectangular", "\n".join(issue.message for issue in report.issues))

    def test_bad_numeric_strength_file_fails_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "strength_0.0_0.0.out").write_text("not numeric\n")
            report = validate_strength_grid(
                root,
                require_rectangular_grid=False,
            )
            self.assertFalse(report.ok)
            self.assertIn("could not read numeric strength table", "\n".join(issue.message for issue in report.issues))

    def test_extra_numeric_strength_column_fails_generic_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "strength_0.0_0.0.out").write_text("0.0 1.0 2.0\n1.0 2.0 3.0\n")
            report = validate_strength_grid(root, require_rectangular_grid=False)
            self.assertFalse(report.ok)
            self.assertIn("expected at most 2 numeric columns", "\n".join(issue.message for issue in report.issues))


class DataSpecRegressionTest(unittest.TestCase):
    def test_strength_grid_spec_validates_toy_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for alpha in ("0.0", "1.0"):
                for beta in ("0.0", "2.0"):
                    (root / f"strength_{beta}_{alpha}.out").write_text("0.0 1.0\n1.0 2.0\n")
            spec = StrengthGridSpec(
                data_dir=str(root),
                filename_regex=r"strength_(?P<beta>[0-9.]+)_(?P<alpha>[0-9.]+)\.out",
                parameter_names=("alpha", "beta"),
            )
            report = spec.validate()
            self.assertTrue(report.ok)
            self.assertEqual(report.parameter_names, ("alpha", "beta"))

    def test_builtin_paper_specs_validate_available_data(self):
        dipole = paper_dipole_em1_spec(strength_dir=str(ROOT / "dipole_polarizability_160Yb/total_strength"))
        beta = paper_beta_em2_spec(data_dir=str(ROOT / "beta_decay_80Ni"))
        self.assertTrue(dipole.validate_data().ok)
        self.assertTrue(beta.validate_data().ok)
        self.assertEqual(dipole.observable.name, "alphaD")
        self.assertEqual(beta.observable.name, "half_life")

    def test_run_spec_splits_by_training_box(self):
        spec = paper_dipole_em1_spec()
        points = np.array([[0.2, 1.0], [0.4, 1.5], [1.0, 2.0], [2.0, 4.5]])
        split = spec.split(points)
        np.testing.assert_array_equal(split.train_mask, [False, True, True, False])

    def test_h2_example_spec_validates_available_data(self):
        spec = h2_2d_strength_spec(strength_dir=str(ROOT / "extra_docs/aaron_H2/total_strength"))
        report = spec.validate_data()
        self.assertTrue(report.ok)
        self.assertEqual(report.parameter_names, ("q", "theta"))


class SerializationRegressionTest(unittest.TestCase):
    def test_save_and_load_emulator_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            params = np.array([1.0, 2.0, 3.0])
            spec = paper_beta_em2_spec(data_dir="beta_decay_80Ni")
            policy = RetainedModePolicy(kind="centered", retain=0.9)
            save_emulator(
                tmp,
                params=params,
                name="toy",
                adapter="PaperBetaDecayAdapter",
                spec=spec,
                retained_modes=policy,
                metadata={"seed": 42},
            )
            loaded = load_emulator(tmp)
            np.testing.assert_array_equal(loaded.params, params)
            self.assertEqual(loaded.record.name, "toy")
            self.assertEqual(loaded.record.retained_mode_policy.indices(10), (0, 9, 9))
            self.assertEqual(loaded.record.metadata["seed"], 42)

    def test_loaded_beta_em2_prediction_is_finite(self):
        with tempfile.TemporaryDirectory() as tmp:
            params_file = ROOT / "Beta_decay_package/runs_em2/n6_seed42/params_6_only_HL.txt"
            package_existing_emulator(
                tmp,
                params_file=params_file,
                name="beta_em2_n6",
                adapter="PaperBetaDecayAdapter",
                spec={"model": {"n": 6}, "central_point": [1.0, 0.5]},
                metadata={"mode": "em2"},
            )
            loaded = load_emulator(tmp)
            pred = loaded.predict_beta_em2([[1.0, 0.5]])["half_life"]
            self.assertEqual(pred.shape, (1,))
            self.assertTrue(np.isfinite(pred[0]))
            self.assertGreater(pred[0], 0.0)


class BetaRunRegressionTest(unittest.TestCase):
    ROOT = ROOT
    RUNS = [
        (
            "em1_n8",
            ROOT / "Beta_decay/runs_em1/n8_retain0p9_w1p0_seed42",
            ROOT / "Beta_decay_package/runs_em1/n8_retain0p9_w1p0_seed42",
            "params_best_n8_retain0.9.txt",
            "seed_42/cost_history_seed42.txt",
            0.53932888642,
        ),
        (
            "em1_n13",
            ROOT / "Beta_decay/runs_em1/n13_retain0p9_w1p0_seed42",
            ROOT / "Beta_decay_package/runs_em1/n13_retain0p9_w1p0_seed42",
            "params_best_n13_retain0.9.txt",
            "seed_42/cost_history_seed42.txt",
            0.086267438514,
        ),
        (
            "em2_n6",
            ROOT / "Beta_decay/runs_em2/n6_seed42",
            ROOT / "Beta_decay_package/runs_em2/n6_seed42",
            "params_6_only_HL.txt",
            None,
            5.57867922325e-05,
        ),
        (
            "em2_n9",
            ROOT / "Beta_decay/runs_em2/n9_seed42",
            ROOT / "Beta_decay_package/runs_em2/n9_seed42",
            "params_9_only_HL.txt",
            None,
            1.45207866796e-05,
        ),
    ]

    def test_original_and_package_best_params_match_bitwise(self):
        for name, original, package, params_file, _, _ in self.RUNS:
            with self.subTest(name=name):
                a = np.loadtxt(original / params_file)
                b = np.loadtxt(package / params_file)
                np.testing.assert_array_equal(a, b)

    def test_original_and_package_cost_histories_match_bitwise_when_available(self):
        for name, original, package, _, cost_file, _ in self.RUNS:
            if cost_file is None:
                continue
            with self.subTest(name=name):
                a = np.loadtxt(original / cost_file)
                b = np.loadtxt(package / cost_file)
                np.testing.assert_array_equal(a, b)

    def test_recorded_best_costs_match_reference(self):
        for name, original, package, _, _, expected_cost in self.RUNS:
            with self.subTest(name=name):
                for run_dir in (original, package):
                    meta = {}
                    for line in (run_dir / "seed_42/meta.txt").read_text().splitlines():
                        if "=" in line:
                            key, value = line.split("=", 1)
                            meta[key] = value
                    self.assertAlmostEqual(float(meta["best_cost"]), expected_cost, delta=max(1e-12, abs(expected_cost) * 1e-9))

    def test_diagnostic_artifacts_exist_and_are_valid_pngs(self):
        for name, original, package, _, _, _ in self.RUNS:
            with self.subTest(name=name):
                for run_dir in (original, package):
                    diag = run_dir / "diagnostics"
                    expected = [
                        "half_life_true_vs_pred.png",
                        "parameter_error_map.png",
                        "detail_spectrum.png" if name.startswith("em1") else "detail_observable.png",
                    ]
                    for filename in expected:
                        path = diag / filename
                        self.assertGreater(path.stat().st_size, 1000)
                        self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_observable_diagnostic_csv_matches_between_original_and_package(self):
        for name, original, package, _, _, _ in self.RUNS:
            with self.subTest(name=name):
                a = (original / "diagnostics/half_life_predictions.csv").read_text()
                b = (package / "diagnostics/half_life_predictions.csv").read_text()
                self.assertEqual(a, b)

    def test_observable_error_metrics_are_within_reference_bounds(self):
        bounds = {
            "em1_n8": 0.014,
            "em1_n13": 0.010,
            "em2_n6": 0.007,
            "em2_n9": 0.003,
        }
        for name, original, package, _, _, _ in self.RUNS:
            with self.subTest(name=name):
                for run_dir in (original, package):
                    summary = {}
                    for line in (run_dir / "diagnostics/summary.txt").read_text().splitlines():
                        key, value = line.split("=", 1)
                        summary[key] = float(value)
                    self.assertLess(summary["max_relative_error"], bounds[name])

    def test_selected_spectrum_reconstruction_is_stable(self):
        adapter = PaperBetaDecayAdapter(n=8)
        run_dir = self.ROOT / "Beta_decay_package/runs_em1/n8_retain0p9_w1p0_seed42"
        params = np.loadtxt(run_dir / "params_best_n8_retain0.9.txt")
        D, S1, S2, v0, eta, x1, x2, x3 = adapter.unpack_em1_parameters(params)
        central_point = ("1.000", "0.500")
        point = ("1.000", "0.500")
        matrix = adapter.em1_matrix(D, S1, S2, point[0], point[1], central_point)
        eigvals, eigvecs = tf.linalg.eigh(matrix)
        left, right, _ = numerics.centered_keep_indices(8, 0.9)
        eigvals = eigvals[left:right]
        eigvecs = eigvecs[:, left:right]
        strengths = tf.square(tf.linalg.matvec(tf.transpose(eigvecs), v0))
        width = adapter.em1_width(eta, x1, x2, x3, float(point[0]), float(point[1]))
        x = tf.constant([-1.0, 0.0, 0.5], dtype=tf.float64)
        spectrum = adapter.lorentzian(x, eigvals, strengths, width).numpy()
        expected = np.array([4.380955784300, 6.755414970570, 4.894853976940])
        np.testing.assert_allclose(spectrum, expected, rtol=0.0, atol=1e-6)


class BetaEM1SmokeParityTest(unittest.TestCase):
    def test_core_backed_package_em1_matches_project_legacy_for_short_training(self):
        from numpy.polynomial.polynomial import Polynomial

        import Beta_decay.helper as legacy
        import Beta_decay_package.src.helper_gpt as package

        nucnam = "Ni_80"
        coeffs = Polynomial(legacy.fit_phase_space(0, 28, 80, 15)).coef
        pattern = re.compile(r"lorm_" + re.escape(nucnam) + r"_([0-9.]+)_([0-9.]+)\.out")
        combined = []
        lorm_dir = ROOT / "beta_decay_80Ni/total_lorm"
        excm_dir = ROOT / "beta_decay_80Ni/total_excm"
        for fname in sorted(os.listdir(lorm_dir)):
            match = pattern.match(fname)
            if not match:
                continue
            beta = match.group(1)
            alpha = match.group(2)
            if 0.1 <= float(beta) <= 0.9 and 0.2 <= float(alpha) <= 1.8:
                combined.append((alpha, beta))

        points = combined[:3]
        values = np.asarray(combined, dtype=float)
        center = 0.5 * (values.min(axis=0) + values.max(axis=0))
        central_point = tuple(combined[int(np.argmin(np.linalg.norm(values - center, axis=1)))])

        for n, retain, weight in ((4, 0.75, 1.0), (6, 0.9, 1.0), (8, 0.9, 0.5)):
            with self.subTest(n=n, retain=retain, weight=weight):
                n_params = n + 2 * int(n * (n + 1) / 2) + n + 4
                initial = np.random.default_rng(42).uniform(0.0, 1.0, size=n_params)
                legacy_lors = []
                legacy_hls = []
                for alpha, beta in points:
                    lorm = np.loadtxt(lorm_dir / f"lorm_{nucnam}_{beta}_{alpha}.out")
                    lorm = lorm[lorm[:, 0] < legacy.del_nH]
                    lorm = lorm[lorm[:, 0] > -10]
                    legacy_lors.append(lorm)
                    excm = np.loadtxt(excm_dir / f"excm_{nucnam}_{beta}_{alpha}.out")
                    excm = excm[excm[:, 0] < legacy.del_nH]
                    excm = excm[excm[:, 0] > -10]
                    legacy_hls.append(legacy.half_life_loss(excm[:, 0], excm[:, 1], coeffs, 1.2))
                os.environ["SMLR_BETA_DATA_DIR"] = str(ROOT / "beta_decay_80Ni")
                package_lors, package_hls = package.data_table(points, coeffs, 1.2, nucnam)

                for legacy_table, package_table in zip(legacy_lors, package_lors):
                    np.testing.assert_array_equal(legacy_table, package_table)
                np.testing.assert_array_equal(legacy_hls, package_hls)

                legacy_params = tf.Variable(initial.copy(), dtype=tf.float64)
                package_params = tf.Variable(initial.copy(), dtype=tf.float64)
                legacy_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
                package_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
                legacy_costs = []
                package_costs = []

                for _ in range(3):
                    with tf.GradientTape() as tape:
                        legacy_cost, *_ = legacy.cost_function(
                            legacy_params, n, points, legacy_lors, legacy_hls, coeffs, 1.2, weight, central_point, retain
                        )
                    grads = tape.gradient(legacy_cost, [legacy_params])
                    legacy_optimizer.apply_gradients(zip(grads, [legacy_params]))
                    legacy_costs.append(legacy_cost.numpy())

                    with tf.GradientTape() as tape:
                        package_cost, *_ = package.cost_function(
                            package_params, n, points, package_lors, package_hls, coeffs, 1.2, weight, central_point, retain
                        )
                    grads = tape.gradient(package_cost, [package_params])
                    package_optimizer.apply_gradients(zip(grads, [package_params]))
                    package_costs.append(package_cost.numpy())

                np.testing.assert_array_equal(legacy_costs, package_costs)
                np.testing.assert_array_equal(legacy_params.numpy(), package_params.numpy())


class DipoleEM1SmokeParityTest(unittest.TestCase):
    def test_core_backed_package_dipole_em1_matches_project_legacy_for_short_training(self):
        sys.path.insert(0, str(ROOT / "Dipole_polarizability"))
        import scrap.helper as legacy
        import Dipole_polarizability.src.helper_gpt as package

        strength_dir = ROOT / "dipole_polarizability_160Yb/total_strength"
        alphaD_dir = ROOT / "dipole_polarizability_160Yb/total_alphaD"
        pattern = re.compile(r"strength_([0-9.]+)_([0-9.]+)\.out")
        combined = []
        for path in sorted(strength_dir.iterdir()):
            match = pattern.match(path.name)
            if not match:
                continue
            beta = match.group(1)
            alpha = match.group(2)
            if 1.5 <= float(beta) <= 4.0 and 0.4 <= float(alpha) <= 1.8:
                combined.append((alpha, beta))

        points = combined[:3]
        values = np.asarray(combined, dtype=float)
        center = 0.5 * (values.min(axis=0) + values.max(axis=0))
        central_point = min(
            combined,
            key=lambda item: (float(item[0]) - center[0]) ** 2 + (float(item[1]) - center[1]) ** 2,
        )

        strengths = []
        alphaD_values = []
        for alpha, beta in points:
            strengths.append(np.loadtxt(strength_dir / f"strength_{beta}_{alpha}.out"))
            alphaD_values.append(np.loadtxt(alphaD_dir / f"alphaD_{beta}_{alpha}.out"))
        alphaD_values = [float(row[2]) for row in np.vstack(alphaD_values)]

        for n, retain, fold in ((4, 0.5, 2.0), (6, 0.5, 2.0), (8, 0.75, 1.5)):
            with self.subTest(n=n, retain=retain, fold=fold):
                n_params = 1 + 3 * n + n + 4 * int(n * (n + 1) / 2) + 4
                initial = np.random.default_rng(42).uniform(0.0, 1.0, size=n_params).astype(np.float32)
                initial[0] = np.float32(fold)
                initial[-4:] = np.array([1, 1, 1, 1], dtype=np.float32)

                legacy_params = tf.Variable(initial.copy(), dtype=tf.float32)
                package_params = tf.Variable(initial.copy(), dtype=tf.float32)
                legacy_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
                package_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
                legacy_costs = []
                package_costs = []

                for _ in range(3):
                    with tf.GradientTape() as tape:
                        legacy_cost, *_ = legacy.cost_function_batched_mixed(
                            legacy_params, n, points, strengths, alphaD_values, central_point, retain, 1, 2, 0, 875.0, 1e-8
                        )
                    grads = tape.gradient(legacy_cost, [legacy_params])
                    legacy_optimizer.apply_gradients(zip(grads, [legacy_params]))
                    legacy_costs.append(legacy_cost.numpy())

                    with tf.GradientTape() as tape:
                        package_cost, *_ = package.cost_function_batched_mixed(
                            package_params, n, points, strengths, alphaD_values, central_point, retain, 1, 2, 0, 875.0, 1e-8
                        )
                    grads = tape.gradient(package_cost, [package_params])
                    package_optimizer.apply_gradients(zip(grads, [package_params]))
                    package_costs.append(package_cost.numpy())

                np.testing.assert_array_equal(legacy_costs, package_costs)
                np.testing.assert_array_equal(legacy_params.numpy(), package_params.numpy())


if __name__ == "__main__":
    unittest.main()
