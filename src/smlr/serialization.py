from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from smlr import __version__
from smlr.core.retention import RetainedModePolicy
from smlr.specs import get_builtin_spec, load_run_spec


METADATA_FILENAME = "emulator.json"
PARAMS_FILENAME = "params.txt"


@dataclass(frozen=True)
class EmulatorRecord:
    """Serializable metadata for a trained emulator."""

    name: str
    adapter: str
    params_file: str = PARAMS_FILENAME
    spec: Mapping[str, Any] | None = None
    retained_modes: Mapping[str, Any] | None = None
    backend: str = "tensorflow"
    package_version: str = __version__
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EmulatorRecord":
        return cls(
            name=str(data["name"]),
            adapter=str(data["adapter"]),
            params_file=str(data.get("params_file", PARAMS_FILENAME)),
            spec=data.get("spec"),
            retained_modes=data.get("retained_modes"),
            backend=str(data.get("backend", "tensorflow")),
            package_version=str(data.get("package_version", "unknown")),
            metadata=dict(data.get("metadata", {})),
        )

    @property
    def retained_mode_policy(self) -> RetainedModePolicy | None:
        if self.retained_modes is None:
            return None
        return RetainedModePolicy.from_dict(self.retained_modes)


@dataclass
class LoadedEmulator:
    """Loaded params plus metadata.

    This is intentionally light for now. Domain adapters will later grow
    prediction methods that consume this object directly.
    """

    record: EmulatorRecord
    params: np.ndarray
    root: Path

    def predict_dipole(self, param_values, central_point=None, energy=None):
        """Predict dipole-like poles, strengths, optional spectra, and alphaD.

        This is currently TensorFlow-backed and intended for emulators whose
        parameters follow the generic dipole packed layout.
        """

        import tensorflow as tf

        from smlr.domains import DipoleAdapter, PaperDipoleAdapter

        model = dict((self.record.spec or {}).get("model", {}))
        strength = dict((self.record.spec or {}).get("strength", {}))
        adapter_cls = PaperDipoleAdapter if self.record.adapter == "PaperDipoleAdapter" else DipoleAdapter
        adapter = adapter_cls(
            strength_dir=strength.get("data_dir", ""),
            observable_dir=((self.record.spec or {}).get("observable") or {}).get("data_dir"),
            n=int(model.get("n", 10)),
            n_params=int(model.get("n_params", np.asarray(param_values).shape[1])),
            retain=float(model.get("retain", 0.5)),
            fold=float(model.get("fold", 2.0)),
            ansatz=model.get("ansatz", "paper_dipole" if adapter_cls is PaperDipoleAdapter else "linear"),
            width_model=model.get("width_model", "affine"),
            use_vector_terms=bool(model.get("use_vector_terms", True)),
        )
        param_values = np.asarray(param_values, dtype=np.float32)
        central_point = _central_point_from_record(self.record, param_values, central_point).astype(np.float32)
        matrices, vectors, widths, _ = adapter.build_model(self.params, param_values, central_point)
        eigvals, eigvecs = tf.linalg.eigh(matrices)
        policy = self.record.retained_mode_policy or RetainedModePolicy(kind="centered", retain=adapter.retain)
        left, right, _ = policy.indices(int(eigvals.shape[1]))
        eigvals = eigvals[:, left:right]
        eigvecs = eigvecs[:, :, left:right]
        projections = tf.matmul(tf.transpose(eigvecs, perm=[0, 2, 1]), vectors[:, :, None])
        strengths = tf.square(tf.squeeze(projections, axis=-1))
        alphaD = [float(adapter.alphaD_from_poles(eigvals[i], strengths[i]).numpy()) for i in range(param_values.shape[0])]
        result = {
            "eigenvalues": eigvals.numpy(),
            "strengths": strengths.numpy(),
            "widths": widths.numpy(),
            "alphaD": np.asarray(alphaD),
        }
        if energy is not None:
            energy_tf = tf.convert_to_tensor(np.asarray(energy, dtype=np.float32), dtype=tf.float32)
            spectra = [
                adapter.lorentzian(energy_tf, eigvals[i], strengths[i], widths[i]).numpy()
                for i in range(param_values.shape[0])
            ]
            result["energy"] = np.asarray(energy)
            result["spectra"] = np.asarray(spectra)
        return result

    def predict_beta_em1(self, points, central_point=None, energy=None):
        """Predict beta-decay EM1 poles, strengths, optional spectra, and widths."""

        import tensorflow as tf

        from smlr.domains import PaperBetaDecayAdapter

        model = dict((self.record.spec or {}).get("model", {}))
        metadata = dict(self.record.metadata or {})
        adapter = PaperBetaDecayAdapter(
            n=int(model.get("n", metadata.get("n", 8))),
            retain=float(model.get("retain", metadata.get("retain", 0.9))),
        )
        points = np.asarray(points, dtype=float)
        central_point = _central_point_from_record(self.record, points, central_point)
        D, S1, S2, v0, eta, x1, x2, x3 = adapter.unpack_em1_parameters(self.params)
        eigs = []
        strengths = []
        widths = []
        spectra = []
        policy = self.record.retained_mode_policy or RetainedModePolicy(kind="centered", retain=adapter.retain)
        for point in points:
            matrix = adapter.em1_matrix(D, S1, S2, point[0], point[1], central_point)
            eigenvalues, eigenvectors = tf.linalg.eigh(matrix)
            left, right, _ = policy.indices(int(eigenvalues.shape[0]))
            eigenvalues = eigenvalues[left:right]
            eigenvectors = eigenvectors[:, left:right]
            B = tf.square(tf.linalg.matvec(tf.transpose(eigenvectors), v0))
            width = adapter.em1_width(eta, x1, x2, x3, float(point[0]), float(point[1]))
            eigs.append(eigenvalues.numpy())
            strengths.append(B.numpy())
            widths.append(float(width.numpy()))
            if energy is not None:
                spectra.append(adapter.lorentzian(tf.constant(energy, dtype=tf.float64), eigenvalues, B, width).numpy())
        result = {"eigenvalues": np.asarray(eigs), "strengths": np.asarray(strengths), "widths": np.asarray(widths)}
        if energy is not None:
            result["energy"] = np.asarray(energy)
            result["spectra"] = np.asarray(spectra)
        return result

    def predict_beta_em2(self, points, central_point=None):
        """Predict beta-decay EM2 half-life-only values."""

        import tensorflow as tf

        from smlr.domains import PaperBetaDecayAdapter

        model = dict((self.record.spec or {}).get("model", {}))
        metadata = dict(self.record.metadata or {})
        adapter = PaperBetaDecayAdapter(n=int(model.get("n", metadata.get("n", 9))))
        points = np.asarray(points, dtype=float)
        central_point = _central_point_from_record(self.record, points, central_point)
        D, S1, S2 = adapter.unpack_em2_parameters(self.params)
        half_lives = []
        for point in points:
            matrix = adapter.em2_matrix(D, S1, S2, point[0], point[1], central_point)
            eigenvalues, _ = tf.linalg.eigh(matrix)
            half_lives.append(float((10 ** eigenvalues[int(adapter.n / 2)]).numpy()))
        return {"half_life": np.asarray(half_lives)}


def _central_point_from_record(record: EmulatorRecord, param_values, override):
    if override is not None:
        return np.asarray(override, dtype=float)
    spec = record.spec or {}
    if spec.get("central_point") is not None:
        return np.asarray(spec["central_point"], dtype=float)
    metadata = record.metadata or {}
    if metadata.get("central_point") is not None:
        return np.asarray(metadata["central_point"], dtype=float)
    values = np.asarray(param_values, dtype=float)
    center = 0.5 * (np.min(values, axis=0) + np.max(values, axis=0))
    return values[int(np.argmin(np.sum((values - center[None, :]) ** 2, axis=1)))]


def save_emulator(
    out_dir: str | Path,
    *,
    params,
    name: str,
    adapter: str,
    spec=None,
    retained_modes: RetainedModePolicy | Mapping[str, Any] | None = None,
    backend: str = "tensorflow",
    metadata: Mapping[str, Any] | None = None,
) -> EmulatorRecord:
    """Save emulator params and metadata into a portable run directory."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / PARAMS_FILENAME, np.asarray(params))
    if hasattr(spec, "to_dict"):
        spec_payload = spec.to_dict()
    else:
        spec_payload = spec
    if isinstance(retained_modes, RetainedModePolicy):
        retained_payload = retained_modes.to_dict()
    else:
        retained_payload = retained_modes
    record = EmulatorRecord(
        name=name,
        adapter=adapter,
        params_file=PARAMS_FILENAME,
        spec=spec_payload,
        retained_modes=retained_payload,
        backend=backend,
        metadata=dict(metadata or {}),
    )
    (out_dir / METADATA_FILENAME).write_text(json.dumps(record.to_dict(), indent=2, sort_keys=True))
    return record


def package_existing_emulator(
    run_dir: str | Path,
    *,
    params_file: str | Path,
    name: str,
    adapter: str,
    spec=None,
    retained_modes: RetainedModePolicy | Mapping[str, Any] | None = None,
    backend: str = "tensorflow",
    metadata: Mapping[str, Any] | None = None,
) -> EmulatorRecord:
    """Create ``emulator.json`` for an already-trained parameter file.

    The original parameter file is copied to ``params.txt`` inside ``run_dir``.
    """

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(params_file, run_dir / PARAMS_FILENAME)
    params = np.loadtxt(run_dir / PARAMS_FILENAME)
    return save_emulator(
        run_dir,
        params=params,
        name=name,
        adapter=adapter,
        spec=spec,
        retained_modes=retained_modes,
        backend=backend,
        metadata=metadata,
    )


def load_emulator(path: str | Path) -> LoadedEmulator:
    """Load an emulator saved with ``save_emulator``."""

    root = Path(path)
    if root.is_file():
        metadata_path = root
        root = root.parent
    else:
        metadata_path = root / METADATA_FILENAME
    record = EmulatorRecord.from_dict(json.loads(metadata_path.read_text()))
    params = np.loadtxt(root / record.params_file)
    return LoadedEmulator(record=record, params=params, root=root)


def spec_from_selector(selector: str | None):
    if selector is None:
        return None
    path = Path(selector)
    if path.exists():
        return load_run_spec(path)
    return get_builtin_spec(selector)
