from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from smlr.core.splitting import ParameterSplit, split_by_parameter_ranges
from smlr.validation import (
    GENERIC_2D_PARAMETER_NAMES,
    GENERIC_2D_STRENGTH_REGEX,
    ValidationReport,
    validate_strength_grid,
)


@dataclass(frozen=True)
class StrengthGridSpec:
    """File layout for a strength-function parameter grid."""

    data_dir: str
    filename_regex: str
    parameter_names: tuple[str, ...]
    min_files: int = 1
    min_columns: int = 2
    max_columns: int | None = 2
    require_rectangular_grid: bool = True
    allow_negative_strength: bool = False

    def validate(self) -> ValidationReport:
        return validate_strength_grid(
            self.data_dir,
            self.filename_regex,
            parameter_names=self.parameter_names,
            min_files=self.min_files,
            min_columns=self.min_columns,
            max_columns=self.max_columns,
            require_rectangular_grid=self.require_rectangular_grid,
            allow_negative_strength=self.allow_negative_strength,
        )

    def resolved(self, base_dir: str | Path) -> "StrengthGridSpec":
        return StrengthGridSpec(
            data_dir=str((Path(base_dir) / self.data_dir).resolve()) if not Path(self.data_dir).is_absolute() else self.data_dir,
            filename_regex=self.filename_regex,
            parameter_names=self.parameter_names,
            min_files=self.min_files,
            min_columns=self.min_columns,
            max_columns=self.max_columns,
            require_rectangular_grid=self.require_rectangular_grid,
            allow_negative_strength=self.allow_negative_strength,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StrengthGridSpec":
        return cls(
            data_dir=str(data["data_dir"]),
            filename_regex=str(data["filename_regex"]),
            parameter_names=tuple(data["parameter_names"]),
            min_files=int(data.get("min_files", 1)),
            min_columns=int(data.get("min_columns", 2)),
            max_columns=data.get("max_columns", 2),
            require_rectangular_grid=bool(data.get("require_rectangular_grid", True)),
            allow_negative_strength=bool(data.get("allow_negative_strength", False)),
        )


@dataclass(frozen=True)
class ObservableSpec:
    """Optional scalar/table observable layout associated with a strength grid."""

    name: str
    data_dir: str | None = None
    filename_regex: str | None = None
    parameter_names: tuple[str, ...] = ()
    min_columns: int = 1
    log_scale: bool = False
    units: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ObservableSpec | None":
        if data is None:
            return None
        return cls(
            name=str(data["name"]),
            data_dir=data.get("data_dir"),
            filename_regex=data.get("filename_regex"),
            parameter_names=tuple(data.get("parameter_names", ())),
            min_columns=int(data.get("min_columns", 1)),
            log_scale=bool(data.get("log_scale", False)),
            units=data.get("units"),
        )


@dataclass(frozen=True)
class EmulatorRunSpec:
    """Domain-neutral run description used by validation, CLI, and notebooks."""

    name: str
    strength: StrengthGridSpec
    observable: ObservableSpec | None = None
    model: Mapping[str, Any] = field(default_factory=dict)
    train_filter_ranges: Mapping[str, Sequence[float]] | None = None
    central_point: tuple[float, ...] | None = None
    output_dir: str = "runs"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate_data(self) -> ValidationReport:
        return self.strength.validate()

    def split(self, param_values) -> ParameterSplit:
        return split_by_parameter_ranges(
            np.asarray(param_values, dtype=float),
            self.strength.parameter_names,
            self.train_filter_ranges,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "strength": self.strength.to_dict(),
            "observable": self.observable.to_dict() if self.observable is not None else None,
            "model": dict(self.model),
            "train_filter_ranges": dict(self.train_filter_ranges) if self.train_filter_ranges is not None else None,
            "central_point": list(self.central_point) if self.central_point is not None else None,
            "output_dir": self.output_dir,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EmulatorRunSpec":
        return cls(
            name=str(data["name"]),
            strength=StrengthGridSpec.from_dict(data["strength"]),
            observable=ObservableSpec.from_dict(data.get("observable")),
            model=dict(data.get("model", {})),
            train_filter_ranges=data.get("train_filter_ranges"),
            central_point=tuple(data["central_point"]) if data.get("central_point") is not None else None,
            output_dir=str(data.get("output_dir", "runs")),
            metadata=dict(data.get("metadata", {})),
        )


def paper_dipole_em1_spec(
    *,
    strength_dir: str = "dipole_polarizability_160Yb/total_strength",
    alphaD_dir: str = "dipole_polarizability_160Yb/total_alphaD",
    output_dir: str = "Dipole_polarizability/runs_em1",
) -> EmulatorRunSpec:
    return EmulatorRunSpec(
        name="paper_dipole_em1",
        strength=StrengthGridSpec(
            data_dir=strength_dir,
            filename_regex=r"strength_(?P<p2>[0-9.]+)_(?P<p1>[0-9.]+)\.out",
            parameter_names=("p1", "p2"),
        ),
        observable=ObservableSpec(
            name="alphaD",
            data_dir=alphaD_dir,
            filename_regex=r"alphaD_(?P<p2>[0-9.]+)_(?P<p1>[0-9.]+)\.out",
            parameter_names=("p1", "p2"),
            min_columns=3,
            units="fm^3",
        ),
        model={"n": 10, "retain": 0.5, "fold": 2.0, "ansatz": "paper_dipole", "width_model": "affine"},
        train_filter_ranges={"p1": (0.4, 1.8), "p2": (1.5, 4.0)},
        output_dir=output_dir,
        metadata={"paper_behavior": True, "domain": "dipole"},
    )


def dipole_2d_spectral_example_spec(
    *,
    strength_dir: str = "dipole_polarizability_160Yb/total_strength",
    alphaD_dir: str = "dipole_polarizability_160Yb/total_alphaD",
    output_dir: str = "runs/dp_2d_spectral",
) -> EmulatorRunSpec:
    """General two-parameter spectral-emulation example using the DP dataset."""

    return EmulatorRunSpec(
        name="dp_2d_spectral_example",
        strength=StrengthGridSpec(
            data_dir=strength_dir,
            filename_regex=r"strength_(?P<p2>[0-9.]+)_(?P<p1>[0-9.]+)\.out",
            parameter_names=("p1", "p2"),
            min_columns=2,
            max_columns=2,
        ),
        observable=ObservableSpec(
            name="alphaD",
            data_dir=alphaD_dir,
            filename_regex=r"alphaD_(?P<p2>[0-9.]+)_(?P<p1>[0-9.]+)\.out",
            parameter_names=("p1", "p2"),
            min_columns=3,
            units="fm^3",
        ),
        model={
            "n": 10,
            "retain": 0.5,
            "fold": 2.0,
            "ansatz": "linear_exp",
            "width_model": "affine",
            "use_vector_terms": True,
            "w_strength": 1.0,
            "w_alphaD": 1.0,
            "w_m1": 0.0,
        },
        train_filter_ranges={"p1": (0.4, 1.8), "p2": (1.5, 4.0)},
        central_point=(1.1, 2.75),
        output_dir=output_dir,
        metadata={
            "domain": "generic_2d_lrt",
            "example": "dipole_polarizability_160Yb",
            "physical_parameter_names": {"p1": "alpha", "p2": "beta"},
            "format": "strength_<p2>_<p1>.out with columns omega, B(E1)",
            "paper_adapter_required": False,
        },
    )


def paper_beta_em1_spec(
    *,
    data_dir: str = "beta_decay_80Ni",
    nucnam: str = "Ni_80",
    output_dir: str = "Beta_decay_package/runs_em1",
) -> EmulatorRunSpec:
    return EmulatorRunSpec(
        name="paper_beta_em1",
        strength=StrengthGridSpec(
            data_dir=str(Path(data_dir) / "total_lorm") if (Path(data_dir) / "total_lorm").is_dir() else data_dir,
            filename_regex=rf"lorm_{nucnam}_(?P<beta>[0-9.]+)_(?P<alpha>[0-9.]+)\.out",
            parameter_names=("alpha", "beta"),
        ),
        observable=ObservableSpec(
            name="half_life",
            data_dir=str(Path(data_dir) / "total_half_life") if (Path(data_dir) / "total_half_life").is_dir() else None,
            filename_regex=rf"half_life_{nucnam}_(?P<beta>[0-9.]+)_(?P<alpha>[0-9.]+)\.txt",
            parameter_names=("alpha", "beta"),
            min_columns=1,
            log_scale=True,
        ),
        model={"n": 8, "retain": 0.9, "ansatz": "paper_beta_decay"},
        output_dir=output_dir,
        metadata={"paper_behavior": True, "domain": "beta_decay", "nucnam": nucnam},
    )


def paper_beta_em2_spec(
    *,
    data_dir: str = "beta_decay_80Ni",
    nucnam: str = "Ni_80",
    output_dir: str = "Beta_decay_package/runs_em2",
) -> EmulatorRunSpec:
    return EmulatorRunSpec(
        name="paper_beta_em2",
        strength=StrengthGridSpec(
            data_dir=str(Path(data_dir) / "total_lorm") if (Path(data_dir) / "total_lorm").is_dir() else data_dir,
            filename_regex=rf"lorm_{nucnam}_(?P<beta>[0-9.]+)_(?P<alpha>[0-9.]+)\.out",
            parameter_names=("alpha", "beta"),
        ),
        observable=ObservableSpec(
            name="half_life",
            data_dir=str(Path(data_dir) / "total_half_life") if (Path(data_dir) / "total_half_life").is_dir() else None,
            filename_regex=rf"half_life_{nucnam}_(?P<beta>[0-9.]+)_(?P<alpha>[0-9.]+)\.txt",
            parameter_names=("alpha", "beta"),
            min_columns=1,
            log_scale=True,
        ),
        model={"n": 9, "ansatz": "paper_beta_decay_half_life_only"},
        output_dir=output_dir,
        metadata={"paper_behavior": True, "domain": "beta_decay", "nucnam": nucnam},
    )


def h2_2d_strength_spec(
    *,
    strength_dir: str = "extra_docs/aaron_H2/total_strength",
    output_dir: str = "extra_docs/aaron_H2/runs_strength",
) -> EmulatorRunSpec:
    """Synthetic two-parameter H2 strength-only example spec."""

    return EmulatorRunSpec(
        name="h2_2d_strength",
        strength=StrengthGridSpec(
            data_dir=strength_dir,
            filename_regex=r"strength_(?P<q>[0-9.]+)_(?P<theta>[0-9.]+)\.out",
            parameter_names=("q", "theta"),
        ),
        observable=None,
        model={
            "n": 13,
            "retain": 1.0,
            "fold": 2.0,
            "ansatz": "linear_exp",
            "width_model": "affine",
            "strength_only": True,
        },
        output_dir=output_dir,
        metadata={"domain": "synthetic_lrt", "example": "aaron_H2"},
    )


def generic_2d_strength_spec(
    *,
    strength_dir: str,
    output_dir: str = "runs_strength",
) -> EmulatorRunSpec:
    """Release-supported generic two-parameter strength-only run spec."""

    return EmulatorRunSpec(
        name="generic_2d_strength",
        strength=StrengthGridSpec(
            data_dir=strength_dir,
            filename_regex=GENERIC_2D_STRENGTH_REGEX,
            parameter_names=GENERIC_2D_PARAMETER_NAMES,
            min_columns=2,
            max_columns=2,
        ),
        observable=None,
        model={
            "ansatz": "linear_exp",
            "width_model": "affine",
            "strength_only": True,
        },
        output_dir=output_dir,
        metadata={
            "domain": "generic_2d_lrt",
            "format": "strength_<p1>_<p2>.out with columns omega, B",
        },
    )


BUILTIN_SPECS = {
    "dipole-2d-example": dipole_2d_spectral_example_spec,
    "dipole-paper-em1": paper_dipole_em1_spec,
    "beta-paper-em1": paper_beta_em1_spec,
    "beta-paper-em2": paper_beta_em2_spec,
    "h2-2d-strength": h2_2d_strength_spec,
}


def load_run_spec(path: str | Path) -> EmulatorRunSpec:
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("YAML specs require PyYAML. Install `smlr[yaml]` to load YAML config files.") from exc
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    return EmulatorRunSpec.from_dict(payload)


def save_run_spec(spec: EmulatorRunSpec, path: str | Path) -> None:
    path = Path(path)
    payload = spec.to_dict()
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("YAML specs require PyYAML. Install `smlr[yaml]` to save YAML config files.") from exc
        path.write_text(yaml.safe_dump(payload, sort_keys=True))
    else:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def get_builtin_spec(name: str) -> EmulatorRunSpec:
    try:
        return BUILTIN_SPECS[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown built-in spec {name!r}. Available: {sorted(BUILTIN_SPECS)}") from exc
