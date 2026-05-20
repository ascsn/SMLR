from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from smlr.core.splitting import ParameterSplit, split_by_parameter_ranges
from smlr.validation import ValidationReport, validate_strength_grid


@dataclass(frozen=True)
class StrengthGridSpec:
    """File layout for a strength-function parameter grid."""

    data_dir: str
    filename_regex: str
    parameter_names: tuple[str, ...]
    min_files: int = 1
    min_columns: int = 2
    require_rectangular_grid: bool = True
    allow_negative_strength: bool = False

    def validate(self) -> ValidationReport:
        return validate_strength_grid(
            self.data_dir,
            self.filename_regex,
            parameter_names=self.parameter_names,
            min_files=self.min_files,
            min_columns=self.min_columns,
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
            require_rectangular_grid=self.require_rectangular_grid,
            allow_negative_strength=self.allow_negative_strength,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def paper_dipole_em1_spec(
    *,
    strength_dir: str = "dipoles_data_all/total_strength",
    alphaD_dir: str = "dipoles_data_all/total_alphaD",
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
            units="fm^3",
        ),
        model={"n": 10, "retain": 0.5, "fold": 2.0, "ansatz": "paper_dipole", "width_model": "affine"},
        train_filter_ranges={"p1": (0.4, 1.8), "p2": (1.5, 4.0)},
        output_dir=output_dir,
        metadata={"paper_behavior": True, "domain": "dipole"},
    )


def paper_beta_em1_spec(
    *,
    data_dir: str = "beta_decay_data_Ni_80",
    nucnam: str = "Ni_80",
    output_dir: str = "Beta_decay_package/runs_em1",
) -> EmulatorRunSpec:
    return EmulatorRunSpec(
        name="paper_beta_em1",
        strength=StrengthGridSpec(
            data_dir=data_dir,
            filename_regex=rf"lorm_{nucnam}_(?P<beta>[0-9.]+)_(?P<alpha>[0-9.]+)\.out",
            parameter_names=("alpha", "beta"),
        ),
        observable=ObservableSpec(name="half_life", parameter_names=("alpha", "beta"), log_scale=True),
        model={"n": 8, "retain": 0.9, "ansatz": "paper_beta_decay"},
        output_dir=output_dir,
        metadata={"paper_behavior": True, "domain": "beta_decay", "nucnam": nucnam},
    )


def paper_beta_em2_spec(
    *,
    data_dir: str = "beta_decay_data_Ni_80",
    nucnam: str = "Ni_80",
    output_dir: str = "Beta_decay_package/runs_em2",
) -> EmulatorRunSpec:
    return EmulatorRunSpec(
        name="paper_beta_em2",
        strength=StrengthGridSpec(
            data_dir=data_dir,
            filename_regex=rf"lorm_{nucnam}_(?P<beta>[0-9.]+)_(?P<alpha>[0-9.]+)\.out",
            parameter_names=("alpha", "beta"),
        ),
        observable=ObservableSpec(name="half_life", parameter_names=("alpha", "beta"), log_scale=True),
        model={"n": 9, "ansatz": "paper_beta_decay_half_life_only"},
        output_dir=output_dir,
        metadata={"paper_behavior": True, "domain": "beta_decay", "nucnam": nucnam},
    )
