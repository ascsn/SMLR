from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Pattern

import numpy as np

GENERIC_2D_STRENGTH_REGEX = r"strength_(?P<p1>[0-9.]+)_(?P<p2>[0-9.]+)\.out"
GENERIC_2D_PARAMETER_NAMES = ("p1", "p2")


@dataclass(frozen=True)
class ValidationIssue:
    """One user-data validation problem."""

    level: str
    message: str
    path: str | None = None


@dataclass
class ValidationReport:
    """Result from validating an emulator input dataset."""

    ok: bool
    files_checked: int
    points: tuple[tuple[str, ...], ...] = ()
    parameter_names: tuple[str, ...] = ()
    issues: list[ValidationIssue] = field(default_factory=list)

    def raise_for_errors(self) -> None:
        errors = [issue for issue in self.issues if issue.level == "error"]
        if errors:
            details = "\n".join(f"- {issue.message}" for issue in errors)
            raise ValueError(f"SMLR data validation failed:\n{details}")

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "files_checked": self.files_checked,
            "points": [list(point) for point in self.points],
            "parameter_names": list(self.parameter_names),
            "issues": [issue.__dict__.copy() for issue in self.issues],
        }


def _compile_regex(pattern: str | Pattern[str]) -> Pattern[str]:
    return re.compile(pattern) if isinstance(pattern, str) else pattern


def _load_numeric_table(path: Path, *, min_columns: int, max_columns: int | None = None) -> np.ndarray:
    table = np.loadtxt(path, comments="#", ndmin=2)
    if table.size == 0:
        raise ValueError("file has no numeric rows")
    if table.shape[1] < min_columns:
        raise ValueError(f"expected at least {min_columns} numeric columns, got {table.shape[1]}")
    if max_columns is not None and table.shape[1] > max_columns:
        raise ValueError(f"expected at most {max_columns} numeric columns, got {table.shape[1]}")
    if not np.all(np.isfinite(table[:, :min_columns])):
        raise ValueError("numeric table contains NaN or inf values")
    return table


def _rectangular_grid_missing(points: Iterable[tuple[str, ...]]) -> list[tuple[str, ...]]:
    points = tuple(points)
    if not points:
        return []
    ndim = len(points[0])
    axes = [sorted({point[i] for point in points}, key=float) for i in range(ndim)]
    expected = {()}
    for axis in axes:
        expected = {prefix + (value,) for prefix in expected for value in axis}
    return sorted(expected.difference(points), key=lambda point: tuple(float(value) for value in point))


def _validate_matching_table_grid(
    data_dir: Path,
    filename_regex: str | Pattern[str],
    *,
    parameter_names: tuple[str, ...],
    expected_points: set[tuple[str, ...]],
    min_columns: int,
    max_columns: int | None = None,
    label: str,
) -> tuple[list[ValidationIssue], int]:
    issues: list[ValidationIssue] = []
    regex = _compile_regex(filename_regex)
    matched: list[tuple[Path, tuple[str, ...]]] = []

    if not data_dir.exists():
        return [ValidationIssue("error", f"{label} directory does not exist: {data_dir}", str(data_dir))], 0
    if not data_dir.is_dir():
        return [ValidationIssue("error", f"{label} path is not a directory: {data_dir}", str(data_dir))], 0

    for path in sorted(data_dir.iterdir()):
        if not path.is_file():
            continue
        match = regex.fullmatch(path.name)
        if not match:
            continue
        groups = match.groupdict()
        point = tuple(groups[name] for name in parameter_names)
        matched.append((path, point))
        try:
            _load_numeric_table(path, min_columns=min_columns, max_columns=max_columns)
        except Exception as exc:
            issues.append(ValidationIssue("error", f"could not read numeric {label} table: {exc}", str(path)))

    points = {point for _, point in matched}
    missing = sorted(expected_points.difference(points), key=lambda point: tuple(float(value) for value in point))
    extra = sorted(points.difference(expected_points), key=lambda point: tuple(float(value) for value in point))
    if missing:
        preview = ", ".join(str(point) for point in missing[:5])
        suffix = "" if len(missing) <= 5 else f", ... ({len(missing)} total)"
        issues.append(ValidationIssue("error", f"{label} files missing strength-grid points: {preview}{suffix}", str(data_dir)))
    if extra:
        preview = ", ".join(str(point) for point in extra[:5])
        suffix = "" if len(extra) <= 5 else f", ... ({len(extra)} total)"
        issues.append(ValidationIssue("warning", f"{label} has files outside strength grid: {preview}{suffix}", str(data_dir)))

    return issues, len(matched)


def validate_strength_grid(
    data_dir: str | Path,
    filename_regex: str | Pattern[str] = GENERIC_2D_STRENGTH_REGEX,
    *,
    parameter_names: tuple[str, ...] | None = None,
    min_files: int = 1,
    min_columns: int = 2,
    max_columns: int | None = 2,
    require_rectangular_grid: bool = True,
    allow_negative_strength: bool = False,
) -> ValidationReport:
    """Validate strength-function files before setting up an emulator run.

    The regex must match filenames in ``data_dir`` and contain named groups for
    each model parameter. By default, this validates the release-supported
    generic 2D format ``strength_<p1>_<p2>.out`` with two numeric columns:
    column 0 is omega and column 1 is B strength. Numeric contents are checked
    with ``numpy.loadtxt``.
    """

    data_path = Path(data_dir)
    regex = _compile_regex(filename_regex)
    issues: list[ValidationIssue] = []
    matched: list[tuple[Path, Mapping[str, str]]] = []

    if not data_path.exists():
        return ValidationReport(
            ok=False,
            files_checked=0,
            issues=[ValidationIssue("error", f"data directory does not exist: {data_path}", str(data_path))],
        )
    if not data_path.is_dir():
        return ValidationReport(
            ok=False,
            files_checked=0,
            issues=[ValidationIssue("error", f"data path is not a directory: {data_path}", str(data_path))],
        )

    for path in sorted(data_path.iterdir()):
        if not path.is_file():
            continue
        match = regex.fullmatch(path.name)
        if match:
            matched.append((path, match.groupdict()))

    if parameter_names is None:
        if matched:
            parameter_names = tuple(matched[0][1].keys())
        else:
            parameter_names = ()

    if len(matched) < min_files:
        issues.append(
            ValidationIssue(
                "error",
                f"expected at least {min_files} matching strength files, found {len(matched)}",
                str(data_path),
            )
        )

    points: list[tuple[str, ...]] = []
    for path, groups in matched:
        missing_names = [name for name in parameter_names if name not in groups]
        if missing_names:
            issues.append(
                ValidationIssue("error", f"filename is missing regex groups: {', '.join(missing_names)}", str(path))
            )
            continue
        point = tuple(groups[name] for name in parameter_names)
        points.append(point)
        try:
            table = _load_numeric_table(path, min_columns=min_columns, max_columns=max_columns)
        except Exception as exc:
            issues.append(ValidationIssue("error", f"could not read numeric strength table: {exc}", str(path)))
            continue
        if not allow_negative_strength and np.any(table[:, 1] < 0.0):
            issues.append(ValidationIssue("error", "strength column contains negative values", str(path)))
        if table.shape[0] < 2:
            issues.append(ValidationIssue("warning", "strength table has fewer than two rows", str(path)))

    if require_rectangular_grid and len(parameter_names) > 1:
        missing = _rectangular_grid_missing(points)
        if missing:
            preview = ", ".join(str(point) for point in missing[:5])
            suffix = "" if len(missing) <= 5 else f", ... ({len(missing)} total)"
            issues.append(ValidationIssue("error", f"parameter grid is not rectangular; missing {preview}{suffix}"))

    errors = [issue for issue in issues if issue.level == "error"]
    return ValidationReport(
        ok=not errors,
        files_checked=len(matched),
        points=tuple(sorted(set(points), key=lambda point: tuple(float(value) for value in point))),
        parameter_names=tuple(parameter_names),
        issues=issues,
    )


def validate_paper_beta_data(data_dir: str | Path = "beta_decay_80Ni", nucnam: str = "Ni_80") -> ValidationReport:
    """Validate the paper beta-decay files used by the Ni-80 examples."""

    escaped = re.escape(nucnam)
    data_dir = Path(data_dir)
    lorm_dir = data_dir / "total_lorm" if (data_dir / "total_lorm").is_dir() else data_dir
    report = validate_strength_grid(
        lorm_dir,
        rf"lorm_{escaped}_(?P<beta>[0-9.]+)_(?P<alpha>[0-9.]+)\.out",
        parameter_names=("alpha", "beta"),
        min_files=1,
        min_columns=2,
        max_columns=2,
        require_rectangular_grid=True,
    )
    expected_points = set(report.points)
    issues = list(report.issues)
    files_checked = report.files_checked
    if expected_points and (data_dir / "total_lorm").is_dir():
        extra_issues, count = _validate_matching_table_grid(
            data_dir / "total_excm",
            rf"excm_{escaped}_(?P<beta>[0-9.]+)_(?P<alpha>[0-9.]+)\.out",
            parameter_names=("alpha", "beta"),
            expected_points=expected_points,
            min_columns=2,
            max_columns=2,
            label="excitation",
        )
        issues.extend(extra_issues)
        files_checked += count
        extra_issues, count = _validate_matching_table_grid(
            data_dir / "total_half_life",
            rf"half_life_{escaped}_(?P<beta>[0-9.]+)_(?P<alpha>[0-9.]+)\.txt",
            parameter_names=("alpha", "beta"),
            expected_points=expected_points,
            min_columns=1,
            max_columns=1,
            label="half-life",
        )
        issues.extend(extra_issues)
        files_checked += count
    errors = [issue for issue in issues if issue.level == "error"]
    return ValidationReport(
        ok=not errors,
        files_checked=files_checked,
        points=report.points,
        parameter_names=report.parameter_names,
        issues=issues,
    )


def validate_paper_dipole_data(
    strength_dir: str | Path = "dipole_polarizability_160Yb/total_strength",
    alphaD_dir: str | Path | None = None,
) -> ValidationReport:
    """Validate the paper dipole files used by the dipole examples."""

    report = validate_strength_grid(
        strength_dir,
        r"strength_(?P<p2>[0-9.]+)_(?P<p1>[0-9.]+)\.out",
        parameter_names=("p1", "p2"),
        min_files=1,
        min_columns=2,
        max_columns=2,
        require_rectangular_grid=True,
    )
    if alphaD_dir is None:
        strength_path = Path(strength_dir)
        alphaD_dir = strength_path.parent / "total_alphaD" if strength_path.name == "total_strength" else None
    if alphaD_dir is None or not report.points:
        return report

    issues = list(report.issues)
    extra_issues, count = _validate_matching_table_grid(
        Path(alphaD_dir),
        r"alphaD_(?P<p2>[0-9.]+)_(?P<p1>[0-9.]+)\.out",
        parameter_names=("p1", "p2"),
        expected_points=set(report.points),
        min_columns=3,
        max_columns=3,
        label="alphaD",
    )
    issues.extend(extra_issues)
    errors = [issue for issue in issues if issue.level == "error"]
    return ValidationReport(
        ok=not errors,
        files_checked=report.files_checked + count,
        points=report.points,
        parameter_names=report.parameter_names,
        issues=issues,
    )
