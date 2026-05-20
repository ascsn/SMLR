from __future__ import annotations

import runpy
import sys
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def _argv(argv):
    old = sys.argv[:]
    sys.argv = list(argv)
    try:
        yield
    finally:
        sys.argv = old


@contextmanager
def _prepend_sys_path(path: Path):
    path_str = str(path)
    inserted = False
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(path_str)
            except ValueError:
                pass


def _run_compat_module(module_name: str, argv: list[str], fallback_path: str) -> None:
    with _argv([module_name, *argv]):
        try:
            runpy.run_module(module_name, run_name="__main__")
            return
        except ImportError:
            path = Path.cwd() / fallback_path
            if not path.exists():
                raise
            sys.argv[0] = str(path)
            with _prepend_sys_path(path.parent):
                runpy.run_path(str(path), run_name="__main__")


def run_dipole_paper_em1(argv: list[str]) -> None:
    _run_compat_module(
        "Dipole_polarizability.src.main_gpt2",
        argv,
        "Dipole_polarizability/src/main_gpt2.py",
    )


def run_beta_paper_em1(argv: list[str]) -> None:
    _run_compat_module(
        "Beta_decay_package.src.main_gpt2",
        argv,
        "Beta_decay_package/src/main_gpt2.py",
    )


def run_beta_paper_em2(argv: list[str]) -> None:
    _run_compat_module(
        "Beta_decay_package.src.main_only_HL_gpt",
        argv,
        "Beta_decay_package/src/main_only_HL_gpt.py",
    )
