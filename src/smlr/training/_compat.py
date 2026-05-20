from __future__ import annotations

import runpy
import sys
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def patched_argv(argv):
    old = sys.argv[:]
    sys.argv = list(argv)
    try:
        yield
    finally:
        sys.argv = old


@contextmanager
def prepended_sys_path(path: Path):
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


def run_legacy_entrypoint(module_name: str, argv: list[str], fallback_path: str) -> None:
    """Run a legacy paper trainer while package-native modules are filled in."""

    with patched_argv([module_name, *argv]):
        try:
            runpy.run_module(module_name, run_name="__main__")
            return
        except ImportError:
            path = Path.cwd() / fallback_path
            if not path.exists():
                raise
            sys.argv[0] = str(path)
            with prepended_sys_path(path.parent):
                runpy.run_path(str(path), run_name="__main__")
