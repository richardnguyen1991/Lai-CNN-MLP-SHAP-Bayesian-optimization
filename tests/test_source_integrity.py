"""Every source file parses, and every module in src/ imports.

This exists because a syntax error reached Kaggle. `session.py` is the entry
point for every run -- the notebook does nothing but fetch the code and call it
-- and no test imported it, so a broken f-string passed a green suite of 217
tests and only surfaced as a papermill traceback after a kernel had been pushed,
polled and failed.

Compiling catches the parse errors without running anything; importing catches
the ones that only appear when a module body executes. Both are cheap, and the
alternative is finding out a full session later.
"""

from __future__ import annotations

import importlib
import py_compile
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SRC))

SOURCES = sorted(SRC.glob("*.py")) + sorted(SCRIPTS.glob("*.py"))
MODULES = sorted(path.stem for path in SRC.glob("*.py"))

assert SOURCES, "no source files found; the layout moved"
assert MODULES, "no modules found in src/"


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_every_source_file_parses(path, tmp_path):
    # Byte-code goes to a scratch directory: compiling in place would drop a
    # .pyc beside every source file in the working tree.
    py_compile.compile(str(path), doraise=True, cfile=str(tmp_path / "out.pyc"))


@pytest.mark.parametrize("name", MODULES)
def test_every_module_imports(name):
    # session.py in particular: the notebook is a five-cell bootstrap that
    # imports this and calls run_session, so an import error here is a failed
    # Kaggle session rather than a failed test.
    importlib.import_module(name)
