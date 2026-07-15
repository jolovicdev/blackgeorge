import subprocess
import sys
import tomllib
from pathlib import Path


def test_chromadb_is_an_optional_vector_dependency() -> None:
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    dependencies = project["project"]["dependencies"]
    vector_dependencies = project["project"]["optional-dependencies"]["vector"]

    assert not any(dependency.startswith("chromadb") for dependency in dependencies)
    assert any(dependency.startswith("chromadb") for dependency in vector_dependencies)


def test_memory_imports_without_chromadb() -> None:
    code = """
import builtins
import sys

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "chromadb" or name.startswith("chromadb."):
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

import blackgeorge.memory
from blackgeorge.memory import InMemoryMemoryStore, SQLiteMemoryStore

assert InMemoryMemoryStore is not None
assert SQLiteMemoryStore is not None
assert "chromadb" not in sys.modules

try:
    from blackgeorge.memory import VectorMemoryStore
except ImportError as exc:
    assert "blackgeorge[vector]" in str(exc)
else:
    raise AssertionError("VectorMemoryStore imported without ChromaDB")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
