"""The runtime package may use NumPy, its own modules and the standard library."""

import ast
import sys
from pathlib import Path


def test_core_uses_only_numpy_and_standard_library():
    package = Path(__file__).resolve().parents[1] / "src" / "ml_from_scratch"
    allowed = sys.stdlib_module_names | {"numpy", "ml_from_scratch", "__future__"}
    for source in package.rglob("*.py"):
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for name in names:
                assert name.split(".")[0] in allowed, f"Unexpected dependency in {source}: {name}"
