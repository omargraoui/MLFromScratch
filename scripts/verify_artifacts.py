"""Execute documented examples and validate repository artifacts.

Run from any working directory after installing the package and dev extras.
Notebooks use a temporary kernel specification pointing to this interpreter.
Use --write-notebooks only when intentionally refreshing checked-in outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import nbformat
import yaml
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def verify_readme() -> None:
    """Execute each README Python block independently, in its displayed order."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    examples = re.findall(r"```python\s*\n(.*?)```", readme, re.DOTALL)
    if not examples:
        raise ValueError("README has no executable Python examples.")
    for index, code in enumerate(examples, start=1):
        exec(compile(code, f"README example {index}", "exec"), {"__name__": "__main__"})
    print(f"README: {len(examples)} independent Python examples passed.")


def verify_links() -> None:
    """Resolve relative Markdown file links, excluding fenced-code examples."""
    paths = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    paths.extend(sorted((ROOT / "benchmarks" / "results").glob("*.md")))
    count = 0
    for path in paths:
        contents = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.DOTALL)
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", contents):
            target = target.strip().split(' "', maxsplit=1)[0].strip("<>")
            parsed = urlparse(target)
            if parsed.scheme or not parsed.path:
                continue
            destination = path.parent / unquote(parsed.path)
            if not destination.exists():
                raise FileNotFoundError(f"Broken link in {path.relative_to(ROOT)}: {target}")
            count += 1
    print(f"Markdown: {count} local file links resolved.")


def verify_workflow() -> None:
    """Validate YAML syntax and the expected CI trigger/job/step structure."""
    # BaseLoader preserves the YAML 1.2 'on' key rather than coercing it to True.
    workflow = yaml.load(
        (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    if not {"push", "pull_request"} <= set(workflow["on"]):
        raise ValueError("CI must run on push and pull_request.")
    for name, job in workflow["jobs"].items():
        if not job.get("runs-on") or not job.get("steps"):
            raise ValueError(f"CI job {name} needs a runner and steps.")
        for step in job["steps"]:
            if not ("uses" in step or "run" in step):
                raise ValueError(f"CI job {name} has a step without uses/run.")
    print("CI: YAML syntax and job structure passed (hosted execution is separate).")


def verify_notebooks(*, write: bool) -> None:
    """Run each notebook from a clean kernel, optionally retaining real outputs."""
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    if len(notebooks) != 5:
        raise ValueError("Expected five educational notebooks.")
    with tempfile.TemporaryDirectory(prefix="ml-scratch-kernel-") as directory:
        temporary = Path(directory)
        kernel_name = "ml-scratch-verification"
        kernel_directory = temporary / kernel_name
        kernel_directory.mkdir()
        specification = {
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "ML from scratch verification",
            "language": "python",
            "env": {
                "IPYTHONDIR": str(temporary / "ipython"),
                "MPLBACKEND": "module://matplotlib_inline.backend_inline",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
        }
        (kernel_directory / "kernel.json").write_text(json.dumps(specification), encoding="utf-8")
        for path in notebooks:
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            manager = KernelManager(
                kernel_name=kernel_name,
                kernel_spec_manager=KernelSpecManager(kernel_dirs=[directory]),
                connection_file=str(temporary / f"{path.stem}.json"),
            )
            client = NotebookClient(
                notebook,
                km=manager,
                timeout=180,
                allow_errors=False,
                resources={"metadata": {"path": str(ROOT)}},
                record_timing=False,
            )
            try:
                client.execute()
            finally:
                # Supplying a manager makes its lifecycle our responsibility.
                if manager.has_kernel:
                    manager.shutdown_kernel(now=True)
                manager.cleanup_resources()
            if write:
                nbformat.write(notebook, path)
            print(f"Notebook passed: {path.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebooks", action="store_true", help="Execute all five notebooks.")
    parser.add_argument(
        "--write-notebooks", action="store_true", help="Save fresh notebook outputs."
    )
    args = parser.parse_args()
    if args.write_notebooks and not args.notebooks:
        parser.error("--write-notebooks requires --notebooks")
    os.chdir(ROOT)
    verify_readme()
    verify_links()
    verify_workflow()
    if args.notebooks:
        verify_notebooks(write=args.write_notebooks)


if __name__ == "__main__":
    main()
