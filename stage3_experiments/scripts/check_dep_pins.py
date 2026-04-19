"""Check that environment.yml and pyproject.toml dependency pins are consistent.

The two files serve different purposes:
  - pyproject.toml declares *minimum* runtime requirements (what pip needs to
    install the package in *any* environment).
  - environment.yml pins a *specific tested range* for the conda dev/CI env
    (stricter upper bounds so we don't accidentally test against a version we
    haven't validated).

The dangerous drift pattern is: someone bumps the lower bound in pyproject.toml
(e.g. "we now require numpy>=1.27 for the new structured-array API") but forgets
to update environment.yml, which still caps at <1.27. The conda env would resolve
to 1.26.x, the pip install in production would require 1.27+, and the mismatch
goes unnoticed until a user hits an AttributeError.

This script catches that case: for every package that appears in both files,
it checks that the environment.yml upper cap is strictly greater than the
pyproject.toml lower bound.

Usage (from repo root):
    python stage3_experiments/scripts/check_dep_pins.py

Exit codes:
    0 — all pins consistent (or no overlap to check)
    1 — at least one conflict detected
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]  # pip install tomli
    except ModuleNotFoundError:
        print("ERROR: Python 3.11+ required, or install tomli: pip install tomli", file=sys.stderr)
        sys.exit(1)

# ---------------------------------------------------------------------------
# conda package name → canonical PyPI name (lower-underscore form).
# Only packages where the names differ need an entry.
# ---------------------------------------------------------------------------
_CONDA_TO_PYPI: dict[str, str] = {
    "pytorch": "torch",
}


def _canon(name: str) -> str:
    """Normalise to PEP 503 canonical form (lower, hyphens→underscores)."""
    return re.sub(r"[-_.]+", "_", name).lower()


def _parse_specifiers(spec_str: str) -> list[tuple[str, tuple[int, ...]]]:
    """Extract (operator, version_tuple) pairs from a version specifier string."""
    pairs = re.findall(r"(>=|<=|!=|~=|==|<|>)\s*([0-9][0-9a-zA-Z.\-]*)", spec_str)
    result = []
    for op, ver in pairs:
        # Only keep numeric parts so "2.2.0" and "2.2" compare equal.
        parts = tuple(int(x) for x in re.findall(r"\d+", ver))
        result.append((op, parts))
    return result


def _load_pyproject_lower_bounds(pyproject_path: Path) -> dict[str, tuple[int, ...]]:
    """Return {canonical_name: lower_bound_tuple} for all [project.dependencies]."""
    with open(pyproject_path, "rb") as fh:
        data = tomllib.load(fh)  # type: ignore[possibly-undefined]

    lower_bounds: dict[str, tuple[int, ...]] = {}
    for dep in data.get("project", {}).get("dependencies", []):
        # Strip extras and env markers before parsing.
        name_spec = re.split(r"[\[;]", dep)[0]
        m = re.match(r"([A-Za-z0-9_.\-]+)(.*)", name_spec.strip())
        if not m:
            continue
        cname = _canon(m.group(1))
        for op, ver in _parse_specifiers(m.group(2)):
            if op == ">=":
                lower_bounds[cname] = ver
    return lower_bounds


def _load_env_upper_bounds(env_path: Path) -> dict[str, tuple[int, ...]]:
    """Return {canonical_name: exclusive_upper_bound_tuple} from environment.yml."""
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        print("ERROR: pyyaml is required — run `pip install pyyaml`.", file=sys.stderr)
        sys.exit(1)

    with open(env_path) as fh:
        env: dict[str, Any] = yaml.safe_load(fh)

    upper_bounds: dict[str, tuple[int, ...]] = {}

    def _process(dep: str) -> None:
        # Strip conda channel prefix, e.g. "pytorch::pytorch>=2.2,<2.6".
        dep = dep.split("::")[-1].strip()
        m = re.match(r"([A-Za-z0-9_.\-]+)(.*)", dep)
        if not m:
            return
        raw_name = m.group(1)
        cname = _CONDA_TO_PYPI.get(_canon(raw_name), _canon(raw_name))
        for op, ver in _parse_specifiers(m.group(2)):
            if op == "<":
                upper_bounds[cname] = ver

    for dep in env.get("dependencies", []):
        if isinstance(dep, dict):
            for pip_dep in dep.get("pip", []):
                _process(pip_dep)
        elif isinstance(dep, str):
            _process(dep)

    return upper_bounds


def main() -> int:
    root = Path(__file__).resolve().parent.parent.parent
    pyproject_path = root / "stage3_experiments" / "pyproject.toml"
    env_path = root / "environment.yml"

    for p in (pyproject_path, env_path):
        if not p.exists():
            print(f"ERROR: expected file not found: {p}", file=sys.stderr)
            return 1

    lower = _load_pyproject_lower_bounds(pyproject_path)
    upper = _load_env_upper_bounds(env_path)

    errors: list[str] = []
    for pkg in sorted(lower):
        if pkg not in upper:
            continue  # no upper cap in environment.yml — always fine
        lo, hi = lower[pkg], upper[pkg]
        if hi <= lo:
            errors.append(
                f"  {pkg}: pyproject.toml requires >={'.'.join(map(str, lo))} "
                f"but environment.yml caps at <{'.'.join(map(str, hi))} — "
                "no satisfying version exists in the conda env."
            )

    if errors:
        print("FAIL: dependency pin conflicts detected:\n" + "\n".join(errors))
        print(
            "\nFix: either lower the pyproject.toml minimum or raise the "
            "environment.yml upper cap.",
        )
        return 1

    checked = sorted(set(lower) & set(upper))
    if checked:
        print(f"OK — {len(checked)} overlapping pins are consistent: {', '.join(checked)}")
    else:
        print("OK — no overlapping packages to check (pyproject.toml and environment.yml "
              "pin disjoint sets).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
