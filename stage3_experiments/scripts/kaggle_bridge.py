"""Run a pathogems experiment on Kaggle's free GPU.

The bridge handles the full round-trip:
  1. Bundle the pathogems source into a tarball.
  2. Embed the experiment config into a generated Jupyter notebook.
  3. Push the kernel to Kaggle and wait for it to complete.
  4. Download outputs and route them to the canonical project directories
     (``stage3_experiments/logs/`` and ``stage3_experiments/checkpoints/``).

Authentication — Kaggle recently introduced a new token system:
    a) Generate a token at https://www.kaggle.com/settings (API → Generate New Token).
    b) Either export it::

        export KAGGLE_API_TOKEN=<token>
        export KAGGLE_USERNAME=<your_username>

       or save the token string to ``~/.kaggle/access_token`` and set
       ``KAGGLE_USERNAME`` via env var or the ``_DEFAULT_USERNAME`` constant
       below.

    The bridge reads the new token and bridges it to the ``KAGGLE_KEY``
    env var that the ``kaggle`` package (used for kernel push/poll/fetch)
    still expects internally.

Dependencies::

    pip install kaggle kagglehub nbformat

Usage::

    python stage3_experiments/scripts/kaggle_bridge.py \\
        --config stage3_experiments/configs/brca_omics_baseline.json \\
        --gpu

    python stage3_experiments/scripts/kaggle_bridge.py \\
        --config stage3_experiments/configs/brca_pathway_mlp.json \\
        --gpu --slug pathogems-pathway
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tarfile
import tempfile
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency checks — fail fast before doing any work
# ---------------------------------------------------------------------------

try:
    import nbformat
    from nbformat.v4 import new_code_cell, new_notebook
except ImportError:
    print("Missing: pip install nbformat")
    sys.exit(1)

try:
    import kagglehub  # noqa: F401  (token detection only)
except ImportError:
    print("Missing: pip install kagglehub")
    sys.exit(1)

try:
    from kaggle.api.kaggle_api_extended import KaggleApiExtended
except ImportError:
    print("Missing: pip install kaggle")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Set via KAGGLE_USERNAME env var, or override the constant here.
_DEFAULT_USERNAME = "your_kaggle_username"

#: Seconds between status polls while waiting for the kernel to finish.
POLL_INTERVAL_SEC = 30

#: Hard timeout — 90 min is generous for a 5-fold CV on TCGA-BRCA CPU/GPU.
MAX_WAIT_SEC = 5400

#: This file lives in stage3_experiments/scripts/, so:
_STAGE3_ROOT = Path(__file__).resolve().parent.parent  # stage3_experiments/
_PROJECT_ROOT = _STAGE3_ROOT.parent  # repo root

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    """Timestamped bridge log line, always flushed."""
    print(f"[bridge {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _kaggle_username() -> str:
    """Return the Kaggle username from env or the module constant."""
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    if not username or username == "your_kaggle_username":
        username = _DEFAULT_USERNAME
    if not username or username == "your_kaggle_username":
        print("\nKaggle username not configured.")
        print("Either:")
        print("  export KAGGLE_USERNAME=your_username")
        print(f"  or edit _DEFAULT_USERNAME in {__file__}")
        sys.exit(1)
    return username


def _read_token_file() -> str | None:
    """Read a new-style token from ``~/.kaggle/access_token``."""
    p = Path.home() / ".kaggle" / "access_token"
    return p.read_text().strip() if p.exists() else None


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def authenticate() -> KaggleApiExtended:
    """Bridge the new Kaggle token to the env vars the ``kaggle`` package needs.

    kagglehub reads from (in priority order):
      1. ``KAGGLE_API_TOKEN`` env var
      2. ``~/.kaggle/access_token`` file

    The older ``kaggle`` package (still required for kernel push/poll/fetch)
    reads ``KAGGLE_USERNAME`` + ``KAGGLE_KEY``, so we set those from the
    token before constructing ``KaggleApiExtended``.
    """
    token = os.environ.get("KAGGLE_API_TOKEN") or _read_token_file()
    if not token:
        print("\nNo Kaggle token found.")
        print("Generate one at https://www.kaggle.com/settings (API → Generate New Token)")
        print("Then either:")
        print("  export KAGGLE_API_TOKEN=your_token")
        print("  or save it to ~/.kaggle/access_token")
        sys.exit(1)

    username = _kaggle_username()
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = token

    api = KaggleApiExtended()
    api.authenticate()
    _log(f"Authenticated with Kaggle as '{username}'")
    return api


# ---------------------------------------------------------------------------
# Source bundling
# ---------------------------------------------------------------------------


def _bundle_source(dest_tar: Path) -> None:
    """Create a gzipped tarball of the stage3 package for upload to Kaggle.

    Includes ``stage3_experiments/src/`` (the pathogems package) and
    ``stage3_experiments/pyproject.toml`` so the kernel can do
    ``pip install -e .`` without needing network access to PyPI for the
    pathogems package itself.
    """
    _log(f"Bundling source → {dest_tar.name}")
    with tarfile.open(dest_tar, "w:gz") as tf:
        tf.add(_STAGE3_ROOT / "src", arcname="src")
        tf.add(_STAGE3_ROOT / "pyproject.toml", arcname="pyproject.toml")
    _log(f"  source tarball: {dest_tar.stat().st_size:,} bytes")


# ---------------------------------------------------------------------------
# Notebook cell templates
# ---------------------------------------------------------------------------

_CELL_SETUP = """\
# Auto-generated by kaggle_bridge.py — do not edit by hand.
import os
import sys
import torch

print(f"Python {sys.version}")
print(f"PyTorch {torch.__version__}  CUDA available: {torch.cuda.is_available()}")
print(f"Working dir: {os.getcwd()}")
"""

_CELL_INSTALL_DEPS = """\
# Install additional dependencies not pre-installed on Kaggle.
# torch / numpy / pandas / scipy are already available in the base image.
import subprocess
import sys

_extra = ["scikit-survival>=0.22", "lifelines>=0.28"]
for _pkg in _extra:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", _pkg],
        check=True,
    )
print("Extra deps installed:", _extra)
"""

_CELL_INSTALL_PATHOGEMS = """\
# Extract the bundled source tarball and install pathogems in editable mode.
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

_tar = Path("/kaggle/working/pathogems_src.tar.gz")
_dst = Path("/tmp/pathogems_src")

if _dst.exists():
    shutil.rmtree(_dst)
_dst.mkdir(parents=True)

with tarfile.open(_tar, "r:gz") as tf:
    tf.extractall(_dst)

result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-e", str(_dst)],
    capture_output=True,
    text=True,
)
if result.returncode != 0:
    print("STDERR:", result.stderr[-2000:])
    raise RuntimeError("pathogems install failed")

import importlib
importlib.import_module("pathogems")
print("pathogems installed and importable")
"""

# URLs mirror fetch_cbioportal_brca.py so the download logic stays in sync.
_STUDY_ID = "brca_tcga_pan_can_atlas_2018"
_BRCA_URLS = [
    f"https://cbioportal.org/study/downloadStudy?id={_STUDY_ID}",
    f"http://download.cbioportal.org/{_STUDY_ID}.tar.gz",
    f"https://cbioportal-datahub.s3.amazonaws.com/{_STUDY_ID}.tar.gz",
]

_CELL_FETCH_DATA = (
    "# Download TCGA-BRCA omics + clinical data from cBioPortal.\n"
    "import shutil\n"
    "import tarfile\n"
    "import urllib.request\n"
    "from pathlib import Path\n"
    "\n"
    "_DATA_DIR = Path('/kaggle/working/brca_data')\n"
    "_DATA_DIR.mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "_EXPECTED = [\n"
    "    'data_mrna_seq_v2_rsem.txt',\n"
    "    'data_clinical_patient.txt',\n"
    "    'data_clinical_sample.txt',\n"
    "]\n"
    "\n"
    "if all((_DATA_DIR / f).exists() for f in _EXPECTED):\n"
    "    print('BRCA data already present — skipping download.')\n"
    "else:\n"
    "    _URLS = " + repr(_BRCA_URLS) + "\n"
    "    _ok = False\n"
    "    for _url in _URLS:\n"
    "        try:\n"
    "            print(f'Trying {_url} ...')\n"
    "            _archive = _DATA_DIR / 'brca.tar.gz'\n"
    "            urllib.request.urlretrieve(_url, _archive)\n"
    "            with tarfile.open(_archive, 'r:gz') as _tf:\n"
    "                _tf.extractall(_DATA_DIR)\n"
    "            # cBioPortal archives nest files inside a sub-directory.\n"
    "            _subdirs = [d for d in _DATA_DIR.iterdir() if d.is_dir()]\n"
    "            for _sub in _subdirs:\n"
    "                for _f in _sub.iterdir():\n"
    "                    shutil.move(str(_f), _DATA_DIR / _f.name)\n"
    "                shutil.rmtree(_sub)\n"
    "            _archive.unlink(missing_ok=True)\n"
    "            _ok = True\n"
    "            print('Download complete.')\n"
    "            break\n"
    "        except Exception as _e:\n"
    "            print(f'  failed: {_e}')\n"
    "    if not _ok:\n"
    "        raise RuntimeError(\n"
    "            'All download URLs failed. '\n"
    "            'Check internet access or download manually.'\n"
    "        )\n"
    "\n"
    "_present = [f for f in _EXPECTED if (_DATA_DIR / f).exists()]\n"
    "print(f'Data files present ({len(_present)}/{len(_EXPECTED)}): {_present}')\n"
    "assert len(_present) == len(_EXPECTED), "
    "f'Missing: {set(_EXPECTED) - set(_present)}'\n"
)


def _make_config_cell(config: dict) -> str:
    """Return a notebook cell source that writes the patched config to /tmp."""
    # Override study_data_dir to the Kaggle-local download location.
    patched = {**config, "study_data_dir": "/kaggle/working/brca_data"}
    # Disable MLflow on Kaggle — no tracking server is available there.
    patched["enable_mlflow"] = False
    config_json = json.dumps(patched, indent=2)
    lines = [
        "import json",
        "from pathlib import Path",
        "",
        f"_CONFIG = {config_json!r}",
        "_CONFIG_PATH = Path('/tmp/run_config.json')",
        "_CONFIG_PATH.write_text(json.dumps(_CONFIG))",
        'print(f"Config written → {_CONFIG_PATH}")',
        "print(f\"  name:    {_CONFIG['name']}\")",
        "print(f\"  model:   {_CONFIG.get('model', 'N/A')}\")",
        "print(f\"  n_folds: {_CONFIG.get('n_folds', 'N/A')}\")",
    ]
    return "\n".join(lines)


_CELL_TRAIN = """\
# Run pathogems-train and stream its output.
import sys
import torch
from pathlib import Path

_LOGS_DIR = Path("/kaggle/working/stage3_experiments/logs")
_LOGS_DIR.mkdir(parents=True, exist_ok=True)

_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {_device}")

# Call the CLI directly via its Python entry point so stdout/stderr stay
# in the notebook output rather than a subprocess pipe.
from pathogems.cli import main as _train_main

_exit_code = _train_main([
    "--config", "/tmp/run_config.json",
    "--logs-dir", str(_LOGS_DIR),
    "--device", _device,
    "--no-report",
])
if _exit_code != 0 and _exit_code is not None:
    raise RuntimeError(f"pathogems-train exited with code {_exit_code}")

print("\\nTraining complete. Logs:")
for _f in sorted(_LOGS_DIR.glob("*.json")):
    print(f"  {_f.name}  ({_f.stat().st_size:,} bytes)")
"""

_CELL_COLLECT_OUTPUTS = """\
# Collect outputs into /kaggle/working/outputs/ so Kaggle bundles them
# for download.  route_outputs() in the bridge maps these back to the
# canonical project directories locally.
import shutil
from pathlib import Path

_OUT = Path("/kaggle/working/outputs")
_OUT.mkdir(exist_ok=True)

_logs = Path("/kaggle/working/stage3_experiments/logs")
for _f in _logs.glob("*.json"):
    shutil.copy2(_f, _OUT / _f.name)
    print(f"  log  → outputs/{_f.name}")

_ckpts = Path("/kaggle/working/stage3_experiments/checkpoints")
if _ckpts.exists():
    for _f in _ckpts.glob("*.pt"):
        shutil.copy2(_f, _OUT / _f.name)
        print(f"  ckpt → outputs/{_f.name}")

print(f"\\nOutputs ready in {_OUT}")
"""


# ---------------------------------------------------------------------------
# Notebook assembly
# ---------------------------------------------------------------------------


def build_notebook(config: dict) -> "nbformat.NotebookNode":
    """Assemble the Kaggle training notebook from the cell templates.

    Args:
        config: Experiment config dict (will have ``study_data_dir`` patched
                to the Kaggle-local data path before embedding).

    Returns:
        A ``nbformat.NotebookNode`` ready to be written to disk.
    """
    nb = new_notebook()
    nb.cells = [
        new_code_cell(_CELL_SETUP),
        new_code_cell(_CELL_INSTALL_DEPS),
        new_code_cell(_CELL_INSTALL_PATHOGEMS),
        new_code_cell(_CELL_FETCH_DATA),
        new_code_cell(_make_config_cell(config)),
        new_code_cell(_CELL_TRAIN),
        new_code_cell(_CELL_COLLECT_OUTPUTS),
    ]
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    return nb


# ---------------------------------------------------------------------------
# Kernel push / poll / fetch
# ---------------------------------------------------------------------------


def _write_kernel_metadata(
    folder: Path,
    kernel_slug: str,
    notebook_filename: str,
    enable_gpu: bool,
    username: str,
) -> None:
    metadata = {
        "id": f"{username}/{kernel_slug}",
        "title": kernel_slug.replace("-", " ").title(),
        "code_file": notebook_filename,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": str(enable_gpu).lower(),
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))


def push_kernel(api: "KaggleApiExtended", folder: Path) -> str:
    """Push the kernel folder to Kaggle and return the kernel reference string."""
    _log("Pushing kernel to Kaggle…")
    api.kernels_push(str(folder))
    meta = json.loads((folder / "kernel-metadata.json").read_text())
    ref = meta["id"]
    _log(f"Pushed → https://www.kaggle.com/code/{ref}")
    return ref


def wait_for_completion(api: "KaggleApiExtended", kernel_ref: str) -> str:
    """Poll until the kernel reaches a terminal state or the timeout fires.

    Returns the final status string: ``"complete"``, ``"error"``,
    ``"cancelAcknowledged"``, or ``"timeout"``.
    """
    _log(f"Polling every {POLL_INTERVAL_SEC}s (timeout {MAX_WAIT_SEC}s)…")
    elapsed = 0
    while elapsed < MAX_WAIT_SEC:
        obj = api.kernels_status(kernel_ref)
        status = (
            obj.get("status", "unknown")
            if isinstance(obj, dict)
            else getattr(obj, "status", "unknown")
        )
        _log(f"  status={status}  elapsed={elapsed}s")
        if status in ("complete", "error", "cancelAcknowledged"):
            return str(status)
        time.sleep(POLL_INTERVAL_SEC)
        elapsed += POLL_INTERVAL_SEC
    return "timeout"


def fetch_outputs(api: "KaggleApiExtended", kernel_ref: str, output_dir: Path) -> list[Path]:
    """Download all kernel output files to ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Fetching outputs → {output_dir}")
    api.kernels_output(kernel_ref, path=str(output_dir), force=True, quiet=False)
    files = [f for f in output_dir.rglob("*") if f.is_file()]
    for f in files:
        _log(f"  {f.name}  ({f.stat().st_size:,} bytes)")
    return files


def _print_fold_summary(log_path: Path) -> None:
    """Parse a run log JSON and print a per-fold results table locally.

    Derives epochs trained and best epoch from the val loss curve length
    and argmin respectively — the same values the training loop tracks —
    so no extra fields are needed beyond what the existing schema stores.

    Silently skips if the file is missing, malformed, or has no metrics
    (e.g. a ``status="failed"`` log).
    """
    try:
        data = json.loads(log_path.read_text())
    except Exception:
        return

    metrics = data.get("metrics", {})
    c_folds: list[float] = metrics.get("c_index_folds", [])
    loss_folds: list[float] = metrics.get("final_loss_folds", [])
    curves: dict = metrics.get("loss_curves", {})

    if not c_folds:
        return  # failed run or empty metrics — nothing to show

    n = len(c_folds)
    print()
    print(f"  {'fold':>4}  {'C-index':>8}  {'val_loss':>9}  {'epochs':>7}  {'best@':>6}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*6}")

    for i in range(n):
        c = c_folds[i]
        loss = loss_folds[i] if i < len(loss_folds) else float("nan")
        val_curve: list[float] = curves.get(str(i), {}).get("val", [])
        epochs = len(val_curve)
        best = int(min(range(epochs), key=lambda e: val_curve[e])) + 1 if epochs else 0
        print(f"  {i + 1:>4}  {c:>8.4f}  {loss:>9.4f}  " f"{epochs:>7d}  {best:>6d}")

    mean = metrics.get("c_index_mean", float("nan"))
    std = metrics.get("c_index_std", float("nan"))
    print(f"  {'-'*4}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*6}")
    print(f"  {'mean':>4}  {mean:>8.4f} ± {std:.4f}  (n_folds={n})")
    print()


def route_outputs(files: list[Path]) -> None:
    """Copy fetched files to the canonical stage3 output directories.

    * ``*.json`` → ``stage3_experiments/logs/``  (then prints fold summary)
    * ``*.pt``   → ``stage3_experiments/checkpoints/``

    Other file types are logged but not moved.
    """
    log_dir = _STAGE3_ROOT / "logs"
    ckpt_dir = _STAGE3_ROOT / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        if f.suffix == ".json":
            dest = log_dir / f.name
            shutil.copy2(f, dest)
            _log(f"  log  → {dest.relative_to(_PROJECT_ROOT)}")
            _print_fold_summary(dest)
        elif f.suffix == ".pt":
            dest = ckpt_dir / f.name
            shutil.copy2(f, dest)
            _log(f"  ckpt → {dest.relative_to(_PROJECT_ROOT)}")
        else:
            _log(f"  (skip) {f.name}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _make_kernel_slug(config: dict, config_path: Path) -> str:
    """Derive a Kaggle-safe kernel slug from the experiment name."""
    exp_name: str = config.get("name", config_path.stem)
    return exp_name.lower().replace("_", "-")[:50]


def dry_run(
    config_path: Path,
    enable_gpu: bool,
    kernel_slug: str | None = None,
) -> bool:
    """Validate every local step without pushing anything to Kaggle.

    Checks credentials, bundles the source, generates the training notebook,
    and writes the output to ``stage3_experiments/kaggle_outputs/dry_run/``
    so you can inspect the notebook before committing a real push.

    Nothing is sent to Kaggle — no quota is consumed.

    Args:
        config_path: Path to the experiment config JSON.
        enable_gpu:  GPU flag to embed in kernel metadata (no effect locally).
        kernel_slug: Kernel slug override.

    Returns:
        ``True`` if all local steps succeeded; ``False`` on any error.
    """
    config = json.loads(config_path.read_text())
    slug = kernel_slug or _make_kernel_slug(config, config_path)
    username = _kaggle_username()

    _log("=== DRY RUN — nothing will be pushed to Kaggle ===")
    _log(f"Experiment: {config.get('name', config_path.stem)}")
    _log(f"Config:     {config_path}")
    _log(f"Slug:       {slug}")
    _log(f"GPU:        {enable_gpu}")

    out_dir = _STAGE3_ROOT / "kaggle_outputs" / "dry_run"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[tuple[str, bool, str]] = []  # (step, ok, detail)

    # Step 1 — credentials
    try:
        authenticate()
        results.append(("credentials", True, f"authenticated as '{username}'"))
    except SystemExit:
        results.append(("credentials", False, "auth failed — check token/username"))
        _print_dry_run_summary(results)
        return False

    # Step 2 — source bundle
    tar_path = out_dir / "pathogems_src.tar.gz"
    try:
        _bundle_source(tar_path)
        size_kb = tar_path.stat().st_size // 1024
        results.append(("source bundle", True, f"{tar_path.name}  ({size_kb} KB)"))
    except Exception as exc:
        results.append(("source bundle", False, str(exc)))
        _print_dry_run_summary(results)
        return False

    # Step 3 — notebook generation
    nb_name = f"{slug}.ipynb"
    nb_path = out_dir / nb_name
    try:
        nb = build_notebook(config)
        nbformat.write(nb, nb_path)
        n_cells = len(nb.cells)
        results.append(("notebook", True, f"{nb_name}  ({n_cells} cells)"))
    except Exception as exc:
        results.append(("notebook", False, str(exc)))
        _print_dry_run_summary(results)
        return False

    # Step 4 — kernel metadata
    meta_path = out_dir / "kernel-metadata.json"
    try:
        _write_kernel_metadata(
            folder=out_dir,
            kernel_slug=slug,
            notebook_filename=nb_name,
            enable_gpu=enable_gpu,
            username=username,
        )
        meta = json.loads(meta_path.read_text())
        results.append(("kernel metadata", True, f"id={meta['id']}"))
    except Exception as exc:
        results.append(("kernel metadata", False, str(exc)))
        _print_dry_run_summary(results)
        return False

    # Step 5 — notebook cell sanity check (parse each cell as valid Python)
    import ast

    bad_cells: list[int] = []
    for idx, cell in enumerate(nb.cells):
        try:
            ast.parse(cell["source"])
        except SyntaxError as exc:
            bad_cells.append(idx + 1)
            _log(f"  cell {idx + 1} syntax error: {exc}")
    if bad_cells:
        results.append(("cell syntax", False, f"cells with errors: {bad_cells}"))
    else:
        results.append(("cell syntax", True, f"all {n_cells} cells parse cleanly"))

    _print_dry_run_summary(results)

    if all(ok for _, ok, _ in results):
        _log(f"Inspect the generated kernel at: {out_dir}")
        _log(f"  notebook:         {nb_name}")
        _log(f"  source tarball:   pathogems_src.tar.gz")
        _log(f"  kernel metadata:  kernel-metadata.json")
        _log("")
        _log("To run for real (CPU, no GPU quota):")
        _log(f"  python {Path(__file__).name} " f"--config {config_path}")
        _log("To run with GPU:")
        _log(f"  python {Path(__file__).name} " f"--config {config_path} --gpu")
        return True

    return False


def _print_dry_run_summary(results: list[tuple[str, bool, str]]) -> None:
    """Print a checklist of dry-run step outcomes."""
    print()
    print("  Dry-run checklist:")
    for step, ok, detail in results:
        mark = "✓" if ok else "✗"
        print(f"    [{mark}] {step:<20}  {detail}")
    print()


def run_bridge(
    config_path: Path,
    enable_gpu: bool,
    kernel_slug: str | None = None,
) -> bool:
    """Execute the full bridge round-trip for one experiment config.

    Args:
        config_path: Path to the experiment config JSON.
        enable_gpu:  Whether to request a Kaggle GPU accelerator.
        kernel_slug: Kaggle kernel slug override.  Defaults to the
                     experiment ``name`` field, truncated to 50 chars.

    Returns:
        ``True`` if the kernel completed successfully and outputs were
        routed; ``False`` on kernel error or timeout.
    """
    config = json.loads(config_path.read_text())
    slug = kernel_slug or _make_kernel_slug(config, config_path)

    _log(f"Experiment: {config.get('name', config_path.stem)}")
    _log(f"Config:     {config_path}")
    _log(f"Slug:       {slug}")
    _log(f"GPU:        {enable_gpu}")

    api = authenticate()
    username = _kaggle_username()

    tmp_dir = Path(tempfile.mkdtemp(prefix="pathogems_kaggle_"))
    try:
        # 1. Bundle the pathogems source so the kernel can install it.
        _bundle_source(tmp_dir / "pathogems_src.tar.gz")

        # 2. Build and write the training notebook.
        nb = build_notebook(config)
        nb_name = f"{slug}.ipynb"
        nb_path = tmp_dir / nb_name
        nbformat.write(nb, nb_path)
        _log(f"Notebook:   {nb_path.name}")

        # 3. Write the Kaggle kernel metadata alongside the notebook.
        _write_kernel_metadata(
            folder=tmp_dir,
            kernel_slug=slug,
            notebook_filename=nb_name,
            enable_gpu=enable_gpu,
            username=username,
        )

        # 4. Push the entire temp dir (notebook + source tarball + metadata).
        kernel_ref = push_kernel(api, tmp_dir)

        # 5. Wait for Kaggle to finish executing the notebook.
        final_status = wait_for_completion(api, kernel_ref)
        _log(f"Kernel finished: {final_status}")

        if final_status != "complete":
            _log(
                f"Kernel did not complete — inspect at: "
                f"https://www.kaggle.com/code/{kernel_ref}"
            )
            return False

        # 6. Download the outputs the notebook staged in /kaggle/working/outputs/.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = _STAGE3_ROOT / "kaggle_outputs" / f"{slug}_{timestamp}"
        files = fetch_outputs(api, kernel_ref, output_dir)

        # 7. Route JSON logs → logs/, .pt files → checkpoints/.
        route_outputs(files)

        _log(f"Done. Run log in {(_STAGE3_ROOT / 'logs').relative_to(_PROJECT_ROOT)}")
        return True

    finally:
        # Always clean up the temp dir, even on failure.
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for ``python kaggle_bridge.py``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    p = argparse.ArgumentParser(
        prog="kaggle_bridge",
        description="Run a pathogems experiment on Kaggle's free GPU.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--config",
        required=True,
        type=Path,
        help=(
            "Path to experiment config JSON "
            "(e.g. stage3_experiments/configs/brca_omics_baseline.json)."
        ),
    )
    p.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help="Enable Kaggle GPU accelerator (counts against 30 hr/week free quota).",
    )
    p.add_argument(
        "--slug",
        default=None,
        help="Kernel slug override. Default: derived from the experiment name field.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Validate credentials, bundle source, and generate the notebook locally "
            "without pushing anything to Kaggle. Writes output to "
            "stage3_experiments/kaggle_outputs/dry_run/ for inspection. "
            "No quota is consumed."
        ),
    )
    args = p.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    if args.dry_run:
        ok = dry_run(
            config_path=config_path,
            enable_gpu=args.gpu,
            kernel_slug=args.slug,
        )
    else:
        ok = run_bridge(
            config_path=config_path,
            enable_gpu=args.gpu,
            kernel_slug=args.slug,
        )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
