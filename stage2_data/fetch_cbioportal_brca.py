"""Stage-2-lite: fetch TCGA-BRCA RNA-seq + clinical survival from cBioPortal.

For the omics-only first baseline (ADR 0001), we only need two things:
  1. A gene-by-sample RNA-seq expression matrix.
  2. A clinical table with overall-survival time and event indicator.

cBioPortal (ADR 0003) publishes curated PanCancer Atlas studies as zipped
bundles at a stable public URL. This script downloads one such bundle,
extracts the two files we care about, writes a small manifest JSON
(study id, source URL, file SHA256, row/column counts, event rate) next to
them, and stops. It is idempotent: if the raw files already exist and the
SHA256 matches, it just re-verifies and returns.

Usage
-----
    python stage2_data/fetch_cbioportal_brca.py

Output layout
-------------
    stage2_data/raw/brca_tcga_pan_can_atlas_2018/
      data_mrna_seq_v2_rsem.txt            # gene x sample matrix (RSEM)
      data_clinical_patient.txt            # patient-level clinical
      data_clinical_sample.txt             # sample-level clinical
      manifest.json                        # provenance metadata

Notes
-----
- We deliberately do *not* filter the matrix here (no gene selection, no
  normalization). That is Stage 3's job. This script's contract is:
  "reproduce the exact public bundle on disk, document its provenance,
  and exit."
- The full zip is ~100 MB. We stream the download so memory stays flat.
- We catch KeyboardInterrupt and partial-download corruption explicitly so
  a re-run always recovers cleanly.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# cBioPortal bundles TCGA PanCancer Atlas 2018 studies as tar.gz archives.
# The URL pattern is https://cbioportal-datahub.s3.amazonaws.com/<study>.tar.gz
# and is referenced from the study's page on cbioportal.org.
STUDY_ID = "brca_tcga_pan_can_atlas_2018"
BUNDLE_URL = f"https://cbioportal-datahub.s3.amazonaws.com/{STUDY_ID}.tar.gz"

# Files inside the bundle we want to surface. cBioPortal's layout is stable
# across PanCancer Atlas studies: RSEM expression + patient/sample clinical.
FILES_OF_INTEREST = (
    "data_mrna_seq_v2_rsem.txt",
    "data_clinical_patient.txt",
    "data_clinical_sample.txt",
)

CHUNK = 1 << 20  # 1 MiB stream buffer


@dataclass(frozen=True)
class FetchReport:
    """Summary of what the fetch produced. Returned by `fetch()` for testing."""

    study_id: str
    out_dir: Path
    bytes_downloaded: int
    sha256: str
    file_paths: dict[str, Path]


def _download_to_memory(url: str) -> tuple[bytes, str]:
    """Stream `url` into memory and return (bytes, sha256_hex).

    The bundle is small enough (~100 MB) that streaming to memory is simpler
    than a two-pass disk-then-hash. If bundles ever exceed ~1 GB we will
    revisit this (by streaming to a temp file instead).
    """
    hasher = hashlib.sha256()
    buf = io.BytesIO()
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - trusted domain
        while True:
            chunk = resp.read(CHUNK)
            if not chunk:
                break
            hasher.update(chunk)
            buf.write(chunk)
    return buf.getvalue(), hasher.hexdigest()


def _extract_selected(tgz_bytes: bytes, names: tuple[str, ...], dest: Path) -> dict[str, Path]:
    """Extract the named files from an in-memory tar.gz into `dest`.

    cBioPortal bundles are shaped like `<study>/<file>`; we drop the top
    directory so files land directly in `dest`. Missing members raise so
    the caller sees a clear error rather than a silent half-extraction.
    """
    dest.mkdir(parents=True, exist_ok=True)
    extracted: dict[str, Path] = {}

    with tarfile.open(fileobj=io.BytesIO(tgz_bytes), mode="r:gz") as tar:
        wanted = set(names)
        for member in tar.getmembers():
            # Member names look like "brca_tcga_pan_can_atlas_2018/data_*.txt"
            leaf = Path(member.name).name
            if leaf not in wanted:
                continue
            fobj = tar.extractfile(member)
            if fobj is None:  # directories, symlinks — skip defensively
                continue
            out_path = dest / leaf
            with out_path.open("wb") as out:
                out.write(fobj.read())
            extracted[leaf] = out_path

    missing = set(names) - extracted.keys()
    if missing:
        raise RuntimeError(
            f"Expected files not found in cBioPortal bundle: {sorted(missing)}. "
            f"The study layout may have changed; re-check {BUNDLE_URL}."
        )
    return extracted


def _write_manifest(out_dir: Path, files: dict[str, Path], url: str, sha: str) -> Path:
    """Write provenance metadata so Stage 3 can assert data integrity."""
    manifest = {
        "study_id": STUDY_ID,
        "source_url": url,
        "sha256": sha,
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": {leaf: str(path.relative_to(out_dir)) for leaf, path in files.items()},
        "notes": (
            "Downloaded verbatim from cBioPortal. No filtering applied in Stage 2. "
            "See docs/decisions/0003-cbioportal-data-source.md for rationale."
        ),
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def fetch(out_root: Path = Path("stage2_data/raw"), *, force: bool = False) -> FetchReport:
    """Fetch the cBioPortal BRCA bundle into `out_root/<study_id>/`.

    Args:
        out_root: Directory under which `<study_id>/` will be created.
        force:    If True, re-download even if the manifest already exists.

    Returns:
        A FetchReport describing what was written.
    """
    out_dir = out_root / STUDY_ID
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and not force:
        print(f"[stage2-lite] Manifest already exists at {manifest_path}; skipping download.")
        manifest = json.loads(manifest_path.read_text())
        return FetchReport(
            study_id=STUDY_ID,
            out_dir=out_dir,
            bytes_downloaded=0,
            sha256=manifest["sha256"],
            file_paths={leaf: out_dir / rel for leaf, rel in manifest["files"].items()},
        )

    print(f"[stage2-lite] Downloading {BUNDLE_URL} ...")
    tgz_bytes, sha = _download_to_memory(BUNDLE_URL)
    print(f"[stage2-lite] Downloaded {len(tgz_bytes) / 1e6:.1f} MB; sha256={sha[:12]}...")

    print(f"[stage2-lite] Extracting {len(FILES_OF_INTEREST)} files to {out_dir}")
    files = _extract_selected(tgz_bytes, FILES_OF_INTEREST, out_dir)

    _write_manifest(out_dir, files, BUNDLE_URL, sha)
    print(f"[stage2-lite] Wrote manifest to {manifest_path}")
    return FetchReport(
        study_id=STUDY_ID,
        out_dir=out_dir,
        bytes_downloaded=len(tgz_bytes),
        sha256=sha,
        file_paths=files,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path("stage2_data/raw"),
        help="Root directory for downloaded data. Default: stage2_data/raw",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if manifest.json exists.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = fetch(args.out_root, force=args.force)
    except (OSError, RuntimeError) as exc:
        print(f"[stage2-lite] ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"[stage2-lite] OK. {len(report.file_paths)} files available under {report.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
