"""Tests for pathogems.pathways.

All tests are hermetic — no network access required.  GMT content is either
written to a tmp_path fixture or the download step is bypassed by pre-seeding
the cache directory.  The one test that exercises _download_gmt patches
urllib so the function never touches the network.
"""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path
from unittest import mock

import pytest
import torch

from pathogems.pathways import (
    _download_gmt,
    _parse_gmt,
    build_connectivity,
    load_gene_sets,
)


# --------------------------------------------------------------------------- #
# Tiny GMT fixture
# --------------------------------------------------------------------------- #
_SAMPLE_GMT = textwrap.dedent(
    """\
    HALLMARK_APOPTOSIS\thttp://www.gsea-msigdb.org\tTP53\tBCL2\tBAX
    HALLMARK_CELL_CYCLE\thttp://www.gsea-msigdb.org\tCDK2\tTP53\tRB1
    HALLMARK_ANGIOGENESIS\thttp://www.gsea-msigdb.org\tVEGFA\tFGFR1
    """
)


@pytest.fixture
def gmt_file(tmp_path: Path) -> Path:
    p = tmp_path / "hallmark.gmt"
    p.write_text(_SAMPLE_GMT, encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# _parse_gmt
# --------------------------------------------------------------------------- #
class TestParseGmt:
    def test_returns_dict_with_correct_keys(self, gmt_file: Path) -> None:
        result = _parse_gmt(gmt_file)
        assert set(result.keys()) == {
            "HALLMARK_APOPTOSIS",
            "HALLMARK_CELL_CYCLE",
            "HALLMARK_ANGIOGENESIS",
        }

    def test_genes_parsed_correctly(self, gmt_file: Path) -> None:
        result = _parse_gmt(gmt_file)
        assert result["HALLMARK_APOPTOSIS"] == ["TP53", "BCL2", "BAX"]
        assert result["HALLMARK_CELL_CYCLE"] == ["CDK2", "TP53", "RB1"]

    def test_skips_short_lines(self, tmp_path: Path) -> None:
        """Lines with fewer than 3 tab-separated fields are silently skipped."""
        p = tmp_path / "partial.gmt"
        p.write_text("ONLY_ONE_FIELD\n", encoding="utf-8")
        result = _parse_gmt(p)
        assert result == {}

    def test_strips_whitespace_from_gene_names(self, tmp_path: Path) -> None:
        p = tmp_path / "whitespace.gmt"
        p.write_text("PW\tdesc\tGENE1 \t GENE2\n", encoding="utf-8")
        result = _parse_gmt(p)
        assert result["PW"] == ["GENE1", "GENE2"]

    def test_empty_file_returns_empty_dict(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.gmt"
        p.write_text("", encoding="utf-8")
        assert _parse_gmt(p) == {}


# --------------------------------------------------------------------------- #
# build_connectivity
# --------------------------------------------------------------------------- #
class TestBuildConnectivity:
    def _small_gene_sets(self) -> dict[str, list[str]]:
        return {
            "PATHWAY_A": ["TP53", "BCL2", "BRCA1"],
            "PATHWAY_B": ["MYC", "TP53"],
        }

    def test_mask_shape(self) -> None:
        selected = ["TP53", "BCL2", "MYC", "KRAS"]
        mask, names, _ = build_connectivity(selected, self._small_gene_sets())
        # 2 named pathways + 1 UNASSIGNED (KRAS is in neither)
        assert mask.shape == (3, 4)

    def test_mask_values_binary(self) -> None:
        selected = ["TP53", "BCL2", "MYC", "KRAS"]
        mask, _, _ = build_connectivity(selected, self._small_gene_sets())
        assert set(mask.unique().tolist()).issubset({0.0, 1.0})

    def test_known_memberships(self) -> None:
        selected = ["TP53", "BCL2", "MYC", "KRAS"]
        gene_sets = self._small_gene_sets()
        mask, names, _ = build_connectivity(selected, gene_sets)
        gene_idx = {g: i for i, g in enumerate(selected)}
        pathway_idx = {n: j for j, n in enumerate(names)}
        # TP53 is in both PATHWAY_A and PATHWAY_B
        assert mask[pathway_idx["PATHWAY_A"], gene_idx["TP53"]] == 1.0
        assert mask[pathway_idx["PATHWAY_B"], gene_idx["TP53"]] == 1.0
        # KRAS is in neither → should be in UNASSIGNED
        assert mask[pathway_idx["UNASSIGNED"], gene_idx["KRAS"]] == 1.0
        # BCL2 is only in PATHWAY_A
        assert mask[pathway_idx["PATHWAY_A"], gene_idx["BCL2"]] == 1.0
        assert mask[pathway_idx["PATHWAY_B"], gene_idx["BCL2"]] == 0.0

    def test_no_unassigned_when_all_covered(self) -> None:
        selected = ["TP53", "BCL2"]
        gene_sets = {"PW": ["TP53", "BCL2", "EXTRA"]}
        mask, names, assigned = build_connectivity(selected, gene_sets)
        assert "UNASSIGNED" not in names
        assert mask.shape == (1, 2)
        assert sorted(assigned) == ["BCL2", "TP53"]

    def test_assigned_list_correct(self) -> None:
        selected = ["TP53", "BCL2", "MYC", "KRAS"]
        _, _, assigned = build_connectivity(selected, self._small_gene_sets())
        # TP53, BCL2 in PATHWAY_A; MYC in PATHWAY_B; KRAS in neither
        assert set(assigned) == {"TP53", "BCL2", "MYC"}

    def test_all_unassigned_creates_single_row(self) -> None:
        selected = ["GENE_X", "GENE_Y"]
        gene_sets = {"PW": ["BRCA1", "MYC"]}
        mask, names, assigned = build_connectivity(selected, gene_sets)
        assert names[-1] == "UNASSIGNED"
        assert assigned == []

    def test_returns_float32_tensor(self) -> None:
        selected = ["TP53"]
        gene_sets = {"PW": ["TP53"]}
        mask, _, _ = build_connectivity(selected, gene_sets)
        assert isinstance(mask, torch.Tensor)
        assert mask.dtype == torch.float32

    def test_deterministic_ordering(self) -> None:
        """Pathway rows must be in sorted order for reproducibility."""
        selected = ["TP53", "MYC"]
        gene_sets = {"ZZZ": ["MYC"], "AAA": ["TP53"]}
        mask, names, _ = build_connectivity(selected, gene_sets)
        # Should be alphabetically sorted: AAA before ZZZ
        assert names[0] == "AAA"
        assert names[1] == "ZZZ"


# --------------------------------------------------------------------------- #
# load_gene_sets
# --------------------------------------------------------------------------- #
class TestLoadGeneSets:
    def test_returns_dict_from_cached_file(self, tmp_path: Path) -> None:
        """load_gene_sets must read from cache without hitting the network."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        # Pre-seed the cache so the download branch is never reached.
        (cache_dir / "hallmark.gmt").write_text(_SAMPLE_GMT, encoding="utf-8")

        result = load_gene_sets(db="hallmark", cache_dir=cache_dir)
        assert isinstance(result, dict)
        assert "HALLMARK_APOPTOSIS" in result

    def test_unknown_db_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unknown gene-set collection"):
            load_gene_sets(db="not_a_real_db", cache_dir=tmp_path)

    def test_creates_cache_dir_if_missing(self, tmp_path: Path) -> None:
        """cache_dir is created automatically when it does not exist."""
        cache_dir = tmp_path / "new_cache"
        # Pre-seed inside the not-yet-existing directory — but we create it
        # first to write the file, then remove it and let load_gene_sets recreate it.
        cache_dir.mkdir()
        (cache_dir / "hallmark.gmt").write_text(_SAMPLE_GMT, encoding="utf-8")
        # Move file out, remove dir, put it back after to test mkdir behaviour.
        gmt_content = (cache_dir / "hallmark.gmt").read_text()
        shutil.rmtree(cache_dir)

        # Patch download so it writes the file instead of hitting the network.
        def _fake_download(db: str, dest: Path) -> None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(gmt_content, encoding="utf-8")

        with mock.patch("pathogems.pathways._download_gmt", side_effect=_fake_download):
            result = load_gene_sets(db="hallmark", cache_dir=cache_dir)
        assert "HALLMARK_APOPTOSIS" in result


# --------------------------------------------------------------------------- #
# _download_gmt — error path (no network needed)
# --------------------------------------------------------------------------- #
class TestDownloadGmt:
    def test_raises_runtime_error_when_all_urls_fail(self, tmp_path: Path) -> None:
        """_download_gmt must raise RuntimeError if every URL attempt fails."""
        dest = tmp_path / "hallmark.gmt"
        with mock.patch(
            "pathogems.pathways.urllib.request.urlopen",
            side_effect=OSError("connection refused"),
        ):
            with pytest.raises(
                RuntimeError, match="All download URLs for 'hallmark' gene sets failed"
            ):
                _download_gmt("hallmark", dest)

    def test_error_message_contains_dest_path(self, tmp_path: Path) -> None:
        dest = tmp_path / "hallmark.gmt"
        with mock.patch(
            "pathogems.pathways.urllib.request.urlopen",
            side_effect=OSError("network down"),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                _download_gmt("hallmark", dest)
        assert str(dest) in str(exc_info.value)

    def test_does_not_create_file_on_failure(self, tmp_path: Path) -> None:
        dest = tmp_path / "hallmark.gmt"
        with mock.patch(
            "pathogems.pathways.urllib.request.urlopen",
            side_effect=OSError("no route to host"),
        ):
            with pytest.raises(RuntimeError):
                _download_gmt("hallmark", dest)
        assert not dest.exists()
