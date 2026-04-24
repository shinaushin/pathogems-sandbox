r"""Generate a self-contained HTML experiment comparison report.

Reads every *.json file in the logs directory, diffs each experiment's
config against the baseline (brca_omics_baseline), and writes a
single HTML file that includes:

  - Summary table: C-index, std, status, wall time, config changes
  - C-index bar chart across all experiments (with literature benchmark band)
  - Per-fold C-index breakdown
  - Config diff table (what changed vs baseline, highlighted)
  - Training / validation loss curves per experiment

Usage (from repo root):
    python stage3_experiments/scripts/experiment_report.py
    python stage3_experiments/scripts/experiment_report.py \\
        --logs-dir stage3_experiments/logs \\
        --out stage3_experiments/reports/experiment_report.html

The output file is self-contained — no server required, open it
directly in any browser.
"""

from __future__ import annotations

import argparse
import html
import json
from datetime import UTC, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Config keys excluded from the diff (infrastructure / metadata, not science)
# ---------------------------------------------------------------------------
_EXCLUDED_KEYS = {
    "name",
    "cohort",
    "seed",
    "study_data_dir",
    "notes",
    "config_version",
    "enable_mlflow",
    "mlflow_tracking_uri",
    "mlflow_experiment_name",
}

BASELINE_NAME = "brca_omics_baseline"
BENCH_LOW = 0.62
BENCH_HIGH = 0.68


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_logs(logs_dir: Path) -> list[dict]:
    """Load all run-log JSONs, sorted chronologically by started_at."""
    runs = []
    for path in sorted(logs_dir.glob("*.json")):
        if path.name == ".gitkeep":
            continue
        try:
            data = json.loads(path.read_text())
            data["_path"] = str(path)
            runs.append(data)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[report] Warning: skipping {path.name}: {exc}")
    runs.sort(key=lambda r: r.get("started_at", ""))
    return runs


def _config_diff(baseline: dict, other: dict) -> list[tuple[str, object, object]]:
    """Return (key, baseline_val, other_val) for every differing config field."""
    all_keys = set(baseline) | set(other)
    diffs = []
    for k in sorted(all_keys):
        if k in _EXCLUDED_KEYS:
            continue
        bv = baseline.get(k, "<absent>")
        ov = other.get(k, "<absent>")
        if bv != ov:
            diffs.append((k, bv, ov))
    return diffs


def _fmt(v: object) -> str:
    """Format a config value for display."""
    if isinstance(v, list):
        return "[" + ", ".join(str(x) for x in v) + "]"
    if v is None:
        return "null"
    return str(v)


def _cindex_color(c: float | None) -> str:
    """Return a CSS class based on C-index relative to benchmark."""
    if c is None:
        return "ci-na"
    if c >= BENCH_HIGH:
        return "ci-great"
    if c >= BENCH_LOW:
        return "ci-good"
    if c >= 0.50:
        return "ci-low"
    return "ci-bad"


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------
def _html_summary_row(run: dict, baseline_cfg: dict | None, rank: int) -> str:
    """One <tr> for the summary table."""
    cfg = run.get("config", {})
    metrics = run.get("metrics", {})
    name = run.get("run_name", run["_path"])
    status = run.get("status", "?")
    wall = run.get("wall_clock_sec")
    wall_str = f"{wall:.1f}s" if wall else "—"
    started = run.get("started_at", "")[:16].replace("T", " ")

    ci_mean = metrics.get("c_index_mean")
    ci_std = metrics.get("c_index_std")
    ci_str = f"{ci_mean:.4f} ± {ci_std:.4f}" if ci_mean is not None else "—"
    ci_cls = _cindex_color(ci_mean)

    best_fold = max(metrics.get("c_index_folds", [float("nan")]))
    worst_fold = min(metrics.get("c_index_folds", [float("nan")]))

    # Config diff summary
    if baseline_cfg and name != BASELINE_NAME:
        diffs = _config_diff(baseline_cfg, cfg)
        if diffs:
            diff_cells = "".join(
                f'<span class="diff-pill">{html.escape(k)}: '
                f"{html.escape(_fmt(bv))} → {html.escape(_fmt(ov))}</span>"
                for k, bv, ov in diffs
            )
        else:
            diff_cells = '<span class="diff-same">identical to baseline</span>'
    elif name == BASELINE_NAME:
        diff_cells = '<span class="diff-baseline">baseline</span>'
    else:
        diff_cells = "—"

    status_cls = "status-ok" if status == "success" else "status-err"
    baseline_marker = " ★" if name == BASELINE_NAME else ""

    return f"""
<tr id="row-{rank}">
  <td class="col-name"><a href="#exp-{rank}">{html.escape(name)}{baseline_marker}</a></td>
  <td class="{ci_cls} col-ci">{ci_str}</td>
  <td class="col-num">{best_fold:.4f}</td>
  <td class="col-num">{worst_fold:.4f}</td>
  <td class="{status_cls} col-status">{status}</td>
  <td class="col-num">{wall_str}</td>
  <td class="col-started">{started}</td>
  <td class="col-diffs">{diff_cells}</td>
</tr>"""


def _html_experiment_section(run: dict, baseline_cfg: dict | None, rank: int) -> str:
    """Detailed per-experiment section: diff table + fold bars + loss curves."""
    cfg = run.get("config", {})
    metrics = run.get("metrics", {})
    name = run.get("run_name", run["_path"])
    folds = metrics.get("c_index_folds", [])
    loss_curves = metrics.get("loss_curves", {})

    # --- Config diff table ---
    if baseline_cfg and name != BASELINE_NAME:
        diffs = _config_diff(baseline_cfg, cfg)
        if diffs:
            diff_rows = "".join(
                f'<tr><td class="diff-key">{html.escape(k)}</td>'
                f'<td class="diff-old">{html.escape(_fmt(bv))}</td>'
                f'<td class="diff-arrow">→</td>'
                f'<td class="diff-new">{html.escape(_fmt(ov))}</td></tr>'
                for k, bv, ov in diffs
            )
            diff_html = f'<table class="diff-table">{diff_rows}</table>'
        else:
            diff_html = '<p class="muted">Config identical to baseline.</p>'
    elif name == BASELINE_NAME:
        diff_html = (
            '<p class="muted">This is the baseline'
            " — all other experiments are diffed against it.</p>"
        )
    else:
        diff_html = ""

    # --- Fold C-index bars ---
    fold_bars = ""
    for i, ci in enumerate(folds):
        pct = min(ci / 0.80 * 100, 100)
        bench_low_pct = BENCH_LOW / 0.80 * 100
        bench_high_pct = BENCH_HIGH / 0.80 * 100
        cls = _cindex_color(ci)
        fold_bars += f"""
<div class="fold-row">
  <span class="fold-label">Fold {i}</span>
  <div class="fold-bar-wrap">
    <div class="bench-band" style="left:{bench_low_pct:.1f}%;width:{bench_high_pct - bench_low_pct:.1f}%"
    ></div>
    <div class="fold-bar {cls}" style="width:{pct:.1f}%"></div>
  </div>
  <span class="fold-val">{ci:.4f}</span>
</div>"""

    # --- Loss curve data (serialised as JS) ---
    fold_colors = ["#3266ad", "#1d9e75", "#ba7517", "#a32d2d", "#534ab7"]
    datasets_js = ""
    for fold_id_str, curves in sorted(loss_curves.items()):
        fi = int(fold_id_str)
        col = fold_colors[fi % len(fold_colors)]
        train = curves.get("train", [])
        val = curves.get("val", [])
        tr_pts = (
            "[" + ",".join(f"{{x:{e+1},y:{v:.4f}}}" for e, v in enumerate(train)) + "]"
        )
        vl_pts = (
            "[" + ",".join(f"{{x:{e+1},y:{v:.4f}}}" for e, v in enumerate(val)) + "]"
        )
        _tr = (
            f"  {{label:'F{fi} train',data:{tr_pts},borderColor:'{col}',"
            f"borderWidth:1.5,borderDash:[],pointRadius:0,tension:0.3}},"
        )
        _vl = (
            f"  {{label:'F{fi} val',  data:{vl_pts},borderColor:'{col}',"
            f"borderWidth:2,borderDash:[5,3],pointRadius:0,tension:0.3}},"
        )
        datasets_js += f"\n{_tr}\n{_vl}"

    chart_html = ""
    if datasets_js:
        canvas_id = f"lc-{rank}"
        chart_html = f"""
<div style="position:relative;width:100%;height:220px;margin-top:12px;">
  <canvas id="{canvas_id}" role="img"
    aria-label="Training and validation loss curves for {html.escape(name)}">
    Loss curves for {html.escape(name)}.
  </canvas>
</div>
<script>
new Chart(document.getElementById('{canvas_id}'),{{
  type:'line',
  data:{{datasets:[{datasets_js}]}},
  options:{{
    responsive:true,maintainAspectRatio:false,parsing:false,animation:false,
    plugins:{{legend:{{display:false}},tooltip:{{mode:'index',intersect:false,
      callbacks:{{label:c=>c.dataset.label+': '+c.parsed.y.toFixed(3)}}}}}},
    scales:{{
      x:{{type:'linear',title:{{display:true,text:'Epoch',font:{{size:11}}}},
          ticks:{{font:{{size:10}},stepSize:10}}}},
      y:{{title:{{display:true,text:'Cox loss',font:{{size:11}}}},
          ticks:{{font:{{size:10}},callback:v=>v.toFixed(2)}}}}
    }}
  }}
}});
</script>"""

    return f"""
<section class="exp-section" id="exp-{rank}">
  <h2 class="exp-title">{html.escape(name)}</h2>
  <div class="exp-meta">
    Status: <strong>{run.get("status","?")}</strong> &nbsp;|&nbsp;
    Started: {run.get("started_at","")[:16].replace("T"," ")} &nbsp;|&nbsp;
    Wall time: {run.get("wall_clock_sec",0):.1f}s &nbsp;|&nbsp;
    Git: <code>{(run.get("git_sha") or "")[:10]}</code>
  </div>

  <h3>Config changes vs baseline</h3>
  {diff_html}

  <h3>C-index by fold <span class="bench-legend">gold band = benchmark 0.62-0.68</span></h3>
  {fold_bars}

  <h3>Loss curves
    <span class="curve-legend">
      <span class="leg-solid">— train</span>
      <span class="leg-dash">- - val</span>
    </span>
  </h3>
  {chart_html}
</section>"""


def _build_comparison_chart_js(runs: list[dict]) -> str:
    """JS for the top-level C-index bar chart comparing all experiments."""
    names = [json.dumps(r.get("run_name", r["_path"])) for r in runs]
    means = [r.get("metrics", {}).get("c_index_mean") or 0 for r in runs]
    stds = [r.get("metrics", {}).get("c_index_std") or 0 for r in runs]
    bar_colors = [
        "#1d9e75" if (m >= BENCH_LOW) else ("#3266ad" if m >= 0.50 else "#d85a30") for m in means
    ]

    names_js = "[" + ",".join(names) + "]"
    means_js = "[" + ",".join(f"{v:.6f}" for v in means) + "]"
    stds_js = "[" + ",".join(f"{v:.6f}" for v in stds) + "]"
    colors_js = "[" + ",".join(f'"{c}"' for c in bar_colors) + "]"

    n = len(runs)
    height = max(200, n * 50 + 80)

    return f"""
<div style="position:relative;width:100%;height:{height}px;">
  <canvas id="cmp-chart" role="img"
    aria-label="C-index comparison bar chart across {n} experiments.">
    Comparison chart of mean C-index across experiments.
  </canvas>
</div>
<script>
(function(){{
  const names   = {names_js};
  const means   = {means_js};
  const stds    = {stds_js};
  const colors  = {colors_js};
  const benchLo = {BENCH_LOW};
  const benchHi = {BENCH_HIGH};

  const errBars = means.map((m,i) => ({{
    x: m, y: names[i],
    xMin: Math.max(0, m - stds[i]),
    xMax: Math.min(1, m + stds[i]),
  }}));

  new Chart(document.getElementById('cmp-chart'), {{
    type: 'bar',
    data: {{
      labels: names,
      datasets: [
        {{
          label: 'Mean C-index',
          data: means,
          backgroundColor: colors,
          borderWidth: 0,
          barThickness: 28,
        }}
      ]
    }},
    options: {{
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {{
        legend: {{ display: false }},
        annotation: {{
          annotations: {{
            benchBand: {{
              type: 'box',
              xMin: benchLo, xMax: benchHi,
              backgroundColor: 'rgba(186,117,23,0.10)',
              borderColor: 'rgba(186,117,23,0.50)',
              borderWidth: 1,
              label: {{
                display: true,
                content: 'Benchmark 0.62-0.68',
                position: 'start',
                font: {{ size: 10 }},
                color: '#ba7517',
              }}
            }}
          }}
        }},
        tooltip: {{
          callbacks: {{
            label: ctx => {{
              const i = ctx.dataIndex;
              return `C-index: ${{means[i].toFixed(4)}} ± ${{stds[i].toFixed(4)}}`;
            }}
          }}
        }}
      }},
      scales: {{
        x: {{
          min: 0.40, max: 0.80,
          title: {{ display: true, text: 'Harrell C-index', font: {{ size: 12 }} }},
          ticks: {{ font: {{ size: 11 }}, callback: v => v.toFixed(2) }}
        }},
        y: {{
          ticks: {{ font: {{ size: 11 }} }}
        }}
      }}
    }}
  }});
}})();
</script>"""


def generate_report(logs_dir: Path, out_path: Path) -> None:
    runs = _load_logs(logs_dir)
    if not runs:
        print(f"[report] No run logs found in {logs_dir}. Nothing to report.")
        return

    baseline_run = next((r for r in runs if r.get("run_name") == BASELINE_NAME), runs[0])
    baseline_cfg = baseline_run.get("config", {})

    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    n = len(runs)

    summary_rows = "".join(_html_summary_row(r, baseline_cfg, i) for i, r in enumerate(runs))
    exp_sections = "".join(_html_experiment_section(r, baseline_cfg, i) for i, r in enumerate(runs))
    comparison_chart = _build_comparison_chart_js(runs)

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PathoGems — Experiment Report</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/chartjs-plugin-annotation/3.0.1/chartjs-plugin-annotation.min.js"></script>
<style>
*, *::before, *::after {{ box-sizing: border-box; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  font-size: 14px;
  line-height: 1.6;
  color: #1a1a2e;
  background: #f8f8f5;
  margin: 0;
  padding: 0;
}}
a {{ color: #185fa5; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
code {{ font-family: "SFMono-Regular", Consolas, monospace; font-size: 12px;
        background: #eee; padding: 1px 5px; border-radius: 3px; }}

/* Layout */
.page-header {{
  background: #1a1a2e;
  color: #fff;
  padding: 2rem 2.5rem 1.5rem;
}}
.page-header h1 {{ margin: 0 0 4px; font-size: 22px; font-weight: 500; }}
.page-header p  {{ margin: 0; font-size: 12px; color: #aaa; }}
.container {{ max-width: 1100px; margin: 0 auto; padding: 2rem 2rem; }}
.section-head {{
  font-size: 11px; font-weight: 500; letter-spacing: .06em; text-transform: uppercase;
  color: #888; margin: 2rem 0 0.75rem; border-bottom: 1px solid #e0e0d8; padding-bottom: 6px;
}}

/* Summary table */
.summary-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
.summary-table th {{
  text-align: left; font-weight: 500; font-size: 11px; color: #666;
  border-bottom: 1.5px solid #ccc; padding: 6px 10px;
}}
.summary-table td {{ padding: 7px 10px; border-bottom: 1px solid #ececec; vertical-align: top; }}
.summary-table tr:hover td {{ background: #f2f2ee; }}
.col-name  {{ min-width: 200px; font-weight: 500; }}
.col-ci    {{ white-space: nowrap; }}
.col-num   {{ text-align: right; white-space: nowrap; }}
.col-status {{ text-align: center; }}
.col-started {{ white-space: nowrap; font-size: 12px; color: #888; }}
.col-diffs {{ font-size: 12px; }}

/* C-index colour coding */
.ci-great {{ color: #085041; font-weight: 500; }}
.ci-good  {{ color: #3b6d11; }}
.ci-low   {{ color: #854f0b; }}
.ci-bad   {{ color: #a32d2d; font-weight: 500; }}
.ci-na    {{ color: #888; }}

/* Status */
.status-ok  {{ color: #085041; }}
.status-err {{ color: #a32d2d; font-weight: 500; }}

/* Diff pills in summary */
.diff-pill {{
  display: inline-block; background: #eef3fb; color: #185fa5;
  border: 1px solid #b5d4f4; border-radius: 12px;
  padding: 1px 8px; margin: 2px 3px 2px 0; font-size: 11px;
  white-space: nowrap;
}}
.diff-same    {{ color: #888; font-style: italic; }}
.diff-baseline {{ color: #634f0b; background: #faeeda; border-radius: 12px;
                  padding: 1px 8px; font-size: 11px; font-weight: 500; }}

/* Experiment sections */
.exp-section {{
  background: #fff;
  border: 1px solid #e0e0d8;
  border-radius: 10px;
  padding: 1.5rem 1.75rem;
  margin-bottom: 1.5rem;
}}
.exp-title {{
  font-size: 17px; font-weight: 500; margin: 0 0 6px; color: #1a1a2e;
}}
.exp-meta {{ font-size: 12px; color: #888; margin-bottom: 1.25rem; }}
.exp-section h3 {{
  font-size: 12px; font-weight: 500; text-transform: uppercase; letter-spacing: .05em;
  color: #666; margin: 1.25rem 0 0.6rem; border-bottom: 1px solid #eee; padding-bottom: 5px;
}}
.muted {{ color: #999; font-style: italic; font-size: 13px; margin: 4px 0; }}

/* Config diff table */
.diff-table {{ border-collapse: collapse; font-size: 13px; margin-top: 4px; }}
.diff-table td {{ padding: 5px 12px; vertical-align: top; }}
.diff-key  {{ color: #555; font-weight: 500; white-space: nowrap; }}
.diff-old  {{ color: #a32d2d; background: #fef2f2; border-radius: 4px; padding: 2px 8px; }}
.diff-new  {{ color: #085041; background: #edf7f3; border-radius: 4px; padding: 2px 8px; }}
.diff-arrow {{ color: #aaa; padding: 5px 4px; }}

/* Fold bars */
.fold-row {{
  display: flex; align-items: center; gap: 10px;
  margin-bottom: 7px; font-size: 13px;
}}
.fold-label {{ width: 50px; color: #666; flex-shrink: 0; }}
.fold-bar-wrap {{
  flex: 1; background: #f0f0ea; border-radius: 4px; height: 20px;
  position: relative; overflow: hidden;
}}
.bench-band {{
  position: absolute; top: 0; height: 20px;
  background: rgba(186,117,23,0.12);
  border-left: 1.5px solid rgba(186,117,23,0.45);
  border-right: 1.5px solid rgba(186,117,23,0.45);
}}
.fold-bar {{
  height: 20px; border-radius: 4px; position: absolute; left: 0; top: 0;
  transition: width .3s;
}}
.fold-bar.ci-great {{ background: #1d9e75; }}
.fold-bar.ci-good  {{ background: #639922; }}
.fold-bar.ci-low   {{ background: #ef9f27; }}
.fold-bar.ci-bad   {{ background: #d85a30; }}
.fold-val {{ width: 52px; text-align: right; color: #333; flex-shrink: 0; font-size: 12px; }}

/* Legend */
.bench-legend {{ font-size: 11px; font-weight: 400; color: #ba7517; margin-left: 8px; }}
.curve-legend {{ font-size: 11px; font-weight: 400; color: #888; margin-left: 8px; }}
.leg-solid {{ border-bottom: 2px solid #888; padding-bottom: 1px; }}
.leg-dash  {{ border-bottom: 2px dashed #888; padding-bottom: 1px; margin-left: 8px; }}

/* Comparison chart wrapper */
.chart-card {{
  background: #fff; border: 1px solid #e0e0d8;
  border-radius: 10px; padding: 1.25rem 1.5rem; margin-bottom: 1.5rem;
}}
</style>
</head>
<body>
<div class="page-header">
  <h1>PathoGems — Experiment Report</h1>
  <p>TCGA-BRCA survival prediction &nbsp;·&nbsp;
     {n} experiment{"s" if n != 1 else ""} &nbsp;·&nbsp; Generated {now}</p>
</div>

<div class="container">

<p class="section-head">C-index comparison</p>
<div class="chart-card">
{comparison_chart}
</div>

<p class="section-head">Summary ({n} experiments)</p>
<div style="overflow-x:auto;">
<table class="summary-table">
  <thead>
    <tr>
      <th>Experiment</th>
      <th>Mean C-index ± std</th>
      <th style="text-align:right">Best fold</th>
      <th style="text-align:right">Worst fold</th>
      <th style="text-align:center">Status</th>
      <th style="text-align:right">Wall time</th>
      <th>Started</th>
      <th>Config changes vs baseline</th>
    </tr>
  </thead>
  <tbody>
{summary_rows}
  </tbody>
</table>
</div>

<p class="section-head">Experiment details</p>
{exp_sections}

</div>
</body>
</html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc, encoding="utf-8")
    print(f"[report] Wrote {n} experiment(s) → {out_path}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("stage3_experiments/logs"),
        help="Directory containing run-log JSON files. Default: stage3_experiments/logs",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("stage3_experiments/reports/experiment_report.html"),
        help="Output HTML path. Default: stage3_experiments/reports/experiment_report.html",
    )
    args = p.parse_args(argv)

    if not args.logs_dir.is_dir():
        print(f"[report] ERROR: logs directory not found: {args.logs_dir}")
        return 1

    generate_report(args.logs_dir, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
