"""Build a compact HTML review page for Megarexus fixed-val checkpoint gates."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
from typing import Any


POSITIVE_METRICS = [
    "bc/action_type_acc",
    "catch/acc",
    "control/acc",
    "fwd/changed_cell_acc",
    "fwd/flow_changed_cell_acc",
    "goose/change_acc",
    "inv/action_acc",
    "level/acc",
    "level/true_pos",
    "settle/iou",
]

LOWER_IS_BETTER = [
    "inv/click_mean_dist_px",
    "level/false_pos",
]

KEY_COLUMNS = [
    ("ckpt", "Checkpoint"),
    ("verdict", "No-Loss"),
    ("bc/action_type_acc", "BC"),
    ("catch/acc", "Catch"),
    ("level/true_pos", "Level TP"),
    ("level/positives", "Level Pos"),
    ("level/false_pos", "Level FP"),
    ("settle/iou", "Settle IoU"),
    ("fwd/changed_cell_acc", "Changed"),
    ("fwd/flow_changed_cell_acc", "Flow Changed"),
    ("inv/click_mean_dist_px", "Click Dist"),
    ("sha256", "SHA256"),
    ("reasons", "Reasons"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def metric(metrics: dict[str, float], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    return float(value)


def assess(
    metrics: dict[str, float],
    baseline: dict[str, float],
    eps: float = 1e-9,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for key in POSITIVE_METRICS:
        cur = metric(metrics, key)
        ref = metric(baseline, key)
        if cur is None or ref is None:
            reasons.append(f"{key}=missing")
        elif cur + eps < ref:
            reasons.append(f"{key} {cur:.6g} < {ref:.6g}")
    for key in LOWER_IS_BETTER:
        cur = metric(metrics, key)
        ref = metric(baseline, key)
        if cur is None or ref is None:
            reasons.append(f"{key}=missing")
        elif cur > ref + eps:
            reasons.append(f"{key} {cur:.6g} > {ref:.6g}")
    return not reasons, reasons


def ckpt_sha(path_text: str, base_dir: Path) -> str:
    path = Path(path_text)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(base_dir / path)
    for candidate in candidates:
        if candidate.exists():
            return sha256_file(candidate)
    return "missing"


def repo_root_for(report_path: Path) -> Path:
    for parent in report_path.parents:
        if (parent / "synthesis_lab" / "orchestration").is_dir():
            return parent
    return report_path.parent


def render(report: dict[str, Any], report_path: Path) -> str:
    rows = report.get("results") or []
    if not rows:
        raise ValueError("report has no results rows")

    baseline = rows[0]
    baseline_metrics = baseline.get("metrics") or {}
    body = []
    pass_count = 0
    for idx, row in enumerate(rows):
        metrics = row.get("metrics") or {}
        if idx == 0:
            passed = True
            reasons = ["baseline"]
        else:
            passed, reasons = assess(metrics, baseline_metrics)
            pass_count += int(passed)
        verdict = "BASELINE" if idx == 0 else "PASS" if passed else "FAIL"
        row_data = {
            "ckpt": Path(row.get("ckpt", "")).name,
            "verdict": verdict,
            "sha256": ckpt_sha(row.get("ckpt", ""), repo_root_for(report_path)),
            "reasons": "; ".join(reasons),
        }
        row_data.update(metrics)
        cls = "baseline" if idx == 0 else "pass" if passed else "fail"
        cells = []
        for key, _label in KEY_COLUMNS:
            value = row_data.get(key)
            if key == "verdict":
                cells.append(f'<td class="{cls}">{esc(value)}</td>')
            elif key == "sha256":
                cells.append(f"<td><code>{esc(value)}</code></td>")
            else:
                cells.append(f"<td>{esc(fmt(value))}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")

    header = "".join(f"<th>{esc(label)}</th>" for _key, label in KEY_COLUMNS)
    shards = "".join(f"<li><code>{esc(s)}</code></li>" for s in report.get("val_shards", []))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Megarexus Fixed-Val Gate</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172026;
      --muted: #5d6770;
      --line: #ccd4da;
      --panel: #f6f8f9;
      --good: #11613a;
      --bad: #9a241d;
      --base: #264f7b;
    }}
    body {{
      margin: 0;
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: #fff;
    }}
    main {{
      max-width: 1360px;
      margin: 0 auto;
      padding: 28px 24px 48px;
    }}
    h1 {{ font-size: 24px; margin: 0 0 6px; }}
    h2 {{ font-size: 16px; margin: 24px 0 8px; }}
    p {{ margin: 6px 0; color: var(--muted); max-width: 980px; }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      overflow-wrap: anywhere;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      margin-top: 10px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 7px 8px;
      text-align: left;
      vertical-align: top;
      overflow-wrap: anywhere;
    }}
    th {{ background: var(--panel); color: var(--muted); font-weight: 600; }}
    .pass {{ color: var(--good); font-weight: 700; }}
    .fail {{ color: var(--bad); font-weight: 700; }}
    .baseline {{ color: var(--base); font-weight: 700; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 10px;
      margin: 18px 0;
    }}
    .card {{ border: 1px solid var(--line); border-radius: 6px; padding: 12px; background: var(--panel); }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .03em; }}
    .value {{ font-weight: 700; margin-top: 4px; overflow-wrap: anywhere; }}
  </style>
</head>
<body>
<main>
  <h1>Megarexus Fixed-Val Gate</h1>
  <p>Compares candidate checkpoints against the first row baseline on the same v8 validation shards. This is checkpoint evidence only; adapter, perturbation, and live gates remain separate.</p>
  <div class="summary">
    <div class="card"><div class="label">Schema</div><div class="value"><code>{esc(report.get("schema"))}</code></div></div>
    <div class="card"><div class="label">Rows</div><div class="value">{len(rows)}</div></div>
    <div class="card"><div class="label">Candidate Passes</div><div class="value">{pass_count}/{max(len(rows) - 1, 0)}</div></div>
    <div class="card"><div class="label">Report SHA256</div><div class="value"><code>{sha256_file(report_path)}</code></div></div>
  </div>
  <h2>Checkpoint Rows</h2>
  <table>
    <thead><tr>{header}</tr></thead>
    <tbody>
      {''.join(body)}
    </tbody>
  </table>
  <h2>Validation Shards</h2>
  <ul>{shards}</ul>
  <h2>Raw Artifact</h2>
  <p><a href="{esc(report_path.name)}">{esc(report_path.name)}</a></p>
</main>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report_path = args.report.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    out_path = args.out.resolve() if args.out else report_path.with_name("index.html")
    out_path.write_text(render(report, report_path), encoding="utf-8")
    print(f"[build_gate_many_review] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
