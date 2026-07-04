"""Build a compact HTML review surface for Megarexus adapter/perturb gates."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def esc(value: Any) -> str:
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def rel_link(target: str | None, base: Path) -> str:
    if not target:
        return ""
    path = Path(target)
    try:
        if path.is_absolute():
            return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()
    return path.as_posix()


def checks_text(row: dict[str, Any]) -> str:
    checks = row.get("checks") or {}
    if not checks:
        return ""
    parts = []
    for key, value in sorted(checks.items()):
        cls = "pass" if bool(value) else "fail"
        parts.append(f'<span class="{cls}">{esc(key)}={esc(value)}</span>')
    return "<br>".join(parts)


def goal_provider_text(row: dict[str, Any]) -> str:
    provider = row.get("goal_provider") or {}
    if not provider:
        return ""
    parts = [
        f"enabled={esc(provider.get('enabled'))}",
        f"updates={esc(provider.get('updates'))}",
        f"latest_target_count={esc(provider.get('latest_target_count'))}",
        f"pour_budget={esc(provider.get('pour_budget_enabled'))}",
    ]
    return "<br>".join(parts)


def source_histogram_text(row: dict[str, Any]) -> str:
    histogram = row.get("source_histogram") or {}
    if not histogram:
        return ""
    return "<br>".join(f"{esc(key)}={esc(value)}" for key, value in sorted(histogram.items()))


def metric_cards(report: dict[str, Any], report_path: Path) -> str:
    verdict = str(report.get("gate_verdict", "UNKNOWN"))
    verdict_cls = "pass" if verdict == "PASS" else "fail" if verdict == "FAIL" else "warn"
    cards = [
        ("Gate Verdict", f'<span class="{verdict_cls}">{esc(verdict)}</span>'),
        ("Model Passed", f"{esc(report.get('model_passed'))}/{esc(report.get('model_episodes'))}"),
        ("Random Levels >= 2", f"{esc(report.get('random_levels_ge_2'))}/{esc(report.get('random_episodes'))}"),
        ("Report SHA256", f"<code>{sha256_file(report_path)}</code>"),
    ]
    if report.get("target_levels") is not None:
        cards.insert(2, ("Target Levels", esc(report.get("target_levels"))))
    if report.get("condition_sha256") is not None:
        cards.append(("Condition SHA256", f"<code>{esc(report.get('condition_sha256'))}</code>"))
    threshold = report.get("model_pass_threshold")
    if threshold is not None:
        cards.insert(2, ("Model Threshold", esc(threshold)))
    return "\n".join(
        f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div></div>'
        for label, value in cards
    )


def episode_table(report: dict[str, Any], base_dir: Path) -> str:
    rows = (report.get("detail") or {}).get("model") or []
    body = []
    for row in rows:
        passed = bool(row.get("passed"))
        status_cls = "pass" if passed else "fail"
        frame_capture = row.get("frame_capture") or {}
        frame_meta = frame_capture.get("meta_json")
        frame_link = ""
        if frame_meta:
            frame_link = f'<a href="{esc(rel_link(frame_meta, base_dir))}">frame meta</a>'
        body.append(
            "<tr>"
            f"<td>{esc(row.get('episode_seed'))}</td>"
            f'<td class="{status_cls}">{esc(passed)}</td>'
            f"<td>{esc(row.get('levels_completed'))}</td>"
            f"<td>{esc(row.get('actions'))}</td>"
            f"<td>{esc(row.get('resets'))}</td>"
            f"<td>{esc(row.get('reached_plumbing'))}</td>"
            f"<td>{esc(row.get('replay_match'))}</td>"
            f"<td>{goal_provider_text(row)}</td>"
            f"<td>{source_histogram_text(row)}</td>"
            f"<td>{checks_text(row)}</td>"
            f"<td>{esc(row.get('status'))}<br>{frame_link}</td>"
            "</tr>"
        )
    if not body:
        body.append('<tr><td colspan="10">No model episode rows in report.</td></tr>')
    return "\n".join(body)


def random_table(report: dict[str, Any]) -> str:
    rows = (report.get("detail") or {}).get("random") or []
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{esc(row.get('episode_seed'))}</td>"
            f"<td>{esc(row.get('levels_completed'))}</td>"
            f"<td>{esc(row.get('actions'))}</td>"
            f"<td>{esc(row.get('resets'))}</td>"
            f"<td>{esc(row.get('status'))}</td>"
            "</tr>"
        )
    if not body:
        body.append('<tr><td colspan="5">No random baseline rows in report.</td></tr>')
    return "\n".join(body)


def render(report: dict[str, Any], report_path: Path, log_path: Path | None) -> str:
    title = report.get("condition") or report.get("schema") or "Megarexus gate review"
    boundary = report.get("boundary") or report.get("evidence") or "Local review artifact."
    raw_links = [
        f'<li><a href="{esc(report_path.name)}">{esc(report_path.name)}</a></li>',
    ]
    if log_path is not None:
        raw_links.append(f'<li><a href="{esc(log_path.name)}">{esc(log_path.name)}</a></li>')
    ckpt = report.get("checkpoint")
    if ckpt:
        raw_links.append(f"<li>Checkpoint: <code>{esc(ckpt)}</code></li>")
    source_hashes = report.get("source_hashes") or {}
    if report.get("script_sha256"):
        source_hashes = {**source_hashes, "adapter_gate.py": report.get("script_sha256")}
    if report.get("policy_sha256"):
        source_hashes = {**source_hashes, "megarexus_live_policy.py": report.get("policy_sha256")}
    source_hash_items = "\n".join(
        f"<li><code>{esc(k)}</code>: <code>{esc(v)}</code></li>"
        for k, v in sorted(source_hashes.items())
    )
    if not source_hash_items:
        source_hash_items = "<li>No source hash block in report.</li>"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #182026;
      --muted: #59646d;
      --line: #cfd6dc;
      --panel: #f7f9fa;
      --good: #11613a;
      --warn: #8a4f00;
      --bad: #9a241d;
    }}
    body {{
      margin: 0;
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: white;
    }}
    main {{
      max-width: 1240px;
      margin: 0 auto;
      padding: 28px 24px 48px;
    }}
    h1 {{ font-size: 24px; margin: 0 0 6px; }}
    h2 {{ font-size: 16px; margin: 24px 0 8px; }}
    p {{ margin: 6px 0; }}
    a {{ color: #075a8c; }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      overflow-wrap: anywhere;
    }}
    .boundary {{ color: var(--muted); max-width: 920px; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 10px;
      margin: 18px 0;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
      background: var(--panel);
    }}
    .label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .03em;
    }}
    .value {{ font-weight: 700; margin-top: 4px; overflow-wrap: anywhere; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 8px;
      table-layout: fixed;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 7px 8px;
      text-align: left;
      vertical-align: top;
      overflow-wrap: anywhere;
    }}
    th {{ color: var(--muted); font-weight: 600; background: #eef2f4; }}
    .pass {{ color: var(--good); font-weight: 700; }}
    .fail {{ color: var(--bad); font-weight: 700; }}
    .warn {{ color: var(--warn); font-weight: 700; }}
  </style>
</head>
<body>
<main>
  <h1>{esc(title)}</h1>
  <p class="boundary">{esc(boundary)}</p>
  <div class="cards">
    {metric_cards(report, report_path)}
  </div>

  <h2>Raw Artifacts</h2>
  <ul>
    {''.join(raw_links)}
  </ul>

  <h2>Model Episodes</h2>
  <table>
    <thead>
      <tr>
        <th>Seed</th><th>Passed</th><th>Levels</th><th>Actions</th>
        <th>Resets</th><th>Reached Plumbing</th><th>Replay Match</th>
        <th>Goal Provider</th><th>Source Histogram</th><th>Checks</th><th>Status / Frames</th>
      </tr>
    </thead>
    <tbody>
      {episode_table(report, report_path.parent)}
    </tbody>
  </table>

  <h2>Random Baseline</h2>
  <table>
    <thead>
      <tr><th>Seed</th><th>Levels</th><th>Actions</th><th>Resets</th><th>Status</th></tr>
    </thead>
    <tbody>
      {random_table(report)}
    </tbody>
  </table>

  <h2>Provenance</h2>
  <ul>
    <li>Schema: <code>{esc(report.get('schema'))}</code></li>
    <li>Evidence: <code>{esc(report.get('evidence'))}</code></li>
    <li>Game: <code>{esc(report.get('game_id'))}</code></li>
    <li>Checkpoint SHA256: <code>{esc(report.get('checkpoint_sha256'))}</code></li>
    <li>Seeds: <code>{esc(report.get('paired_seeds'))}</code></li>
    <li>Target levels: <code>{esc(report.get('target_levels'))}</code></li>
    <li>Condition SHA256: <code>{esc(report.get('condition_sha256'))}</code></li>
  </ul>
  <h2>Source Hashes</h2>
  <ul>
    {source_hash_items}
  </ul>
</main>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()

    report_path = args.report.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    out_path = args.out.resolve() if args.out is not None else report_path.with_name("index.html")
    log_path = args.log.resolve() if args.log is not None else None
    out_path.write_text(render(report, report_path, log_path), encoding="utf-8")
    print(f"[build_gate_review] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
