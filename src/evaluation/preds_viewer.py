"""Tiny local UI for benchmark + question-wise prediction inspection.

Run:
  python -m src.evaluation.preds_viewer --preds-dir data/preds --benchmarks-dir data/benchmarks
Then open http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


def _norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def _contains_ref(pred: str, ref: str) -> bool:
    nref = _norm(ref)
    return bool(nref) and nref in _norm(pred)


def _topic(question: str) -> str:
    q = (question or "").lower()
    if "magnetic force" in q or "magnets" in q:
        return "magnetism"
    if "punnett" in q or "offspring" in q or "ratio" in q:
        return "genetics_ratio"
    if "kinetic energy" in q or "temperature" in q:
        return "thermal_kinetic"
    if "weather" in q or "climate" in q:
        return "weather_climate"
    if "which of the following could" in q:
        return "experimental_design"
    return "other"


def _latex_health(text: str) -> str:
    t = text or ""
    dollars = 0
    escaped = False
    for ch in t:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == "$":
            dollars += 1
    if dollars % 2 == 1:
        return "suspicious"
    if t.count(r"\(") != t.count(r"\)") or t.count(r"\[") != t.count(r"\]"):
        return "suspicious"
    return "ok"


def _table_html(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _load_preds(preds_dir: Path) -> tuple[list[dict], dict[str, list[dict]]]:
    by_cfg: dict[str, list[dict]] = {}
    flat: list[dict] = []
    for p in sorted(preds_dir.glob("preds_*.json")):
        cfg = p.stem.split("_")[1] if "_" in p.stem else p.stem
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        rows = []
        for i, r in enumerate(data):
            row = {
                "config": cfg,
                "idx": i,
                "question": str(r.get("question", "")),
                "reference": str(r.get("reference", "")),
                "prediction": str(r.get("prediction", "")),
            }
            rows.append(row)
            flat.append(row)
        by_cfg[cfg] = rows
    return flat, by_cfg


def _load_benchmarks(benchmarks_dir: Path) -> list[dict]:
    rows: list[dict] = []
    if not benchmarks_dir.exists():
        return rows
    for p in sorted(benchmarks_dir.glob("benchmark_*.json")):
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            rows.append(data)
        elif isinstance(data, list):
            rows.extend(data)
    return rows


def _load_posthoc_summary(posthoc_dir: Path) -> list[dict]:
    csv_path = posthoc_dir / "posthoc_summary.csv"
    if not csv_path.exists():
        return []
    out: list[dict] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(row)
    return out


def _build_left_join_compare(by_cfg: dict[str, list[dict]], cfg_order: list[str]) -> list[dict]:
    """Index-based left join across configs to preserve all rows."""
    max_len = max((len(v) for v in by_cfg.values()), default=0)
    rows: list[dict] = []
    for i in range(max_len):
        question = ""
        reference = ""
        answers: dict[str, str] = {}
        for cfg in cfg_order:
            cfg_rows = by_cfg.get(cfg, [])
            if i < len(cfg_rows):
                r = cfg_rows[i]
                answers[cfg] = r["prediction"]
                if not question:
                    question = r["question"]
                if not reference:
                    reference = r["reference"]
            else:
                answers[cfg] = ""
        if not question:
            for cfg_rows in by_cfg.values():
                if i < len(cfg_rows):
                    question = cfg_rows[i]["question"]
                    reference = cfg_rows[i]["reference"]
                    break
        rows.append(
            {
                "idx": i,
                "question": question,
                "reference": reference,
                "topic": _topic(question),
                "answers": answers,
            }
        )
    return rows


def _base_css() -> str:
    return """
body { font-family: Inter, Arial, sans-serif; margin: 20px; background: #f5f7fb; color: #111827; }
a { color: #1d4ed8; text-decoration: none; }
a:hover { text-decoration: underline; }
h2, h3 { margin: 6px 0 10px 0; }
.muted { color: #6b7280; font-size: 13px; }
.card { background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 14px; margin-bottom: 14px; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #e5e7eb; padding: 8px; font-size: 13px; text-align: left; vertical-align: top; }
th { background: #f3f4f6; }
.controls { display:grid; gap: 8px; margin-bottom: 12px; }
.badge { padding: 2px 8px; border-radius: 999px; font-size: 12px; border: 1px solid transparent; }
.hit { background:#dcfce7; border-color:#86efac; }
.miss { background:#fee2e2; border-color:#fca5a5; }
.missing { background:#f3f4f6; border-color:#d1d5db; color:#6b7280; }
.topic { background:#e0e7ff; border-color:#c7d2fe; }
.latex-ok { background:#e0f2fe; border-color:#7dd3fc; }
.latex-suspicious { background:#fef3c7; border-color:#fcd34d; }
.pager { margin: 12px 0; display:flex; gap: 10px; align-items:center; }
.pred { font-size: 13px; line-height: 1.35; white-space: pre-wrap; max-height: 220px; overflow: auto; border:1px solid #eef2ff; border-radius: 6px; padding: 8px; background:#fafaff; }
pre { white-space: pre-wrap; max-height: 220px; overflow:auto; }
.plot-grid { display:grid; grid-template-columns: repeat(2, minmax(380px, 1fr)); gap: 12px; }
.plot-frame { width:100%; height:340px; object-fit: contain; background:#fff; border:1px solid #e5e7eb; border-radius:8px; }
.diagram-grid { display:grid; grid-template-columns: repeat(2, minmax(360px, 1fr)); gap: 12px; }
.diagram-frame { width:100%; height:300px; object-fit: contain; background:#fff; border:1px solid #e5e7eb; border-radius:8px; }
@media (max-width: 960px) { .plot-grid { grid-template-columns: 1fr; } .plot-frame { height:300px; } }
"""


def make_handler(
    rows: list[dict],
    by_cfg: dict[str, list[dict]],
    benchmark_rows: list[dict],
    posthoc_rows: list[dict],
    plots_dir: Path,
    posthoc_dir: Path,
    diagrams_dir: Path,
):
    cfg_order = ["B1", "B2", "M1", "M2", "M3"]
    cfgs_all = sorted({r["config"] for r in rows}, key=lambda c: (cfg_order.index(c) if c in cfg_order else 999, c))
    compare_rows = _build_left_join_compare(by_cfg, cfg_order)

    class Handler(BaseHTTPRequestHandler):
        def _send_html(self, content: str):
            payload = content.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path
            q = parse_qs(parsed.query)

            if path == "/api/benchmarks":
                payload = json.dumps(benchmark_rows, indent=2).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            if path == "/api/posthoc":
                payload = json.dumps(posthoc_rows, indent=2).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            if path.startswith("/static/plots/"):
                name = unquote(path.split("/static/plots/", 1)[1])
                target = (plots_dir / name).resolve()
                if not target.exists() or target.parent != plots_dir.resolve():
                    self.send_response(404)
                    self.end_headers()
                    return
                with open(target, "rb") as f:
                    data = f.read()
                mime = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return

            if path.startswith("/static/posthoc/"):
                name = unquote(path.split("/static/posthoc/", 1)[1])
                target = (posthoc_dir / name).resolve()
                if not target.exists() or target.parent != posthoc_dir.resolve():
                    self.send_response(404)
                    self.end_headers()
                    return
                with open(target, "rb") as f:
                    data = f.read()
                mime = mimetypes.guess_type(str(target))[0] or "text/plain"
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return

            if path.startswith("/static/diagrams/"):
                name = unquote(path.split("/static/diagrams/", 1)[1])
                target = (diagrams_dir / name).resolve()
                if not target.exists() or target.parent != diagrams_dir.resolve():
                    self.send_response(404)
                    self.end_headers()
                    return
                with open(target, "rb") as f:
                    data = f.read()
                mime = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return

            if path in ("/", "/index.html"):
                cfg_stats = []
                for cfg in cfgs_all:
                    rr = by_cfg.get(cfg, [])
                    hits = sum(1 for x in rr if _contains_ref(x["prediction"], x["reference"]))
                    avg_words = (sum(len((x["prediction"] or "").split()) for x in rr) / len(rr)) if rr else 0.0
                    cfg_stats.append([cfg, str(len(rr)), f"{hits / max(len(rr), 1):.3f}", f"{avg_words:.1f}"])

                bench_rows = []
                for b in sorted(benchmark_rows, key=lambda x: x.get("config", "")):
                    gj = b.get("gpt4o_judge", {})
                    bench_rows.append(
                        [
                            html.escape(str(b.get("config", "N/A"))),
                            f"{float(b.get('contains_accuracy', 0.0)):.3f}",
                            f"{float(b.get('rouge_l', {}).get('f1', 0.0)):.3f}",
                            f"{float(b.get('bert_score', {}).get('f1', 0.0)):.3f}",
                            f"{float(gj.get('correctness', 0.0)):.2f}",
                            f"{float(gj.get('reasoning', 0.0)):.2f}",
                            f"{float(b.get('elapsed_seconds', 0.0))/60.0:.1f}",
                        ]
                    )

                posthoc_tbl = []
                for r in sorted(posthoc_rows, key=lambda x: x.get("config", "")):
                    posthoc_tbl.append(
                        [
                            html.escape(r.get("config", "")),
                            f"{float(r.get('contains_ref_acc', 0.0)):.3f}",
                            f"{float(r.get('token_f1_mean', 0.0)):.3f}",
                            f"{float(r.get('rougeL_f1_mean', 0.0)):.3f}",
                            f"{float(r.get('avg_pred_words', 0.0)):.1f}",
                            f"{float(r.get('repetition_rate', 0.0)):.3f}",
                        ]
                    )

                preferred_plot_order = [
                    "accuracy_comparison.png",
                    "metric_heatmap.png",
                    "contains_vs_bertscore.png",
                    "runtime_comparison.png",
                    "quality_profile_radar.png",
                    "efficiency_frontier.png",
                    "delta_vs_m1.png",
                    "metric_rankings.png",
                    "posthoc_length_vs_contains.png",
                    "posthoc_repetition_rate.png",
                    "posthoc_first_sentence_delta.png",
                    "posthoc_topic_contains_heatmap.png",
                    "posthoc_word_distribution.png",
                ]
                allowed_plot_ext = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"}
                discovered_plot_files = (
                    [p.name for p in sorted(plots_dir.iterdir()) if p.is_file() and p.suffix.lower() in allowed_plot_ext]
                    if plots_dir.exists()
                    else []
                )
                existing_plots = [p for p in preferred_plot_order if p in discovered_plot_files]
                existing_plots.extend([p for p in discovered_plot_files if p not in existing_plots])
                plot_html = "".join(
                    f'<div class="card"><h3>{html.escape(p)}</h3><img src="/static/plots/{html.escape(p)}" style="max-width:100%; border:1px solid #e5e7eb; border-radius:8px;" /></div>'
                    for p in existing_plots
                )

                page = f"""<!doctype html>
<html><head><meta charset="utf-8"/><title>Eval Dashboard</title><style>{_base_css()}</style></head>
<body>
  <h2>Evaluation Dashboard</h2>
  <div class="muted">Benchmark + posthoc + question-level exploration</div>

  <div class="card">
    <h3>Benchmark Metrics</h3>
    {_table_html(["Config", "Contains", "ROUGE-L", "BERTScore", "GPT Corr", "GPT Reas", "Time (min)"], bench_rows)}
    <p class="muted">JSON APIs: <a href="/api/benchmarks">benchmarks</a> | <a href="/api/posthoc">posthoc</a></p>
  </div>

  <div class="card">
    <h3>Benchmark + Posthoc Matrix</h3>
    {_table_html(
        ["Config", "Contains", "ROUGE-L", "BERTScore", "GPT Corr", "GPT Reas", "Token F1", "Avg Words", "Repetition"],
        [
            [
                html.escape(str(b.get("config", "N/A"))),
                f"{float(b.get('contains_accuracy', 0.0)):.3f}",
                f"{float(b.get('rouge_l', {}).get('f1', 0.0)):.3f}",
                f"{float(b.get('bert_score', {}).get('f1', 0.0)):.3f}",
                f"{float(b.get('gpt4o_judge', {}).get('correctness', 0.0)):.2f}",
                f"{float(b.get('gpt4o_judge', {}).get('reasoning', 0.0)):.2f}",
                (
                    next((f"{float(p.get('token_f1_mean', 0.0)):.3f}" for p in posthoc_rows if p.get("config") == b.get("config")), "N/A")
                ),
                (
                    next((f"{float(p.get('avg_pred_words', 0.0)):.1f}" for p in posthoc_rows if p.get("config") == b.get("config")), "N/A")
                ),
                (
                    next((f"{float(p.get('repetition_rate', 0.0)):.3f}" for p in posthoc_rows if p.get("config") == b.get("config")), "N/A")
                ),
            ]
            for b in sorted(benchmark_rows, key=lambda x: x.get("config", ""))
        ],
    )}
  </div>

  <div class="card">
    <h3>Prediction Set Stats</h3>
    {_table_html(["Config", "Rows", "Contains Hit Rate", "Avg Pred Words"], cfg_stats)}
    <p><a href="/questions">Question browser</a> | <a href="/compare">Side-by-side compare (left join by index)</a> | <a href="/experiment">Experiment diagrams</a></p>
  </div>

  <div class="card">
    <h3>Posthoc Summary</h3>
    {_table_html(["Config", "Contains", "Token F1", "ROUGE-L", "Avg Words", "Repetition"], posthoc_tbl) if posthoc_tbl else "<p class='muted'>No posthoc summary found. Run posthoc_analysis first.</p>"}
    <p class="muted">Downloads: <a href="/static/posthoc/posthoc_summary.csv">posthoc_summary.csv</a> | <a href="/static/posthoc/topic_breakdown.csv">topic_breakdown.csv</a></p>
  </div>

  {f'<div class="card"><h3>Plots</h3><div class="plot-grid">{ "".join(f"<div><div class=muted>{html.escape(p)}</div><img class=plot-frame src=/static/plots/{html.escape(p)} /></div>" for p in existing_plots) }</div></div>' if existing_plots else '<div class="card"><h3>Plots</h3><p class="muted">No plot files found in data/eval/plots. Run visualize_results first.</p></div>'}
</body></html>"""
                self._send_html(page)
                return

            if path == "/experiment":
                allowed = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
                diagrams = [p for p in sorted(diagrams_dir.iterdir()) if p.is_file() and p.suffix.lower() in allowed] if diagrams_dir.exists() else []
                cards = "".join(
                    f"""
                    <div class="card">
                      <h3>{html.escape(p.stem)}</h3>
                      <img class="diagram-frame" src="/static/diagrams/{html.escape(p.name)}" />
                      <div class="muted">file: {html.escape(p.name)}</div>
                    </div>
                    """
                    for p in diagrams
                )
                page = f"""<!doctype html>
<html><head><meta charset="utf-8"/><title>Experiment Diagrams</title><style>{_base_css()}</style></head>
<body>
  <h2>Experiment Diagrams</h2>
  <p><a href="/">Back to dashboard</a> | <a href="/questions">Question browser</a> | <a href="/compare">Compare</a></p>
  {f'<div class="diagram-grid">{cards}</div>' if cards else '<div class="card"><p class="muted">No diagram images found.</p></div>'}
</body></html>"""
                self._send_html(page)
                return

            if path == "/compare":
                topic = q.get("topic", ["all"])[0]
                term = q.get("q", [""])[0].strip().lower()
                page_num = max(1, int(q.get("page", ["1"])[0]))
                per_page = 8
                filtered = []
                for r in compare_rows:
                    if topic != "all" and r["topic"] != topic:
                        continue
                    blob = (r["question"] + " " + r["reference"] + " " + " ".join(r["answers"].values())).lower()
                    if term and term not in blob:
                        continue
                    filtered.append(r)
                total = len(filtered)
                total_pages = max(1, (total + per_page - 1) // per_page)
                page_num = min(page_num, total_pages)
                start = (page_num - 1) * per_page
                end = min(start + per_page, total)
                page_rows = filtered[start:end]
                topics = sorted({r["topic"] for r in compare_rows})

                def esc(s: str) -> str:
                    return html.escape(s).replace("\n", "<br>")

                cards = []
                for row in page_rows:
                    columns = []
                    for cfg in cfg_order:
                        pred = row["answers"].get(cfg, "")
                        if not pred:
                            columns.append('<div class="ans"><div class="hdr"><b>' + cfg + '</b> <span class="badge missing">missing</span></div></div>')
                            continue
                        hit = _contains_ref(pred, row["reference"])
                        health = _latex_health(pred)
                        columns.append(
                            f"""
                            <div class="ans">
                              <div class="hdr"><b>{cfg}</b> <span class="badge {'hit' if hit else 'miss'}">{'hit' if hit else 'miss'}</span> <span class="badge {'latex-ok' if health=='ok' else 'latex-suspicious'}">latex:{health}</span></div>
                              <div class="pred">{esc(pred)}</div>
                              <details><summary>Raw text</summary><pre>{html.escape(pred)}</pre></details>
                            </div>
                            """
                        )
                    cards.append(
                        f"""
                        <div class="card">
                          <div class="muted">row #{row['idx']} <span class="badge topic">{row['topic']}</span></div>
                          <p><b>Q:</b> {esc(row['question'])}</p>
                          <p><b>Ref:</b> {esc(row['reference'])}</p>
                          <div style="display:grid; grid-template-columns: repeat(5, minmax(220px, 1fr)); gap:8px;">{''.join(columns)}</div>
                        </div>
                        """
                    )

                prev_page = max(1, page_num - 1)
                next_page = min(total_pages, page_num + 1)
                query_base = f"topic={topic}&q={html.escape(term)}"
                page = f"""<!doctype html>
<html><head><meta charset="utf-8"/><title>Compare</title><style>{_base_css()}</style></head>
<body>
  <h2>Side-by-Side Compare (index left join)</h2>
  <p><a href="/">Back to dashboard</a> | <a href="/questions">Question browser</a> | <a href="/experiment">Experiment diagrams</a></p>
  <form method="get" class="controls" style="grid-template-columns:180px 1fr 120px;">
    <input type="hidden" name="page" value="1"/>
    <select name="topic"><option value="all">all topics</option>{"".join(f'<option value="{t}" {"selected" if t==topic else ""}>{t}</option>' for t in topics)}</select>
    <input type="text" name="q" value="{html.escape(term)}" placeholder="search question/reference/predictions"/>
    <button type="submit">Apply</button>
  </form>
  <div><b>{total}</b> rows matched (showing {start + 1 if total else 0}-{end})</div>
  <div class="pager"><a href="?{query_base}&page={prev_page}">Prev</a><span>Page {page_num} / {total_pages}</span><a href="?{query_base}&page={next_page}">Next</a></div>
  {''.join(cards) if cards else "<p>No rows.</p>"}
</body></html>"""
                self._send_html(page)
                return

            if path == "/questions":
                cfg = q.get("config", ["all"])[0]
                topic = q.get("topic", ["all"])[0]
                status = q.get("status", ["all"])[0]
                term = q.get("q", [""])[0].strip().lower()
                page_num = max(1, int(q.get("page", ["1"])[0]))
                per_page = 20
                filtered = []
                for r in rows:
                    hit = _contains_ref(r["prediction"], r["reference"])
                    t = _topic(r["question"])
                    if cfg != "all" and r["config"] != cfg:
                        continue
                    if topic != "all" and t != topic:
                        continue
                    if status == "hit" and not hit:
                        continue
                    if status == "miss" and hit:
                        continue
                    if term and term not in (r["question"] + " " + r["prediction"]).lower():
                        continue
                    rr = dict(r)
                    rr["hit"] = hit
                    rr["topic"] = t
                    filtered.append(rr)

                total = len(filtered)
                total_pages = max(1, (total + per_page - 1) // per_page)
                page_num = min(page_num, total_pages)
                start = (page_num - 1) * per_page
                end = min(start + per_page, total)
                page_rows = filtered[start:end]
                topics = sorted({_topic(r["question"]) for r in rows})

                cards = []
                for r in page_rows:
                    health = _latex_health(r["prediction"])
                    cards.append(
                        f"""
                        <div class="card">
                          <div class="muted"><span class="badge topic">{r['config']}</span> <span class="badge topic">{r['topic']}</span> <span class="badge {'hit' if r['hit'] else 'miss'}">{'hit' if r['hit'] else 'miss'}</span> <span class="badge {'latex-ok' if health=='ok' else 'latex-suspicious'}">latex:{health}</span> idx#{r['idx']}</div>
                          <p><b>Q:</b> {html.escape(r['question'])}</p>
                          <p><b>Ref:</b> {html.escape(r['reference'])}</p>
                          <div class="pred">{html.escape(r['prediction']).replace(chr(10), '<br>')}</div>
                          <details><summary>Raw text</summary><pre>{html.escape(r['prediction'])}</pre></details>
                        </div>
                        """
                    )

                prev_page = max(1, page_num - 1)
                next_page = min(total_pages, page_num + 1)
                query_base = f"config={cfg}&topic={topic}&status={status}&q={html.escape(term)}"
                page = f"""<!doctype html>
<html><head><meta charset="utf-8"/><title>Questions</title><style>{_base_css()}</style></head>
<body>
  <h2>Question Browser</h2>
  <p><a href="/">Back to dashboard</a> | <a href="/compare">Side-by-side compare</a> | <a href="/experiment">Experiment diagrams</a></p>
  <form method="get" class="controls" style="grid-template-columns:repeat(5,minmax(120px,1fr));">
    <input type="hidden" name="page" value="1"/>
    <select name="config"><option value="all">all configs</option>{"".join(f'<option value="{c}" {"selected" if c==cfg else ""}>{c}</option>' for c in cfgs_all)}</select>
    <select name="topic"><option value="all">all topics</option>{"".join(f'<option value="{t}" {"selected" if t==topic else ""}>{t}</option>' for t in topics)}</select>
    <select name="status">
      <option value="all" {"selected" if status=="all" else ""}>all</option>
      <option value="hit" {"selected" if status=="hit" else ""}>contains hit</option>
      <option value="miss" {"selected" if status=="miss" else ""}>contains miss</option>
    </select>
    <input type="text" name="q" value="{html.escape(term)}" placeholder="search question/prediction"/>
    <button type="submit">Apply</button>
  </form>
  <div><b>{total}</b> rows matched (showing {start + 1 if total else 0}-{end})</div>
  <div class="pager"><a href="?{query_base}&page={prev_page}">Prev</a><span>Page {page_num} / {total_pages}</span><a href="?{query_base}&page={next_page}">Next</a></div>
  {''.join(cards) if cards else "<p>No rows.</p>"}
</body></html>"""
                self._send_html(page)
                return

            self.send_response(404)
            self.end_headers()

        def log_message(self, format, *args):  # noqa: A003
            return

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Local browser for question-wise predictions.")
    parser.add_argument("--preds-dir", type=Path, default=Path("data/preds"))
    parser.add_argument("--benchmarks-dir", type=Path, default=Path("data/benchmarks"))
    parser.add_argument("--plots-dir", type=Path, default=Path("data/eval/plots"))
    parser.add_argument("--posthoc-dir", type=Path, default=Path("data/eval/posthoc"))
    parser.add_argument("--diagrams-dir", type=Path, default=Path("paper/Diagrams"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    rows, by_cfg = _load_preds(args.preds_dir)
    if not rows:
        raise FileNotFoundError(f"No preds_*.json found in {args.preds_dir}")
    benchmark_rows = _load_benchmarks(args.benchmarks_dir)
    posthoc_rows = _load_posthoc_summary(args.posthoc_dir)

    handler = make_handler(
        rows,
        by_cfg,
        benchmark_rows,
        posthoc_rows,
        args.plots_dir,
        args.posthoc_dir,
        args.diagrams_dir,
    )
    httpd = HTTPServer((args.host, args.port), handler)
    print(
        f"Serving dashboard at http://{args.host}:{args.port} "
        f"(rows={len(rows)}, benchmarks={len(benchmark_rows)}, posthoc={len(posthoc_rows)})"
    )
    httpd.serve_forever()


if __name__ == "__main__":
    main()
