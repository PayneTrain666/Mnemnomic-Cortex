#!/usr/bin/env python3
"""
Plain-language summary
----------------------
What this file is for: Live training metrics dashboard for copy/reverse runs.
How it fits in the system: Reads metrics JSONL written by copy_task_gpu_train.py.
Status: WORKING
Important notes for non-coders: Open the local URL while training to see charts and advice.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.training_metrics_catalog import catalog_rows, interpret_training


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Mnemonic Cortex Training Dashboard</title>
  <style>
    :root {
      --bg: #111315;
      --panel: #1a1d21;
      --text: #e8eaed;
      --muted: #9aa0a6;
      --line: #2a2f36;
      --good: #7cb87c;
      --warn: #d0a15c;
      --info: #7ea7d0;
      --accent: #c4c7c5;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", "Helvetica Neue", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    header {
      padding: 18px 22px 8px;
      border-bottom: 1px solid var(--line);
    }
    h1 { font-size: 22px; margin: 0 0 6px; font-weight: 600; }
    .sub { color: var(--muted); font-size: 13px; }
    main { padding: 16px 22px 40px; display: grid; gap: 16px; }
    .grid { display: grid; gap: 12px; grid-template-columns: repeat(4, minmax(0, 1fr)); }
    .stat, .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
    }
    .stat .k { color: var(--muted); font-size: 12px; }
    .stat .v { font-size: 22px; margin-top: 4px; font-variant-numeric: tabular-nums; }
    .card h2 { margin: 0 0 10px; font-size: 15px; font-weight: 600; }
    .advice { display: grid; gap: 8px; }
    .advice-item {
      border-left: 3px solid var(--info);
      padding: 8px 10px;
      background: #15181c;
    }
    .advice-item.good { border-color: var(--good); }
    .advice-item.warn { border-color: var(--warn); }
    .advice-item .t { font-weight: 600; margin-bottom: 4px; }
    .advice-item .d { color: var(--muted); font-size: 13px; line-height: 1.4; }
    canvas.chart {
      width: 100%;
      height: 220px;
      background: #14171b;
      border: 1px solid var(--line);
      border-radius: 6px;
    }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { border-bottom: 1px solid var(--line); padding: 6px 8px; text-align: left; vertical-align: top; }
    th { color: var(--muted); font-weight: 600; }
    .two { display: grid; gap: 12px; grid-template-columns: 1.2fr 1fr; }
    .metric-groups { display: grid; gap: 12px; grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .metric-group h3 { margin: 0 0 8px; font-size: 13px; color: var(--accent); font-weight: 600; }
    .metric-group table td:first-child { width: 55%; color: var(--muted); font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 11px; }
    .metric-group table td:last-child { font-variant-numeric: tabular-nums; word-break: break-all; }
    .scroll { max-height: 420px; overflow: auto; }
    .event-log { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 11px; white-space: pre-wrap; color: var(--text); line-height: 1.45; }
    @media (max-width: 1100px) {
      .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .two { grid-template-columns: 1fr; }
      .metric-groups { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Live training metrics</h1>
    <div class="sub" id="subtitle">Waiting for metrics…</div>
  </header>
  <main>
    <section class="grid" id="stats"></section>
    <section class="two">
      <div class="card">
        <h2>Loss and accuracy</h2>
        <canvas class="chart" id="chartMain"></canvas>
      </div>
      <div class="card">
        <h2>Advice and insight</h2>
        <div class="advice" id="advice"></div>
      </div>
    </section>
    <section class="two">
      <div class="card">
        <h2>Gradients and learning rate</h2>
        <canvas class="chart" id="chartOpt"></canvas>
      </div>
      <div class="card">
        <h2>Latest epoch validation</h2>
        <div id="epochBox" class="sub">No epoch summary yet.</div>
      </div>
    </section>
    <section class="card">
      <h2>All latest metrics (<span id="latestCount">0</span> keys)</h2>
      <div class="sub" style="margin-bottom:8px;">Full payload from the newest train_step / epoch_end event.</div>
      <div class="metric-groups" id="allMetrics"></div>
    </section>
    <section class="card">
      <h2>Event log</h2>
      <div class="scroll" data-scroll-key="event-log">
        <div class="event-log" id="eventLog">No events yet.</div>
      </div>
    </section>
    <section class="card">
      <h2>Metric glossary (filtered to keys seen in this run)</h2>
      <div class="scroll" data-scroll-key="glossary">
        <table>
          <thead><tr><th>Key</th><th>Meaning</th><th>Healthy</th><th>Watch for</th></tr></thead>
          <tbody id="glossary"></tbody>
        </table>
      </div>
    </section>
  </main>
  <script>
    const params = new URLSearchParams(location.search);
    const metricsPath = params.get("metrics") || "";

    function fmt(v, digits=4) {
      if (v === null || v === undefined || Number.isNaN(v)) return "—";
      if (typeof v === "number") return v.toFixed(digits);
      return String(v);
    }

    function captureScrollPositions() {
      const positions = {
        window: { top: window.scrollY || 0, left: window.scrollX || 0 },
        panels: {},
      };
      document.querySelectorAll(".scroll[data-scroll-key]").forEach((el) => {
        positions.panels[el.getAttribute("data-scroll-key")] = {
          top: el.scrollTop,
          left: el.scrollLeft,
        };
      });
      return positions;
    }

    function restoreScrollPositions(positions) {
      if (!positions) return;
      const apply = () => {
        document.querySelectorAll(".scroll[data-scroll-key]").forEach((el) => {
          const key = el.getAttribute("data-scroll-key");
          const saved = positions.panels[key];
          if (!saved) return;
          el.scrollTop = saved.top;
          el.scrollLeft = saved.left;
        });
        if (positions.window) {
          window.scrollTo(positions.window.left, positions.window.top);
        }
      };
      // Restore after layout from innerHTML replacements.
      apply();
      requestAnimationFrame(apply);
    }

    function drawChart(canvas, seriesList, yLabel) {
      const ctx = canvas.getContext("2d");
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const w = rect.width, h = rect.height;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#14171b";
      ctx.fillRect(0, 0, w, h);
      const pad = {l: 42, r: 12, t: 14, b: 28};
      const colors = ["#c4c7c5", "#7ea7d0", "#7cb87c", "#d0a15c"];
      let allY = [];
      seriesList.forEach(s => { allY = allY.concat(s.values.filter(Number.isFinite)); });
      if (!allY.length) {
        ctx.fillStyle = "#9aa0a6";
        ctx.fillText("No series yet", pad.l, h/2);
        return;
      }
      const minY = Math.min(...allY);
      const maxY = Math.max(...allY);
      const spanY = Math.max(1e-9, maxY - minY);
      const maxX = Math.max(...seriesList.map(s => Math.max(0, s.values.length - 1)), 1);
      function xAt(i) { return pad.l + (w - pad.l - pad.r) * (i / maxX); }
      function yAt(v) { return pad.t + (h - pad.t - pad.b) * (1 - (v - minY) / spanY); }
      ctx.strokeStyle = "#2a2f36";
      ctx.beginPath();
      ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, h - pad.b); ctx.lineTo(w - pad.r, h - pad.b);
      ctx.stroke();
      ctx.fillStyle = "#9aa0a6";
      ctx.font = "11px Segoe UI";
      ctx.fillText(yLabel, pad.l, 12);
      seriesList.forEach((s, idx) => {
        if (!s.values.length) return;
        ctx.strokeStyle = colors[idx % colors.length];
        ctx.beginPath();
        s.values.forEach((v, i) => {
          const x = xAt(i), y = yAt(v);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.stroke();
        ctx.fillStyle = colors[idx % colors.length];
        ctx.fillText(s.name, pad.l + 8 + idx * 90, h - 10);
      });
    }

    async function refresh() {
      const url = "/api/snapshot" + (metricsPath ? ("?metrics=" + encodeURIComponent(metricsPath)) : "");
      const res = await fetch(url);
      const data = await res.json();
      const savedScroll = captureScrollPositions();
      const h = data.highlights || {};
      document.getElementById("subtitle").textContent =
        `Source: ${data.metrics_path || "—"} · steps=${h.steps_logged || 0} · epochs=${h.epochs_logged || 0} · refreshed ${new Date().toLocaleTimeString()}`;

      const stats = [
        ["Step loss", fmt(h.latest_loss)],
        ["Mean loss", fmt(h.latest_mean_loss)],
        ["Token acc", fmt(h.latest_acc)],
        ["Seq acc", fmt(h.latest_seq_acc)],
        ["LR", fmt(h.latest_lr, 6)],
        ["Pre-grad", fmt(h.latest_pre_grad, 2)],
        ["Loss trend", h.loss_trend || "—"],
        ["Best val@curr", fmt(h.best_val_acc_curriculum)],
      ];
      document.getElementById("stats").innerHTML = stats.map(([k,v]) =>
        `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`
      ).join("");

      const advice = data.advice || [];
      document.getElementById("advice").innerHTML = advice.map(a =>
        `<div class="advice-item ${a.tone || "info"}"><div class="t">${a.title}</div><div class="d">${a.detail}</div></div>`
      ).join("");

      const series = data.series || {};
      drawChart(document.getElementById("chartMain"), [
        {name: "loss", values: series.loss || []},
        {name: "mean_loss", values: series.mean_loss || []},
        {name: "acc", values: series.acc || []},
        {name: "seq_acc", values: series.seq_acc || []},
      ], "loss / acc");
      drawChart(document.getElementById("chartOpt"), [
        {name: "pre_grad", values: series.pre_grad_norm || []},
        {name: "lr*1000", values: (series.lr || []).map(v => v * 1000)},
        {name: "topo_fit", values: series.topology_fitness_ema || []},
      ], "opt diagnostics");

      const epochs = data.epochs || [];
      if (epochs.length) {
        const e = epochs[epochs.length - 1];
        document.getElementById("epochBox").innerHTML = `
          <div>Epoch ${e.epoch}</div>
          <div>train_mean_loss=${fmt(e.train_mean_loss)} train_acc=${fmt(e.train_acc)}</div>
          <div>val@curr loss=${fmt(e.val_loss_curriculum_len)} acc=${fmt(e.val_acc_curriculum_len)}</div>
          <div>val@8 acc=${fmt(e.val_acc_len8)} · val@16 acc=${fmt(e.val_acc_len16)}</div>
          <div>seq_acc@curr=${fmt(e.val_seq_acc_curriculum_len)}</div>
        `;
      }

      const glossary = data.glossary || [];
      document.getElementById("glossary").innerHTML = glossary.map(row =>
        `<tr><td><code>${row.key}</code></td><td>${row.meaning}</td><td>${row.healthy}</td><td>${row.watch}</td></tr>`
      ).join("");

      const latest = data.latest || {};
      const keys = Object.keys(latest).sort();
      document.getElementById("latestCount").textContent = String(keys.length);
      const groups = {};
      keys.forEach((k) => {
        let g = "core";
        if (k.startsWith("broker_")) g = "broker / CMS";
        else if (k.startsWith("hg_episodic") || k.startsWith("shared_")) g = "shared / episodic";
        else if (k.startsWith("hg_") || k.startsWith("cgmn_") || k.startsWith("curved_") || k.startsWith("ltm_") || k.startsWith("spatial") || k.startsWith("ltm_spatial")) g = "LTM banks";
        else if (k.startsWith("router_") || k.startsWith("aux_spec") || k.startsWith("aux_")) g = "router / aux";
        else if (k.startsWith("diag_") || k.startsWith("cms_depth") || k.startsWith("qdt_")) g = "diagnostics / depth";
        else if (k.startsWith("val_") || k.startsWith("train_")) g = "train / val";
        const bucket = groups[g] || (groups[g] = []);
        bucket.push([k, latest[k]]);
      });
      const groupOrder = ["core", "train / val", "LTM banks", "shared / episodic", "broker / CMS", "router / aux", "diagnostics / depth"];
      const ordered = groupOrder.filter((g) => groups[g]).concat(Object.keys(groups).filter((g) => !groupOrder.includes(g)));
      document.getElementById("allMetrics").innerHTML = ordered.map((g) => {
        const rows = groups[g].map(([k, v]) => {
          let shown = v;
          if (typeof v === "number") shown = Number.isInteger(v) ? String(v) : v.toFixed(6);
          else if (v === null || v === undefined) shown = "—";
          else shown = String(v);
          return `<tr><td>${k}</td><td>${shown}</td></tr>`;
        }).join("");
        const scrollKey = "metrics:" + g;
        return `<div class="metric-group card" style="padding:10px;"><h3>${g} (${groups[g].length})</h3><div class="scroll" data-scroll-key="${scrollKey}"><table><tbody>${rows}</tbody></table></div></div>`;
      }).join("");

      const events = data.events || [];
      if (!events.length) {
        document.getElementById("eventLog").textContent = "No events yet.";
      } else {
        document.getElementById("eventLog").textContent = events.map((ev, idx) => {
          const kind = ev.kind || "?";
          if (kind === "run_start") {
            const a = ev.args || {};
            return `#${idx} run_start profile=${a.qdt_hardware_profile} batch=${a.batch_size} steps=${a.total_steps} d_model=${a.d_model}`;
          }
          if (kind === "train_step") {
            return (
              `#${idx} train_step gstep=${ev.global_step} ep=${ev.epoch} len=${ev.seq_len} ` +
              `loss=${fmt(ev.loss)} mean=${fmt(ev.mean_loss)} acc=${fmt(ev.acc)} seq=${fmt(ev.seq_acc)} ` +
              `recall=${fmt(ev.recall_loss)} cms=${fmt(ev.cms_loss)} pre_grad=${fmt(ev.pre_grad_norm,2)} ` +
              `lr=${fmt(ev.lr,6)} topo=${fmt(ev.topology_fitness_ema)} ` +
              `router(hg/cgmn/curved/spcp/spatial)=${fmt(ev.ltm_router_hg)}/${fmt(ev.ltm_router_cgmn)}/${fmt(ev.ltm_router_curved)}/${fmt(ev.ltm_router_spcp)}/${fmt(ev.ltm_router_spatial)} ` +
              `shared_used=${fmt(ev.shared_mem_used_slots,0)} epi_writes=${fmt(ev.shared_slot_episodic_writes_total,0)} ` +
              `keys=${Object.keys(ev).length}`
            );
          }
          if (kind === "epoch_end") {
            return (
              `#${idx} epoch_end ep=${ev.epoch} train_acc=${fmt(ev.train_acc)} ` +
              `val_curr=${fmt(ev.val_acc_curriculum_len)} val8=${fmt(ev.val_acc_len8)} val16=${fmt(ev.val_acc_len16)}`
            );
          }
          return `#${idx} ${kind} keys=${Object.keys(ev).length}`;
        }).join("\n");
      }
      restoreScrollPositions(savedScroll);
    }

    refresh();
    setInterval(refresh, 2000);
  </script>
</body>
</html>
"""


def _read_events(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    events: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                events.append(payload)
    return events


def build_snapshot(metrics_path: Path, *, target_grad_norm: float = 0.8) -> Dict[str, Any]:
    events = _read_events(metrics_path)
    interpretation = interpret_training(events, target_grad_norm=target_grad_norm)
    seen_keys = set()
    for event in events:
        seen_keys.update(event.keys())
    glossary = [row for row in catalog_rows() if row["key"] in seen_keys]
    # Keep glossary usable even before first step by showing core metrics.
    if not glossary:
        glossary = [row for row in catalog_rows() if row["group"] in {"core", "validation"}][:24]
    # Cap event log payload for long runs; always keep full latest in interpretation.
    event_log = events[-80:] if len(events) > 80 else events
    return {
        "metrics_path": str(metrics_path),
        "mtime": metrics_path.stat().st_mtime if metrics_path.exists() else None,
        "event_count": len(events),
        "highlights": interpretation["highlights"],
        "advice": interpretation["advice"],
        "latest": interpretation["latest"],
        "epochs": interpretation["epochs"],
        "series": interpretation["series"],
        "start": interpretation["start"],
        "events": event_log,
        "glossary": glossary,
    }


def serve(
    metrics_path: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    target_grad_norm: float = 0.8,
) -> None:
    metrics_path = metrics_path.resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                self._send(200, DASHBOARD_HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/snapshot":
                qs = parse_qs(parsed.query)
                chosen = metrics_path
                if qs.get("metrics"):
                    candidate = Path(qs["metrics"][0])
                    if not candidate.is_absolute():
                        candidate = (Path.cwd() / candidate).resolve()
                    chosen = candidate
                payload = build_snapshot(chosen, target_grad_norm=target_grad_norm)
                self._send(
                    200,
                    json.dumps(payload).encode("utf-8"),
                    "application/json; charset=utf-8",
                )
                return
            if parsed.path == "/api/catalog":
                self._send(
                    200,
                    json.dumps(catalog_rows()).encode("utf-8"),
                    "application/json; charset=utf-8",
                )
                return
            self._send(404, b"not found", "text/plain; charset=utf-8")

    server = ThreadingHTTPServer((host, int(port)), Handler)
    url = f"http://{host}:{port}/?metrics={metrics_path.as_posix()}"
    print(f"[dashboard] serving {url}", flush=True)
    print(f"[dashboard] watching metrics file: {metrics_path}", flush=True)
    if open_browser:
        threading.Thread(target=lambda: (time.sleep(0.4), webbrowser.open(url)), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[dashboard] stopped", flush=True)
        server.shutdown()


def launch_in_background(
    metrics_path: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    target_grad_norm: float = 0.8,
) -> threading.Thread:
    path = Path(metrics_path)

    def _run() -> None:
        serve(
            path,
            host=host,
            port=port,
            open_browser=open_browser,
            target_grad_norm=target_grad_norm,
        )

    thread = threading.Thread(target=_run, name="training-metrics-dashboard", daemon=True)
    thread.start()
    return thread


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Live copy/reverse training metrics dashboard")
    p.add_argument(
        "--metrics_jsonl",
        default="logs/copy_reverse_smoke/metrics.jsonl",
        help="Path to trainer metrics JSONL",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no_browser", action="store_true")
    p.add_argument("--target_grad_norm", type=float, default=0.8)
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    serve(
        Path(args.metrics_jsonl),
        host=str(args.host),
        port=int(args.port),
        open_browser=not bool(args.no_browser),
        target_grad_norm=float(args.target_grad_norm),
    )


if __name__ == "__main__":
    main()
