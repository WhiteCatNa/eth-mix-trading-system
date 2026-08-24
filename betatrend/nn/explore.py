"""TorchExplorer 看板：盯 GRPO Actor 的层输入/输出/参数/梯度，并叠训练日志。

不进训练子进程。当前 walk-forward 已经在跑时，用 ``explore-nn --live``
对着最新 fold checkpoint 做一次真实窗口的前向+反向，直方图随折更新。
"""
from __future__ import annotations

import json
import re
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

from betatrend.config import ROOT
from betatrend.nn.dataset import N_FEAT, SEQ_LEN
from betatrend.nn.model import GRPOActor

_FOLD = re.compile(
    r"fold (\d+)/(\d+) train_idx=\[(\d+), (\d+)\) n_train=(\d+) n_test=(\d+)"
)
_EPOCH = re.compile(r"GRPO seed=(\d+) epoch (\d+)/(\d+) mean_r=([-+0-9.eE]+)")
_HEADER = re.compile(
    r"walk-forward folds=(\d+) warmup=(\d+) min_train=(\d+) test_h=(\d+) seeds=(\d+) epochs=(\d+)"
)
_CKPT = re.compile(r"saved checkpoint (.+\.pt)")
_ERR = re.compile(r"(Traceback|ValueError|RuntimeError|KeyboardInterrupt)")
_FOLD_CKPT = re.compile(r"eth_decision_fold(\d+)_s(\d+)_e(\d+)\.pt$")

_BOARD_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>GRPO train board</title>
<style>
  :root { color-scheme: dark; --bg:#0f1115; --fg:#e6e6e6; --muted:#8b909a; --line:#2a2e36; --accent:#6ea8fe; --ok:#3dd68c; --warn:#e6b450; }
  * { box-sizing: border-box; }
  body { margin:0; font: 13px/1.45 ui-sans-serif, system-ui, sans-serif; background: var(--bg); color: var(--fg); }
  header { padding: 16px 20px 8px; border-bottom: 1px solid var(--line); }
  h1 { font-size: 18px; font-weight: 600; margin: 0 0 4px; }
  .sub { color: var(--muted); }
  .stats { display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; padding: 16px 20px; }
  .stat { border: 1px solid var(--line); padding: 10px 12px; }
  .stat .k { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .04em; }
  .stat .v { font-size: 20px; font-weight: 600; margin-top: 4px; }
  .ok { color: var(--ok); } .warn { color: var(--warn); }
  .row { display: grid; grid-template-columns: 1.4fr 0.8fr; gap: 0; }
  section { padding: 12px 20px 20px; }
  h2 { font-size: 13px; font-weight: 600; margin: 0 0 8px; color: var(--muted); }
  #chart { width: 100%; height: 220px; background: #0b0d11; border: 1px solid var(--line); }
  iframe { width: 100%; height: 72vh; border: 0; background: #111; border-top: 1px solid var(--line); }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 4px 6px; border-bottom: 1px solid var(--line); font-variant-numeric: tabular-nums; }
  th { color: var(--muted); font-weight: 500; }
  a { color: var(--accent); }
  .mod { border: 1px solid var(--line); margin-bottom: 12px; }
  .modh { padding: 8px 10px; border-bottom: 1px solid var(--line); font-weight: 600; }
  .modh span { color: var(--muted); font-weight: 400; margin-left: 8px; }
  .hrow { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 8px; padding: 10px; }
  .hcell canvas { width: 100%; height: 72px; background: #0b0d11; border: 1px solid var(--line); }
  .hk { color: var(--muted); font-size: 11px; margin-bottom: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .hr { color: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
</style>
</head>
<body>
<header>
  <h1>GRPO training board</h1>
  <div class="sub">上面是层输入/输出/参数直方图（每折 checkpoint 更新）。下面结构图已自动钉上 LSTM / Linear / mu_raw，不必再拖节点。</div>
</header>
<div class="stats" id="stats"></div>
<div class="row">
  <section>
    <h2>In-sample group reward (mean_r)</h2>
    <canvas id="chart" width="720" height="220"></canvas>
  </section>
  <section>
    <h2>Latest epochs</h2>
    <table id="tbl"><thead><tr><th>fold</th><th>seed</th><th>epoch</th><th>mean_r</th></tr></thead><tbody></tbody></table>
  </section>
</div>
<section>
  <h2>Layer histograms (io / params)</h2>
  <div id="hists"></div>
</section>
<section style="padding-top:0">
  <h2>TorchExplorer <a id="exlink" href="http://127.0.0.1:8080" target="_blank">open 8080</a></h2>
  <iframe id="explorer" src="http://127.0.0.1:8080" title="TorchExplorer"></iframe>
</section>
<script>
async function load() {
  const r = await fetch("metrics.json?t=" + Date.now());
  const m = await r.json();
  const st = m.status || "unknown";
  const cls = st === "error" ? "warn" : "ok";
  document.getElementById("stats").innerHTML = [
    ["fold", (m.fold || "—") + " / " + (m.n_folds || "—")],
    ["n_train", m.n_train ?? "—"],
    ["epoch", m.epoch ? (m.epoch + " / " + m.epochs) : "—"],
    ["mean_r", m.mean_r == null ? "—" : Number(m.mean_r).toFixed(4)],
    ["ckpt fold", m.ckpt_fold == null ? "—" : m.ckpt_fold],
    ["status", st],
  ].map(([k,v],i) => `<div class="stat"><div class="k">${k}</div><div class="v ${i===5?cls:""}">${v}</div></div>`).join("");
  if (m.explorer_url) {
    document.getElementById("exlink").href = m.explorer_url;
    const ifr = document.getElementById("explorer");
    if (ifr.getAttribute("src") !== m.explorer_url) ifr.src = m.explorer_url;
  }
  const rows = (m.points || []).slice(-12).reverse();
  document.querySelector("#tbl tbody").innerHTML = rows.map(p =>
    `<tr><td>${p.fold}</td><td>${p.seed}</td><td>${p.epoch}</td><td>${Number(p.mean_r).toFixed(4)}</td></tr>`
  ).join("");
  draw(m.points || []);
}
function draw(points) {
  const c = document.getElementById("chart");
  const ctx = c.getContext("2d");
  const w = c.width, h = c.height;
  ctx.clearRect(0,0,w,h);
  if (!points.length) return;
  const ys = points.map(p => p.mean_r);
  let ymin = Math.min(0, ...ys), ymax = Math.max(0, ...ys);
  if (ymin === ymax) { ymin -= 0.1; ymax += 0.1; }
  const pad = 0.08 * (ymax - ymin);
  ymin -= pad; ymax += pad;
  const x = i => 24 + (w-36) * i / Math.max(points.length-1, 1);
  const y = v => h-18 - (h-32) * (v - ymin) / (ymax - ymin);
  ctx.strokeStyle = "#2a2e36"; ctx.beginPath(); ctx.moveTo(24,y(0)); ctx.lineTo(w-8,y(0)); ctx.stroke();
  ctx.strokeStyle = "#6ea8fe"; ctx.beginPath();
  points.forEach((p,i) => { const X=x(i), Y=y(p.mean_r); i?ctx.lineTo(X,Y):ctx.moveTo(X,Y); });
  ctx.stroke();
  ctx.fillStyle = "#8b909a"; ctx.font = "11px ui-sans-serif";
  ctx.fillText(ymax.toFixed(2), 2, 12);
  ctx.fillText(ymin.toFixed(2), 2, h-6);
}
function drawHist(canvas, bins) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width = canvas.clientWidth || 220;
  const h = canvas.height = 72;
  ctx.clearRect(0,0,w,h);
  if (!bins || !bins.length) return;
  const max = Math.max(...bins, 1e-9);
  const bw = w / bins.length;
  ctx.fillStyle = "#6ea8fe";
  bins.forEach((v,i) => {
    const bh = (h-4) * v / max;
    ctx.fillRect(i*bw, h-bh, Math.max(bw-0.5, 1), bh);
  });
}
async function loadHists() {
  const r = await fetch("hists.json?t=" + Date.now());
  if (!r.ok) { document.getElementById("hists").textContent = "直方图还没写出来，等下一折 checkpoint。"; return; }
  const data = await r.json();
  const mods = data.modules || [];
  const root = document.getElementById("hists");
  root.innerHTML = mods.map(m => {
    const cards = (m.hists || []).slice(0, 6);
    return `<div class="mod"><div class="modh">${m.name}<span>${m.path}</span></div><div class="hrow">${
      cards.map((h,i) => `<div class="hcell"><div class="hk">${h.group} · ${h.label}</div><canvas data-k="${m.id}-${i}"></canvas><div class="hr">${Number(h.min).toFixed(2)} … ${Number(h.max).toFixed(2)}</div></div>`).join("")
    }</div></div>`;
  }).join("") || "<div class='sub'>还没有层直方图</div>";
  mods.forEach(m => (m.hists || []).slice(0,6).forEach((h,i) => {
    const c = document.querySelector(`canvas[data-k="${m.id}-${i}"]`);
    if (c) drawHist(c, h.bins);
  }));
}
load();
loadHists();
setInterval(load, 4000);
setInterval(loadHists, 15000);
</script>
</body>
</html>
"""


def parse_train_log(text: str) -> dict[str, Any]:
    """把 train-nn 日志收成看板用的快照。"""
    n_folds = epochs = seeds = None
    fold = n_train = epoch = None
    mean_r = None
    points: list[dict[str, Any]] = []
    last_ckpt = None
    status = "running"
    error = None
    for line in text.splitlines():
        if _ERR.search(line) and "INFO" not in line:
            status = "error"
            error = line.strip()[:400]
        mh = _HEADER.search(line)
        if mh:
            n_folds = int(mh.group(1))
            seeds = int(mh.group(5))
            epochs = int(mh.group(6))
        mf = _FOLD.search(line)
        if mf:
            fold = int(mf.group(1))
            n_folds = int(mf.group(2))
            n_train = int(mf.group(5))
        me = _EPOCH.search(line)
        if me:
            epoch = int(me.group(2))
            epochs = int(me.group(3))
            mean_r = float(me.group(4))
            points.append(
                {
                    "fold": fold,
                    "seed": int(me.group(1)),
                    "epoch": epoch,
                    "mean_r": mean_r,
                }
            )
        mc = _CKPT.search(line)
        if mc:
            last_ckpt = mc.group(1).strip()
        if "oos_neural" in line or "ETH decision net" in line:
            status = "done"
    return {
        "n_folds": n_folds,
        "fold": fold,
        "n_train": n_train,
        "epochs": epochs,
        "epoch": epoch,
        "seeds": seeds,
        "mean_r": mean_r,
        "points": points,
        "last_ckpt": last_ckpt,
        "status": status,
        "error": error,
    }


def parse_histogram_field(blob: str, group: str) -> list[dict[str, Any]]:
    """Decode one TorchExplorer histogram string into collapsed 1-D bins."""
    out: list[dict[str, Any]] = []
    if not blob:
        return out
    for part in blob.split("|"):
        fields = part.split("!!")
        if len(fields) < 7:
            continue
        try:
            lo, hi = (float(x) for x in fields[2].split("::")[:2])
        except ValueError:
            continue
        collapsed: list[float] | None = None
        for row in fields[6].split(";"):
            if not row.strip():
                continue
            vals = [float(x) for x in row.split("::") if x != ""]
            if not vals:
                continue
            if collapsed is None:
                collapsed = [0.0] * len(vals)
            for i, v in enumerate(vals):
                if i < len(collapsed):
                    collapsed[i] += v
        if not collapsed:
            continue
        out.append(
            {
                "group": group,
                "label": fields[0],
                "min": lo,
                "max": hi,
                "bins": collapsed,
            }
        )
    return out


def summarize_explorer_hists(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep GRPOActor 直接子模块（LSTM / Linear / mu_raw …），丢掉 Input/Output 叶子。"""
    modules: list[dict[str, Any]] = []
    for r in rows:
        if r.get("type") != "nodes":
            continue
        name = str(r.get("nodes:display_name") or "")
        stack = str(r.get("nodes:parent_stack") or "")
        if stack.count(";") != 1 or name in {"Input", "Output", "GRPOActor"}:
            continue
        hists = (
            parse_histogram_field(str(r.get("nodes:input_histograms") or ""), "input")
            + parse_histogram_field(str(r.get("nodes:output_histograms") or ""), "output")
            + parse_histogram_field(str(r.get("nodes:param_histograms") or ""), "params")
        )
        keep = [
            h
            for h in hists
            if not (h["group"] == "params" and "grad" in str(h["label"]).lower())
        ]
        if not keep:
            continue
        modules.append(
            {
                "id": int(r["nodes:id"]),
                "name": name,
                "path": stack.replace("::", "#").replace(";", " › "),
                "hists": keep[:8],
            }
        )
    return {"modules": modules}


def panel_node_ids(rows: list[dict[str, Any]]) -> list[int]:
    """Pick up to 5 modules to auto-pin on TorchExplorer's right-hand columns."""
    order = ("LSTM", "Linear", "mu_raw", "LayerNorm", "Sequential", "Dropout")
    scored: list[tuple[int, int]] = []
    for mod in summarize_explorer_hists(rows)["modules"]:
        name = str(mod["name"])
        rank = next((i for i, key in enumerate(order) if key in name), 99)
        scored.append((rank, int(mod["id"])))
    scored.sort()
    ids = [i for _, i in scored[:5]]
    return (ids + [-1, -1, -1, -1, -1])[:5]


def latest_fold_ckpt(models_dir: Path) -> Path | None:
    """最新写完的 fold checkpoint（忽略 .tmp）。"""
    cands = []
    for p in Path(models_dir).glob("eth_decision_fold*_s*_e*.pt"):
        if p.name.endswith(".tmp"):
            continue
        m = _FOLD_CKPT.search(p.name)
        if not m:
            continue
        cands.append((p.stat().st_mtime, int(m.group(1)), int(m.group(3)), int(m.group(2)), p))
    if not cands:
        return None
    cands.sort()
    return cands[-1][-1]


def load_actor(path: Path) -> tuple[GRPOActor, dict]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    seq = int(blob.get("seq_len") or SEQ_LEN)
    n_feat = int(blob.get("n_feat") or N_FEAT)
    dropout = float(blob.get("dropout", 0.2))
    net = GRPOActor(n_feat=n_feat, seq_len=seq, dropout=dropout)
    states = blob["states"]
    net.load_state_dict(states[0] if isinstance(states, (list, tuple)) else states)
    return net, blob


def attach_tensor(tensor: torch.Tensor, parent: torch.nn.Module, name: str) -> torch.Tensor:
    import torchexplorer

    return torchexplorer.attach(tensor, parent, name)


def _patch_lstm_seq_only(net: torch.nn.Module) -> None:
    """TorchExplorer 的 hook 假定 forward 输出是 Tensor，LSTM 返回 (out, (h,c)) 会炸。"""
    import types

    real = torch.nn.LSTM.forward

    def wrapped(self, input, hx=None):  # noqa: ANN001
        output, _state = real(self, input, hx)
        return output

    for m in net.modules():
        if isinstance(m, torch.nn.LSTM):
            m.forward = types.MethodType(wrapped, m)


def watch_actor(
    net: GRPOActor,
    *,
    out_dir: Path,
    port: int = 8080,
    log_freq: int = 1,
) -> None:
    """只应调用一次。后续用 load_state_dict + probe 更新直方图。"""
    import shutil

    import torchexplorer

    out_dir = Path(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    # StandaloneBackend 只在目录不存在时 copytree。空目录会让 import app 失败。
    if out_dir.exists() and not (out_dir / "app.py").exists():
        shutil.rmtree(out_dir)
    _patch_lstm_seq_only(net)
    net._torchexplore = True  # type: ignore[attr-defined]
    try:
        torchexplorer.watch(
            net,
            log=["io", "io_grad", "params", "params_grad"],
            log_freq=int(log_freq),
            delay_log_multi_backward=False,
            backend="standalone",
            standalone_dir=str(out_dir),
            standalone_port=int(port),
            verbose=True,
        )
    except RuntimeError:
        logger.warning("io_grad failed (likely inplace); watching io+params only")
        torchexplorer.watch(
            net,
            log=["io", "params", "params_grad"],
            log_freq=int(log_freq),
            delay_log_multi_backward=False,
            backend="standalone",
            standalone_dir=str(out_dir),
            standalone_port=int(port),
            verbose=True,
        )
    _patch_standalone_fetch(out_dir)


def _patch_standalone_fetch(out_dir: Path) -> None:
    """Standalone 模板写死 localhost，iframe 用 127.0.0.1 时 fetch 跨域，Vega 一直 Waiting on data。"""
    html = Path(out_dir) / "templates" / "index.html"
    if not html.exists():
        return
    text = html.read_text(encoding="utf-8")
    needle = 'var dataAddress = "http://localhost:" + String(port) + "/data/data.json";'
    if needle in text:
        html.write_text(
            text.replace(needle, 'var dataAddress = "/data/data.json";'),
            encoding="utf-8",
        )
    app_py = Path(out_dir) / "app.py"
    if app_py.exists() and "Access-Control-Allow-Origin" not in app_py.read_text(encoding="utf-8"):
        src = app_py.read_text(encoding="utf-8")
        src = src.replace(
            "app = Flask(__name__)\n",
            'app = Flask(__name__)\napp.config["TEMPLATES_AUTO_RELOAD"] = True\n'
            "app.jinja_env.auto_reload = True\napp.jinja_env.cache = None\n",
        )
        src = src.replace(
            "    return send_from_directory('data', path)\n",
            "    resp = send_from_directory('data', path)\n"
            "    resp.headers['Access-Control-Allow-Origin'] = '*'\n"
            "    resp.headers['Cache-Control'] = 'no-store'\n"
            "    return resp\n",
        )
        app_py.write_text(src, encoding="utf-8")
    _pin_explorer_panels(out_dir, [2, 60, 9, 6, 54])


def _pin_explorer_panels(out_dir: Path, ids: list[int]) -> None:
    """Right-hand Vega columns start empty (Waiting on data) until a node is dragged. Pin modules that already have histograms."""
    vega = Path(out_dir) / "vega_dataless.json"
    if not vega.exists():
        return
    padded = (list(ids) + [-1, -1, -1, -1, -1])[:5]
    parents = [0 if i >= 0 else -1 for i in padded]
    text = vega.read_text(encoding="utf-8")
    text = re.sub(
        r'\{ "name": "panels_node_id_all", "value": \[[^\]]*\]',
        '{ "name": "panels_node_id_all", "value": ' + json.dumps(padded),
        text,
        count=1,
    )
    text = re.sub(
        r'\{ "name": "panels_parents_node_id_all",\s*"value": \[[^\]]*\]',
        '{ "name": "panels_parents_node_id_all",\n      "value": ' + json.dumps(parents),
        text,
        count=1,
    )
    vega.write_text(text, encoding="utf-8")


def _probe_windows(blob: dict, n: int = 32) -> torch.Tensor:
    seq = int(blob.get("seq_len") or SEQ_LEN)
    panel_path = ROOT / "data/cache/panel_binance_ETHUSDT_1h.parquet"
    if panel_path.exists() and blob.get("median") is not None:
        import pandas as pd

        from betatrend.nn.dataset import build_feature_frame, make_windows
        from betatrend.nn.train import _robust_scale

        feats = build_feature_frame(pd.read_parquet(panel_path))
        x = _robust_scale(
            feats.to_numpy(dtype=np.float64),
            np.asarray(blob["median"], dtype=np.float32),
            np.asarray(blob["iqr"], dtype=np.float32),
        )
        win = make_windows(x, seq_len=seq)
        if len(win) > n:
            idx = np.linspace(max(len(win) // 5, 0), len(win) - 1, n, dtype=int)
            return torch.from_numpy(np.ascontiguousarray(win[idx], dtype=np.float32))
    return torch.randn(n, seq, int(blob.get("n_feat") or N_FEAT))


def probe(net: GRPOActor, x: torch.Tensor) -> None:
    """一次 train-mode 前向+反向，推动 TorchExplorer 直方图。"""
    net.train()
    net.zero_grad(set_to_none=True)
    y = net(x)
    y.sum().backward()
    # delay_log 时直方图要等到下一次 forward 才落盘；多推一次保证 data.json 立刻有内容
    with torch.no_grad():
        net(x)


def _serve_board(board_dir: Path, port: int) -> ThreadingHTTPServer:
    board_dir.mkdir(parents=True, exist_ok=True)
    (board_dir / "index.html").write_text(_BOARD_HTML, encoding="utf-8")

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(board_dir), **kwargs)

        def end_headers(self) -> None:
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    httpd = ThreadingHTTPServer(("127.0.0.1", int(port)), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def run_live(
    *,
    log_path: Path,
    models_dir: Path,
    board_dir: Path,
    explorer_dir: Path,
    explorer_port: int = 8080,
    board_port: int = 8081,
    poll: float = 5.0,
) -> None:
    """对着正在跑的 train-nn 起看板。Ctrl-C 退出，不影响训练进程。"""
    log_path = Path(log_path)
    models_dir = Path(models_dir)
    board_dir = Path(board_dir)
    explorer_dir = Path(explorer_dir)
    gv_bin = Path("/opt/homebrew/bin")
    if gv_bin.exists():
        import os

        os.environ["PATH"] = str(gv_bin) + os.pathsep + os.environ.get("PATH", "")
    _serve_board(board_dir, board_port)
    explorer_url = f"http://127.0.0.1:{int(explorer_port)}"
    board_url = f"http://127.0.0.1:{int(board_port)}"
    logger.info("training board {}  |  TorchExplorer {}", board_url, explorer_url)

    net: GRPOActor | None = None
    windows: torch.Tensor | None = None
    last_ckpt: Path | None = None

    while True:
        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        snap = parse_train_log(text)
        ckpt = latest_fold_ckpt(models_dir)
        if ckpt is not None:
            m = _FOLD_CKPT.search(ckpt.name)
            snap["ckpt"] = str(ckpt)
            snap["ckpt_fold"] = int(m.group(1)) + 1 if m else None
            snap["ckpt_seed"] = int(m.group(2)) if m else None
        snap["explorer_url"] = explorer_url
        (board_dir / "metrics.json").write_text(json.dumps(snap), encoding="utf-8")
        data_json = explorer_dir / "data" / "data.json"
        if data_json.exists():
            try:
                rows = json.loads(data_json.read_text(encoding="utf-8"))
                (board_dir / "hists.json").write_text(
                    json.dumps(summarize_explorer_hists(rows)), encoding="utf-8"
                )
            except Exception:
                logger.exception("failed to summarize explorer histograms")

        if ckpt is not None and ckpt != last_ckpt:
            try:
                loaded, blob = load_actor(ckpt)
                fold_i = int(_FOLD_CKPT.search(ckpt.name).group(1)) if _FOLD_CKPT.search(ckpt.name) else 0
                if net is None:
                    net = loaded
                    windows = _probe_windows(blob)
                    watch_actor(net, out_dir=explorer_dir, port=explorer_port, log_freq=1)
                else:
                    net.load_state_dict(loaded.state_dict())
                net._explore_fold = fold_i  # type: ignore[attr-defined]
                assert windows is not None
                probe(net, windows)
                last_ckpt = ckpt
                if data_json.exists():
                    rows = json.loads(data_json.read_text(encoding="utf-8"))
                    _pin_explorer_panels(explorer_dir, panel_node_ids(rows))
                    (board_dir / "hists.json").write_text(
                        json.dumps(summarize_explorer_hists(rows)), encoding="utf-8"
                    )
                logger.info("TorchExplorer probed {}", ckpt.name)
            except Exception:
                logger.exception("failed to probe {}", ckpt)

        time.sleep(float(poll))
