//! JSON-lines env worker. Protocol matches stage-0 `env_spec` / `ResetCfg`.
//!
//! stdin/stdout JSON lines. Same-host Unix socket and cross-host gRPC are reserved
//! for a later cut; field names stay identical so PyO3 is the local implementation
//! of this protocol.
//!
//! Ops: `spec`, `reset`, `step`, `last_window`, `overlay_rewards`.

use betatrend_env::bars::{bars_from_matrix, ResetCfg};
use betatrend_env::env::{OverlayEnv, TradingEnv};
use betatrend_env::exec::overlay_rewards;
use betatrend_env::features::last_window;
use betatrend_env::spec::default_spec;
use serde::Deserialize;
use std::io::{self, BufRead, Write};

#[derive(Debug, Deserialize)]
struct Msg {
    op: String,
    #[serde(default)]
    cfg: Option<ResetCfg>,
    #[serde(default)]
    bars: Option<Vec<Vec<f64>>>,
    #[serde(default)]
    flags: Option<Vec<bool>>,
    #[serde(default)]
    action: Option<f64>,
    #[serde(default)]
    seq_len: Option<usize>,
    #[serde(default)]
    actions: Option<Vec<f64>>,
    #[serde(default)]
    y: Option<Vec<f64>>,
    #[serde(default)]
    lev: Option<Vec<f64>>,
    #[serde(default)]
    vol: Option<Vec<f64>>,
    #[serde(default)]
    cost: Option<f64>,
    #[serde(default)]
    eta: Option<f64>,
    #[serde(default)]
    dd_inc: Option<f64>,
    #[serde(default)]
    dd_level: Option<f64>,
    #[serde(default)]
    clip: Option<f64>,
}

fn flags5(v: Option<Vec<bool>>) -> Option<[bool; 5]> {
    v.and_then(|x| {
        if x.len() == 5 {
            Some([x[0], x[1], x[2], x[3], x[4]])
        } else {
            None
        }
    })
}

fn flatten(mat: &[Vec<f64>]) -> (Vec<f64>, usize) {
    let n = mat.len();
    let mut flat = Vec::with_capacity(n * 13);
    for row in mat {
        flat.extend_from_slice(row);
    }
    (flat, n)
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut env: Option<OverlayEnv> = None;
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(s) => s,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }
        let msg: Msg = match serde_json::from_str(&line) {
            Ok(m) => m,
            Err(e) => {
                let _ = writeln!(stdout, "{}", serde_json::json!({"error": e.to_string()}));
                let _ = stdout.flush();
                continue;
            }
        };
        let resp = match msg.op.as_str() {
            "spec" => serde_json::to_value(default_spec()).unwrap_or_else(|_| serde_json::json!({})),
            "reset" => match msg.bars {
                None => serde_json::json!({"error": "reset requires bars"}),
                Some(bars) => {
                    let cfg = msg.cfg.unwrap_or_default();
                    let (flat, n) = flatten(&bars);
                    if n == 0 || flat.len() != n * 13 {
                        serde_json::json!({"error": "bars must be (n, 13)"})
                    } else {
                        let f = flags5(msg.flags);
                        let b = bars_from_matrix(&flat, n, f);
                        let mut e = OverlayEnv::from_bars(b, cfg.clone(), f);
                        let obs = e.reset(cfg);
                        env = Some(e);
                        serde_json::json!({"obs": obs})
                    }
                }
            },
            "step" => match env.as_mut() {
                Some(e) => {
                    let s = e.step(msg.action.unwrap_or(0.0) as f32);
                    serde_json::json!({
                        "obs": s.obs,
                        "reward": s.reward,
                        "done": s.done,
                        "pnl": s.pnl,
                        "t": s.t
                    })
                }
                None => serde_json::json!({"error": "reset first"}),
            },
            "last_window" => match msg.bars {
                None => serde_json::json!({"error": "last_window requires bars"}),
                Some(bars) => {
                    let (flat, n) = flatten(&bars);
                    let f = flags5(msg.flags);
                    let b = bars_from_matrix(&flat, n, f);
                    let w = last_window(&b, msg.seq_len.unwrap_or(7), f);
                    serde_json::json!({"obs": w})
                }
            },
            "overlay_rewards" => match (msg.actions, msg.y, msg.lev, msg.vol) {
                (Some(actions), Some(y), Some(lev), Some(vol)) => {
                    let (pnl, rew) = overlay_rewards(
                        &actions,
                        &y,
                        &lev,
                        &vol,
                        msg.cost.unwrap_or(0.0008),
                        msg.eta.unwrap_or(1.0 / 72.0),
                        msg.dd_inc.unwrap_or(1.0),
                        msg.dd_level.unwrap_or(0.05),
                        msg.clip.unwrap_or(5.0),
                    );
                    serde_json::json!({"pnl": pnl, "reward": rew})
                }
                _ => serde_json::json!({"error": "overlay_rewards requires actions,y,lev,vol"}),
            },
            other => serde_json::json!({
                "error": format!("unknown op {other}"),
                "reserved": "grpc_unix_socket"
            }),
        };
        let _ = writeln!(stdout, "{resp}");
        let _ = stdout.flush();
    }
}
