//! Golden-file contract: Rust overlay env vs Python `dump_golden` CSV.

use betatrend_env::bars::{parse_utc_datetime, Bar, ResetCfg};
use betatrend_env::exec::overlay_rewards;
use betatrend_env::features::{compute_features, FEATURE_NAMES};
use betatrend_env::env::{OverlayEnv, TradingEnv};
use betatrend_env::{N_FEAT, SEQ_LEN};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("golden")
}

fn split_csv_line(line: &str) -> Vec<String> {
    line.split(',').map(|s| s.trim().to_string()).collect()
}

fn parse_f64(s: &str) -> f64 {
    s.parse().unwrap_or(f64::NAN)
}

fn load_bars() -> Vec<Bar> {
    let raw = fs::read_to_string(golden_dir().join("bars.csv")).expect("bars.csv");
    let mut lines = raw.lines();
    let header = split_csv_line(lines.next().expect("header"));
    let col = |name: &str| header.iter().position(|h| h == name);
    let i_open = col("open").expect("open");
    let i_high = col("high").expect("high");
    let i_low = col("low").expect("low");
    let i_close = col("close").expect("close");
    let i_vol = col("volume").expect("volume");
    let i_trades = col("trades");
    let i_taker = col("taker_buy_base");
    let i_fund = col("funding_rate");
    let i_mark = col("mark_close");
    let i_index = col("index_close");
    let i_oi = col("open_interest");
    let i_lsr = col("long_short_ratio");
    let mut bars = Vec::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let c = split_csv_line(line);
        let ts = parse_utc_datetime(&c[0]).unwrap_or(0);
        let close = parse_f64(&c[i_close]);
        bars.push(Bar {
            ts_unix: ts,
            open: parse_f64(&c[i_open]),
            high: parse_f64(&c[i_high]),
            low: parse_f64(&c[i_low]),
            close,
            volume: parse_f64(&c[i_vol]),
            funding: i_fund.map(|i| parse_f64(&c[i])).unwrap_or(0.0),
            taker_buy_base: i_taker.map(|i| parse_f64(&c[i])).unwrap_or(0.0),
            trades: i_trades.map(|i| parse_f64(&c[i])).unwrap_or(0.0),
            mark_close: i_mark.map(|i| parse_f64(&c[i])).unwrap_or(close),
            index_close: i_index.map(|i| parse_f64(&c[i])).unwrap_or(close),
            open_interest: i_oi.map(|i| parse_f64(&c[i])).unwrap_or(0.0),
            long_short_ratio: i_lsr.map(|i| parse_f64(&c[i])).unwrap_or(0.0),
            has_taker: i_taker.is_some(),
            has_trades: i_trades.is_some(),
            has_index: i_index.is_some(),
            has_oi: i_oi.is_some(),
            has_lsr: i_lsr.is_some(),
        });
    }
    bars
}

fn load_steps() -> (Vec<String>, Vec<HashMap<String, f64>>) {
    let raw = fs::read_to_string(golden_dir().join("steps.csv")).expect("steps.csv");
    let mut lines = raw.lines();
    let header = split_csv_line(lines.next().expect("header"));
    let mut rows = Vec::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let vals = split_csv_line(line);
        let mut m = HashMap::new();
        for (h, v) in header.iter().zip(vals.iter()) {
            m.insert(h.clone(), parse_f64(v));
        }
        rows.push(m);
    }
    (header, rows)
}

fn max_abs(a: f64, b: f64) -> f64 {
    (a - b).abs()
}

#[test]
fn golden_overlay_pnl_and_reward() {
    let cfg: ResetCfg = serde_json::from_str(
        &fs::read_to_string(golden_dir().join("reset_cfg.json")).unwrap(),
    )
    .unwrap();
    let (_hdr, rows) = load_steps();
    assert_eq!(rows.len(), 218);
    let actions: Vec<f64> = rows.iter().map(|r| r["action"]).collect();
    let y: Vec<f64> = rows.iter().map(|r| r["y"]).collect();
    let lev: Vec<f64> = rows.iter().map(|r| r["lev"]).collect();
    let vol: Vec<f64> = rows.iter().map(|r| r["vol"]).collect();
    let (pnl, rew) = overlay_rewards(
        &actions,
        &y,
        &lev,
        &vol,
        cfg.cost,
        cfg.down_lambda,
        cfg.dd_inc,
        cfg.dd_level,
        cfg.clip,
    );
    let mut max_pnl: f64 = 0.0;
    let mut max_rew: f64 = 0.0;
    for (i, row) in rows.iter().enumerate() {
        max_pnl = max_pnl.max(max_abs(pnl[i], row["pnl"]));
        max_rew = max_rew.max(max_abs(rew[i], row["reward"]));
    }
    assert!(max_pnl < 1e-6, "pnl err {max_pnl}");
    assert!(max_rew < 1e-6, "reward err {max_rew}");
}

#[test]
fn golden_features_match_x_columns() {
    let bars = load_bars();
    let flags = Some([true, true, true, true, true]);
    let x = compute_features(&bars, flags);
    let (_hdr, rows) = load_steps();
    let mut worst = 0.0;
    let mut where_ = String::new();
    for row in &rows {
        let t = row["t"] as usize;
        for (j, name) in FEATURE_NAMES.iter().enumerate() {
            let key = format!("x_{name}");
            let got = x[t][j];
            let exp = row[&key];
            let d = max_abs(got, exp);
            if d > worst {
                worst = d;
                where_ = format!("t={t} {name} got={got} exp={exp}");
            }
        }
    }
    assert!(worst < 1e-6, "feature err {worst} at {where_}");
}

#[test]
fn golden_overlay_env_stepwise() {
    let bars = load_bars();
    let cfg: ResetCfg = serde_json::from_str(
        &fs::read_to_string(golden_dir().join("reset_cfg.json")).unwrap(),
    )
    .unwrap();
    let (_hdr, rows) = load_steps();
    let mut env = OverlayEnv::from_bars(bars, cfg.clone(), Some([true; 5]));
    let mut obs = env.reset(cfg);
    assert_eq!(obs.len(), SEQ_LEN * N_FEAT);
    let mut max_pnl: f64 = 0.0;
    let mut max_rew: f64 = 0.0;
    let mut max_obs: f64 = 0.0;
    let mut max_x: f64 = 0.0;
    for row in &rows {
        let t = row["t"] as usize;
        for (j, name) in FEATURE_NAMES.iter().enumerate() {
            let key = format!("x_{name}");
            max_x = max_x.max(max_abs(env.x()[t][j], row[&key]));
        }
        for j in 0..obs.len() {
            let key = format!("obs_{j}");
            max_obs = max_obs.max(max_abs(obs[j] as f64, row[&key]));
        }
        let step = env.step(row["action"] as f32);
        max_pnl = max_pnl.max(max_abs(step.pnl, row["pnl"]));
        max_rew = max_rew.max(max_abs(step.reward, row["reward"]));
        assert_eq!(step.done, row["done"] > 0.5);
        obs = step.obs;
    }
    assert!(max_x < 1e-6, "x err {max_x}");
    assert!(max_pnl < 1e-6, "pnl err {max_pnl}");
    assert!(max_rew < 1e-6, "reward err {max_rew}");
    assert!(max_obs < 1e-6, "obs err {max_obs}");
}
