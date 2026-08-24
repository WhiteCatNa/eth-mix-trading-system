//! Gym-like overlay / backtest environments. Matches Python `OverlayEnv` / `BacktestEnv`.

use crate::bars::{execution_aligned_y, Bar};
use crate::exec::{vol_leverage, BacktestExec, OverlayExec};
use crate::features::{compute_features, make_windows};
use crate::reward::RewardMachine;
use crate::scaler::Scaler;
use crate::spec::{default_spec, FEAT_CLIP, N_FEAT};

pub use crate::bars::ResetCfg;
pub use crate::spec::EnvSpec;

const VOL_24: usize = 6;

#[derive(Debug, Clone)]
pub struct Step {
    pub obs: Vec<f32>,
    pub reward: f64,
    pub done: bool,
    pub pnl: f64,
    pub t: usize,
}

pub trait TradingEnv {
    fn spec(&self) -> EnvSpec;
    fn reset(&mut self, cfg: ResetCfg) -> Vec<f32>;
    fn step(&mut self, action: f32) -> Step;
    fn rollout_ready(&self) -> bool;
}

#[derive(Debug, Clone)]
pub struct OverlayEnv {
    bars: Vec<Bar>,
    cfg: ResetCfg,
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    vol: Vec<f64>,
    lev: Vec<f64>,
    scaler: Scaler,
    windows: Vec<Vec<f32>>,
    exec: OverlayExec,
    rm: RewardMachine,
    t: usize,
    end: usize,
}

impl OverlayEnv {
    pub fn from_bars(bars: Vec<Bar>, cfg: ResetCfg, flags: Option<[bool; 5]>) -> Self {
        let x = compute_features(&bars, flags);
        let opens: Vec<f64> = bars.iter().map(|b| b.open).collect();
        let y = execution_aligned_y(&opens);
        let n = x.len();
        let vol: Vec<f64> = (0..n)
            .map(|t| {
                if t < x.len() && x[t].len() > VOL_24 {
                    x[t][VOL_24]
                } else {
                    0.0
                }
            })
            .collect();
        let lev: Vec<f64> = vol
            .iter()
            .map(|v| vol_leverage(*v, cfg.target_vol, cfg.max_leverage))
            .collect();
        let scaler = Scaler::fit(&x);
        let exec = OverlayExec::new(y.clone(), lev.clone(), vol.clone(), cfg.cost);
        let rm = RewardMachine::new(cfg.eta, cfg.dd_inc, cfg.dd_level, cfg.clip);
        let mut env = Self {
            bars,
            cfg,
            x,
            y,
            vol,
            lev,
            scaler,
            windows: vec![],
            exec,
            rm,
            t: 0,
            end: n,
        };
        env.rebuild_windows();
        env
    }

    fn rebuild_windows(&mut self) {
        let seq = self.cfg.seq_len.max(1);
        let scaled: Vec<Vec<f64>> = self
            .x
            .iter()
            .map(|row| {
                self.scaler
                    .transform_clip(row, FEAT_CLIP)
                    .into_iter()
                    .map(|v| v as f64)
                    .collect()
            })
            .collect();
        self.windows = make_windows(&scaled, seq);
    }

    fn fit_scaler(&mut self) {
        if let Some(te) = self.cfg.train_end {
            let te = te.min(self.x.len());
            if te > 0 {
                self.scaler = Scaler::fit(&self.x[..te]);
                return;
            }
        }
        self.scaler = Scaler::fit(&self.x);
    }

    fn window_at(&self, t: usize) -> Vec<f32> {
        if self.windows.is_empty() {
            return vec![0.0; self.cfg.seq_len.max(1) * N_FEAT];
        }
        let i = t.min(self.windows.len() - 1);
        self.windows[i].clone()
    }

    pub fn x(&self) -> &[Vec<f64>] {
        &self.x
    }

    pub fn y(&self) -> &[f64] {
        &self.y
    }

    pub fn lev(&self) -> &[f64] {
        &self.lev
    }

    pub fn vol(&self) -> &[f64] {
        &self.vol
    }

    pub fn bars(&self) -> &[Bar] {
        &self.bars
    }

    /// Fit scaler on an arbitrary train-index set (walk-forward fold).
    pub fn reset_fold(&mut self, train_idx: &[usize]) -> Vec<f32> {
        let rows: Vec<Vec<f64>> = train_idx
            .iter()
            .copied()
            .filter(|i| *i < self.x.len())
            .map(|i| self.x[i].clone())
            .collect();
        if !rows.is_empty() {
            self.scaler = Scaler::fit(&rows);
        }
        self.rebuild_windows();
        self.exec.reset();
        self.rm.reset();
        self.t = self.cfg.start;
        self.end = self.cfg.end.unwrap_or_else(|| self.x.len().saturating_sub(2).max(self.t));
        self.window_at(self.t)
    }

    pub fn obs_batch(&self) -> &[Vec<f32>] {
        &self.windows
    }
}

impl TradingEnv for OverlayEnv {
    fn spec(&self) -> EnvSpec {
        default_spec()
    }

    fn reset(&mut self, cfg: ResetCfg) -> Vec<f32> {
        self.cfg = cfg;
        self.lev = self
            .vol
            .iter()
            .map(|v| vol_leverage(*v, self.cfg.target_vol, self.cfg.max_leverage))
            .collect();
        self.exec = OverlayExec::new(self.y.clone(), self.lev.clone(), self.vol.clone(), self.cfg.cost);
        self.rm = RewardMachine::new(self.cfg.eta, self.cfg.dd_inc, self.cfg.dd_level, self.cfg.clip);
        self.fit_scaler();
        self.rebuild_windows();
        self.exec.reset();
        self.rm.reset();
        self.t = self.cfg.start;
        self.end = self
            .cfg
            .end
            .unwrap_or_else(|| self.x.len().saturating_sub(2).max(self.t));
        self.window_at(self.t)
    }

    fn step(&mut self, action: f32) -> Step {
        let t = self.t.min(self.y.len().saturating_sub(1));
        let pnl = self.exec.step_pnl(t, action as f64);
        let rew = self.rm.step(pnl, self.vol[t]);
        self.t = self.t.saturating_add(1);
        let done = self.t >= self.end;
        let obs = self.window_at(self.t.min(self.x.len().saturating_sub(1)));
        Step {
            obs,
            reward: rew,
            done,
            pnl,
            t: self.t,
        }
    }

    fn rollout_ready(&self) -> bool {
        self.end > self.t && self.t < self.x.len()
    }
}

#[derive(Debug, Clone)]
pub struct BacktestEnv {
    cfg: ResetCfg,
    x: Vec<Vec<f64>>,
    vol: Vec<f64>,
    scaler: Scaler,
    windows: Vec<Vec<f32>>,
    exec: BacktestExec,
    rm: RewardMachine,
    t: usize,
    end: usize,
}

impl BacktestEnv {
    pub fn from_bars(bars: Vec<Bar>, cfg: ResetCfg, flags: Option<[bool; 5]>) -> Self {
        let x = compute_features(&bars, flags);
        let n = x.len();
        let vol: Vec<f64> = (0..n)
            .map(|t| {
                if t < x.len() && x[t].len() > VOL_24 {
                    x[t][VOL_24]
                } else {
                    0.0
                }
            })
            .collect();
        let scaler = Scaler::fit(&x);
        let exec = BacktestExec::new(
            bars,
            vol.clone(),
            cfg.fee_rate,
            cfg.slip_bps,
            cfg.funding_interval_hours,
            cfg.initial_equity,
            cfg.target_vol,
            cfg.max_leverage,
            cfg.risk_budget,
            cfg.turnover_band_equity,
        );
        let rm = RewardMachine::new(cfg.eta, cfg.dd_inc, cfg.dd_level, cfg.clip);
        let mut env = Self {
            cfg,
            x,
            vol,
            scaler,
            windows: vec![],
            exec,
            rm,
            t: 0,
            end: n,
        };
        env.rebuild_windows();
        env
    }

    fn rebuild_windows(&mut self) {
        let seq = self.cfg.seq_len.max(1);
        let scaled: Vec<Vec<f64>> = self
            .x
            .iter()
            .map(|row| {
                self.scaler
                    .transform_clip(row, FEAT_CLIP)
                    .into_iter()
                    .map(|v| v as f64)
                    .collect()
            })
            .collect();
        self.windows = make_windows(&scaled, seq);
    }

    fn window_at(&self, t: usize) -> Vec<f32> {
        if self.windows.is_empty() {
            return vec![0.0; self.cfg.seq_len.max(1) * N_FEAT];
        }
        let i = t.min(self.windows.len() - 1);
        self.windows[i].clone()
    }
}

impl TradingEnv for BacktestEnv {
    fn spec(&self) -> EnvSpec {
        default_spec()
    }

    fn reset(&mut self, cfg: ResetCfg) -> Vec<f32> {
        self.cfg = cfg;
        self.rm = RewardMachine::new(self.cfg.eta, self.cfg.dd_inc, self.cfg.dd_level, self.cfg.clip);
        self.exec.fee_rate = self.cfg.fee_rate;
        self.exec.slip_bps = self.cfg.slip_bps;
        self.exec.funding_interval_hours = self.cfg.funding_interval_hours;
        self.exec.initial_equity = self.cfg.initial_equity;
        self.exec.target_vol = self.cfg.target_vol;
        self.exec.max_leverage = self.cfg.max_leverage;
        self.exec.risk_budget = self.cfg.risk_budget;
        self.exec.turnover_band_equity = self.cfg.turnover_band_equity;
        self.t = self.cfg.start;
        self.end = self.cfg.end.unwrap_or(self.x.len());
        self.exec.reset_at(self.t);
        self.rm.reset();
        if let Some(te) = self.cfg.train_end {
            let te = te.min(self.x.len());
            if te > 0 {
                self.scaler = Scaler::fit(&self.x[..te]);
                self.rebuild_windows();
            }
        }
        self.window_at(self.t)
    }

    fn step(&mut self, action: f32) -> Step {
        let t = self.t.min(self.exec.bars.len().saturating_sub(1));
        let pnl = self.exec.step(t, action as f64);
        let rew = self.rm.step(pnl / self.cfg.initial_equity.max(1.0), self.vol[t]);
        self.t = self.t.saturating_add(1);
        let done = self.t >= self.end;
        let obs = self.window_at(self.t.min(self.x.len().saturating_sub(1)));
        Step {
            obs,
            reward: rew,
            done,
            pnl,
            t: self.t,
        }
    }

    fn rollout_ready(&self) -> bool {
        self.t < self.end
    }
}
