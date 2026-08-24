//! Stepwise differential-Sharpe minus drawdown penalty. Matches Python `RewardMachine`.

use crate::spec::{BARS_PER_YEAR, PNL_CLIP, R_VOL_CLIP};

#[derive(Debug, Clone)]
pub struct RewardMachine {
    pub eta: f64,
    pub dd_inc: f64,
    pub dd_level: f64,
    pub clip: f64,
    pub bars_per_year: f64,
    a: f64,
    b: f64,
    equity: f64,
    peak: f64,
    prev_depth: f64,
}

impl RewardMachine {
    pub fn new(eta: f64, dd_inc: f64, dd_level: f64, clip: f64) -> Self {
        Self {
            eta: eta.clamp(1e-4, 0.5),
            dd_inc,
            dd_level,
            clip,
            bars_per_year: BARS_PER_YEAR,
            a: 0.0,
            b: 1.0,
            equity: 1.0,
            peak: 1.0,
            prev_depth: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.a = 0.0;
        self.b = 1.0;
        self.equity = 1.0;
        self.peak = 1.0;
        self.prev_depth = 0.0;
    }

    pub fn step(&mut self, pnl: f64, vol_ann: f64) -> f64 {
        let hourly_vol = (vol_ann / self.bars_per_year.sqrt()).max(1e-6);
        let rt = (pnl / hourly_vol).clamp(-R_VOL_CLIP, R_VOL_CLIP);
        let d_a = rt - self.a;
        let d_b = rt * rt - self.b;
        let var = (self.b - self.a * self.a).max(1e-8);
        let d_sharpe = (self.b * d_a - 0.5 * self.a * d_b) / var.powf(1.5);
        self.a += self.eta * d_a;
        self.b += self.eta * d_b;
        self.equity *= 1.0 + pnl.clamp(-PNL_CLIP, PNL_CLIP);
        self.peak = self.peak.max(self.equity);
        let depth = (self.peak - self.equity) / self.peak.max(1e-12);
        let deepen = (depth - self.prev_depth).max(0.0);
        self.prev_depth = depth;
        let dd_pen = self.dd_inc * deepen + self.dd_level * depth;
        (d_sharpe - dd_pen).clamp(-self.clip, self.clip)
    }
}
