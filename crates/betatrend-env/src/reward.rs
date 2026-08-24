//! Stepwise mean − downside variance − optional drawdown. Matches Python `RewardMachine`.

use crate::spec::{BARS_PER_YEAR, PNL_CLIP, R_VOL_CLIP};

#[derive(Debug, Clone)]
pub struct RewardMachine {
    pub down_lambda: f64,
    pub dd_inc: f64,
    pub dd_level: f64,
    pub clip: f64,
    pub bars_per_year: f64,
    equity: f64,
    peak: f64,
    prev_depth: f64,
}

impl RewardMachine {
    pub fn new(down_lambda: f64, dd_inc: f64, dd_level: f64, clip: f64) -> Self {
        Self {
            down_lambda,
            dd_inc,
            dd_level,
            clip,
            bars_per_year: BARS_PER_YEAR,
            equity: 1.0,
            peak: 1.0,
            prev_depth: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.equity = 1.0;
        self.peak = 1.0;
        self.prev_depth = 0.0;
    }

    pub fn step(&mut self, pnl: f64, vol_ann: f64) -> f64 {
        let hourly_vol = (vol_ann / self.bars_per_year.sqrt()).max(1e-6);
        let rt = (pnl / hourly_vol).clamp(-R_VOL_CLIP, R_VOL_CLIP);
        let down = rt.min(0.0);
        self.equity *= 1.0 + pnl.clamp(-PNL_CLIP, PNL_CLIP);
        self.peak = self.peak.max(self.equity);
        let depth = (self.peak - self.equity) / self.peak.max(1e-12);
        let deepen = (depth - self.prev_depth).max(0.0);
        self.prev_depth = depth;
        let dd_pen = self.dd_inc * deepen + self.dd_level * depth;
        (rt - self.down_lambda * down * down - dd_pen).clamp(-self.clip, self.clip)
    }
}
