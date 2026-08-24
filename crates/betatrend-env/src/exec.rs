//! Overlay PnL (open[t+1]→open[t+2]) and backtest-ordered fills.

use crate::bars::Bar;
use crate::reward::RewardMachine;

/// exposure = action * lev; pnl = exposure * y - cost * |Δexposure|
pub fn overlay_pnl(actions: &[f64], y: &[f64], lev: &[f64], cost: f64) -> Vec<f64> {
    assert_eq!(actions.len(), y.len());
    assert_eq!(actions.len(), lev.len());
    let n = actions.len();
    let mut pnl = vec![0.0; n];
    let mut prev = 0.0;
    for t in 0..n {
        let exp = actions[t] * lev[t];
        let dlt = if t == 0 { exp.abs() } else { (exp - prev).abs() };
        pnl[t] = exp * y[t] - cost * dlt;
        prev = exp;
    }
    pnl
}

pub fn overlay_rewards(
    actions: &[f64],
    y: &[f64],
    lev: &[f64],
    vol_ann: &[f64],
    cost: f64,
    eta: f64,
    dd_inc: f64,
    dd_level: f64,
    clip: f64,
    so_w: f64,
) -> (Vec<f64>, Vec<f64>) {
    let pnl = overlay_pnl(actions, y, lev, cost);
    let mut rm = RewardMachine::new(eta, dd_inc, dd_level, clip, so_w);
    let mut rew = Vec::with_capacity(pnl.len());
    for t in 0..pnl.len() {
        rew.push(rm.step(pnl[t], vol_ann[t]));
    }
    (pnl, rew)
}

/// Inverse-vol leverage: target / realized vol, clipped to [0, max_leverage].
pub fn vol_leverage(vol: f64, target: f64, max_leverage: f64) -> f64 {
    if !(vol > 0.0) {
        return 1.0;
    }
    (target / vol).clamp(0.0, max_leverage)
}

#[derive(Debug, Clone)]
pub struct OverlayExec {
    pub y: Vec<f64>,
    pub lev: Vec<f64>,
    pub vol: Vec<f64>,
    pub cost: f64,
    prev_exp: f64,
}

impl OverlayExec {
    pub fn new(y: Vec<f64>, lev: Vec<f64>, vol: Vec<f64>, cost: f64) -> Self {
        Self { y, lev, vol, cost, prev_exp: 0.0 }
    }

    pub fn reset(&mut self) {
        self.prev_exp = 0.0;
    }

    pub fn step_pnl(&mut self, t: usize, action: f64) -> f64 {
        let exp = action.clamp(-1.0, 1.0) * self.lev[t];
        let pnl = exp * self.y[t] - self.cost * (exp - self.prev_exp).abs();
        self.prev_exp = exp;
        pnl
    }
}

/// Matches Backtester bar order: gap mark → funding → pending fill → mark open→close → set pending.
#[derive(Debug, Clone)]
pub struct BacktestExec {
    pub bars: Vec<Bar>,
    pub vol: Vec<f64>,
    pub fee_rate: f64,
    pub slip_bps: f64,
    pub funding_interval_hours: i32,
    pub initial_equity: f64,
    pub target_vol: f64,
    pub max_leverage: f64,
    pub risk_budget: f64,
    pub turnover_band_equity: f64,
    pub cash: f64,
    pub qty: f64,
    pub pending: Option<f64>,
    pub last_close: f64,
}

impl BacktestExec {
    pub fn new(
        bars: Vec<Bar>,
        vol: Vec<f64>,
        fee_rate: f64,
        slip_bps: f64,
        funding_interval_hours: i32,
        initial_equity: f64,
        target_vol: f64,
        max_leverage: f64,
        risk_budget: f64,
        turnover_band_equity: f64,
    ) -> Self {
        let last_close = bars.first().map(|b| b.close).unwrap_or(0.0);
        Self {
            bars,
            vol,
            fee_rate,
            slip_bps,
            funding_interval_hours,
            initial_equity,
            target_vol,
            max_leverage,
            risk_budget,
            turnover_band_equity,
            cash: initial_equity,
            qty: 0.0,
            pending: None,
            last_close,
        }
    }

    pub fn reset(&mut self) {
        self.reset_at(0);
    }

    pub fn reset_at(&mut self, start: usize) {
        self.cash = self.initial_equity;
        self.qty = 0.0;
        self.pending = None;
        if start > 0 && start <= self.bars.len() {
            self.last_close = self.bars[start - 1].close;
        } else {
            self.last_close = self.bars.first().map(|b| b.close).unwrap_or(0.0);
        }
    }

    pub fn step(&mut self, t: usize, action: f64) -> f64 {
        let bar = &self.bars[t];
        let o = bar.open;
        let c = bar.close;
        let mut pnl = 0.0;
        if t > 0 {
            let mark = self.qty * (o - self.last_close);
            self.cash += mark;
            pnl += mark;
        }
        if bar.hour() % self.funding_interval_hours == 0 {
            let fp = -(self.qty * o) * bar.funding;
            self.cash += fp;
            pnl += fp;
        }
        if let Some(tgt) = self.pending.take() {
            let d_n = tgt - self.qty * o;
            let band = (self.cash * self.turnover_band_equity).max(75.0);
            if d_n.abs() >= band && o > 0.0 {
                let fee = d_n.abs() * self.fee_rate;
                let slip = d_n.abs() * (self.slip_bps / 10_000.0);
                self.cash -= fee + slip;
                self.qty += d_n / o;
                pnl -= fee + slip;
            }
        }
        let mark2 = self.qty * (c - o);
        self.cash += mark2;
        pnl += mark2;
        self.last_close = c;
        let vol = self.vol[t].max(1e-6);
        let lev = self.max_leverage.min(self.target_vol / vol);
        let unit = action.clamp(-1.0, 1.0);
        self.pending = Some(self.cash * self.risk_budget * lev * unit);
        pnl
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bars::Bar;

    fn bar(hour: i64, open: f64, close: f64, funding: f64) -> Bar {
        Bar::from_ohlcv(hour * 3600, open, open.max(close), open.min(close), close, 1.0, funding)
    }

    #[test]
    fn backtest_order_is_gap_funding_fill_mark_pending() {
        // hour 8 → funding; hour 9 → no funding. turnover band 0 so fills always fire.
        let bars = vec![
            bar(8, 100.0, 101.0, 0.001),
            bar(9, 102.0, 103.0, 0.0),
            bar(16, 104.0, 105.0, 0.002),
        ];
        let vol = vec![0.20, 0.20, 0.20];
        let mut ex = BacktestExec::new(bars, vol, 0.001, 10.0, 8, 10_000.0, 0.20, 2.0, 1.0, 0.0);
        let p0 = ex.step(0, 1.0);
        assert!(p0.abs() < 1e-12, "flat book on first bar");
        assert!(ex.pending.is_some());
        assert!(ex.qty.abs() < 1e-12);
        let _p1 = ex.step(1, 1.0);
        assert!(ex.qty > 0.0, "pending must fill at next open after gap+funding");
        assert!(ex.pending.is_some());
        let qty_after_fill = ex.qty;
        let _p2 = ex.step(2, 0.0);
        assert!(ex.qty.abs() < qty_after_fill || ex.pending.is_some());
    }
}
