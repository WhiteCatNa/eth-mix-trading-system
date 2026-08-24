use serde::{Deserialize, Serialize};

/// Series-level optional-column flags (column exists, not per-bar non-zero).
#[derive(Debug, Clone, Copy)]
pub struct FeatureFlags {
    pub has_taker: bool,
    pub has_trades: bool,
    pub has_index: bool,
    pub has_oi: bool,
    pub has_lsr: bool,
}

impl Default for FeatureFlags {
    fn default() -> Self {
        Self::all_false()
    }
}

impl FeatureFlags {
    pub fn all_true() -> Self {
        Self {
            has_taker: true,
            has_trades: true,
            has_index: true,
            has_oi: true,
            has_lsr: true,
        }
    }

    pub fn all_false() -> Self {
        Self {
            has_taker: false,
            has_trades: false,
            has_index: false,
            has_oi: false,
            has_lsr: false,
        }
    }

    pub fn from_array(a: [bool; 5]) -> Self {
        Self {
            has_taker: a[0],
            has_trades: a[1],
            has_index: a[2],
            has_oi: a[3],
            has_lsr: a[4],
        }
    }

    pub fn from_bars(bars: &[Bar]) -> Self {
        Self {
            has_taker: bars.iter().any(|b| b.has_taker),
            has_trades: bars.iter().any(|b| b.has_trades),
            has_index: bars.iter().any(|b| b.has_index),
            has_oi: bars.iter().any(|b| b.has_oi),
            has_lsr: bars.iter().any(|b| b.has_lsr),
        }
    }

    pub fn apply(self, bars: &mut [Bar]) {
        for b in bars {
            b.has_taker = self.has_taker;
            b.has_trades = self.has_trades;
            b.has_index = self.has_index;
            b.has_oi = self.has_oi;
            b.has_lsr = self.has_lsr;
        }
    }
}

/// One hourly bar. Optional columns default to 0 / close.
#[derive(Debug, Clone, Default)]
pub struct Bar {
    pub ts_unix: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub funding: f64,
    pub taker_buy_base: f64,
    pub trades: f64,
    pub mark_close: f64,
    pub index_close: f64,
    pub open_interest: f64,
    pub long_short_ratio: f64,
    pub has_taker: bool,
    pub has_trades: bool,
    pub has_index: bool,
    pub has_oi: bool,
    pub has_lsr: bool,
}

impl Bar {
    pub fn from_ohlcv(ts_unix: i64, open: f64, high: f64, low: f64, close: f64, volume: f64, funding: f64) -> Self {
        Self {
            ts_unix,
            open,
            high,
            low,
            close,
            volume,
            funding,
            mark_close: close,
            index_close: close,
            ..Default::default()
        }
    }

    pub fn hour(&self) -> i32 {
        let s = ((self.ts_unix % 86400) + 86400) % 86400;
        (s / 3600) as i32
    }

    /// Python pandas `dayofweek`: Monday=0.
    pub fn dow(&self) -> i32 {
        let days = self.ts_unix.div_euclid(86400);
        ((days + 3).rem_euclid(7)) as i32
    }
}

/// Packed array layout used by PyO3: (n, 13)
/// ts, open, high, low, close, volume, funding, taker, trades, mark, index, oi, lsr
///
/// `flags` is series-level column existence: [taker, trades, index, oi, lsr].
/// If omitted, a column is treated as present when any row is non-zero.
pub fn bars_from_matrix(mat: &[f64], n: usize, flags: Option<[bool; 5]>) -> Vec<Bar> {
    assert_eq!(mat.len(), n * 13);
    let inferred = flags.unwrap_or_else(|| {
        let mut any = [false; 5];
        for i in 0..n {
            let r = &mat[i * 13..(i + 1) * 13];
            any[0] |= r[7].abs() > 0.0;
            any[1] |= r[8].abs() > 0.0;
            any[2] |= r[10].abs() > 0.0;
            any[3] |= r[11].abs() > 0.0;
            any[4] |= r[12].abs() > 0.0;
        }
        any
    });
    let ff = FeatureFlags::from_array(inferred);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let r = &mat[i * 13..(i + 1) * 13];
        let close = r[4];
        out.push(Bar {
            ts_unix: r[0] as i64,
            open: r[1],
            high: r[2],
            low: r[3],
            close,
            volume: r[5],
            funding: r[6],
            taker_buy_base: r[7],
            trades: r[8],
            mark_close: if r[9].abs() > 0.0 { r[9] } else { close },
            index_close: if r[10].abs() > 0.0 { r[10] } else { close },
            open_interest: r[11],
            long_short_ratio: r[12],
            has_taker: ff.has_taker,
            has_trades: ff.has_trades,
            has_index: ff.has_index,
            has_oi: ff.has_oi,
            has_lsr: ff.has_lsr,
        });
    }
    out
}

/// Parse `2023-01-01 00:00:00+00:00` / RFC3339-ish UTC into unix seconds.
pub fn parse_utc_datetime(s: &str) -> Option<i64> {
    let s = s.trim().trim_matches('"');
    let s = s.replace('T', " ");
    let (date, rest) = s.split_once(' ')?;
    let mut ymd = date.split('-');
    let y: i32 = ymd.next()?.parse().ok()?;
    let m: u32 = ymd.next()?.parse().ok()?;
    let d: u32 = ymd.next()?.parse().ok()?;
    let time = rest.split(['+', 'Z']).next().unwrap_or(rest);
    let mut hms = time.split(':');
    let h: i64 = hms.next()?.parse().ok()?;
    let min: i64 = hms.next()?.parse().ok()?;
    let sec_s = hms.next().unwrap_or("0");
    let sec: i64 = sec_s.split('.').next()?.parse().ok()?;
    Some(days_from_civil(y, m, d) * 86400 + h * 3600 + min * 60 + sec)
}

fn days_from_civil(y: i32, m: u32, d: u32) -> i64 {
    let mut y = y;
    let m = m as i32;
    let d = d as i32;
    y -= i32::from(m <= 2);
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u32;
    let mp = if m > 2 { m - 3 } else { m + 9 } as u32;
    let doy = (153 * mp + 2) / 5 + d as u32 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era as i64 * 146097 + doe as i64 - 719468
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ResetCfg {
    pub symbol: String,
    pub start: usize,
    pub end: Option<usize>,
    pub cost: f64,
    pub eta: f64,
    pub dd_inc: f64,
    pub dd_level: f64,
    pub clip: f64,
    pub so_w: f64,
    pub seq_len: usize,
    pub fold_id: i64,
    pub seed: u64,
    pub exec_mode: String,
    pub fee_rate: f64,
    pub slip_bps: f64,
    pub funding_interval_hours: i32,
    pub initial_equity: f64,
    pub target_vol: f64,
    pub max_leverage: f64,
    pub risk_budget: f64,
    pub turnover_band_equity: f64,
    pub train_end: Option<usize>,
}

impl Default for ResetCfg {
    fn default() -> Self {
        Self {
            symbol: "ETHUSDT".into(),
            start: 0,
            end: None,
            cost: 0.0008,
            eta: 1.0 / 72.0,
            dd_inc: 1.0,
            dd_level: 0.05,
            clip: 5.0,
            so_w: 1.0,
            seq_len: 7,
            fold_id: 0,
            seed: 0,
            exec_mode: "overlay".into(),
            fee_rate: 0.0005,
            slip_bps: 1.5,
            funding_interval_hours: 8,
            initial_equity: 100_000.0,
            target_vol: 0.20,
            max_leverage: 2.0,
            risk_budget: 1.0,
            turnover_band_equity: 0.015,
            train_end: None,
        }
    }
}

pub fn execution_aligned_y(opens: &[f64]) -> Vec<f64> {
    let n = opens.len();
    let mut y = vec![0.0; n];
    for t in 0..n {
        let i1 = t + 1;
        let i2 = t + 2;
        if i2 < n && opens[i1] > 0.0 {
            y[t] = opens[i2] / opens[i1] - 1.0;
        }
    }
    y
}
