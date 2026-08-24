//! Causal features matching Python `build_feature_frame` (pandas rolling / ewm).
//!
//! Rolling std uses ddof=1. EWM uses span with `adjust=False` (`alpha = 2/(span+1)`).
//! Optional columns are gated by series-level flags, not per-bar non-zero.

use crate::bars::{Bar, FeatureFlags};

pub use crate::spec::{FEATURE_NAMES, N_FEAT, SEQ_LEN};

pub const MAX_LOOKBACK: usize = 168;
pub const TAIL_BARS: usize = MAX_LOOKBACK + 48;
const ANN: f64 = 24.0 * 365.0;

fn finite(x: f64) -> bool {
    x.is_finite()
}

fn nz(x: f64, fill: f64) -> f64 {
    if finite(x) { x } else { fill }
}

fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

fn pct_change(x: &[f64], k: usize) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![f64::NAN; n];
    for t in k..n {
        if x[t - k] != 0.0 && finite(x[t]) && finite(x[t - k]) {
            out[t] = x[t] / x[t - k] - 1.0;
        }
    }
    out
}

fn shift(x: &[f64], k: usize) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![f64::NAN; n];
    for t in k..n {
        out[t] = x[t - k];
    }
    out
}

fn ewm(x: &[f64], span: usize) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return vec![];
    }
    let alpha = 2.0 / (span as f64 + 1.0);
    let mut out = vec![0.0; n];
    out[0] = x[0];
    for t in 1..n {
        out[t] = alpha * x[t] + (1.0 - alpha) * out[t - 1];
    }
    out
}

fn rolling_mean(x: &[f64], window: usize, min_periods: usize) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![f64::NAN; n];
    for t in 0..n {
        let lo = t.saturating_sub(window - 1);
        let mut s = 0.0;
        let mut c = 0usize;
        for i in lo..=t {
            if finite(x[i]) {
                s += x[i];
                c += 1;
            }
        }
        if c >= min_periods && c > 0 {
            out[t] = s / c as f64;
        }
    }
    out
}

fn rolling_std(x: &[f64], window: usize, min_periods: usize) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![f64::NAN; n];
    for t in 0..n {
        let lo = t.saturating_sub(window - 1);
        let mut s = 0.0;
        let mut c = 0usize;
        for i in lo..=t {
            if finite(x[i]) {
                s += x[i];
                c += 1;
            }
        }
        if c < min_periods.max(2) {
            continue;
        }
        let mean = s / c as f64;
        let mut ss = 0.0;
        for i in lo..=t {
            if finite(x[i]) {
                let d = x[i] - mean;
                ss += d * d;
            }
        }
        out[t] = (ss / (c - 1) as f64).sqrt();
    }
    out
}

fn rolling_max(x: &[f64], window: usize, min_periods: usize) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![f64::NAN; n];
    for t in 0..n {
        let lo = t.saturating_sub(window - 1);
        let mut m = f64::NEG_INFINITY;
        let mut c = 0usize;
        for i in lo..=t {
            if finite(x[i]) {
                m = m.max(x[i]);
                c += 1;
            }
        }
        if c >= min_periods {
            out[t] = m;
        }
    }
    out
}

fn rolling_min(x: &[f64], window: usize, min_periods: usize) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![f64::NAN; n];
    for t in 0..n {
        let lo = t.saturating_sub(window - 1);
        let mut m = f64::INFINITY;
        let mut c = 0usize;
        for i in lo..=t {
            if finite(x[i]) {
                m = m.min(x[i]);
                c += 1;
            }
        }
        if c >= min_periods {
            out[t] = m;
        }
    }
    out
}

fn rolling_corr(a: &[f64], b: &[f64], window: usize, min_periods: usize) -> Vec<f64> {
    let n = a.len();
    let mut out = vec![f64::NAN; n];
    for t in 0..n {
        let lo = t.saturating_sub(window - 1);
        let mut sa = 0.0;
        let mut sb = 0.0;
        let mut c = 0usize;
        for i in lo..=t {
            if finite(a[i]) && finite(b[i]) {
                sa += a[i];
                sb += b[i];
                c += 1;
            }
        }
        if c < min_periods.max(2) {
            continue;
        }
        let ma = sa / c as f64;
        let mb = sb / c as f64;
        let mut cov = 0.0;
        let mut va = 0.0;
        let mut vb = 0.0;
        for i in lo..=t {
            if finite(a[i]) && finite(b[i]) {
                let da = a[i] - ma;
                let db = b[i] - mb;
                cov += da * db;
                va += da * da;
                vb += db * db;
            }
        }
        let den = (va * vb).sqrt();
        if den > 0.0 {
            out[t] = cov / den;
        }
    }
    out
}

fn rsi(close: &[f64], n: usize) -> Vec<f64> {
    let m = close.len();
    let mut delta = vec![f64::NAN; m];
    for t in 1..m {
        delta[t] = close[t] - close[t - 1];
    }
    let up: Vec<f64> = delta.iter().map(|d| if finite(*d) { d.max(0.0) } else { f64::NAN }).collect();
    let down: Vec<f64> = delta
        .iter()
        .map(|d| if finite(*d) { -d.min(0.0) } else { f64::NAN })
        .collect();
    let au = rolling_mean(&up, n, n);
    let ad = rolling_mean(&down, n, n);
    let mut out = vec![0.0; m];
    for t in 0..m {
        let rs = if finite(au[t]) && finite(ad[t]) {
            if ad[t] == 0.0 {
                f64::NAN
            } else {
                au[t] / ad[t]
            }
        } else {
            f64::NAN
        };
        let rsi = if finite(rs) {
            100.0 - 100.0 / (1.0 + rs)
        } else {
            f64::NAN
        };
        out[t] = nz((rsi - 50.0) / 50.0, 0.0);
    }
    out
}

fn ret_streak(ret: &[f64]) -> Vec<f64> {
    let n = ret.len();
    let mut out = vec![0.0; n];
    if n == 0 {
        return out;
    }
    let mut count: f64 = 0.0;
    for t in 0..n {
        if t == 0 || sign(ret[t]) != sign(ret[t - 1]) {
            count = 1.0;
        } else {
            count += 1.0;
        }
        out[t] = (sign(ret[t]) * count.ln_1p()).clamp(-4.0, 4.0);
    }
    out
}

/// `(n, N_FEAT)` rows. No look-ahead: row t uses bars[0..=t] only.
pub fn compute_features(bars: &[Bar], flags: Option<[bool; 5]>) -> Vec<Vec<f64>> {
    let n = bars.len();
    if n == 0 {
        return vec![];
    }
    let ff = flags
        .map(FeatureFlags::from_array)
        .unwrap_or_else(|| FeatureFlags::from_bars(bars));

    let close: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let open: Vec<f64> = bars.iter().map(|b| b.open).collect();
    let high: Vec<f64> = bars.iter().map(|b| b.high).collect();
    let low: Vec<f64> = bars.iter().map(|b| b.low).collect();
    let volume: Vec<f64> = bars.iter().map(|b| b.volume).collect();
    let funding: Vec<f64> = bars.iter().map(|b| nz(b.funding, 0.0)).collect();
    let taker: Vec<f64> = bars.iter().map(|b| b.taker_buy_base).collect();
    let trades: Vec<f64> = bars.iter().map(|b| b.trades).collect();
    let mark: Vec<f64> = bars.iter().map(|b| b.mark_close).collect();
    let index: Vec<f64> = bars.iter().map(|b| b.index_close).collect();
    let oi: Vec<f64> = bars.iter().map(|b| b.open_interest).collect();
    let lsr: Vec<f64> = bars.iter().map(|b| b.long_short_ratio).collect();
    let hours: Vec<f64> = bars.iter().map(|b| b.hour() as f64).collect();
    let dows: Vec<f64> = bars.iter().map(|b| b.dow() as f64).collect();

    let ret = {
        let mut r = pct_change(&close, 1);
        for v in &mut r {
            *v = nz(*v, 0.0);
        }
        r
    };

    let ret_1 = {
        let mut v = pct_change(&close, 1);
        v.iter_mut().for_each(|x| *x = nz(*x, 0.0));
        v
    };
    let ret_4 = {
        let mut v = pct_change(&close, 4);
        v.iter_mut().for_each(|x| *x = nz(*x, 0.0));
        v
    };
    let ret_12 = {
        let mut v = pct_change(&close, 12);
        v.iter_mut().for_each(|x| *x = nz(*x, 0.0));
        v
    };
    let ret_24 = {
        let mut v = pct_change(&close, 24);
        v.iter_mut().for_each(|x| *x = nz(*x, 0.0));
        v
    };
    let ret_72 = {
        let mut v = pct_change(&close, 72);
        v.iter_mut().for_each(|x| *x = nz(*x, 0.0));
        v
    };
    let ret_168 = {
        let mut v = pct_change(&close, 168);
        v.iter_mut().for_each(|x| *x = nz(*x, 0.0));
        v
    };

    let sqrt_ann = ANN.sqrt();
    let vol_24: Vec<f64> = rolling_std(&ret, 24, (24 / 2).max(8))
        .iter()
        .map(|v| nz(*v, 0.0) * sqrt_ann)
        .collect();
    let vol_72: Vec<f64> = rolling_std(&ret, 72, (72 / 2).max(8))
        .iter()
        .map(|v| nz(*v, 0.0) * sqrt_ann)
        .collect();
    let vol_168: Vec<f64> = rolling_std(&ret, 168, (168 / 2).max(8))
        .iter()
        .map(|v| nz(*v, 0.0) * sqrt_ann)
        .collect();
    let vol_ratio: Vec<f64> = (0..n)
        .map(|t| {
            if vol_168[t] == 0.0 {
                1.0
            } else {
                (vol_24[t] / vol_168[t]).clamp(0.2, 5.0)
            }
        })
        .collect();

    let ema24 = ewm(&close, 24);
    let ema72 = ewm(&close, 72);
    let ema_gap_24: Vec<f64> = (0..n)
        .map(|t| {
            if close[t] == 0.0 {
                0.0
            } else {
                nz((close[t] - ema24[t]) / close[t], 0.0)
            }
        })
        .collect();
    let ema_gap_72: Vec<f64> = (0..n)
        .map(|t| {
            if close[t] == 0.0 {
                0.0
            } else {
                nz((close[t] - ema72[t]) / close[t], 0.0)
            }
        })
        .collect();
    let rsi_14 = rsi(&close, 14);

    let tr: Vec<f64> = (0..n)
        .map(|t| {
            if close[t] == 0.0 {
                f64::NAN
            } else {
                (high[t] - low[t]) / close[t]
            }
        })
        .collect();
    let range_24: Vec<f64> = rolling_mean(&tr, 24, 8).iter().map(|v| nz(*v, 0.0)).collect();

    let vol_ma = rolling_mean(&volume, 24, 8);
    let vol_sd = rolling_std(&volume, 24, 8);
    let volx_z: Vec<f64> = (0..n)
        .map(|t| {
            if !finite(vol_sd[t]) || vol_sd[t] == 0.0 {
                0.0
            } else {
                ((volume[t] - vol_ma[t]) / vol_sd[t]).clamp(-5.0, 5.0)
            }
        })
        .collect();

    let funding_ma: Vec<f64> = rolling_mean(&funding, 24, 4).iter().map(|v| nz(*v, 0.0)).collect();
    let two_pi = std::f64::consts::PI * 2.0;
    let tod_sin: Vec<f64> = hours.iter().map(|h| (two_pi * h / 24.0).sin()).collect();
    let tod_cos: Vec<f64> = hours.iter().map(|h| (two_pi * h / 24.0).cos()).collect();
    let dow_sin: Vec<f64> = dows.iter().map(|d| (two_pi * d / 7.0).sin()).collect();
    let dow_cos: Vec<f64> = dows.iter().map(|d| (two_pi * d / 7.0).cos()).collect();

    let c24 = shift(&close, 24);
    let c192 = shift(&close, 24 + 168);
    let ret_skip: Vec<f64> = (0..n)
        .map(|t| {
            if finite(c24[t]) && finite(c192[t]) && c192[t] != 0.0 {
                let v = c24[t] / c192[t] - 1.0;
                nz(v, 0.0)
            } else {
                0.0
            }
        })
        .collect();
    let mom_agree: Vec<f64> = (0..n)
        .map(|t| (sign(ret_24[t]) + sign(ret_72[t]) + sign(ret_168[t])) / 3.0)
        .collect();
    let vov: Vec<f64> = rolling_std(&vol_24, 72, 24).iter().map(|v| nz(*v, 0.0)).collect();

    let f_ma72 = rolling_mean(&funding, 72, 12);
    let f_sd72 = rolling_std(&funding, 72, 12);
    let funding_z: Vec<f64> = (0..n)
        .map(|t| {
            if !finite(f_sd72[t]) || f_sd72[t] == 0.0 {
                0.0
            } else {
                ((funding[t] - f_ma72[t]) / f_sd72[t]).clamp(-5.0, 5.0)
            }
        })
        .collect();
    let fma_shift = shift(&funding_ma, 8);
    let funding_d8: Vec<f64> = (0..n)
        .map(|t| nz(funding_ma[t] - fma_shift[t], 0.0))
        .collect();

    let hh = rolling_max(&high, 24, 8);
    let ll = rolling_min(&low, 24, 8);
    let range_pos: Vec<f64> = (0..n)
        .map(|t| {
            let den = hh[t] - ll[t];
            let pos = if finite(den) && den != 0.0 && finite(hh[t]) && finite(ll[t]) {
                (close[t] - ll[t]) / den
            } else {
                0.5
            };
            nz(pos, 0.5) * 2.0 - 1.0
        })
        .collect();
    let ret_streak = ret_streak(&ret);
    let ret_lag = shift(&ret, 1);
    let trend_persist: Vec<f64> = rolling_corr(&ret, &ret_lag, 24, 12)
        .iter()
        .map(|v| nz(*v, 0.0).clamp(-1.0, 1.0))
        .collect();
    let c_ma = rolling_mean(&close, 24, 8);
    let c_sd = rolling_std(&close, 24, 8);
    let close_z: Vec<f64> = (0..n)
        .map(|t| {
            if !finite(c_sd[t]) || c_sd[t] == 0.0 {
                0.0
            } else {
                ((close[t] - c_ma[t]) / c_sd[t]).clamp(-5.0, 5.0)
            }
        })
        .collect();

    let taker_imb: Vec<f64> = if ff.has_taker {
        (0..n)
            .map(|t| {
                if volume[t] == 0.0 {
                    0.0
                } else {
                    let v = 2.0 * taker[t] / volume[t] - 1.0;
                    nz(v, 0.0).clamp(-1.0, 1.0)
                }
            })
            .collect()
    } else {
        vec![0.0; n]
    };
    let taker_imb_ma: Vec<f64> = rolling_mean(&taker_imb, 24, 8).iter().map(|v| nz(*v, 0.0)).collect();

    let span: Vec<f64> = (0..n)
        .map(|t| {
            let s = high[t] - low[t];
            if s == 0.0 { f64::NAN } else { s }
        })
        .collect();
    let body: Vec<f64> = (0..n)
        .map(|t| nz((close[t] - open[t]) / span[t], 0.0).clamp(-1.0, 1.0))
        .collect();
    let wick_imb: Vec<f64> = (0..n)
        .map(|t| {
            let upper = high[t] - open[t].max(close[t]);
            let lower = open[t].min(close[t]) - low[t];
            nz((upper - lower) / span[t], 0.0).clamp(-1.0, 1.0)
        })
        .collect();
    let prev_c = shift(&close, 1);
    let gap: Vec<f64> = (0..n)
        .map(|t| {
            if finite(prev_c[t]) && prev_c[t] != 0.0 {
                nz(open[t] / prev_c[t] - 1.0, 0.0).clamp(-0.05, 0.05)
            } else {
                0.0
            }
        })
        .collect();
    let true_range: Vec<f64> = (0..n)
        .map(|t| {
            let hl = high[t] - low[t];
            if t == 0 || !finite(prev_c[t]) {
                hl
            } else {
                hl.max((high[t] - prev_c[t]).abs())
                    .max((low[t] - prev_c[t]).abs())
            }
        })
        .collect();
    let atr = rolling_mean(&true_range, 14, 7);
    let atr_n: Vec<f64> = (0..n)
        .map(|t| {
            if close[t] == 0.0 {
                0.0
            } else {
                nz(atr[t] / close[t], 0.0).clamp(0.0, 0.2)
            }
        })
        .collect();

    let trades_z: Vec<f64> = if ff.has_trades {
        let t_ma = rolling_mean(&trades, 24, 8);
        let t_sd = rolling_std(&trades, 24, 8);
        (0..n)
            .map(|t| {
                if !finite(t_sd[t]) || t_sd[t] == 0.0 {
                    0.0
                } else {
                    ((trades[t] - t_ma[t]) / t_sd[t]).clamp(-5.0, 5.0)
                }
            })
            .collect()
    } else {
        vec![0.0; n]
    };

    let basis: Vec<f64> = if ff.has_index {
        (0..n)
            .map(|t| {
                if index[t] == 0.0 {
                    0.0
                } else {
                    nz((mark[t] - index[t]) / index[t], 0.0).clamp(-0.05, 0.05)
                }
            })
            .collect()
    } else {
        vec![0.0; n]
    };
    let b_ma = rolling_mean(&basis, 72, 12);
    let b_sd = rolling_std(&basis, 72, 12);
    let basis_z: Vec<f64> = (0..n)
        .map(|t| {
            if !finite(b_sd[t]) || b_sd[t] == 0.0 {
                0.0
            } else {
                ((basis[t] - b_ma[t]) / b_sd[t]).clamp(-5.0, 5.0)
            }
        })
        .collect();

    let (oi_chg, oi_z) = if ff.has_oi {
        let chg: Vec<f64> = {
            let mut v = pct_change(&oi, 1);
            v.iter_mut().for_each(|x| *x = nz(*x, 0.0).clamp(-0.2, 0.2));
            v
        };
        let oi_ma = rolling_mean(&oi, 72, 12);
        let oi_sd = rolling_std(&oi, 72, 12);
        let z: Vec<f64> = (0..n)
            .map(|t| {
                if !finite(oi_sd[t]) || oi_sd[t] == 0.0 {
                    0.0
                } else {
                    ((oi[t] - oi_ma[t]) / oi_sd[t]).clamp(-5.0, 5.0)
                }
            })
            .collect();
        (chg, z)
    } else {
        (vec![0.0; n], vec![0.0; n])
    };
    let lsr_dev: Vec<f64> = if ff.has_lsr {
        lsr.iter().map(|v| (v - 1.0).clamp(-2.0, 2.0)).map(|v| nz(v, 0.0)).collect()
    } else {
        vec![0.0; n]
    };

    let cols: [&Vec<f64>; N_FEAT] = [
        &ret_1,
        &ret_4,
        &ret_12,
        &ret_24,
        &ret_72,
        &ret_168,
        &vol_24,
        &vol_72,
        &vol_168,
        &vol_ratio,
        &ema_gap_24,
        &ema_gap_72,
        &rsi_14,
        &range_24,
        &volx_z,
        &funding,
        &funding_ma,
        &tod_sin,
        &tod_cos,
        &dow_sin,
        &dow_cos,
        &ret_skip,
        &mom_agree,
        &vov,
        &funding_z,
        &funding_d8,
        &range_pos,
        &ret_streak,
        &trend_persist,
        &close_z,
        &taker_imb,
        &taker_imb_ma,
        &body,
        &wick_imb,
        &gap,
        &atr_n,
        &trades_z,
        &basis,
        &basis_z,
        &oi_chg,
        &oi_z,
        &lsr_dev,
    ];
    let mut out = vec![vec![0.0; N_FEAT]; n];
    for t in 0..n {
        for j in 0..N_FEAT {
            let v = cols[j][t];
            out[t][j] = if finite(v) { v } else { 0.0 };
        }
    }
    out
}

/// Causal windows: row t is `[t-seq_len+1, t]` left-padded with the first row.
pub fn make_windows(x: &[Vec<f64>], seq_len: usize) -> Vec<Vec<f32>> {
    let n = x.len();
    if n == 0 {
        return vec![];
    }
    let f = x[0].len();
    if seq_len <= 1 {
        return x
            .iter()
            .map(|row| row.iter().map(|v| *v as f32).collect())
            .collect();
    }
    let mut wins = Vec::with_capacity(n);
    for t in 0..n {
        let mut w = vec![0.0f32; seq_len * f];
        for i in 0..seq_len {
            let src = t as isize - (seq_len as isize - 1) + i as isize;
            let row = if src < 0 { &x[0] } else { &x[src as usize] };
            for j in 0..f {
                w[i * f + j] = row[j] as f32;
            }
        }
        wins.push(w);
    }
    wins
}

/// Last `seq_len` causal feature rows as a flat `seq_len * n_feat` vector.
/// Uses the same tail as Python `last_feature_window`.
pub fn last_window(bars: &[Bar], seq_len: usize, flags: Option<[bool; 5]>) -> Vec<f32> {
    let n = bars.len();
    let tail = if n > TAIL_BARS {
        let start = n.saturating_sub(TAIL_BARS.max(256));
        &bars[start..]
    } else {
        bars
    };
    let mut owned = tail.to_vec();
    if let Some(f) = flags {
        FeatureFlags::from_array(f).apply(&mut owned);
    }
    let x = compute_features(&owned, flags);
    make_windows(&x, seq_len.max(1))
        .pop()
        .unwrap_or_else(|| vec![0.0; seq_len.max(1) * N_FEAT])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bars::Bar;

    fn dummy_bars(n: usize) -> Vec<Bar> {
        (0..n)
            .map(|i| {
                let px = 2000.0 + i as f64;
                Bar::from_ohlcv(1_672_531_200 + i as i64 * 3600, px, px + 1.0, px - 1.0, px + 0.2, 1000.0, 0.00001)
            })
            .collect()
    }

    #[test]
    fn features_do_not_look_ahead() {
        let mut bars = dummy_bars(80);
        let base = compute_features(&bars, None);
        bars[50].close *= 1.08;
        bars[50].high *= 1.08;
        let after = compute_features(&bars, None);
        for t in 0..50 {
            for j in 0..N_FEAT {
                let d = (base[t][j] - after[t][j]).abs();
                assert!(d < 1e-12, "t={t} j={j} d={d}");
            }
        }
        let mut changed = false;
        for t in 50..80 {
            for j in 0..N_FEAT {
                if (base[t][j] - after[t][j]).abs() > 1e-12 {
                    changed = true;
                }
            }
        }
        assert!(changed);
    }

    #[test]
    fn last_window_shape() {
        let bars = dummy_bars(40);
        let w = last_window(&bars, SEQ_LEN, None);
        assert_eq!(w.len(), SEQ_LEN * N_FEAT);
    }
}
