use serde::{Deserialize, Serialize};

pub const SEQ_LEN: usize = 7;
pub const N_FEAT: usize = 42;
pub const BARS_PER_YEAR: f64 = 24.0 * 365.0;
pub const FEAT_CLIP: f64 = 8.0;
pub const PNL_CLIP: f64 = 0.4;
pub const R_VOL_CLIP: f64 = 8.0;

pub const FEATURE_NAMES: [&str; N_FEAT] = [
    "ret_1",
    "ret_4",
    "ret_12",
    "ret_24",
    "ret_72",
    "ret_168",
    "vol_24",
    "vol_72",
    "vol_168",
    "vol_ratio",
    "ema_gap_24",
    "ema_gap_72",
    "rsi_14",
    "range_24",
    "volx_z",
    "funding",
    "funding_ma",
    "tod_sin",
    "tod_cos",
    "dow_sin",
    "dow_cos",
    "ret_skip",
    "mom_agree",
    "vov",
    "funding_z",
    "funding_d8",
    "range_pos",
    "ret_streak",
    "trend_persist",
    "close_z",
    "taker_imb",
    "taker_imb_ma",
    "body",
    "wick_imb",
    "gap",
    "atr_n",
    "trades_z",
    "basis",
    "basis_z",
    "oi_chg",
    "oi_z",
    "lsr_dev",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvSpec {
    pub obs_shape: [usize; 2],
    pub n_feat: usize,
    pub seq_len: usize,
    pub action_low: f64,
    pub action_high: f64,
    pub y_definition: String,
    pub cost_bps: f64,
    pub feat_clip: f64,
}

pub fn default_spec() -> EnvSpec {
    EnvSpec {
        obs_shape: [SEQ_LEN, N_FEAT],
        n_feat: N_FEAT,
        seq_len: SEQ_LEN,
        action_low: -1.0,
        action_high: 1.0,
        y_definition: "open[t+2] / open[t+1] - 1".into(),
        cost_bps: 8.0,
        feat_clip: FEAT_CLIP,
    }
}

pub fn load_spec_json(raw: &str) -> Result<serde_json::Value, serde_json::Error> {
    serde_json::from_str(raw)
}
