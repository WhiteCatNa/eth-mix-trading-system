//! ETH timing environment: causal features, overlay/backtest execution, stepwise reward.
//!
//! Python `betatrend.nn.env` is the reference. Golden files under `golden/` pin the contract.

pub mod bars;
pub mod env;
pub mod exec;
pub mod features;
pub mod reward;
pub mod scaler;
pub mod spec;

pub use env::{BacktestEnv, OverlayEnv, ResetCfg, Step, TradingEnv};
pub use exec::{overlay_pnl, overlay_rewards, vol_leverage, BacktestExec, OverlayExec};
pub use features::{compute_features, last_window, make_windows, FEATURE_NAMES, N_FEAT, SEQ_LEN};
pub use reward::RewardMachine;
pub use spec::{default_spec, load_spec_json, EnvSpec};
