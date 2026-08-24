//! PyO3 module `betatrend_env`: same protocol as `betatrend.nn.env`.

use ::betatrend_env::bars::{bars_from_matrix, ResetCfg};
use ::betatrend_env::env::{OverlayEnv as RustOverlay, TradingEnv};
use ::betatrend_env::exec::overlay_rewards as rust_overlay_rewards;
use ::betatrend_env::features::last_window as rust_last_window;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn flags5(v: Option<Vec<bool>>) -> Option<[bool; 5]> {
    v.and_then(|x| {
        if x.len() == 5 {
            Some([x[0], x[1], x[2], x[3], x[4]])
        } else {
            None
        }
    })
}

fn flatten_matrix(mat: Vec<Vec<f64>>) -> PyResult<(Vec<f64>, usize)> {
    let n = mat.len();
    if n == 0 {
        return Ok((vec![], 0));
    }
    let mut flat = Vec::with_capacity(n * 13);
    for row in &mat {
        if row.len() != 13 {
            return Err(PyValueError::new_err(format!(
                "expected 13 columns per bar, got {}",
                row.len()
            )));
        }
        flat.extend_from_slice(row);
    }
    Ok((flat, n))
}

#[pyfunction]
#[pyo3(signature = (actions, y, lev, vol_ann, cost, down_lambda=0.5, dd_inc=0.0, dd_level=0.0, clip=5.0))]
fn overlay_rewards(
    actions: Vec<f64>,
    y: Vec<f64>,
    lev: Vec<f64>,
    vol_ann: Vec<f64>,
    cost: f64,
    down_lambda: f64,
    dd_inc: f64,
    dd_level: f64,
    clip: f64,
) -> (Vec<f64>, Vec<f64>) {
    rust_overlay_rewards(
        &actions,
        &y,
        &lev,
        &vol_ann,
        cost,
        down_lambda,
        dd_inc,
        dd_level,
        clip,
    )
}

#[pyfunction]
#[pyo3(signature = (mat, seq_len, flags=None))]
fn last_window(mat: Vec<Vec<f64>>, seq_len: usize, flags: Option<Vec<bool>>) -> PyResult<Vec<f32>> {
    let (flat, n) = flatten_matrix(mat)?;
    let f = flags5(flags);
    let bars = bars_from_matrix(&flat, n, f);
    Ok(rust_last_window(&bars, seq_len.max(1), f))
}

#[pyclass(name = "OverlayEnv")]
struct PyOverlayEnv {
    inner: RustOverlay,
}

#[pymethods]
impl PyOverlayEnv {
    #[new]
    #[pyo3(signature = (mat, cfg_json, flags=None))]
    fn new(mat: Vec<Vec<f64>>, cfg_json: String, flags: Option<Vec<bool>>) -> PyResult<Self> {
        let cfg: ResetCfg = serde_json::from_str(&cfg_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let (flat, n) = flatten_matrix(mat)?;
        let f = flags5(flags);
        let bars = bars_from_matrix(&flat, n, f);
        Ok(Self {
            inner: RustOverlay::from_bars(bars, cfg, f),
        })
    }

    fn spec(&self) -> String {
        serde_json::to_string(&self.inner.spec()).unwrap_or_else(|_| "{}".into())
    }

    fn reset(&mut self, cfg_json: String) -> PyResult<Vec<f32>> {
        let cfg: ResetCfg = serde_json::from_str(&cfg_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(self.inner.reset(cfg))
    }

    fn step(&mut self, action: f32) -> (Vec<f32>, f64, bool, f64, usize) {
        let s = self.inner.step(action);
        (s.obs, s.reward, s.done, s.pnl, s.t)
    }

    fn rollout_ready(&self) -> bool {
        self.inner.rollout_ready()
    }

    fn reset_fold(&mut self, train_idx: Vec<usize>) -> Vec<f32> {
        self.inner.reset_fold(&train_idx)
    }

    fn obs_batch(&self) -> Vec<Vec<f32>> {
        self.inner.obs_batch().to_vec()
    }
}

#[pymodule]
fn betatrend_env(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(overlay_rewards, m)?)?;
    m.add_function(wrap_pyfunction!(last_window, m)?)?;
    m.add_class::<PyOverlayEnv>()?;
    Ok(())
}
