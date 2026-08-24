//! Median / IQR scaler, fit on train rows only.

#[derive(Debug, Clone)]
pub struct Scaler {
    pub median: Vec<f64>,
    pub iqr: Vec<f64>,
}

impl Scaler {
    pub fn fit(rows: &[Vec<f64>]) -> Self {
        if rows.is_empty() {
            return Self { median: vec![], iqr: vec![] };
        }
        let f = rows[0].len();
        let mut median = vec![0.0; f];
        let mut iqr = vec![1.0; f];
        for j in 0..f {
            let mut col: Vec<f64> = rows.iter().map(|r| r[j]).collect();
            col.sort_by(|a, b| a.partial_cmp(b).unwrap());
            median[j] = quantile(&col, 0.5);
            let q1 = quantile(&col, 0.25);
            let q3 = quantile(&col, 0.75);
            iqr[j] = (q3 - q1).max(1e-6);
        }
        Self { median, iqr }
    }

    pub fn transform_clip(&self, row: &[f64], clip: f64) -> Vec<f32> {
        row.iter()
            .enumerate()
            .map(|(j, x)| {
                let z = (x - self.median[j]) / self.iqr[j];
                z.clamp(-clip, clip) as f32
            })
            .collect()
    }
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let n = sorted.len();
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let w = pos - lo as f64;
        sorted[lo] * (1.0 - w) + sorted[hi] * w
    }
}
