use std::f64::consts::PI;

use crate::{utils::{harden_threshold, kmeans_plus_plus}, Cover};

pub fn run(x: &[Vec<f64>], k: usize, seed: u64) -> Result<Cover, String> {
    if x.is_empty() || k == 0 { return Err("GMM requires non-empty data and k > 0".into()); }
    let n = x.len();
    let d = x[0].len();
    let mut means = kmeans_plus_plus(x, k, seed)?;
    let mut global_mean = vec![0.0; d];
    for row in x { for j in 0..d { global_mean[j] += row[j] / n as f64; } }
    let mut global_var = vec![0.0; d];
    for row in x { for j in 0..d { let a = row[j] - global_mean[j]; global_var[j] += a*a / n as f64; } }
    for v in &mut global_var { *v = (*v).max(1e-5); }
    let mut vars = vec![global_var.clone(); k];
    let mut weights = vec![1.0 / k as f64; k];
    let mut resp = vec![vec![1.0 / k as f64; k]; n];
    for _ in 0..100 {
        for i in 0..n {
            let mut logs = vec![0.0; k];
            let mut mx = f64::NEG_INFINITY;
            for c in 0..k {
                let mut lp = weights[c].max(1e-15).ln();
                for j in 0..d {
                    let v = vars[c][j].max(1e-5);
                    let diff = x[i][j] - means[c][j];
                    lp += -0.5 * ((2.0 * PI * v).ln() + diff * diff / v);
                }
                logs[c] = lp;
                mx = mx.max(lp);
            }
            let sum: f64 = logs.iter().map(|&v| (v - mx).exp()).sum();
            for c in 0..k { resp[i][c] = (logs[c] - mx).exp() / sum.max(1e-300); }
        }
        let mut nk = vec![0.0; k];
        for i in 0..n { for c in 0..k { nk[c] += resp[i][c]; } }
        for c in 0..k {
            weights[c] = (nk[c] / n as f64).max(1e-12);
            if nk[c] <= 1e-10 { continue; }
            for j in 0..d {
                means[c][j] = (0..n).map(|i| resp[i][c] * x[i][j]).sum::<f64>() / nk[c];
            }
            for j in 0..d {
                let vv = (0..n).map(|i| { let diff=x[i][j]-means[c][j]; resp[i][c]*diff*diff }).sum::<f64>() / nk[c];
                vars[c][j] = vv.max(1e-5);
            }
        }
    }
    let membership = harden_threshold(&resp, 0.2, true);
    Ok(Cover { membership, scores: resp })
}
