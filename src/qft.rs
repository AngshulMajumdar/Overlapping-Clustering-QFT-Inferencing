use crate::{utils::{clip, kmeans_plus_plus, median, norm2, sigmoid, squared_distance}, Cover};

#[derive(Clone, Debug)]
pub struct QftConfig {
    pub alpha: f64,
    pub beta: f64,
    pub mu: f64,
    pub rho: f64,
    pub damping: f64,
    pub max_iter: usize,
    pub tolerance: f64,
    pub clip_eps: f64,
}

impl Default for QftConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
            mu: 0.5,
            rho: 1.0,
            damping: 0.5,
            max_iter: 250,
            tolerance: 1e-7,
            clip_eps: 1e-8,
        }
    }
}

pub fn proposed_qft(x: &[Vec<f64>], k: usize, seed: u64, cfg: &QftConfig) -> Result<Cover, String> {
    if x.is_empty() || k == 0 { return Err("QFT requires non-empty data and k > 0".into()); }
    let n = x.len();
    let d = x[0].len();
    if x.iter().any(|r| r.len() != d) { return Err("non-rectangular feature matrix".into()); }
    if cfg.alpha <= 0.0 || cfg.beta <= 0.0 || cfg.rho < 0.0 || cfg.damping <= 0.0 || cfg.damping > 1.0 {
        return Err("invalid QFT parameters".into());
    }

    let centers = kmeans_plus_plus(x, k, seed)?;
    let nearest: Vec<f64> = x.iter().map(|row| centers.iter().map(|c| squared_distance(row, c).sqrt()).fold(f64::INFINITY, f64::min)).collect();
    let mut tau = median(nearest);
    if tau <= 1e-12 { tau = 1.0; }
    let mut m = vec![vec![0.0; k]; n];
    for i in 0..n {
        for c in 0..k {
            m[i][c] = clip((-squared_distance(&x[i], &centers[c]) / (2.0 * tau * tau)).exp(), cfg.clip_eps, 1.0 - cfg.clip_eps);
        }
    }

    for _ in 0..cfg.max_iter {
        let mut v = vec![vec![0.0; k]; n];
        let mut n_tot = vec![0.0; k];
        let mut s_tot = vec![vec![0.0; d]; k];
        let mut u_tot = vec![0.0; k];
        let mut q_tot = vec![vec![0.0; d]; k];
        let mut rdiag_tot = vec![vec![0.0; d]; k];
        let mut row_m = vec![0.0; n];
        let mut row_v = vec![0.0; n];
        for i in 0..n {
            for c in 0..k {
                let mij = m[i][c];
                let vij = mij * (1.0 - mij);
                v[i][c] = vij;
                n_tot[c] += mij;
                u_tot[c] += vij;
                row_m[i] += mij;
                row_v[i] += vij;
                for j in 0..d {
                    s_tot[c][j] += mij * x[i][j];
                    q_tot[c][j] += vij * x[i][j];
                    rdiag_tot[c][j] += vij * x[i][j] * x[i][j];
                }
            }
        }

        let mut raw = vec![vec![0.0; k]; n];
        for i in 0..n {
            let xi2 = norm2(&x[i]);
            for c in 0..k {
                let mij = m[i][c];
                let vij = v[i][c];
                let nc = n_tot[c] - mij;
                let uc = u_tot[c] - vij;
                let rc = row_m[i] - mij;
                let omega = row_v[i] - vij;
                let a = cfg.alpha + cfg.beta * nc;
                let b = a + cfg.beta;
                let mut s = vec![0.0; d];
                let mut q = vec![0.0; d];
                let mut rd = vec![0.0; d];
                for j in 0..d {
                    s[j] = s_tot[c][j] - mij * x[i][j];
                    q[j] = q_tot[c][j] - vij * x[i][j];
                    rd[j] = rdiag_tot[c][j] - vij * x[i][j] * x[i][j];
                }
                let spx: Vec<f64> = s.iter().zip(&x[i]).map(|(aa, bb)| aa + bb).collect();
                let delta = 0.5 * cfg.beta * xi2
                    - 0.5 * cfg.beta * cfg.beta * (norm2(&spx) / b - norm2(&s) / a)
                    + 0.5 * d as f64 * (b / a).ln();
                let mut gs = vec![0.0; d];
                let mut hns = vec![0.0; d];
                for j in 0..d {
                    gs[j] = -cfg.beta.powi(2) * (spx[j] / b - s[j] / a);
                    hns[j] = cfg.beta.powi(3) * (spx[j] / (b * b) - s[j] / (a * a));
                }
                let gn = 0.5 * cfg.beta.powi(3) * (norm2(&spx) / (b * b) - norm2(&s) / (a * a))
                    + 0.5 * d as f64 * cfg.beta * (1.0 / b - 1.0 / a);
                let hss = cfg.beta.powi(2) * (1.0 / a - 1.0 / b);
                let hnn = cfg.beta.powi(4) * (-norm2(&spx) / b.powi(3) + norm2(&s) / a.powi(3))
                    + 0.5 * d as f64 * cfg.beta.powi(2) * (1.0 / (a * a) - 1.0 / (b * b));
                let hns_q: f64 = hns.iter().zip(&q).map(|(aa, bb)| aa * bb).sum();
                let trace_r: f64 = rd.iter().sum();
                let mean_delta = delta + 0.5 * hnn * uc + hns_q + 0.5 * hss * trace_r;
                let gs_q: f64 = gs.iter().zip(&q).map(|(aa, bb)| aa * bb).sum();
                let gs_r_gs: f64 = gs.iter().zip(&rd).map(|(g, r)| g * g * r).sum();
                let variance_delta = gn * gn * uc + 2.0 * gn * gs_q + gs_r_gs + cfg.rho * cfg.rho * omega;
                let logit = -mean_delta + cfg.mu - cfg.rho * rc + 0.5 * variance_delta;
                raw[i][c] = sigmoid(logit);
            }
        }

        let mut max_change: f64 = 0.0;
        for i in 0..n {
            for c in 0..k {
                let newv = clip((1.0 - cfg.damping) * m[i][c] + cfg.damping * raw[i][c], cfg.clip_eps, 1.0 - cfg.clip_eps);
                max_change = max_change.max((newv - m[i][c]).abs());
                m[i][c] = newv;
            }
        }
        if max_change <= cfg.tolerance { break; }
    }

    let mut membership = vec![vec![false; k]; n];
    for i in 0..n {
        for c in 0..k { membership[i][c] = m[i][c] >= 0.5; }
        if !membership[i].iter().any(|&z| z) {
            let mut best = 0usize;
            let mut best_v = f64::NEG_INFINITY;
            for c in 0..k {
                if m[i][c] > best_v { best_v = m[i][c]; best = c; }
            }
            membership[i][best] = true;
        }
    }
    Ok(Cover { membership, scores: m })
}
