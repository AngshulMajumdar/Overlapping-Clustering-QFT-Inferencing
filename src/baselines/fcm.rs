use crate::{utils::{harden_threshold, kmeans_plus_plus, squared_distance}, Cover};

pub fn run(x: &[Vec<f64>], k: usize, seed: u64) -> Result<Cover, String> {
    if x.is_empty() || k == 0 { return Err("FCM requires non-empty data and k > 0".into()); }
    let n = x.len();
    let d = x[0].len();
    let mut c = kmeans_plus_plus(x, k, seed)?;
    let mut u = vec![vec![1.0 / k as f64; k]; n];
    for _ in 0..300 {
        let prev = u.clone();
        for i in 0..n {
            let dist: Vec<f64> = c.iter().map(|cc| squared_distance(&x[i], cc).sqrt().max(1e-12)).collect();
            if let Some(zero) = dist.iter().position(|&v| v <= 1e-10) {
                u[i].fill(0.0);
                u[i][zero] = 1.0;
            } else {
                for a in 0..k {
                    let denom: f64 = (0..k).map(|b| (dist[a] / dist[b]).powi(2)).sum();
                    u[i][a] = 1.0 / denom.max(1e-15);
                }
            }
        }
        let mut next = vec![vec![0.0; d]; k];
        let mut denom = vec![0.0; k];
        for i in 0..n {
            for a in 0..k {
                let w = u[i][a] * u[i][a];
                denom[a] += w;
                for j in 0..d { next[a][j] += w * x[i][j]; }
            }
        }
        for a in 0..k {
            if denom[a] > 1e-15 { for j in 0..d { next[a][j] /= denom[a]; } }
            else { next[a] = c[a].clone(); }
        }
        c = next;
        let maxdiff = u.iter().zip(&prev).flat_map(|(a,b)| a.iter().zip(b).map(|(x,y)| (x-y).abs())).fold(0.0, f64::max);
        if maxdiff <= 1e-5 { break; }
    }
    let membership = harden_threshold(&u, 0.8, true);
    Ok(Cover { membership, scores: u })
}
