use crate::{utils::{kmeans_plus_plus, squared_distance}, Cover};

pub fn run(x: &[Vec<f64>], k: usize, seed: u64) -> Result<Cover, String> {
    if x.is_empty() || k == 0 { return Err("NEO-style requires non-empty data and k > 0".into()); }
    let n = x.len();
    let d = x[0].len();
    let mut centroids = kmeans_plus_plus(x, k, seed)?;
    let mut z = vec![vec![false; k]; n];
    for _ in 0..30 {
        for row in &mut z { row.fill(false); }
        let mut candidates = Vec::new();
        for i in 0..n {
            let mut best = 0usize;
            let mut best_d = f64::INFINITY;
            for c in 0..k {
                let dd = squared_distance(&x[i], &centroids[c]);
                if dd < best_d { best_d = dd; best = c; }
            }
            z[i][best] = true;
            for c in 0..k {
                if c != best { candidates.push((squared_distance(&x[i], &centroids[c]), i, c)); }
            }
        }
        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let extra = ((1.5 * n as f64).round() as usize).min(candidates.len());
        for &(_, i, c) in candidates.iter().take(extra) { z[i][c] = true; }
        let mut next = vec![vec![0.0; d]; k];
        let mut counts = vec![0usize; k];
        for i in 0..n {
            for c in 0..k {
                if z[i][c] {
                    counts[c] += 1;
                    for j in 0..d { next[c][j] += x[i][j]; }
                }
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for j in 0..d { next[c][j] /= counts[c] as f64; }
            } else {
                next[c] = centroids[c].clone();
            }
        }
        centroids = next;
    }
    let scores = z.iter().map(|r| r.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect()).collect();
    Ok(Cover { membership: z, scores })
}
