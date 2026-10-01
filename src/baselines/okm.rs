use nalgebra::DMatrix;

use crate::{utils::{kmeans_plus_plus, squared_distance}, Cover};

pub fn run(x: &[Vec<f64>], k: usize, seed: u64) -> Result<Cover, String> {
    if x.is_empty() || k == 0 { return Err("OKM requires non-empty data and k > 0".into()); }
    let n = x.len();
    let d = x[0].len();
    let mut centroids = kmeans_plus_plus(x, k, seed)?;
    let mut z = vec![vec![false; k]; n];
    for i in 0..n {
        let mut best = 0usize;
        let mut best_d = f64::INFINITY;
        for c in 0..k {
            let dd = squared_distance(&x[i], &centroids[c]);
            if dd < best_d { best_d = dd; best = c; }
        }
        z[i][best] = true;
    }
    for _ in 0..20 {
        centroids = least_squares_centroids(x, &z, k, 1e-6)?;
        let mut changed = false;
        for i in 0..n {
            loop {
                let active: Vec<usize> = (0..k).filter(|&c| z[i][c]).collect();
                if active.len() >= 5 { break; }
                let current = reconstruction_error(&x[i], &active, &centroids);
                let mut best_c = None;
                let mut best_err = current;
                for c in 0..k {
                    if z[i][c] { continue; }
                    let mut cand = active.clone();
                    cand.push(c);
                    let err = reconstruction_error(&x[i], &cand, &centroids);
                    if current - err > 1e-8 && err < best_err {
                        best_err = err;
                        best_c = Some(c);
                    }
                }
                match best_c {
                    Some(c) => { z[i][c] = true; changed = true; },
                    None => break,
                }
            }
        }
        if !changed { break; }
    }
    let scores = z.iter().map(|r| r.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect()).collect();
    Ok(Cover { membership: z, scores })
}

fn reconstruction_error(x: &[f64], active: &[usize], centroids: &[Vec<f64>]) -> f64 {
    if active.is_empty() { return f64::INFINITY; }
    let d = x.len();
    let mut mean = vec![0.0; d];
    for &c in active {
        for j in 0..d { mean[j] += centroids[c][j] / active.len() as f64; }
    }
    squared_distance(x, &mean)
}

fn least_squares_centroids(x: &[Vec<f64>], z: &[Vec<bool>], k: usize, reg: f64) -> Result<Vec<Vec<f64>>, String> {
    let n = x.len();
    let d = x[0].len();
    let mut w = DMatrix::<f64>::zeros(n, k);
    for i in 0..n {
        let m = z[i].iter().filter(|&&b| b).count().max(1) as f64;
        for c in 0..k { if z[i][c] { w[(i, c)] = 1.0 / m; } }
    }
    let mut xm = DMatrix::<f64>::zeros(n, d);
    for i in 0..n { for j in 0..d { xm[(i, j)] = x[i][j]; } }
    let wt = w.transpose();
    let mut lhs = &wt * &w;
    for c in 0..k { lhs[(c, c)] += reg; }
    let rhs = &wt * &xm;
    let sol = lhs.lu().solve(&rhs).ok_or("OKM least-squares solve failed")?;
    let mut out = vec![vec![0.0; d]; k];
    for c in 0..k { for j in 0..d { out[c][j] = sol[(c, j)]; } }
    Ok(out)
}
