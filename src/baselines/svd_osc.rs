use nalgebra::DMatrix;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

use crate::{utils::harden_threshold, Cover};

pub fn run(x: &[Vec<f64>], k: usize, seed: u64) -> Result<Cover, String> {
    if x.is_empty() || k == 0 { return Err("SVD-OSC-style requires non-empty data and k > 0".into()); }
    let n = x.len();
    let d = x[0].len();
    let mut xn = vec![vec![0.0; d]; n];
    for i in 0..n {
        let norm = x[i].iter().map(|v| v*v).sum::<f64>().sqrt();
        if norm > 1e-15 { for j in 0..d { xn[i][j] = x[i][j] / norm; } }
    }
    let mut affinity = DMatrix::<f64>::zeros(n, n);
    let knn = 20usize.min(n.saturating_sub(1));
    for i in 0..n {
        let mut vals = Vec::with_capacity(n.saturating_sub(1));
        for j in 0..n {
            if i == j { continue; }
            let s: f64 = xn[i].iter().zip(&xn[j]).map(|(a,b)| a*b).sum::<f64>().max(0.0);
            vals.push((s, j));
        }
        vals.sort_by(|a,b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        for &(s,j) in vals.iter().take(knn) {
            if s > 0.0 { affinity[(i,j)] = s; }
        }
    }
    for i in 0..n { for j in (i+1)..n {
        let s = affinity[(i,j)].max(affinity[(j,i)]);
        affinity[(i,j)] = s; affinity[(j,i)] = s;
    }}
    let embedding = randomized_left_singular_vectors(&affinity, k, 5, seed)?;
    let mut scores = vec![vec![0.0; k]; n];
    for i in 0..n { for c in 0..k { scores[i][c] = embedding[(i,c)].abs(); } }
    let membership = harden_threshold(&scores, 0.5, true);
    Ok(Cover { membership, scores })
}

fn randomized_left_singular_vectors(a: &DMatrix<f64>, k: usize, power: usize, seed: u64) -> Result<DMatrix<f64>, String> {
    let n = a.nrows();
    let l = (k + 5).min(n).max(k);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut omega = DMatrix::<f64>::zeros(n, l);
    for i in 0..n { for j in 0..l { omega[(i,j)] = StandardNormal.sample(&mut rng); } }
    let mut y = a * &omega;
    y = orthonormalize(&y);
    for _ in 0..power {
        let at = a.transpose();
        let aty = &at * &y;
        y = a * aty;
        y = orthonormalize(&y);
    }
    let yt = y.transpose();
    let b = &yt * a;
    let svd = b.svd(true, false);
    let uhat = svd.u.ok_or("small SVD failed")?;
    let take = k.min(uhat.ncols());
    let small = uhat.columns(0, take).into_owned();
    Ok(y * small)
}

fn orthonormalize(a: &DMatrix<f64>) -> DMatrix<f64> {
    let n = a.nrows();
    let p = a.ncols();
    let mut q = DMatrix::<f64>::zeros(n, p);
    for j in 0..p {
        let mut v: Vec<f64> = (0..n).map(|i| a[(i,j)]).collect();
        for r in 0..j {
            let dot: f64 = (0..n).map(|i| q[(i,r)] * v[i]).sum();
            for i in 0..n { v[i] -= dot * q[(i,r)]; }
        }
        let norm = v.iter().map(|x| x*x).sum::<f64>().sqrt();
        if norm > 1e-12 {
            for i in 0..n { q[(i,j)] = v[i] / norm; }
        }
    }
    q
}
