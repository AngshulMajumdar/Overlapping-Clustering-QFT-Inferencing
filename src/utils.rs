use rand::{rngs::StdRng, Rng, SeedableRng};

pub fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| {
        let d = x - y;
        d * d
    }).sum()
}

pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub fn norm2(a: &[f64]) -> f64 {
    dot(a, a)
}

pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

pub fn clip(x: f64, lo: f64, hi: f64) -> f64 {
    x.max(lo).min(hi)
}

pub fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

pub fn row_argmin_dist(x: &[f64], centroids: &[Vec<f64>]) -> usize {
    let mut best = 0usize;
    let mut best_d = f64::INFINITY;
    for (k, c) in centroids.iter().enumerate() {
        let d = squared_distance(x, c);
        if d < best_d {
            best_d = d;
            best = k;
        }
    }
    best
}

pub fn kmeans_plus_plus(x: &[Vec<f64>], k: usize, seed: u64) -> Result<Vec<Vec<f64>>, String> {
    if x.is_empty() || k == 0 {
        return Err("k-means++ requires non-empty data and k > 0".into());
    }
    if k > x.len() {
        return Err("k cannot exceed number of observations".into());
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mut centers = Vec::with_capacity(k);
    centers.push(x[rng.gen_range(0..x.len())].clone());
    while centers.len() < k {
        let mut weights = Vec::with_capacity(x.len());
        let mut total = 0.0;
        for row in x {
            let d = centers.iter().map(|c| squared_distance(row, c)).fold(f64::INFINITY, f64::min);
            weights.push(d.max(0.0));
            total += d.max(0.0);
        }
        let idx = if total <= 1e-15 {
            rng.gen_range(0..x.len())
        } else {
            let mut u = rng.gen::<f64>() * total;
            let mut chosen = x.len() - 1;
            for (i, w) in weights.iter().enumerate() {
                u -= *w;
                if u <= 0.0 {
                    chosen = i;
                    break;
                }
            }
            chosen
        };
        centers.push(x[idx].clone());
    }
    Ok(centers)
}

pub fn harden_threshold(scores: &[Vec<f64>], threshold: f64, relative_to_row_max: bool) -> Vec<Vec<bool>> {
    let mut out = Vec::with_capacity(scores.len());
    for row in scores {
        let row_max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let cut = if relative_to_row_max { threshold * row_max } else { threshold };
        let mut z: Vec<bool> = row.iter().map(|&v| v >= cut).collect();
        if !z.iter().any(|&b| b) && !row.is_empty() {
            let mut best = 0usize;
            let mut best_v = f64::NEG_INFINITY;
            for (j, &v) in row.iter().enumerate() {
                if v > best_v {
                    best_v = v;
                    best = j;
                }
            }
            z[best] = true;
        }
        out.push(z);
    }
    out
}

pub fn ensure_rectangular(x: &[Vec<f64>]) -> Result<usize, String> {
    if x.is_empty() {
        return Err("empty matrix".into());
    }
    let d = x[0].len();
    if d == 0 {
        return Err("matrix has zero columns".into());
    }
    if x.iter().any(|r| r.len() != d) {
        return Err("matrix is not rectangular".into());
    }
    Ok(d)
}
