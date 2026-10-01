use crate::{utils::{harden_threshold, median, squared_distance}, Cover};

pub fn run(x: &[Vec<f64>], k: usize, _seed: u64) -> Result<Cover, String> {
    if x.is_empty() || k == 0 { return Err("OC-DD-style requires non-empty data and k > 0".into()); }
    let n = x.len();
    let neighbors = 15usize.min(n.saturating_sub(1)).max(1);
    let mut kth = vec![0.0; n];
    let mut local = vec![Vec::<(f64,usize)>::new(); n];
    for i in 0..n {
        let mut ds = Vec::with_capacity(n.saturating_sub(1));
        for j in 0..n { if i != j { ds.push((squared_distance(&x[i], &x[j]).sqrt(), j)); } }
        ds.sort_by(|a,b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        kth[i] = ds[neighbors-1].0;
        local[i] = ds.into_iter().take(neighbors).collect();
    }
    let mut h = median(kth);
    if h <= 1e-12 { h = 1.0; }
    let mut density = vec![0.0; n];
    for i in 0..n {
        density[i] = local[i].iter().map(|(dist,_)| (-(dist/h).powi(2)).exp()).sum::<f64>();
    }
    let mut attractors = Vec::new();
    let first = (0..n).max_by(|&a,&b| density[a].partial_cmp(&density[b]).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(0);
    attractors.push(first);
    while attractors.len() < k.min(n) {
        let mut best = None;
        let mut best_score = f64::NEG_INFINITY;
        for i in 0..n {
            if attractors.contains(&i) { continue; }
            let md = attractors.iter().map(|&a| squared_distance(&x[i], &x[a]).sqrt()).fold(f64::INFINITY, f64::min);
            let score = density[i] * md;
            if score > best_score { best_score = score; best = Some(i); }
        }
        if let Some(i) = best { attractors.push(i); } else { break; }
    }
    while attractors.len() < k { attractors.push(first); }
    let mut scores = vec![vec![0.0; k]; n];
    for i in 0..n {
        for c in 0..k {
            let a = attractors[c];
            let dist = squared_distance(&x[i], &x[a]).sqrt();
            scores[i][c] = density[a] * (-(dist / h).powi(2)).exp();
        }
    }
    let membership = harden_threshold(&scores, 0.55, true);
    Ok(Cover { membership, scores })
}
