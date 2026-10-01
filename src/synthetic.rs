use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

use crate::data::Dataset;

pub fn overlapping_gaussians(n: usize, d: usize, k: usize, seed: u64) -> Dataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut centers = vec![vec![0.0; d]; k];
    for c in &mut centers {
        for v in c { *v = rng.gen_range(-3.0..3.0); }
    }
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let max_members = k.min(3).max(1);
        let m = rng.gen_range(1..=max_members);
        let mut active = vec![false; k];
        let mut selected = Vec::new();
        while selected.len() < m {
            let idx = rng.gen_range(0..k);
            if !active[idx] {
                active[idx] = true;
                selected.push(idx);
            }
        }
        let mut row = vec![0.0; d];
        for &c in &selected {
            for j in 0..d { row[j] += centers[c][j] / m as f64; }
        }
        for v in &mut row {
            let z: f64 = StandardNormal.sample(&mut rng);
            *v += 0.45 * z;
        }
        x.push(row);
        y.push(active);
    }
    Dataset { x, y: Some(y) }
}
