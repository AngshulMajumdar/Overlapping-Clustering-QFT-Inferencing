use crate::utils::ensure_rectangular;

#[derive(Clone, Copy, Debug)]
pub enum Preprocess {
    None,
    Standard,
    IdfL2,
}

pub fn apply(x: &mut [Vec<f64>], mode: Preprocess) -> Result<(), String> {
    match mode {
        Preprocess::None => Ok(()),
        Preprocess::Standard => standardize(x),
        Preprocess::IdfL2 => idf_l2(x),
    }
}

pub fn standardize(x: &mut [Vec<f64>]) -> Result<(), String> {
    let d = ensure_rectangular(x)?;
    let n = x.len() as f64;
    let mut mean = vec![0.0; d];
    for row in x.iter() {
        for j in 0..d { mean[j] += row[j]; }
    }
    for m in &mut mean { *m /= n; }
    let mut var = vec![0.0; d];
    for row in x.iter() {
        for j in 0..d {
            let a = row[j] - mean[j];
            var[j] += a * a;
        }
    }
    for v in &mut var {
        *v = (*v / n).sqrt();
        if *v <= 1e-15 { *v = 1.0; }
    }
    for row in x.iter_mut() {
        for j in 0..d { row[j] = (row[j] - mean[j]) / var[j]; }
    }
    Ok(())
}

pub fn idf_l2(x: &mut [Vec<f64>]) -> Result<(), String> {
    let d = ensure_rectangular(x)?;
    let n = x.len() as f64;
    let mut df = vec![0usize; d];
    for row in x.iter() {
        for j in 0..d {
            if row[j] != 0.0 { df[j] += 1; }
        }
    }
    let idf: Vec<f64> = df.iter().map(|&c| ((1.0 + n) / (1.0 + c as f64)).ln() + 1.0).collect();
    for row in x.iter_mut() {
        for j in 0..d { row[j] *= idf[j]; }
        let norm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 1e-15 {
            for v in row.iter_mut() { *v /= norm; }
        }
    }
    Ok(())
}
