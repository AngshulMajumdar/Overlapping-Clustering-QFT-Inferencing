#[derive(Clone, Debug)]
pub struct Metrics {
    pub onmi: f64,
    pub example_f1: f64,
    pub jaccard: f64,
    pub exact_cardinality: f64,
    pub cardinality_mae: f64,
    pub mean_memberships: f64,
}

pub fn evaluate_cover(pred: &[Vec<bool>], truth: &[Vec<bool>]) -> Metrics {
    assert_eq!(pred.len(), truth.len());
    let onmi = onmi(pred, truth);
    let matched = match_predicted_to_truth(pred, truth);
    let mut f1 = 0.0;
    let mut jac = 0.0;
    let mut exact = 0.0;
    let mut mae = 0.0;
    let mut mean = 0.0;
    for i in 0..pred.len() {
        let pcount = pred[i].iter().filter(|&&b| b).count();
        let tcount = truth[i].iter().filter(|&&b| b).count();
        let mut inter = 0usize;
        let mut union = 0usize;
        for j in 0..truth[i].len() {
            let p = matched[i][j];
            let t = truth[i][j];
            if p && t { inter += 1; }
            if p || t { union += 1; }
        }
        let denom = pcount + tcount;
        f1 += if denom == 0 { 1.0 } else { 2.0 * inter as f64 / denom as f64 };
        jac += if union == 0 { 1.0 } else { inter as f64 / union as f64 };
        if pcount == tcount { exact += 1.0; }
        mae += (pcount as i64 - tcount as i64).abs() as f64;
        mean += pcount as f64;
    }
    let n = pred.len() as f64;
    Metrics { onmi, example_f1: f1/n, jaccard: jac/n, exact_cardinality: exact/n, cardinality_mae: mae/n, mean_memberships: mean/n }
}

pub fn match_predicted_to_truth(pred: &[Vec<bool>], truth: &[Vec<bool>]) -> Vec<Vec<bool>> {
    if pred.is_empty() { return Vec::new(); }
    let kp = pred[0].len();
    let kt = truth[0].len();
    let size = kp.max(kt);
    let mut weight = vec![vec![0.0; size]; size];
    for a in 0..kp {
        let pa = (0..pred.len()).filter(|&i| pred[i][a]).count();
        for b in 0..kt {
            let tb = (0..truth.len()).filter(|&i| truth[i][b]).count();
            let inter = (0..pred.len()).filter(|&i| pred[i][a] && truth[i][b]).count();
            let denom = pa + tb;
            weight[a][b] = if denom == 0 { 0.0 } else { 2.0 * inter as f64 / denom as f64 };
        }
    }
    let assignment = hungarian_max(&weight);
    let mut out = vec![vec![false; kt]; pred.len()];
    for a in 0..kp {
        let b = assignment[a];
        if b < kt {
            for i in 0..pred.len() { if pred[i][a] { out[i][b] = true; } }
        }
    }
    out
}

fn hungarian_max(w: &[Vec<f64>]) -> Vec<usize> {
    let n = w.len();
    let maxw = w.iter().flat_map(|r| r.iter()).copied().fold(0.0, f64::max);
    let mut u = vec![0.0; n+1];
    let mut v = vec![0.0; n+1];
    let mut p = vec![0usize; n+1];
    let mut way = vec![0usize; n+1];
    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![f64::INFINITY; n+1];
        let mut used = vec![false; n+1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0usize;
            for j in 1..=n {
                if used[j] { continue; }
                let cur = (maxw - w[i0-1][j-1]) - u[i0] - v[j];
                if cur < minv[j] { minv[j] = cur; way[j] = j0; }
                if minv[j] < delta { delta = minv[j]; j1 = j; }
            }
            for j in 0..=n {
                if used[j] { u[p[j]] += delta; v[j] -= delta; }
                else { minv[j] -= delta; }
            }
            j0 = j1;
            if p[j0] == 0 { break; }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 { break; }
        }
    }
    let mut ans = vec![0usize; n];
    for j in 1..=n { if p[j] > 0 { ans[p[j]-1] = j-1; } }
    ans
}

pub fn onmi(a: &[Vec<bool>], b: &[Vec<bool>]) -> f64 {
    if a.is_empty() || b.is_empty() { return 0.0; }
    let ca = columns(a);
    let cb = columns(b);
    let hab = normalized_conditional_cover_entropy(&ca, &cb, a.len());
    let hba = normalized_conditional_cover_entropy(&cb, &ca, a.len());
    (1.0 - 0.5 * (hab + hba)).clamp(0.0, 1.0)
}

fn columns(rows: &[Vec<bool>]) -> Vec<Vec<bool>> {
    let k = rows[0].len();
    let mut out = vec![vec![false; rows.len()]; k];
    for i in 0..rows.len() { for j in 0..k { out[j][i] = rows[i][j]; } }
    out
}

fn normalized_conditional_cover_entropy(x: &[Vec<bool>], y: &[Vec<bool>], n: usize) -> f64 {
    if x.is_empty() { return 0.0; }
    let mut sum = 0.0;
    for xc in x {
        let hx = binary_entropy(xc.iter().filter(|&&z| z).count() as f64 / n as f64);
        if hx <= 1e-15 { continue; }
        let mut best = hx;
        for yc in y {
            best = best.min(lfk_conditional_entropy(xc, yc));
        }
        sum += best / hx;
    }
    sum / x.len() as f64
}

fn lfk_conditional_entropy(x: &[bool], y: &[bool]) -> f64 {
    let n = x.len() as f64;
    let mut a=0.0; let mut b=0.0; let mut c=0.0; let mut d=0.0;
    for (&xx,&yy) in x.iter().zip(y) {
        match (xx,yy) { (true,true)=>a+=1.0, (true,false)=>b+=1.0, (false,true)=>c+=1.0, (false,false)=>d+=1.0 }
    }
    let hx = binary_entropy((a+b)/n);
    let hy = binary_entropy((a+c)/n);
    let hxy = entropy_parts(&[a/n,b/n,c/n,d/n]);
    let informative = entropy_parts(&[a/n,d/n]) > entropy_parts(&[b/n,c/n]);
    if informative { (hxy - hy).max(0.0).min(hx) } else { hx }
}

fn binary_entropy(p: f64) -> f64 {
    entropy_parts(&[p, 1.0-p])
}

fn entropy_parts(ps: &[f64]) -> f64 {
    ps.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.log2()).sum()
}
