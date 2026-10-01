use std::{fs::File, io::{BufRead, BufReader}, path::Path};

use crate::utils::ensure_rectangular;

#[derive(Clone, Debug)]
pub struct Dataset {
    pub x: Vec<Vec<f64>>,
    pub y: Option<Vec<Vec<bool>>>,
}

pub fn load_csv(features: &str, labels: Option<&str>, has_headers: bool) -> Result<Dataset, String> {
    let x = read_numeric_csv(features, has_headers)?;
    let y = match labels {
        Some(path) => {
            let raw = read_numeric_csv(path, has_headers)?;
            if raw.len() != x.len() { return Err("feature and label row counts differ".into()); }
            Some(raw.into_iter().map(|r| r.into_iter().map(|v| v > 0.5).collect()).collect())
        }
        None => None,
    };
    Ok(Dataset { x, y })
}

fn read_numeric_csv(path: &str, has_headers: bool) -> Result<Vec<Vec<f64>>, String> {
    let mut rdr = csv::ReaderBuilder::new().has_headers(has_headers).from_path(path).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for rec in rdr.records() {
        let rec = rec.map_err(|e| e.to_string())?;
        let mut row = Vec::with_capacity(rec.len());
        for cell in rec.iter() {
            row.push(cell.trim().parse::<f64>().map_err(|_| format!("non-numeric CSV value: {cell}"))?);
        }
        out.push(row);
    }
    ensure_rectangular(&out)?;
    Ok(out)
}

pub fn load_mulan_arff(path: &str, label_count: usize) -> Result<Dataset, String> {
    let file = File::open(Path::new(path)).map_err(|e| e.to_string())?;
    let reader = BufReader::new(file);
    let mut attribute_count = 0usize;
    let mut in_data = false;
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for line in reader.lines() {
        let line = line.map_err(|e| e.to_string())?;
        let t = line.trim();
        if t.is_empty() || t.starts_with('%') { continue; }
        let lower = t.to_ascii_lowercase();
        if !in_data {
            if lower.starts_with("@attribute") { attribute_count += 1; }
            if lower.starts_with("@data") { in_data = true; }
            continue;
        }
        let row = if t.starts_with('{') {
            parse_sparse_arff_row(t, attribute_count)?
        } else {
            parse_dense_arff_row(t)?
        };
        if row.len() != attribute_count {
            return Err(format!("ARFF row has {} values but {} attributes were declared", row.len(), attribute_count));
        }
        rows.push(row);
    }
    if rows.is_empty() { return Err("no ARFF data rows found".into()); }
    if label_count == 0 || label_count >= attribute_count {
        return Err("label_count must be in 1..attribute_count".into());
    }
    let feature_count = attribute_count - label_count;
    let mut x = Vec::with_capacity(rows.len());
    let mut y = Vec::with_capacity(rows.len());
    for row in rows {
        x.push(row[..feature_count].to_vec());
        y.push(row[feature_count..].iter().map(|&v| v > 0.5).collect());
    }
    Ok(Dataset { x, y: Some(y) })
}

fn parse_dense_arff_row(line: &str) -> Result<Vec<f64>, String> {
    line.split(',').map(|s| {
        let t = s.trim();
        if t == "?" { Ok(0.0) }
        else { t.parse::<f64>().map_err(|_| format!("unsupported ARFF value: {t}")) }
    }).collect()
}

fn parse_sparse_arff_row(line: &str, p: usize) -> Result<Vec<f64>, String> {
    let body = line.trim().trim_start_matches('{').trim_end_matches('}').trim();
    let mut row = vec![0.0; p];
    if body.is_empty() { return Ok(row); }
    for item in body.split(',') {
        let mut parts = item.split_whitespace();
        let idx: usize = parts.next().ok_or("bad sparse ARFF entry")?.parse().map_err(|_| "bad sparse ARFF index")?;
        let raw = parts.next().ok_or("bad sparse ARFF value")?;
        if idx >= p { return Err("sparse ARFF index out of range".into()); }
        row[idx] = if raw == "?" { 0.0 } else { raw.parse::<f64>().map_err(|_| format!("unsupported ARFF value: {raw}"))? };
    }
    Ok(row)
}
