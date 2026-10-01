pub mod baselines;
pub mod data;
pub mod metrics;
pub mod preprocess;
pub mod qft;
pub mod synthetic;
pub mod utils;

use std::time::Instant;

use baselines::{fcm, gmm, neo, ocdd, okm, svd_osc};
use metrics::{evaluate_cover, Metrics};
use qft::{proposed_qft, QftConfig};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Qft,
    Okm,
    Neo,
    Fcm,
    Gmm,
    SvdOsc,
    OcDd,
}

impl Method {
    pub fn all() -> [Method; 7] {
        [
            Method::Qft,
            Method::Okm,
            Method::Neo,
            Method::Fcm,
            Method::Gmm,
            Method::SvdOsc,
            Method::OcDd,
        ]
    }

    pub fn name(self) -> &'static str {
        match self {
            Method::Qft => "Proposed QFT",
            Method::Okm => "OKM",
            Method::Neo => "NEO-style",
            Method::Fcm => "FCM-Overlap",
            Method::Gmm => "GMM-Overlap",
            Method::SvdOsc => "SVD-OSC-style",
            Method::OcDd => "OC-DD-style",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Cover {
    pub membership: Vec<Vec<bool>>,
    pub scores: Vec<Vec<f64>>,
}

#[derive(Clone, Debug)]
pub struct RunResult {
    pub method: Method,
    pub cover: Cover,
    pub seconds: f64,
    pub metrics: Option<Metrics>,
}

pub fn run_method(
    method: Method,
    x: &[Vec<f64>],
    labels: Option<&[Vec<bool>]>,
    k: usize,
    seed: u64,
    qft_config: &QftConfig,
) -> Result<RunResult, String> {
    let start = Instant::now();
    let cover = match method {
        Method::Qft => proposed_qft(x, k, seed, qft_config)?,
        Method::Okm => okm::run(x, k, seed)?,
        Method::Neo => neo::run(x, k, seed)?,
        Method::Fcm => fcm::run(x, k, seed)?,
        Method::Gmm => gmm::run(x, k, seed)?,
        Method::SvdOsc => svd_osc::run(x, k, seed)?,
        Method::OcDd => ocdd::run(x, k, seed)?,
    };
    let seconds = start.elapsed().as_secs_f64();
    let metrics = labels.map(|y| evaluate_cover(&cover.membership, y));
    Ok(RunResult { method, cover, seconds, metrics })
}

pub fn run_all(
    x: &[Vec<f64>],
    labels: Option<&[Vec<bool>]>,
    k: usize,
    seed: u64,
    qft_config: &QftConfig,
) -> Result<Vec<RunResult>, String> {
    let mut out = Vec::new();
    for method in Method::all() {
        out.push(run_method(method, x, labels, k, seed, qft_config)?);
    }
    Ok(out)
}
