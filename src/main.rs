use clap::{Args, Parser, Subcommand, ValueEnum};
use field_theoretic_overlapping_clustering::{
    data::{load_csv, load_mulan_arff, Dataset},
    preprocess::{self, Preprocess},
    qft::QftConfig,
    run_all, run_method, Method,
    synthetic,
};

#[derive(Parser, Debug)]
#[command(name = "field-theoretic-clustering")]
#[command(about = "Rust implementations of Proposed QFT and all overlapping-clustering baselines in the paper")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    Demo(DemoArgs),
    Arff(ArffArgs),
    Csv(CsvArgs),
}

#[derive(Args, Debug)]
struct DemoArgs {
    #[arg(long, default_value_t = 240)] n: usize,
    #[arg(long, default_value_t = 8)] d: usize,
    #[arg(long, default_value_t = 5)] k: usize,
    #[arg(long, default_value_t = 7)] seed: u64,
    #[command(flatten)] qft: QftArgs,
}

#[derive(Args, Debug)]
struct ArffArgs {
    #[arg(long)] path: String,
    #[arg(long)] label_count: usize,
    #[arg(long, value_enum, default_value_t = PrepArg::Standard)] preprocess: PrepArg,
    #[arg(long, value_enum, default_value_t = MethodArg::All)] method: MethodArg,
    #[arg(long, default_value_t = 7)] seed: u64,
    #[command(flatten)] qft: QftArgs,
}

#[derive(Args, Debug)]
struct CsvArgs {
    #[arg(long)] features: String,
    #[arg(long)] labels: Option<String>,
    #[arg(long)] clusters: usize,
    #[arg(long, default_value_t = false)] headers: bool,
    #[arg(long, value_enum, default_value_t = PrepArg::Standard)] preprocess: PrepArg,
    #[arg(long, value_enum, default_value_t = MethodArg::All)] method: MethodArg,
    #[arg(long, default_value_t = 7)] seed: u64,
    #[command(flatten)] qft: QftArgs,
}

#[derive(Args, Debug, Clone)]
struct QftArgs {
    #[arg(long, default_value_t = 1.0)] alpha: f64,
    #[arg(long, default_value_t = 1.0)] beta: f64,
    #[arg(long, default_value_t = 0.5)] mu: f64,
    #[arg(long, default_value_t = 1.0)] rho: f64,
    #[arg(long, default_value_t = 0.5)] damping: f64,
    #[arg(long, default_value_t = 250)] qft_max_iter: usize,
    #[arg(long, default_value_t = 1e-7)] qft_tolerance: f64,
}

impl QftArgs {
    fn config(&self) -> QftConfig {
        QftConfig { alpha:self.alpha, beta:self.beta, mu:self.mu, rho:self.rho, damping:self.damping,
            max_iter:self.qft_max_iter, tolerance:self.qft_tolerance, ..QftConfig::default() }
    }
}

#[derive(ValueEnum, Clone, Copy, Debug)]
enum PrepArg { None, Standard, IdfL2 }
impl From<PrepArg> for Preprocess {
    fn from(v: PrepArg) -> Self { match v { PrepArg::None=>Preprocess::None, PrepArg::Standard=>Preprocess::Standard, PrepArg::IdfL2=>Preprocess::IdfL2 } }
}

#[derive(ValueEnum, Clone, Copy, Debug)]
enum MethodArg { All, Qft, Okm, Neo, Fcm, Gmm, Svd, Ocdd }
impl MethodArg {
    fn method(self) -> Option<Method> {
        match self {
            MethodArg::All=>None, MethodArg::Qft=>Some(Method::Qft), MethodArg::Okm=>Some(Method::Okm), MethodArg::Neo=>Some(Method::Neo),
            MethodArg::Fcm=>Some(Method::Fcm), MethodArg::Gmm=>Some(Method::Gmm), MethodArg::Svd=>Some(Method::SvdOsc), MethodArg::Ocdd=>Some(Method::OcDd),
        }
    }
}

fn main() -> Result<(), String> {
    let cli = Cli::parse();
    match cli.command {
        Command::Demo(args) => {
            let mut ds = synthetic::overlapping_gaussians(args.n, args.d, args.k, args.seed);
            preprocess::apply(&mut ds.x, Preprocess::Standard)?;
            let results = run_all(&ds.x, ds.y.as_deref(), args.k, args.seed, &args.qft.config())?;
            print_results(&results);
        }
        Command::Arff(args) => {
            let mut ds = load_mulan_arff(&args.path, args.label_count)?;
            preprocess::apply(&mut ds.x, args.preprocess.into())?;
            execute(ds, args.label_count, args.method, args.seed, &args.qft.config())?;
        }
        Command::Csv(args) => {
            let mut ds = load_csv(&args.features, args.labels.as_deref(), args.headers)?;
            preprocess::apply(&mut ds.x, args.preprocess.into())?;
            execute(ds, args.clusters, args.method, args.seed, &args.qft.config())?;
        }
    }
    Ok(())
}

fn execute(ds: Dataset, k: usize, method: MethodArg, seed: u64, cfg: &QftConfig) -> Result<(), String> {
    let results = if let Some(m) = method.method() {
        vec![run_method(m, &ds.x, ds.y.as_deref(), k, seed, cfg)?]
    } else {
        run_all(&ds.x, ds.y.as_deref(), k, seed, cfg)?
    };
    print_results(&results);
    Ok(())
}

fn print_results(results: &[field_theoretic_overlapping_clustering::RunResult]) {
    println!("{:<18} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}", "Method", "ONMI", "F1", "Jaccard", "ExactCard", "CardMAE", "MeanM", "Seconds");
    for r in results {
        match &r.metrics {
            Some(m) => println!("{:<18} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>9.4}", r.method.name(), m.onmi, m.example_f1, m.jaccard, m.exact_cardinality, m.cardinality_mae, m.mean_memberships, r.seconds),
            None => {
                let mean = r.cover.membership.iter().map(|row| row.iter().filter(|&&z| z).count() as f64).sum::<f64>() / r.cover.membership.len() as f64;
                println!("{:<18} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9.4} {:>9.4}", r.method.name(), "-", "-", "-", "-", "-", mean, r.seconds);
            }
        }
    }
}
