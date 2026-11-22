//! `cargo run -p tentzhen-littlebit -- export <run-folder>`, from the repo root, writes the run's
//! integer network to `<run-folder>/integer.json` (`tentzhen_littlebit::export`). It reads MNIST's
//! training images from the folder MNIST_DIR names, checked against their pins, and then prints
//! what the integer network gets right of the test images beside what the float run got.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use tentzhen_littlebit::{export, mnist, run};

fn export(folder: &Path) -> Result<String, String> {
    let mnist_dir = std::env::var_os("MNIST_DIR")
        .map(PathBuf::from)
        .ok_or("MNIST_DIR names the folder MNIST is fetched into")?;
    let run = run::Run::load(folder)?;
    let (train, test) = mnist::load(Path::new("."), &mnist_dir)?;
    let net = export::export(&run, &[0, 2], 4, &train)?;
    let out = folder.join(export::FILE);
    std::fs::write(&out, export::to_json(&net)).map_err(|e| format!("{}: {e}", out.display()))?;
    let predictions = net.predict(&test)?;
    let mut report = format!("wrote {}\n", out.display());
    for (j, l) in net.layers.iter().enumerate() {
        report += &format!(
            "  hidden layer {}: g 2^{}, l 2^{}, latent shift {}, h 2^{}, output shift {}\n",
            j + 1,
            l.g_exponent,
            l.l_exponent,
            l.latent_shift,
            l.h_exponent,
            l.output_shift
        );
    }
    report += &format!(
        "  test images right: integer {}, float {}; latents saturated on them: {}\n",
        predictions.correct(&test),
        run.summary.test_correct,
        predictions.saturated
    );
    Ok(report)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.as_slice() {
        [command, folder] if command == "export" => export(Path::new(folder)),
        _ => Err("usage: tentzhen-littlebit export <run-folder>".into()),
    };
    match result {
        Ok(report) => {
            print!("{report}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
