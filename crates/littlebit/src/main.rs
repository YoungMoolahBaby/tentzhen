//! `cargo run -p tentzhen-littlebit -- export <run-folder>`, from the repo root, writes the run's
//! integer network to `<run-folder>/integer.json` (`tentzhen_littlebit::export`). It reads MNIST's
//! training images from the folder MNIST_DIR names, checked against their pins, and then prints
//! what the integer network gets right of the test images beside what the float run got.
//!
//! `cargo run -p tentzhen-littlebit -- vectors`, from the repo root, writes the gateware's test
//! vectors (`tentzhen_littlebit::vectors`). They need no MNIST.
//!
//! `cargo run -p tentzhen-littlebit -- mnist-vectors <count> <file>`, from the repo root, writes
//! r16's scores and digits for MNIST's first `<count>` test images, with the images, to `<file>`,
//! for the full-network cocotb test (`tentzhen_littlebit::vectors::mnist`). `<file>` belongs
//! outside the repo, since MNIST is never committed; MNIST comes from the folder MNIST_DIR names.
//!
//! `cargo run -p tentzhen-littlebit -- layer <port> vectors`, or `... layer <port> mnist <count>`,
//! from the repo root, sends r16's first layer on the board (`gateware/bringup/littlebit_layer.py`)
//! its inputs over the serial port `<port>` at 115,200 baud, a frame at a time, and checks each
//! answer against the golden model (`tentzhen_littlebit::board`). The inputs are the gateware's
//! test vectors for that layer, or the first `<count>` MNIST test images, each pixel shifted right
//! by the network's pixel shift, from the folder MNIST_DIR names. It talks to the board; loading
//! the board is a flash, done by hand.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};
use tentzhen_littlebit::{board, export, integer, mnist, run, vectors};

fn mnist_dir() -> Result<PathBuf, String> {
    std::env::var_os("MNIST_DIR")
        .map(PathBuf::from)
        .ok_or_else(|| "MNIST_DIR names the folder MNIST is fetched into".into())
}

fn layer(port: &str, source: &[String]) -> Result<String, String> {
    let text =
        std::fs::read_to_string(vectors::R16).map_err(|e| format!("{}: {e}", vectors::R16))?;
    let net = integer::Network::parse(&text)?;
    let layer = &net.layers[0];
    let inputs: Vec<Vec<u8>> = match source {
        [s] if s == "vectors" => {
            let text = std::fs::read_to_string(vectors::FILE)
                .map_err(|e| format!("{}: {e}", vectors::FILE))?;
            let v: vectors::Vectors =
                serde_json::from_str(&text).map_err(|e| format!("{}: {e}", vectors::FILE))?;
            let case = v
                .cases
                .into_iter()
                .find(|c| c.name == "r16-layer-1")
                .ok_or("the vectors have no case r16-layer-1")?;
            case.inputs
        }
        [s, count] if s == "mnist" => {
            let count: usize = count.parse().map_err(|e| format!("count {count}: {e}"))?;
            let (_, test) = mnist::load(Path::new("."), &mnist_dir()?)?;
            if count > test.len() {
                return Err(format!("{count} images, of the {} there are", test.len()));
            }
            (0..count)
                .map(|k| {
                    test.image(k)
                        .iter()
                        .map(|&p| p >> net.pixel_shift)
                        .collect()
                })
                .collect()
        }
        _ => return Err(USAGE.into()),
    };
    let mut serial = serialport::new(port, 115_200)
        .timeout(Duration::from_secs(2))
        .open()
        .map_err(|e| format!("{port}: {e}"))?;
    let (mut sent, mut received) = (Vec::new(), Vec::new());
    let mut differences = Vec::new();
    let start = Instant::now();
    for (k, x) in inputs.iter().enumerate() {
        let frame = board::frame(x);
        let mut answer = vec![0; board::answer_len(layer.outputs)];
        serial
            .clear(serialport::ClearBuffer::Input)
            .and_then(|()| Ok(serial.write_all(&frame)?))
            .and_then(|()| Ok(serial.read_exact(&mut answer)?))
            .map_err(|e| format!("input {k}: {e}"))?;
        let outputs =
            board::parse_answer(&answer, layer.outputs).map_err(|e| format!("input {k}: {e}"))?;
        if let Some(d) = board::difference(&outputs, &layer.trace(x)?) {
            differences.push(format!("input {k}: {d}"));
        }
        sent.extend_from_slice(&frame);
        received.extend_from_slice(&answer);
    }
    let mut report = format!(
        "{port}: {} inputs; {} answered exactly as the golden model, every y and z; {} not; in \
         {:.1} s\n  sent {} bytes, sha256 {}\n  received {} bytes, sha256 {}\n",
        inputs.len(),
        inputs.len() - differences.len(),
        differences.len(),
        start.elapsed().as_secs_f64(),
        sent.len(),
        mnist::sha256_hex(&sent),
        received.len(),
        mnist::sha256_hex(&received),
    );
    for d in differences.iter().take(10) {
        report += &format!("  {d}\n");
    }
    if differences.is_empty() {
        Ok(report)
    } else {
        Err(report)
    }
}

const USAGE: &str = "usage: tentzhen-littlebit export <run-folder> | vectors | \
                     mnist-vectors <count> <file> | layer <port> (vectors | mnist <count>)";

fn write_mnist_vectors(count: &str, file: &Path) -> Result<String, String> {
    let count: usize = count.parse().map_err(|e| format!("count {count}: {e}"))?;
    let text =
        std::fs::read_to_string(vectors::R16).map_err(|e| format!("{}: {e}", vectors::R16))?;
    let r16 = integer::Network::parse(&text)?;
    let (_, test) = mnist::load(Path::new("."), &mnist_dir()?)?;
    let case = vectors::mnist(&r16, &test, count)?;
    std::fs::write(file, vectors::to_json(&case))
        .map_err(|e| format!("{}: {e}", file.display()))?;
    let right = case
        .digits
        .iter()
        .zip(&test.labels)
        .filter(|&(&d, &l)| d == usize::from(l))
        .count();
    Ok(format!(
        "wrote {}: r16's scores for MNIST's first {count} test images, {right} of them right\n",
        file.display()
    ))
}

fn write_vectors() -> Result<String, String> {
    let text =
        std::fs::read_to_string(vectors::R16).map_err(|e| format!("{}: {e}", vectors::R16))?;
    let r16 = integer::Network::parse(&text)?;
    let v = vectors::vectors(&r16)?;
    std::fs::write(vectors::FILE, vectors::to_json(&v))
        .map_err(|e| format!("{}: {e}", vectors::FILE))?;
    let mut report = format!("wrote {}\n", vectors::FILE);
    for case in &v.cases {
        let saturated: usize = case.steps.iter().map(|s| s.saturated).sum();
        report += &format!(
            "  {}: {} inputs, {} latents saturated\n",
            case.name,
            case.inputs.len(),
            saturated
        );
    }
    for case in &v.dense_cases {
        report += &format!(
            "  {}: {} inputs, digits {:?}\n",
            case.name,
            case.inputs.len(),
            case.digits
        );
    }
    for case in &v.network_cases {
        report += &format!(
            "  {}: {} images, digits {:?}\n",
            case.name,
            case.images.len(),
            case.digits
        );
    }
    Ok(report)
}

fn export(folder: &Path) -> Result<String, String> {
    let run = run::Run::load(folder)?;
    let (train, test) = mnist::load(Path::new("."), &mnist_dir()?)?;
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
        [command] if command == "vectors" => write_vectors(),
        [command, count, file] if command == "mnist-vectors" => {
            write_mnist_vectors(count, Path::new(file))
        }
        [command, port, source @ ..] if command == "layer" => layer(port, source),
        _ => Err(USAGE.into()),
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
