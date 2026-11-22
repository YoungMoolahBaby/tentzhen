//! A training run, as `models/littlebit/runs/<run>/` holds it: `summary.json`, what was run and
//! what it scored, and `weights.f32`, its parameters as little-endian float32 in the order and
//! shapes the summary's `layout` gives, whose sha256 the summary names.

use crate::mnist::{Images, sha256_hex};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

/// What `models/littlebit/e1_fp32.py` writes as `summary.json`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Summary {
    pub run: String,
    pub model: String,
    /// The weight matrices' entries, biases left out.
    pub weights: usize,
    /// Every parameter, biases too.
    pub parameters: usize,
    pub recipe: Recipe,
    pub mnist_sha256: BTreeMap<String, String>,
    pub train_correct: usize,
    pub train_images: usize,
    pub test_correct: usize,
    pub test_images: usize,
    pub curve: Vec<Epoch>,
    pub layout: Vec<Tensor>,
    pub weights_sha256: String,
    pub seconds: f64,
    pub device: String,
    pub versions: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Recipe {
    pub optimizer: String,
    pub learning_rate: f64,
    pub schedule: String,
    pub epochs: usize,
    pub batch: usize,
    pub seed: u64,
    pub pixels: String,
    pub deterministic_algorithms: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Epoch {
    pub epoch: usize,
    pub train_loss: f64,
    pub test_correct: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tensor {
    pub name: String,
    pub shape: Vec<usize>,
}

impl Summary {
    pub fn parse(text: &str) -> Result<Self, String> {
        serde_json::from_str(text).map_err(|e| format!("summary.json: {e}"))
    }
}

/// A run's summary and its parameters, by name, each checked against the summary.
pub struct Run {
    pub summary: Summary,
    pub tensors: BTreeMap<String, Vec<f32>>,
}

impl Run {
    /// The run in `folder`: its weights refused unless their sha256 is the summary's and their size
    /// is the layout's.
    pub fn load(folder: &Path) -> Result<Self, String> {
        let text = fs::read_to_string(folder.join("summary.json"))
            .map_err(|e| format!("{}: {e}", folder.display()))?;
        let summary = Summary::parse(&text)?;
        let bytes = fs::read(folder.join("weights.f32"))
            .map_err(|e| format!("{}: {e}", folder.display()))?;
        let got = sha256_hex(&bytes);
        if got != summary.weights_sha256 {
            return Err(format!(
                "weights.f32: its sha256 is {got}, not the summary's {}",
                summary.weights_sha256
            ));
        }
        let mut values = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()));
        let mut tensors = BTreeMap::new();
        for t in &summary.layout {
            let n: usize = t.shape.iter().product();
            let v: Vec<f32> = values.by_ref().take(n).collect();
            if v.len() != n {
                return Err(format!("weights.f32 ends inside {}", t.name));
            }
            tensors.insert(t.name.clone(), v);
        }
        if values.next().is_some() || bytes.len() % 4 != 0 {
            return Err("weights.f32 is longer than the layout".into());
        }
        Ok(Run { summary, tensors })
    }

    /// The images `net` gets right, where `net` is the run's MLP: each Linear's weight (out x in,
    /// row-major) and bias, named `<i>.weight` and `<i>.bias` for `i` in `linears`, with a ReLU
    /// after each but the last. Pixels enter as x / 255, in f32, and every product and sum is f32.
    pub fn correct(&self, linears: &[usize], set: &Images) -> Result<usize, String> {
        let layer = |i: usize| -> Result<(&[f32], &[f32]), String> {
            let w = self.tensors.get(&format!("{i}.weight"));
            let b = self.tensors.get(&format!("{i}.bias"));
            match (w, b) {
                (Some(w), Some(b)) if !b.is_empty() && w.len() % b.len() == 0 => Ok((w, b)),
                _ => Err(format!("no Linear {i} in the layout")),
            }
        };
        let layers: Vec<(&[f32], &[f32])> = linears
            .iter()
            .map(|&i| layer(i))
            .collect::<Result<_, _>>()?;
        let mut right = 0;
        for k in 0..set.len() {
            let mut a: Vec<f32> = set.image(k).iter().map(|&p| p as f32 / 255.0).collect();
            for (j, (w, b)) in layers.iter().enumerate() {
                let n_in = a.len();
                if w.len() != b.len() * n_in {
                    return Err(format!(
                        "Linear {} takes {} inputs, not {n_in}",
                        linears[j],
                        w.len() / b.len()
                    ));
                }
                let mut out: Vec<f32> = b
                    .iter()
                    .enumerate()
                    .map(|(o, &bias)| {
                        w[o * n_in..(o + 1) * n_in]
                            .iter()
                            .zip(&a)
                            .fold(0.0f32, |s, (&wi, &ai)| s + wi * ai)
                            + bias
                    })
                    .collect();
                if j + 1 < layers.len() {
                    out.iter_mut().for_each(|v| *v = v.max(0.0));
                }
                a = out;
            }
            let best = (0..a.len()).fold(0, |m, i| if a[i] > a[m] { i } else { m });
            right += usize::from(best == usize::from(set.labels[k]));
        }
        Ok(right)
    }
}
