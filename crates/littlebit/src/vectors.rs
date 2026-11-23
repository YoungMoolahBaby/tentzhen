//! Test vectors for the gateware's cocotb tests (docs/littlebit.md, "cocotb tests"): inputs for a
//! LittleBit layer, and what the golden model computes from each, step by step, for the layer
//! engine in `gateware/cores/littlebit.py` to match bit for bit. `cargo run -p tentzhen-littlebit
//! -- vectors`, from the repo root, writes them to [`FILE`]; a test fails if the committed file is
//! not what it writes.
//!
//! No MNIST image is in them, since MNIST's terms are unread and its files are never committed:
//! the inputs are drawn from a fixed seed. Three cases: r16's two hidden layers, as
//! `models/littlebit/runs/r16/integer.json` has them, and a small convolution whose layer is made
//! up here, applied to each window of its grid ([`integer::windows`]).

use crate::integer::{self, LittleBitLayer, Network, Signs};
use serde::{Deserialize, Serialize};

/// Where the vectors are, relative to the repo root.
pub const FILE: &str = "gateware/cores/tests/littlebit_vectors.json";

/// Where r16's integer network is, relative to the repo root.
pub const R16: &str = "models/littlebit/runs/r16/integer.json";

/// What [`write`] writes.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Vectors {
    /// How to write them again.
    pub written_by: String,
    pub cases: Vec<Case>,
}

/// One layer, the inputs it is given in order, and what the golden model makes of each.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Case {
    pub name: String,
    /// Where the layer is: a network file, relative to the repo root, and the layer's index in it.
    /// A layer made up for the test is here instead, in `layer`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub network: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer_index: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer: Option<LittleBitLayer>,
    /// For a convolution: the grid its inputs are the windows of.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub convolution: Option<Convolution>,
    pub inputs: Vec<Vec<u8>>,
    /// For each input, in order.
    pub steps: Vec<Steps>,
}

/// A grid `height` x `width` x `channels`, stored row by row and a position's channels together,
/// and the kernel its windows are taken with.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Convolution {
    pub height: usize,
    pub width: usize,
    pub channels: usize,
    pub kernel: usize,
    pub grid: Vec<u8>,
}

/// A LittleBit layer's vectors for one input after V, â„“, U and h ([`integer::Trace`], less p,
/// which the engine never holds), and how many latents were saturated.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Steps {
    pub a: Vec<i32>,
    pub t: Vec<i16>,
    pub s: Vec<i32>,
    pub z: Vec<i32>,
    pub y: Vec<u8>,
    pub saturated: usize,
}

/// SplitMix64 (Steele, Lea and Flood, 2014), as a fixed sequence of numbers to draw inputs from.
struct Draw(u64);

impl Draw {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A number from `low` to `high`, both included.
    fn between(&mut self, low: i64, high: i64) -> i64 {
        low + (self.next() % (high - low + 1) as u64) as i64
    }

    fn bytes(&mut self, n: usize, high: u8) -> Vec<u8> {
        (0..n).map(|_| self.between(0, high.into()) as u8).collect()
    }

    fn int8s(&mut self, n: usize) -> Vec<i8> {
        (0..n).map(|_| self.between(-128, 127) as i8).collect()
    }

    fn signs(&mut self, rows: usize, rank: usize) -> Vec<Signs> {
        (0..rows)
            .map(|_| {
                Signs(
                    (0..rank)
                        .map(|_| if self.next() & 1 == 0 { 1 } else { -1 })
                        .collect(),
                )
            })
            .collect()
    }
}

fn steps(layer: &LittleBitLayer, inputs: &[Vec<u8>]) -> Result<Vec<Steps>, String> {
    inputs
        .iter()
        .map(|x| {
            let t = layer.trace(x)?;
            Ok(Steps {
                a: t.a,
                t: t.t,
                s: t.s,
                z: t.z,
                y: t.y,
                saturated: t.saturated,
            })
        })
        .collect()
}

/// A hidden layer of `net` given all zeros, every input at `high`, and four inputs drawn from 0
/// to `high`.
fn hidden(net: &Network, index: usize, high: u8, draw: &mut Draw) -> Result<Case, String> {
    let layer = &net.layers[index];
    let mut inputs = vec![vec![0; layer.inputs], vec![high; layer.inputs]];
    inputs.extend((0..4).map(|_| draw.bytes(layer.inputs, high)));
    Ok(Case {
        name: format!("{}-layer-{}", net.run, index + 1),
        network: Some(R16.into()),
        layer_index: Some(index),
        layer: None,
        convolution: None,
        steps: steps(layer, &inputs)?,
        inputs,
    })
}

/// A 3 x 3 convolution from 2 channels to 4, at rank 3, over a 5 x 5 grid. Its scales and biases
/// are drawn across their whole range, and its shifts set so that some latents saturate and some
/// outputs clamp at each end.
fn convolution(draw: &mut Draw) -> Result<Case, String> {
    let (height, width, channels, kernel, rank, outputs) = (5, 5, 2, 3, 3, 4);
    let inputs = kernel * kernel * channels;
    let (latent_shift, output_shift) = (8, 13);
    let layer = LittleBitLayer {
        inputs,
        outputs,
        rank,
        input_exponent: 0,
        g: draw.int8s(inputs),
        g_exponent: 0,
        v: draw.signs(inputs, rank),
        l: draw.int8s(rank),
        l_exponent: 0,
        latent_shift,
        u: draw.signs(outputs, rank),
        h: draw.int8s(outputs),
        h_exponent: 0,
        bias: (0..outputs)
            .map(|_| draw.between(-(1 << 20), 1 << 20) as i32)
            .collect(),
        output_shift,
        output_exponent: (latent_shift + output_shift) as i32,
    };
    let grid = draw.bytes(height * width * channels, 255);
    let windows = integer::windows(&grid, height, width, channels, kernel)?;
    Ok(Case {
        name: "convolution-3x3".into(),
        network: None,
        layer_index: None,
        steps: steps(&layer, &windows)?,
        layer: Some(layer),
        convolution: Some(Convolution {
            height,
            width,
            channels,
            kernel,
            grid,
        }),
        inputs: windows,
    })
}

/// The vectors, from r16's integer network.
pub fn vectors(r16: &Network) -> Result<Vectors, String> {
    let mut draw = Draw(0x7E47_2E4E);
    // r16's first layer takes pixels shifted right by 1, so 0 to 127.
    let high = 255u8 >> r16.pixel_shift;
    Ok(Vectors {
        written_by: "cargo run -p tentzhen-littlebit -- vectors".into(),
        cases: vec![
            hidden(r16, 0, high, &mut draw)?,
            hidden(r16, 1, 255, &mut draw)?,
            convolution(&mut draw)?,
        ],
    })
}

/// The vectors as [`FILE`] holds them: JSON with one input or step per line, and a newline at the
/// end.
pub fn to_json(vectors: &Vectors) -> String {
    let mut text = serde_json::to_string_pretty(vectors).expect("vectors serialize");
    // serde_json writes each number of an array on a line of its own; fold each array of numbers
    // onto one line, so a 784-input vector takes one line, not 786.
    let mut folded = String::with_capacity(text.len());
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        if line.trim_end().ends_with('[')
            && lines.peek().is_some_and(|next| {
                next.trim()
                    .starts_with(|c: char| c == '-' || c.is_ascii_digit())
            })
        {
            folded.push_str(line.trim_end());
            for item in lines.by_ref() {
                let item = item.trim();
                if item.starts_with(']') {
                    folded.push_str(item);
                    break;
                }
                folded.push_str(item);
                if item.ends_with(',') {
                    folded.push(' ');
                }
            }
        } else {
            folded.push_str(line);
        }
        folded.push('\n');
    }
    text = folded;
    text
}
