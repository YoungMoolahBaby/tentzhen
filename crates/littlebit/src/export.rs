//! The exporter: a LittleBit run's float weights made the integer network that the golden model
//! runs and the gateware will build ([`crate::integer`]), written beside the run as
//! `integer.json` (docs/littlebit.md, "Fixed-point formats"). It reads only the run's committed
//! files and the 60,000 training images, never the test images, and makes each choice as
//! follows:
//!
//! - g, l, h and the output layer's weights become int8, each vector or matrix on the finest grid
//!   2^e that holds its largest magnitude as at most 127 steps ([`int8`]), each value rounded to
//!   the nearest step.
//! - V's and U's signs are the latent weights' signs: +1 where the latent is >= 0, as
//!   `models/littlebit/littlebit.py` has them in training.
//! - A layer's latent shift is the smallest that saturates no latent on the training images.
//! - A bias becomes an int32 on the grid of what it is added to, rounded to the nearest step.
//! - The output shift puts a layer's outputs on the grid the run trained them on, its
//!   `activation_exponents`, so each output is floor(y / 2^e) clamped to 0..255 as in training.

use crate::integer::{DenseLayer, LittleBitLayer, Network, Signs, v_stage};
use crate::mnist::Images;
use crate::run::Run;

/// Where a run's export is written, in its folder.
pub const FILE: &str = "integer.json";

/// 2^e, exactly.
fn step(e: i32) -> f64 {
    2f64.powi(e)
}

/// `values` as int8 on the finest grid 2^e at which the largest magnitude is at most 127 steps,
/// with e: each value is rounded to the nearest step, a half away from zero.
pub fn int8(values: &[f32]) -> Result<(Vec<i8>, i32), String> {
    let most = values.iter().fold(0f64, |m, &v| m.max(f64::from(v).abs()));
    if !(most > 0.0 && most.is_finite()) {
        return Err(format!("no grid holds a largest magnitude of {most}"));
    }
    let mut e = (most / 127.0).log2().ceil() as i32;
    while most > 127.0 * step(e) {
        e += 1;
    }
    while most <= 127.0 * step(e - 1) {
        e -= 1;
    }
    let q = values
        .iter()
        .map(|&v| (f64::from(v) / step(e)).round() as i8)
        .collect();
    Ok((q, e))
}

/// `value` as an int32 on the grid 2^e, rounded to the nearest step, a half away from zero.
fn int32(value: f32, e: i32) -> Result<i32, String> {
    let q = (f64::from(value) / step(e)).round();
    if q.abs() > f64::from(i32::MAX) {
        return Err(format!(
            "{value} is too large for an int32 on a grid of 2^{e}"
        ));
    }
    Ok(q as i32)
}

/// The rows of a row-major matrix of latent weights as signs: +1 where the latent is >= 0.
fn signs(latent: &[f32], rank: usize) -> Vec<Signs> {
    latent
        .chunks_exact(rank)
        .map(|row| Signs(row.iter().map(|&w| if w >= 0.0 { 1 } else { -1 }).collect()))
        .collect()
}

/// The smallest shift s that puts every product in `least..=most` inside an int16.
fn latent_shift(least: i64, most: i64) -> u32 {
    (0..=62)
        .find(|&s| most >> s <= i64::from(i16::MAX) && least >> s >= i64::from(i16::MIN))
        .unwrap_or(62)
}

/// The integer network of `run`: its LittleBit layers `littlebit`, in order, then its dense layer
/// `output`, with every choice made as this module's documentation says, on `train`.
pub fn export(
    run: &Run,
    littlebit: &[usize],
    output: usize,
    train: &Images,
) -> Result<Network, String> {
    let summary = &run.summary;
    let lb = summary
        .littlebit
        .as_ref()
        .ok_or("the run is not a LittleBit run")?;
    if summary.recipe.pixels != "floor(p / 2) / 128" {
        return Err(format!(
            "pixels enter the run as {:?}, not as floor(p / 2) / 128",
            summary.recipe.pixels
        ));
    }
    if lb.activation_exponents.len() != littlebit.len() {
        return Err(format!(
            "{} activation exponents for {} LittleBit layers",
            lb.activation_exponents.len(),
            littlebit.len()
        ));
    }
    let tensor = |name: String| {
        run.tensors
            .get(&name)
            .map(Vec::as_slice)
            .ok_or(format!("the run has no {name}"))
    };
    let pixel_shift = 1;
    let mut xs: Vec<Vec<u8>> = (0..train.len())
        .map(|k| train.image(k).iter().map(|&p| p >> pixel_shift).collect())
        .collect();
    let mut input_exponent = pixel_shift as i32 - 8;
    let mut layers = Vec::new();
    for (&i, &output_exponent) in littlebit.iter().zip(&lb.activation_exponents) {
        let [g, v, l, u, h, b] =
            ["g", "v", "l", "u", "h", "bias"].map(|name| tensor(format!("{i}.{name}")));
        let (g, v, l, u, h, b) = (g?, v?, l?, u?, h?, b?);
        let (rank, inputs, outputs) = (lb.rank, g.len(), h.len());
        if l.len() != rank
            || v.len() != inputs * rank
            || u.len() != outputs * rank
            || b.len() != outputs
        {
            return Err(format!(
                "layer {i}'s shapes are not a LittleBit layer at r = {rank}"
            ));
        }
        let (g, g_exponent) = int8(g)?;
        let (l, l_exponent) = int8(l)?;
        let (v, u) = (signs(v, rank), signs(u, rank));
        let (mut least, mut most) = (0i64, 0i64);
        for x in &xs {
            let (_, a) = v_stage(x, &g, &v, rank)?;
            for (&a, &l) in a.iter().zip(&l) {
                let wide = i64::from(a) * i64::from(l);
                (least, most) = (least.min(wide), most.max(wide));
            }
        }
        let latent_shift = latent_shift(least, most);
        let (h, h_exponent) = int8(h)?;
        let z_exponent =
            input_exponent + g_exponent + l_exponent + latent_shift as i32 + h_exponent;
        let bias = b
            .iter()
            .map(|&b| int32(b, z_exponent))
            .collect::<Result<_, _>>()?;
        let output_shift = u32::try_from(output_exponent - z_exponent).map_err(|_| {
            format!("layer {i}'s outputs are on a coarser grid, 2^{z_exponent}, than they train on")
        })?;
        let layer = LittleBitLayer {
            inputs,
            outputs,
            rank,
            input_exponent,
            g,
            g_exponent,
            v,
            l,
            l_exponent,
            latent_shift,
            u,
            h,
            h_exponent,
            bias,
            output_shift,
            output_exponent,
        };
        xs = xs
            .iter()
            .map(|x| layer.trace(x).map(|t| t.y))
            .collect::<Result<_, _>>()?;
        input_exponent = output_exponent;
        layers.push(layer);
    }
    let w = tensor(format!("{output}.weight"))?;
    let b = tensor(format!("{output}.bias"))?;
    let inputs = layers.last().map_or(784, |l| l.outputs);
    if b.is_empty() || w.len() != b.len() * inputs {
        return Err(format!("layer {output} does not take {inputs} inputs"));
    }
    let (w, weight_exponent) = int8(w)?;
    let net = Network {
        run: summary.run.clone(),
        weights_sha256: summary.weights_sha256.clone(),
        pixel_shift,
        layers,
        output: DenseLayer {
            inputs,
            outputs: b.len(),
            input_exponent,
            weight: w.chunks_exact(inputs).map(<[i8]>::to_vec).collect(),
            weight_exponent,
            bias: b
                .iter()
                .map(|&b| int32(b, input_exponent + weight_exponent))
                .collect::<Result<_, _>>()?,
        },
    };
    net.check()?;
    Ok(net)
}

/// The network as `integer.json` holds it: pretty-printed JSON, ending in a newline.
pub fn to_json(net: &Network) -> String {
    serde_json::to_string_pretty(net).expect("a network is always JSON") + "\n"
}
