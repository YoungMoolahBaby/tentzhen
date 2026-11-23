//! The computer's side of one LittleBit layer on the board (`gateware/bringup/littlebit_layer.py`):
//! the frame it sends for an input vector, and the answer it reads back, checked against the
//! golden model ([`crate::integer::LittleBitLayer::trace`]).
//!
//! A frame is 0xA5, then the layer's inputs, a byte each. The answer is 0x5A, then for each
//! output, in order, y, a byte, and z, the int32 before the shift and clamp, four bytes, least
//! significant first. `cargo run -p tentzhen-littlebit -- layer` sends them over a serial port.

use crate::integer::Trace;

/// The byte a frame starts with.
pub const FRAME: u8 = 0xA5;
/// The byte an answer starts with.
pub const ANSWER: u8 = 0x5A;

/// The frame for one input vector.
pub fn frame(x: &[u8]) -> Vec<u8> {
    let mut f = Vec::with_capacity(1 + x.len());
    f.push(FRAME);
    f.extend_from_slice(x);
    f
}

/// How many bytes the answer for a layer of `outputs` outputs takes.
pub fn answer_len(outputs: usize) -> usize {
    1 + 5 * outputs
}

/// Each output's y and z, from an answer for a layer of `outputs` outputs.
pub fn parse_answer(answer: &[u8], outputs: usize) -> Result<Vec<(u8, i32)>, String> {
    if answer.len() != answer_len(outputs) {
        return Err(format!(
            "an answer of {} bytes, not {}",
            answer.len(),
            answer_len(outputs)
        ));
    }
    if answer[0] != ANSWER {
        return Err(format!(
            "an answer starting {:#04x}, not {ANSWER:#04x}",
            answer[0]
        ));
    }
    Ok(answer[1..]
        .chunks_exact(5)
        .map(|c| (c[0], i32::from_le_bytes([c[1], c[2], c[3], c[4]])))
        .collect())
}

/// How the board's outputs differ from the golden model's, if they do: the first output whose y
/// or z is not the golden model's, and how many are not.
pub fn difference(board: &[(u8, i32)], golden: &Trace) -> Option<String> {
    if board.len() != golden.y.len() {
        return Some(format!("{} outputs, not {}", board.len(), golden.y.len()));
    }
    let golden: Vec<(u8, i32)> = golden
        .y
        .iter()
        .copied()
        .zip(golden.z.iter().copied())
        .collect();
    let differ: Vec<usize> = (0..board.len())
        .filter(|&o| board[o] != golden[o])
        .collect();
    let &o = differ.first()?;
    Some(format!(
        "{} of {} outputs differ; the first, output {o}: y {}, z {}, where the golden model's are \
         y {}, z {}",
        differ.len(),
        board.len(),
        board[o].0,
        board[o].1,
        golden[o].0,
        golden[o].1
    ))
}
