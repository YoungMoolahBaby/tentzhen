//! Project #3's golden model (docs/littlebit.md, "Verification"): MNIST read only as its pins
//! allow ([`mnist`]); a training run's committed weights, dense or LittleBit, re-scored here in f32
//! ([`run`]); the exporter, which makes a LittleBit run an integer network ([`export`]); the integer
//! golden model that runs it ([`integer`]); the test vectors it writes for the gateware
//! ([`vectors`]); and the computer's side of a layer on the board, checked against it ([`board`]).

pub mod board;
pub mod export;
pub mod integer;
pub mod mnist;
pub mod run;
pub mod vectors;

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::{Path, PathBuf};

    fn root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn e1() -> PathBuf {
        root().join("models/littlebit/runs/e1")
    }

    fn r16() -> PathBuf {
        root().join("models/littlebit/runs/r16")
    }

    /// The folder MNIST was fetched into, from MNIST_DIR. A test that needs MNIST fails without
    /// it rather than passing unchecked; the Quality Gate fetches MNIST and sets it.
    fn mnist_dir() -> PathBuf {
        PathBuf::from(std::env::var_os("MNIST_DIR").expect(
            "MNIST_DIR names the folder MNIST is fetched into: in models/littlebit, \
             `uv run --locked python mnist.py <folder>`",
        ))
    }

    /// A fresh folder for one test, under the system's temp folder.
    fn scratch(test: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("tentzhen-littlebit-{}-{test}", std::process::id()));
        if dir.exists() {
            fs::remove_dir_all(&dir).unwrap();
        }
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn the_mnist_pins_name_its_four_files() {
        let pins = mnist::Pins::load(&root()).unwrap();
        assert_eq!(pins.sha256.len(), 4);
        let text = fs::read_to_string(root().join(mnist::PINS)).unwrap();
        let extra = text.replace("[sha256]", "extra = 1\n\n[sha256]");
        assert!(
            mnist::Pins::parse(&extra)
                .unwrap_err()
                .contains("unknown field")
        );
        let renamed = text.replace("\"t10k-labels-idx1-ubyte.gz\"", "\"t10k-labels.gz\"");
        assert!(mnist::Pins::parse(&renamed).unwrap_err().contains("not"));
    }

    #[test]
    fn a_run_summary_refuses_a_key_it_does_not_know() {
        let text = fs::read_to_string(e1().join("summary.json")).unwrap();
        run::Summary::parse(&text).unwrap();
        let top = text.replacen('{', "{\"extra\": 1,", 1);
        assert!(
            run::Summary::parse(&top)
                .unwrap_err()
                .contains("unknown field")
        );
        let nested = text.replacen("\"optimizer\"", "\"extra\": 1, \"optimizer\"", 1);
        assert!(
            run::Summary::parse(&nested)
                .unwrap_err()
                .contains("unknown field")
        );
        let text = fs::read_to_string(r16().join("summary.json")).unwrap();
        assert!(run::Summary::parse(&text).unwrap().littlebit.is_some());
        for (key, with_extra) in [
            ("\"rank\"", "\"extra\": 1, \"rank\""),
            (
                "\"weights_sha256\": \"92",
                "\"extra\": 1, \"weights_sha256\": \"92",
            ),
        ] {
            let nested = text.replacen(key, with_extra, 1);
            assert_ne!(nested, text);
            assert!(
                run::Summary::parse(&nested)
                    .unwrap_err()
                    .contains("unknown field")
            );
        }
    }

    /// A run is scored only by arithmetic this crate knows: a summary naming another pixel
    /// mapping, or activation grids for other than the gaps between its layers, is refused.
    #[test]
    fn a_run_is_scored_only_as_its_summary_says() {
        let run = run::Run::load(&r16()).unwrap();
        let text = fs::read_to_string(r16().join("summary.json")).unwrap();
        let one = mnist::Images {
            pixels: vec![0; 784],
            labels: vec![0],
        };
        let refused = |text: String| {
            let changed = run::Run {
                summary: run::Summary::parse(&text).unwrap(),
                tensors: run.tensors.clone(),
            };
            changed.correct(&[0, 2, 4], &one).unwrap_err()
        };
        assert!(run.correct(&[0, 2, 4], &one).is_ok());
        let pixels = text.replace("floor(p / 2) / 128", "p / 255");
        assert!(refused(pixels).contains("no mapping this crate knows"));
        let grids = text.replace("-5,\n      -4\n", "-5\n");
        assert!(refused(grids).contains("activation exponents"));
        let e1 = run::Run::load(&e1()).unwrap();
        assert!(e1.correct(&[0, 2], &one).is_ok());
        assert!(run.correct(&[0, 2], &one).unwrap_err().contains("gaps"));
    }

    /// E1's committed weights are the ones its summary names, in the shapes of the 784 -> 256 ->
    /// 128 -> 10 MLP docs/littlebit.md gives (234,752 weights), trained on the MNIST the pins
    /// name. A weights file changed by one bit is refused.
    #[test]
    fn e1_weights_are_the_ones_its_summary_names() {
        let run = run::Run::load(&e1()).unwrap();
        let s = &run.summary;
        let shapes: Vec<(&str, &[usize])> = s
            .layout
            .iter()
            .map(|t| (t.name.as_str(), t.shape.as_slice()))
            .collect();
        let want: [(&str, &[usize]); 6] = [
            ("0.weight", &[256, 784]),
            ("0.bias", &[256]),
            ("2.weight", &[128, 256]),
            ("2.bias", &[128]),
            ("4.weight", &[10, 128]),
            ("4.bias", &[10]),
        ];
        assert_eq!(shapes, want);
        assert_eq!((s.weights, s.parameters), (234_752, 235_146));
        assert_eq!(s.mnist_sha256, mnist::Pins::load(&root()).unwrap().sha256);

        let dir = scratch("e1-weights");
        fs::copy(e1().join("summary.json"), dir.join("summary.json")).unwrap();
        let mut bytes = fs::read(e1().join("weights.f32")).unwrap();
        bytes[1000] ^= 1;
        fs::write(dir.join("weights.f32"), bytes).unwrap();
        let refused = run::Run::load(&dir).err().unwrap();
        fs::remove_dir_all(&dir).unwrap();
        assert!(refused.contains("its sha256 is"), "{refused}");
    }

    /// Each MNIST file is checked against its pin: one changed by a bit is refused.
    #[test]
    fn mnist_refuses_a_changed_file() {
        let dir = scratch("mnist");
        for f in mnist::FILES {
            fs::copy(mnist_dir().join(f), dir.join(f)).unwrap();
        }
        mnist::load(&root(), &dir).unwrap();
        let path = dir.join(mnist::FILES[3]);
        let mut bytes = fs::read(&path).unwrap();
        let n = bytes.len();
        bytes[n - 9] ^= 1;
        fs::write(&path, bytes).unwrap();
        let refused = mnist::load(&root(), &dir).err().unwrap();
        fs::remove_dir_all(&dir).unwrap();
        assert!(refused.contains("its sha256 is"), "{refused}");
    }

    /// E1, re-scored here from its committed weights, gets right the images its summary says:
    /// 9,831 of the 10,000 test images (98.31 %, docs/littlebit.md) and all 60,000 training
    /// images. The summary's counts came from PyTorch on a GPU; these come from this crate's own
    /// f32 arithmetic on the CPU.
    #[test]
    fn e1_scores_what_its_summary_says() {
        let run = run::Run::load(&e1()).unwrap();
        let (train, test) = mnist::load(&root(), &mnist_dir()).unwrap();
        let linears = [0, 2, 4];
        assert_eq!(
            run.correct(&linears, &test).unwrap(),
            run.summary.test_correct
        );
        assert_eq!(run.summary.test_correct, 9_831);
        assert_eq!(
            run.correct(&linears, &train).unwrap(),
            run.summary.train_correct
        );
        assert_eq!(run.summary.train_correct, 60_000);
    }

    /// r16's committed weights are the ones its summary names, in the shapes of the LittleBit MLP
    /// at r = 16 (docs/littlebit.md, "Model"): layers 0 and 2 as g, V (in x r), l, U (out x r), h
    /// and a bias, and layer 4 dense. It started from E1's committed weights, trained on the
    /// MNIST the pins name.
    #[test]
    fn r16_weights_are_the_ones_its_summary_names() {
        let run = run::Run::load(&r16()).unwrap();
        let s = &run.summary;
        let shapes: Vec<(&str, &[usize])> = s
            .layout
            .iter()
            .map(|t| (t.name.as_str(), t.shape.as_slice()))
            .collect();
        let want: [(&str, &[usize]); 14] = [
            ("0.g", &[784]),
            ("0.v", &[784, 16]),
            ("0.l", &[16]),
            ("0.u", &[256, 16]),
            ("0.h", &[256]),
            ("0.bias", &[256]),
            ("2.g", &[256]),
            ("2.v", &[256, 16]),
            ("2.l", &[16]),
            ("2.u", &[128, 16]),
            ("2.h", &[128]),
            ("2.bias", &[128]),
            ("4.weight", &[10, 128]),
            ("4.bias", &[10]),
        ];
        assert_eq!(shapes, want);
        assert_eq!((s.weights, s.parameters), (234_752, 25_914));
        let lb = s.littlebit.as_ref().unwrap();
        assert_eq!(lb.rank, 16);
        let e1 = run::Run::load(&e1()).unwrap();
        assert_eq!(lb.start.run, e1.summary.run);
        assert_eq!(lb.start.weights_sha256, e1.summary.weights_sha256);
        assert_eq!(s.mnist_sha256, mnist::Pins::load(&root()).unwrap().sha256);
    }

    /// r16, the LittleBit MLP at r = 16, re-scored here from its committed weights, gets right
    /// the images its summary says: 9,713 of the 10,000 test images (97.13 %) and 58,798 of the
    /// 60,000 training images. The summary's counts came from PyTorch on a GPU; these come from
    /// this crate's own f32 arithmetic on the CPU, with the factors' signs and the activations'
    /// uint8 grid as `models/littlebit/littlebit.py` has them.
    #[test]
    fn r16_scores_what_its_summary_says() {
        let run = run::Run::load(&r16()).unwrap();
        let (train, test) = mnist::load(&root(), &mnist_dir()).unwrap();
        let layers = [0, 2, 4];
        assert_eq!(
            run.correct(&layers, &test).unwrap(),
            run.summary.test_correct
        );
        assert_eq!(run.summary.test_correct, 9_713);
        assert_eq!(
            run.correct(&layers, &train).unwrap(),
            run.summary.train_correct
        );
        assert_eq!(run.summary.train_correct, 58_798);
    }

    fn r16_integer() -> integer::Network {
        integer::Network::parse(&fs::read_to_string(r16().join(export::FILE)).unwrap()).unwrap()
    }

    /// One small LittleBit layer, worked by hand: each step is what `integer::LittleBitLayer`
    /// says, a right shift rounds toward minus infinity, a latent too large for an int16 is
    /// saturated and counted, and the outputs are clamped to 0..255.
    #[test]
    fn a_littlebit_layer_computes_what_its_steps_say() {
        use integer::Signs;
        let layer = integer::LittleBitLayer {
            inputs: 2,
            outputs: 3,
            rank: 2,
            input_exponent: 0,
            g: vec![3, -2],
            g_exponent: 0,
            v: vec![Signs(vec![1, -1]), Signs(vec![1, 1])],
            l: vec![5, 127],
            l_exponent: 0,
            latent_shift: 1,
            u: vec![Signs(vec![1, 1]), Signs(vec![-1, 1]), Signs(vec![1, -1])],
            h: vec![-2, 1, -1],
            h_exponent: 0,
            bias: vec![1, 0, 31_318],
            output_shift: 2,
            output_exponent: 3,
        };
        // p = (1 * 3, 255 * -2); a_0 = 3 - 510, a_1 = -3 - 510; t_0 = -2,535 >> 1 = -1,268, not
        // the -1,267 truncating would give; t_1 = -65,151 >> 1 = -32,576, which fits.
        let t = layer.trace(&[1, 255]).unwrap();
        assert_eq!(t.p, [3, -510]);
        assert_eq!(t.a, [-507, -513]);
        assert_eq!(t.t, [-1_268, -32_576]);
        assert_eq!(t.s, [-33_844, -31_308, 31_308]);
        assert_eq!(t.z, [67_689, -31_308, 10]);
        // 67,689 >> 2 = 16,922, clamped to 255; -7,827 clamped to 0; 10 >> 2 = 2.
        assert_eq!(t.y, [255, 0, 2]);
        assert_eq!(t.saturated, 0);
        // a_1 = -765 - 510; -1,275 * 127 >> 1 = -80,963, saturated to -32,768.
        let t = layer.trace(&[255, 255]).unwrap();
        assert_eq!(t.a, [255, -1_275]);
        assert_eq!(t.t, [637, -32_768]);
        assert_eq!(t.saturated, 1);
        assert_eq!(t.y, [255, 0, 0]);
    }

    /// A 2 x 3 grid of 2 channels, worked by hand: each window is the 3 x 3 patch around its
    /// position, row by row and then by channel, with zeros past the edge; and the windows come
    /// in the grid's order.
    #[test]
    fn windows_take_each_position_s_patch_with_zeros_past_the_edge() {
        // Position (row, column) holds channel 0 = 10 * row + column + 1, channel 1 = 100 + that.
        let grid: Vec<u8> = (0..2)
            .flat_map(|r| (0..3).flat_map(move |c| [10 * r + c + 1, 110 + 10 * r + c + 1]))
            .collect();
        let w = integer::windows(&grid, 2, 3, 2, 3).unwrap();
        assert_eq!(w.len(), 6);
        assert!(w.iter().all(|w| w.len() == 18));
        // Position (0, 0): the patch's top row and left column are past the edge.
        #[rustfmt::skip]
        assert_eq!(w[0], [
            0, 0,   0, 0,     0, 0,
            0, 0,   1, 111,   2, 112,
            0, 0,   11, 121,  12, 122,
        ]);
        // Position (1, 2): its bottom row and right column are.
        #[rustfmt::skip]
        assert_eq!(w[5], [
            2, 112,   3, 113,   0, 0,
            12, 122,  13, 123,  0, 0,
            0, 0,     0, 0,     0, 0,
        ]);
        assert!(integer::windows(&grid, 3, 3, 2, 3).is_err());
        assert!(integer::windows(&grid, 2, 3, 2, 2).is_err());
    }

    /// A frame is 0xA5 and the inputs; an answer is 0x5A and each output's y and z, z an int32
    /// least significant byte first. An answer of the wrong length or header is refused, and one
    /// that differs from the golden model says where.
    #[test]
    fn a_board_answer_is_read_as_its_frame_says() {
        assert_eq!(board::frame(&[1, 2, 255]), [0xA5, 1, 2, 255]);
        assert_eq!(board::answer_len(256), 1_281);
        let answer = [0x5A, 7, 0x10, 0x00, 0x00, 0x00, 0, 0xFE, 0xFF, 0xFF, 0xFF];
        let outputs = board::parse_answer(&answer, 2).unwrap();
        assert_eq!(outputs, [(7, 16), (0, -2)]);
        assert!(board::parse_answer(&answer[..10], 2).is_err());
        let mut wrong = answer;
        wrong[0] = 0x5B;
        assert!(board::parse_answer(&wrong, 2).is_err());
        let golden = integer::Trace {
            p: vec![],
            a: vec![],
            t: vec![],
            s: vec![],
            z: vec![16, -2],
            y: vec![7, 0],
            saturated: 0,
        };
        assert_eq!(board::difference(&outputs, &golden), None);
        let d = board::difference(&[(7, 16), (0, -3)], &golden).unwrap();
        assert!(
            d.contains("1 of 2 outputs differ") && d.contains("output 1"),
            "{d}"
        );
        assert!(board::difference(&outputs[..1], &golden).is_some());
    }

    /// The gateware's test vectors are what the golden model writes, and load as their schema.
    /// Each case exercises the engine's edges: some latent saturates, and some output clamps to
    /// 0, some to 255, and some falls between.
    #[test]
    fn littlebit_vectors_are_what_the_golden_model_writes() {
        let v = vectors::vectors(&r16_integer()).unwrap();
        let committed = fs::read_to_string(root().join(vectors::FILE)).unwrap();
        assert!(
            committed == vectors::to_json(&v),
            "{} is stale: run `cargo run -p tentzhen-littlebit -- vectors`",
            vectors::FILE
        );
        let loaded: vectors::Vectors = serde_json::from_str(&committed).unwrap();
        assert_eq!(loaded, v);
        let names: Vec<&str> = v.cases.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["r16-layer-1", "r16-layer-2", "convolution-3x3"]);
        for case in &v.cases {
            assert_eq!(case.inputs.len(), case.steps.len(), "{}", case.name);
            let saturated: usize = case.steps.iter().map(|s| s.saturated).sum();
            let y: Vec<u8> = case.steps.iter().flat_map(|s| s.y.clone()).collect();
            assert!(saturated > 0, "{}: no latent saturates", case.name);
            assert!(
                y.contains(&0) && y.contains(&255),
                "{}: no clamp",
                case.name
            );
            assert!(y.iter().any(|&y| 0 < y && y < 255), "{}", case.name);
        }
        let conv = &v.cases[2];
        let c = conv.convolution.as_ref().unwrap();
        let windows = integer::windows(&c.grid, c.height, c.width, c.channels, c.kernel).unwrap();
        assert_eq!(conv.inputs, windows);
        assert_eq!(conv.inputs.len(), 25);
    }

    /// `int8` takes the finest power-of-two grid that holds the largest magnitude in 127 steps,
    /// and rounds each value to the nearest step, a half away from zero.
    #[test]
    fn int8_takes_the_finest_grid_that_holds_the_largest() {
        assert_eq!(
            export::int8(&[1.0, -0.5, 0.01]).unwrap(),
            (vec![64, -32, 1], -6)
        );
        assert_eq!(
            export::int8(&[127.0, 2.5, -2.5]).unwrap(),
            (vec![127, 3, -3], 0)
        );
        assert_eq!(export::int8(&[127.5]).unwrap(), (vec![64], 1));
        assert!(export::int8(&[0.0, 0.0]).is_err());
    }

    /// integer.json is refused unless its keys are known, its signs are signs, its shapes fit
    /// together and its exponents add up.
    #[test]
    fn an_integer_network_refuses_what_does_not_fit() {
        let text = fs::read_to_string(r16().join(export::FILE)).unwrap();
        integer::Network::parse(&text).unwrap();
        let v = text.find("\"v\": [").unwrap();
        let sign = v + 6 + text[v + 6..].find('"').unwrap() + 1;
        let mut not_a_sign = text.clone();
        not_a_sign.replace_range(sign..sign + 1, "x");
        let mut short = text.clone();
        short.remove(sign);
        let cases = [
            (
                text.replacen("\"rank\"", "\"extra\": 1, \"rank\"", 1),
                "unknown field",
            ),
            (not_a_sign, "a sign is"),
            (short, "not as long as its rank"),
            (
                text.replacen("\"latent_shift\": 10,", "\"latent_shift\": 11,", 1),
                "add up",
            ),
            (
                text.replacen("\"pixel_shift\": 1,", "\"pixel_shift\": 2,", 1),
                "not the outputs before it",
            ),
        ];
        for (bad, why) in cases {
            assert_ne!(bad, text);
            let refused = integer::Network::parse(&bad).unwrap_err();
            assert!(refused.contains(why), "{refused}");
        }
    }

    /// r16's integer.json is what the exporter writes from r16's committed weights and the
    /// 60,000 training images.
    #[test]
    fn r16_integer_json_is_what_the_exporter_writes() {
        let run = run::Run::load(&r16()).unwrap();
        let (train, _) = mnist::load(&root(), &mnist_dir()).unwrap();
        let net = export::export(&run, &[0, 2], 4, &train).unwrap();
        assert_eq!(net.weights_sha256, run.summary.weights_sha256);
        let committed = fs::read_to_string(r16().join(export::FILE)).unwrap();
        assert!(
            committed == export::to_json(&net),
            "runs/r16/{} is stale: run `cargo run -p tentzhen-littlebit -- export \
             models/littlebit/runs/r16`",
            export::FILE
        );
    }

    /// On the training images, which chose its latent shifts, r16's integer network saturates no
    /// latent, and gets 58,809 of the 60,000 right, against the float run's 58,798.
    #[test]
    fn r16_integer_saturates_no_latent_on_the_training_images() {
        let (train, _) = mnist::load(&root(), &mnist_dir()).unwrap();
        let p = r16_integer().predict(&train).unwrap();
        assert_eq!(p.saturated, 0);
        assert_eq!(p.correct(&train), 58_809);
    }

    /// Check 1 at r = 16 (docs/littlebit.md, "Verification"): the integer golden model, run from
    /// r16's integer.json, gets at most 0.5 points fewer of the 10,000 test images right than
    /// the float run. It gets 9,717 against the float's 9,713, and the two pick the same digit on
    /// 9,980 of them. No latent is saturated.
    #[test]
    fn r16_integer_is_within_half_a_point_of_float() {
        let run = run::Run::load(&r16()).unwrap();
        let (_, test) = mnist::load(&root(), &mnist_dir()).unwrap();
        let p = r16_integer().predict(&test).unwrap();
        let (integer, float) = (p.correct(&test), run.summary.test_correct);
        assert!(
            integer + 50 >= float,
            "{integer} test images right, more than 50 under the float run's {float}"
        );
        assert_eq!((integer, float), (9_717, 9_713));
        assert_eq!(p.saturated, 0);
        let float_digits = run.predict(&[0, 2, 4], &test).unwrap();
        let same = float_digits
            .iter()
            .zip(&p.digits)
            .filter(|(f, i)| f == i)
            .count();
        assert_eq!(same, 9_980);
    }
}
