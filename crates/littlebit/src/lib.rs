//! Project #3's golden model (docs/littlebit.md, "Verification"). So far it holds what grading a
//! float run needs: MNIST read only as its pins allow ([`mnist`]), and a training run's committed
//! weights, re-scored here in f32 ([`run`]). The integer model comes with the LittleBit layer.

pub mod mnist;
pub mod run;

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
}
