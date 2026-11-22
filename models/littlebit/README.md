# models/littlebit

Project #3's model training, in PyTorch (`docs/littlebit.md`, "Languages and layout"). E1, the
FP32 baseline, is the first run; the LittleBit layer comes next (`docs/littlebit.md`, "First
steps").

## Running E1

Fetch MNIST into a folder outside the repo, and train. `mnist.py` refuses any file whose sha256 is
not its pin in `mnist.toml`; MNIST's terms are unread, so its files are never committed.

    uv run --locked python mnist.py D:\tentzhen-data\mnist
    uv run --locked python e1_fp32.py D:\tentzhen-data\mnist runs\e1

The second writes `runs/e1/weights.f32` (every parameter, as little-endian float32) and
`runs/e1/summary.json` (what was run and what it scored, with the weights' sha256), which are
committed. On the lab computer it took 16 s and got 9,831 of the 10,000 test images right
(98.31 %); six runs gave the same weights, byte for byte (2026-11-22).

That figure is `tested`: `e1_scores_what_its_summary_says`, in `crates/littlebit`, re-scores the
committed weights with its own arithmetic, and the Quality Gate runs it. To run it yourself, point
`MNIST_DIR` at the folder MNIST is in; the tests that need MNIST fail without it, rather than pass
unchecked:

    $env:MNIST_DIR = "D:\tentzhen-data\mnist"
    cargo test -p tentzhen-littlebit

## Setting it up

You need [uv](https://docs.astral.sh/uv/). From this folder:

    uv sync --locked

That installs Python 3.12.15 and the packages `uv.lock` pins: PyTorch 2.15.0 and NumPy 2.5.5
among them. On Windows and Linux PyTorch is its CUDA 13.2 build, from PyTorch's own index; on
macOS it is PyPI's. `--locked` stops if `uv.lock` no longer matches `pyproject.toml`, so nothing
installs that the lock does not name.

On Windows the environment came to 3.0 GB (2026-11-22). To keep it off a small system drive, point
uv at another one before you run it, in PowerShell for example:

    $env:UV_CACHE_DIR = "D:\uv\cache"
    $env:UV_PYTHON_INSTALL_DIR = "D:\uv\python"
    $env:UV_PROJECT_ENVIRONMENT = "D:\venvs\tentzhen-littlebit"

To see that PyTorch finds the GPU:

    uv run --locked python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

On the lab computer, an RTX 4070 Ti with driver 621.01, that printed `2.15.0+cu132 True`. A GPU is
not required: the networks in `docs/littlebit.md` are small, and PyTorch runs them on the CPU too.
How long a run takes either way is unknown until one is timed.
