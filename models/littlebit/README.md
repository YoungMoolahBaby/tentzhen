# models/littlebit

Project #3's model training, in PyTorch (`docs/littlebit.md`, "Languages and layout"). Nothing is
trained here yet: the first run is E1, the FP32 baseline (`docs/littlebit.md`, "First steps").

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
