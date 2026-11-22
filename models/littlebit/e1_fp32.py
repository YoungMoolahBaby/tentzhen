"""E1, the FP32 ceiling (docs/littlebit.md, "Experiments"): the 784 -> 256 -> 128 -> 10 MLP, dense
and in float, trained by the recipe's step 1 (Adam, learning rate 1e-3, cosine decay, 20 epochs),
and its accuracy on the 10,000 test images. From this folder:

    uv run --locked python e1_fp32.py <mnist-folder> <out-folder>

It writes two files to the out folder: weights.f32, every parameter as little-endian float32 one
after another in the order and shapes summary.json's `layout` gives, and summary.json, what was
run and what it scored, with the weights file's sha256. No pickle: crates/littlebit reads both and
re-scores the weights (`e1_scores_what_its_summary_says`). The model kept is the one after the
last epoch: the test set is scored after each epoch to show the curve, and nothing is chosen by it.

What the recipe leaves open is chosen here: pixels as x / 255, batches of 128 in an order shuffled
each epoch, PyTorch's default initialization, no weight decay, dropout or augmentation, the learning
rate decayed by step from 1e-3 to 0 over the 20 epochs, seed 0, and PyTorch's deterministic
algorithms, so a second run on the same machine can be compared with the first.
"""

import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

# cuBLAS gives the same results run to run only with a fixed workspace, set before CUDA starts.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import mnist

SEED = 0
EPOCHS = 20
BATCH = 128
LEARNING_RATE = 1e-3
WIDTHS = [784, 256, 128, 10]


def model():
    layers = []
    for i, (n_in, n_out) in enumerate(zip(WIDTHS, WIDTHS[1:])):
        layers.append(nn.Linear(n_in, n_out))
        if i < len(WIDTHS) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def write_weights(net, path):
    """Writes every parameter, in the state dict's order, as little-endian float32, row-major in
    PyTorch's shapes (a Linear's weight is out x in); gives the layout and the file's sha256."""
    layout, data = [], bytearray()
    for name, t in net.state_dict().items():
        layout.append({"name": name, "shape": list(t.shape)})
        data += t.detach().cpu().contiguous().numpy().astype("<f4").tobytes()
    path.write_bytes(bytes(data))
    return layout, hashlib.sha256(data).hexdigest()


@torch.no_grad()
def correct(net, x, y):
    net.eval()
    return int((net(x).argmax(1) == y).sum())


def main(mnist_folder, out_folder):
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"

    (train_x, train_y), (test_x, test_y) = mnist.load(mnist_folder)

    def images(a):
        return torch.from_numpy(a.reshape(len(a), -1).astype(np.float32) / 255).to(device)

    def labels(a):
        return torch.from_numpy(a.astype(np.int64)).to(device)

    xtr, ytr, xte, yte = images(train_x), labels(train_y), images(test_x), labels(test_y)

    net = model().to(device)
    weights = sum(m.weight.numel() for m in net if isinstance(m, nn.Linear))
    assert weights == 234_752, weights  # docs/littlebit.md, "Model"
    steps_per_epoch = math.ceil(len(xtr) / BATCH)
    opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * steps_per_epoch)
    order = torch.Generator().manual_seed(SEED)

    curve = []
    start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        net.train()
        perm = torch.randperm(len(xtr), generator=order).to(device)
        total_loss = 0.0
        for i in range(0, len(xtr), BATCH):
            batch = perm[i : i + BATCH]
            loss = F.cross_entropy(net(xtr[batch]), ytr[batch])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total_loss += float(loss.detach()) * len(batch)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / len(xtr),
            "test_correct": correct(net, xte, yte),
        }
        curve.append(row)
        print(f"epoch {epoch:2d}  train loss {row['train_loss']:.4f}  test {row['test_correct']}/10000")
    seconds = time.perf_counter() - start

    out = Path(out_folder)
    out.mkdir(parents=True, exist_ok=True)
    layout, sha256 = write_weights(net, out / "weights.f32")
    summary = {
        "run": "E1",
        "model": "784 -> 256 -> 128 -> 10 MLP, dense FP32, ReLU between layers",
        "weights": weights,
        "parameters": sum(p.numel() for p in net.parameters()),
        "recipe": {
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "schedule": "cosine, by step, to 0",
            "epochs": EPOCHS,
            "batch": BATCH,
            "seed": SEED,
            "pixels": "x / 255",
            "deterministic_algorithms": True,
        },
        "mnist_sha256": mnist.SHA256,
        "train_correct": correct(net, xtr, ytr),
        "train_images": len(xtr),
        "test_correct": curve[-1]["test_correct"],
        "test_images": len(xte),
        "curve": curve,
        "layout": layout,
        "weights_sha256": sha256,
        "seconds": seconds,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor(),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "numpy": np.__version__,
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"test accuracy {summary['test_correct'] / 100:.2f}%  "
        f"train {summary['train_correct']}/{len(xtr)}  "
        f"weights sha256 {summary['weights_sha256'][:16]}  {seconds:.1f} s"
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
