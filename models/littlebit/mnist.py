"""MNIST, fetched from the mirror torchvision uses and checked against pinned sha256s before use.

MNIST's terms of use are unread (docs/littlebit.md, "Open questions"), so its files are never
committed: they live in a folder named on the command line. The pins are the sha256s of the files
as fetched on 2026-11-22, whose md5s were the ones torchvision lists for them (pytorch/vision at
4e0198c3, torchvision/datasets/mnist.py).
"""

import gzip
import hashlib
import urllib.request
from pathlib import Path

import numpy as np

MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"

SHA256 = {
    "train-images-idx3-ubyte.gz": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    "train-labels-idx1-ubyte.gz": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
    "t10k-images-idx3-ubyte.gz": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
    "t10k-labels-idx1-ubyte.gz": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
}


def fetch(folder):
    """Downloads each file missing from `folder`, then checks them all."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name in SHA256:
        path = folder / name
        if not path.exists():
            urllib.request.urlretrieve(MIRROR + name, path)
    check(folder)


def check(folder):
    """Refuses a file whose sha256 is not its pin."""
    for name, pin in SHA256.items():
        got = hashlib.sha256((Path(folder) / name).read_bytes()).hexdigest()
        if got != pin:
            raise ValueError(f"{name}: its sha256 is {got}, not the pinned {pin}")


def _idx(path, magic, shape):
    """An IDX file's array of uint8: its magic number and sizes are checked against `shape`."""
    data = gzip.decompress(path.read_bytes())
    dims = len(shape)
    words = [int.from_bytes(data[4 * i : 4 * i + 4], "big") for i in range(dims + 1)]
    if words != [magic, *shape]:
        raise ValueError(f"{path.name}: header {words}, not {[magic, *shape]}")
    return np.frombuffer(data, dtype=np.uint8, offset=4 * (dims + 1)).reshape(shape)


def load(folder):
    """The training and test sets, after `check`: images as uint8 (n, 28, 28), labels as uint8 (n,)."""
    folder = Path(folder)
    check(folder)
    return (
        (
            _idx(folder / "train-images-idx3-ubyte.gz", 2051, (60000, 28, 28)),
            _idx(folder / "train-labels-idx1-ubyte.gz", 2049, (60000,)),
        ),
        (
            _idx(folder / "t10k-images-idx3-ubyte.gz", 2051, (10000, 28, 28)),
            _idx(folder / "t10k-labels-idx1-ubyte.gz", 2049, (10000,)),
        ),
    )
