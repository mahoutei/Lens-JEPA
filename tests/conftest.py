import numpy as np
import pytest
import torch

torch.set_num_threads(4)


@pytest.fixture
def fake_root(tmp_path):
    """Three classes of synthetic images, one folder per class.

    Each class carries a wide Gaussian blob at its own centre, so the classes are
    separable: a tiny model can beat chance. That matters for the agent campaign,
    which flags a run at chance level as collapsed and then has no configuration
    left to select — identical classes would make every run collapse.
    """
    rng = np.random.RandomState(0)
    ys, xs = np.mgrid[0:64, 0:64]
    for index, name in enumerate(("axion", "cdm", "no_sub")):
        (tmp_path / name).mkdir()
        cy, cx = [(16, 16), (16, 48), (48, 16)][index]
        blob = np.exp(-(((ys - cy) ** 2 + (xs - cx) ** 2) / 600.0)).astype(np.float32)
        for i in range(12):
            image = rng.rand(64, 64).astype(np.float32) + 10.0 * blob
            if name == "axion":
                payload = np.empty(2, dtype=object)
                payload[0], payload[1] = image, 10 ** rng.uniform(-24, -22)
                np.save(tmp_path / name / f"{i}.npy", payload, allow_pickle=True)
            else:
                np.save(tmp_path / name / f"{i}.npy", image)
    return tmp_path

