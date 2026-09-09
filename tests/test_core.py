from pathlib import Path

import torch

from hlrepair.geometry import hl_centers, normalize
from hlrepair.model import HLRepair, gated_repair, load_checkpoint


def test_codebook_has_26_unit_directions():
    centers = hl_centers()
    assert centers.shape == (26, 3)
    torch.testing.assert_close(centers.norm(dim=-1), torch.ones(26))


def test_forward_shape_and_unit_norm():
    model = HLRepair(hidden=24, layers=1, heads=3).eval()
    x = normalize(torch.randn(2, 17, 20, 3))
    with torch.inference_mode():
        repaired, logits = model(x)
    assert repaired.shape == x.shape
    assert logits.shape == (2, 17, 20, 26)
    torch.testing.assert_close(repaired.norm(dim=-1), torch.ones(2, 17, 20), atol=1e-5, rtol=1e-5)


def test_released_checkpoint_loads_and_runs():
    path = Path(__file__).parents[1] / "checkpoints/hl_repair_stb.pt"
    model = load_checkpoint(path)
    x = normalize(torch.randn(1, 17, 20, 3))
    with torch.inference_mode():
        repaired, _, trigger = gated_repair(model, x)
    assert repaired.shape == x.shape
    assert trigger.shape == (1,)
