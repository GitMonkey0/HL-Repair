"""Repair precomputed local bone-direction sequences stored as NumPy arrays."""

import argparse

import numpy as np
import torch

from .model import gated_repair, load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help=".npy array shaped [N,17,20,3] or [17,20,3]")
    parser.add_argument("output")
    parser.add_argument("--checkpoint", default="checkpoints/hl_repair_stb.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=3.75)
    args = parser.parse_args()
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401
    array = np.load(args.input).astype("float32")
    single = array.ndim == 3
    tensor = torch.from_numpy(array[None] if single else array).to(args.device)
    model = load_checkpoint(args.checkpoint, args.device)
    torch.backends.mha.set_fastpath_enabled(False)
    with torch.inference_mode():
        repaired, _, trigger = gated_repair(model, tensor, args.threshold)
    result = repaired.cpu().numpy()[0] if single else repaired.cpu().numpy()
    np.save(args.output, result)
    print(f"saved {args.output}; repaired {int(trigger.sum())}/{len(trigger)} windows")


if __name__ == "__main__":
    main()
