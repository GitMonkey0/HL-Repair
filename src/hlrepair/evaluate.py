"""Evaluate a released checkpoint on the official STB test annotations."""

import argparse
import json
from pathlib import Path

import torch

from .data import corrupt, load_stb_windows
from .metrics import centered_median, evaluate_tensors
from .model import gated_repair, load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/hl_repair_stb.pt")
    parser.add_argument("--output", default="outputs/evaluation.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=9917)
    args = parser.parse_args()
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401
    clean, sequences = load_stb_windows(args.annotations, stride=17)
    noisy, outliers, stale = corrupt(clean, args.seed)
    model = load_checkpoint(args.checkpoint, args.device)
    predictions, triggers = [], []
    torch.backends.mha.set_fastpath_enabled(False)
    with torch.inference_mode():
        for start in range(0, len(noisy), args.batch_size):
            x = noisy[start:start + args.batch_size].to(args.device)
            y, _, trigger = gated_repair(model, x)
            predictions.append(y.cpu())
            triggers.append(trigger.cpu())
    prediction = torch.cat(predictions)
    report = {
        "annotations": str(args.annotations), "corruption_seed": args.seed,
        "sequences": sequences, "windows": len(clean),
        "realized_outliers": int(outliers.sum()), "realized_stale": int(stale.sum()),
        "trigger_percent": float(torch.cat(triggers).float().mean() * 100),
        "corrupted": evaluate_tensors(noisy, clean),
        "median_9": evaluate_tensors(centered_median(noisy), clean),
        "hl_repair": evaluate_tensors(prediction, clean),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
