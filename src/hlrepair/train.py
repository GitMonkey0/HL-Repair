"""Train HL-Repair on official STB annotations."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import corrupt, load_stb_windows
from .geometry import hl_centers
from .metrics import evaluate_tensors
from .model import HLRepair


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-annotations", required=True)
    parser.add_argument("--test-annotations", required=True)
    parser.add_argument("--output", default="outputs/hl_repair.pt")
    parser.add_argument("--report", default="outputs/train_report.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2801)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hl-weight", type=float, default=0.003)
    parser.add_argument("--hl-geometry-weight", type=float, default=1.0)
    parser.add_argument("--hl-smoothing-window", type=int, default=3)
    parser.add_argument("--hl-temperature", type=float, default=0.24)
    parser.add_argument("--max-train-windows", type=int, default=None,
                        help="development smoke-test limit; omit for paper reproduction")
    parser.add_argument("--max-test-windows", type=int, default=None,
                        help="development smoke-test limit; omit for paper reproduction")
    args = parser.parse_args()
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401
    seed_all(args.seed)
    clean_base, train_sequences = load_stb_windows(args.train_annotations, stride=4)
    clean_test, test_sequences = load_stb_windows(args.test_annotations, stride=17)
    if args.max_train_windows is not None:
        clean_base = clean_base[:args.max_train_windows]
    if args.max_test_windows is not None:
        clean_test = clean_test[:args.max_test_windows]
    schedule = ((0, 0, 0, 0), (0.03, 0.10, 0, 0), (0, 0, 0.10, 0),
                (0, 0, 0, 0.08), (0.02, 0.05, 0.03, 0.03),
                (0.03, 0.10, 0.10, 0.08), (0.03, 0.10, 0.10, 0.08),
                (0.05, 0.15, 0.20, 0.15))
    noisy_parts = [corrupt(clean_base, args.seed + 10 + i,
                           (row[0], row[1]), row[2], row[3])[0]
                   for i, row in enumerate(schedule)]
    clean_train = clean_base.repeat(len(schedule), 1, 1, 1)
    noisy_train = torch.cat(noisy_parts)
    noisy_test, _, _ = corrupt(clean_test, 9917)
    device = torch.device(args.device)
    model = HLRepair(args.hidden, args.layers, args.heads,
                     hl_smoothing_window=args.hl_smoothing_window,
                     hl_temperature=args.hl_temperature).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    centers = hl_centers(device)
    labels = torch.einsum("ntbd,kd->ntbk", clean_train.to(device), centers).argmax(-1).cpu()
    correlation = 1 - centers @ centers.T
    generator = torch.Generator().manual_seed(args.seed + 99)
    history, started = [], time.time()
    for epoch in range(args.epochs):
        model.train()
        losses = []
        order = torch.randperm(len(clean_train), generator=generator)
        for start in range(0, len(clean_train), args.batch_size):
            index = order[start:start + args.batch_size]
            x, y, target = noisy_train[index].to(device), clean_train[index].to(device), labels[index].to(device)
            prediction, logits = model(x)
            reconstruction = (1 - (prediction * y).sum(-1)).mean()
            velocity = F.smooth_l1_loss(prediction[:, 1:] - prediction[:, :-1],
                                        y[:, 1:] - y[:, :-1])
            ce = F.cross_entropy(logits.flatten(0, 2), target.flatten())
            costs = correlation[target].reshape(-1, 26)
            geometry = (logits.reshape(-1, 26).softmax(-1) * costs).sum(-1).mean()
            loss = reconstruction + 0.2 * velocity + args.hl_weight * (
                ce + args.hl_geometry_weight * geometry)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)))
        print(f"epoch {epoch + 1:02d}/{args.epochs}: loss={history[-1]:.6f}")
    model.eval()
    predictions = []
    torch.backends.mha.set_fastpath_enabled(False)
    with torch.inference_mode():
        for start in range(0, len(clean_test), args.batch_size):
            predictions.append(model(noisy_test[start:start + args.batch_size].to(device))[0].cpu())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "args": vars(args),
                "use_hl_input": True, "smooth_hl_input": True,
                "output_mode": "continuous"}, output)
    report = {"seed": args.seed, "test_corruption_seed": 9917,
              "train_sequences": train_sequences, "test_sequences": test_sequences,
              "train_windows": len(clean_base), "test_windows": len(clean_test),
              "training_seconds": time.time() - started, "loss": history,
              "metrics": evaluate_tensors(torch.cat(predictions), clean_test)}
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
