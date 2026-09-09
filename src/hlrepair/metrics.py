"""Metrics and deterministic filtering baselines."""

import math

import torch

from .geometry import hl_centers, normalize


def evaluate_tensors(prediction, target):
    centers = hl_centers(prediction.device, prediction.dtype)
    angles = torch.rad2deg(torch.acos((prediction * target).sum(-1).clamp(-1, 1))).flatten()
    pred_labels = torch.einsum("ntbd,kd->ntbk", prediction, centers).argmax(-1)
    target_labels = torch.einsum("ntbd,kd->ntbk", target, centers).argmax(-1)
    target_acc = target[:, 2:] - 2 * target[:, 1:-1] + target[:, :-2]
    pred_acc = prediction[:, 2:] - 2 * prediction[:, 1:-1] + prediction[:, :-2]
    return {
        "mean_angular_error_deg": float(angles.mean()),
        "p95_angular_error_deg": float(torch.quantile(angles, 0.95)),
        "hl_symbol_accuracy_percent": float((pred_labels == target_labels).float().mean() * 100),
        "direction_acceleration_error": float((pred_acc - target_acc).norm(dim=-1).mean()),
    }


def centered_median(directions, window=9):
    half, output = window // 2, []
    for frame in range(directions.shape[1]):
        start, stop = max(0, frame - half), min(directions.shape[1], frame + half + 1)
        output.append(normalize(directions[:, start:stop].median(dim=1).values))
    return torch.stack(output, dim=1)
