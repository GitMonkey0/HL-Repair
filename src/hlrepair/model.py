"""HL-Repair network and checkpoint loading."""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .geometry import hl_centers, normalize


def median_anomaly_score(directions: torch.Tensor, window=9):
    """Mean angular deviation from a centered median, in degrees per clip."""
    values, half = [], window // 2
    for frame in range(directions.shape[1]):
        start, stop = max(0, frame - half), min(directions.shape[1], frame + half + 1)
        values.append(normalize(directions[:, start:stop].median(dim=1).values))
    reference = torch.stack(values, dim=1)
    angles = torch.acos((directions * reference).sum(-1).clamp(-1, 1))
    return torch.rad2deg(angles).mean(dim=(1, 2))


def gated_repair(model, directions: torch.Tensor, threshold_deg=3.75):
    repaired, symbols = model(directions)
    trigger = median_anomaly_score(directions) >= threshold_deg
    output = torch.where(trigger[:, None, None, None], repaired, directions)
    return output, symbols, trigger


class HLRepair(nn.Module):
    """17-frame temporal repairer used in the paper."""

    def __init__(self, hidden=192, layers=4, heads=6, *,
                 hl_smoothing_window=3, hl_temperature=0.24):
        super().__init__()
        self.hl_smoothing_window = hl_smoothing_window
        self.hl_temperature = hl_temperature
        self.register_buffer("hl_centers", hl_centers(), persistent=False)
        self.input = nn.Linear(60, hidden)
        self.hl_embedding = nn.Embedding(26, 8)
        self.hl_input = nn.Linear(160, hidden, bias=False)
        self.position = nn.Parameter(torch.randn(1, 17, hidden) * 0.01)
        layer = nn.TransformerEncoderLayer(
            hidden, heads, hidden * 4, dropout=0.1, batch_first=True,
            activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.direction = nn.Linear(hidden, 60)
        self.symbol = nn.Linear(hidden, 20 * 26)
        nn.init.zeros_(self.direction.weight)
        nn.init.zeros_(self.direction.bias)

    def forward(self, directions: torch.Tensor):
        if directions.ndim != 4 or directions.shape[1:] != (17, 20, 3):
            raise ValueError("expected input shape [batch, 17, 20, 3]")
        batch, frames = directions.shape[:2]
        features = self.input(directions.reshape(batch, frames, 60))
        scores = torch.einsum("ntjc,kc->ntjk", directions, self.hl_centers)
        probabilities = (scores / self.hl_temperature).softmax(-1)
        probabilities = probabilities.permute(0, 2, 3, 1).reshape(batch * 20, 26, frames)
        half = self.hl_smoothing_window // 2
        probabilities = F.avg_pool1d(
            F.pad(probabilities, (half, half), mode="replicate"),
            self.hl_smoothing_window, stride=1)
        probabilities = probabilities.reshape(batch, 20, 26, frames).permute(0, 3, 1, 2)
        symbolic = torch.einsum(
            "ntjk,ke->ntje", probabilities, self.hl_embedding.weight
        ).reshape(batch, frames, 160)
        encoded = self.encoder(features + self.hl_input(symbolic) + self.position[:, :frames])
        symbols = self.symbol(encoded).reshape(batch, frames, 20, 26)
        residual = self.direction(encoded).reshape(batch, frames, 20, 3)
        return normalize(directions + residual), symbols


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    args = checkpoint["args"]
    model = HLRepair(
        hidden=args["hidden"], layers=args["layers"], heads=args["heads"],
        hl_smoothing_window=args.get("hl_smoothing_window", 3),
        hl_temperature=args.get("hl_temperature", 0.24))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval().to(device)
