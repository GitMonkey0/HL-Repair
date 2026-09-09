"""STB annotation loading and the paper's synthetic tracker corruption."""

import json
from collections import defaultdict
from pathlib import Path

import torch

from .geometry import HL_PALM_PARENTS, joints_to_directions, normalize

STB_TO_STANDARD = (0, 17, 18, 19, 20, 13, 14, 15, 16, 9, 10,
                   11, 12, 5, 6, 7, 8, 1, 2, 3, 4)


def load_stb_windows(path, stride, prefixes=()):
    data = json.loads(Path(path).read_text())
    images = {item["id"]: item for item in data["images"]}
    grouped = defaultdict(list)
    for annotation in data["annotations"]:
        image = images[annotation["image_id"]]
        if "_left_" not in image["file_name"]:
            continue
        if prefixes and not image["seq_name"].startswith(tuple(prefixes)):
            continue
        grouped[image["seq_name"]].append(
            (annotation["image_id"], annotation["joint_cam"]))
    windows, counts = [], {}
    for name, records in sorted(grouped.items()):
        records.sort(key=lambda x: x[0])
        joints = torch.tensor([x[1] for x in records], dtype=torch.float32)
        joints = joints[:, STB_TO_STANDARD]
        joints[..., 0] *= -1
        directions, _, _ = joints_to_directions(joints, HL_PALM_PARENTS)
        current = [directions[i:i + 17] for i in range(0, len(directions) - 16, stride)]
        windows.extend(current)
        counts[name] = {"frames": len(directions), "windows": len(current)}
    if not windows:
        raise ValueError(f"no 17-frame STB windows found in {path}")
    return torch.stack(windows), counts


def corrupt(clean, seed=9917, jitter=(0.03, 0.10), outlier_p=0.10, stale_p=0.08):
    generator = torch.Generator().manual_seed(seed)
    sigma = torch.empty((len(clean), 1, 1, 1)).uniform_(*jitter, generator=generator)
    noisy = normalize(clean + torch.randn(clean.shape, generator=generator) * sigma)
    outlier = torch.rand((*clean.shape[:-1], 1), generator=generator) < outlier_p
    noisy = torch.where(outlier, normalize(torch.randn(clean.shape, generator=generator)), noisy)
    stale = torch.rand((*clean.shape[:-1], 1), generator=generator) < stale_p
    stale[:, 0] = False
    for frame in range(1, clean.shape[1]):
        noisy[:, frame] = torch.where(stale[:, frame], noisy[:, frame - 1], noisy[:, frame])
    return noisy, outlier, stale
