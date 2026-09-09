"""Hand geometry and the fixed 26-direction HL codebook."""

import torch

HL_PALM_PARENTS = torch.tensor(
    [0, 0, 1, 2, 3, 9, 5, 6, 7, 0, 9, 10, 11, 9, 13, 14, 15, 13, 17, 18, 19]
)


def normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def hl_centers(device=None, dtype=torch.float32) -> torch.Tensor:
    values = [(x, y, z) for x in (-1.0, 0.0, 1.0)
              for y in (-1.0, 0.0, 1.0) for z in (-1.0, 0.0, 1.0)
              if (x, y, z) != (0.0, 0.0, 0.0)]
    return normalize(torch.tensor(values, device=device, dtype=dtype))


def palm_frame(joints: torch.Tensor) -> torch.Tensor:
    """Return a right-handed local frame for 21-joint hands."""
    y = normalize(joints[:, 9] - joints[:, 0])
    across = normalize(joints[:, 17] - joints[:, 5])
    z = normalize(torch.linalg.cross(across, y, dim=-1))
    x = torch.linalg.cross(y, z, dim=-1)
    return torch.stack((x, y, z), dim=-1)


def joints_to_directions(joints: torch.Tensor, parents=HL_PALM_PARENTS):
    parents = parents.to(joints.device)
    vectors = joints[:, 1:] - joints[:, parents[1:]]
    lengths = vectors.norm(dim=-1)
    world = vectors / lengths[..., None].clamp_min(1e-9)
    frame = palm_frame(joints)
    local = torch.einsum("nij,nki->nkj", frame, world)
    return local, lengths, frame
