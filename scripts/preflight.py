#!/usr/bin/env python3
"""Lightweight release audit for the public repository."""

import hashlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md", "LICENSE", "CITATION.cff", "MODEL_CARD.md",
    "REPRODUCIBILITY.md", "WAN.md", "pyproject.toml",
    "checkpoints/hl_repair_stb.pt", "paper/HL_Repair.pdf",
    "results/stb_summary.json", "results/wan_summary.json",
)
CHECKPOINT_SHA256 = "ce93d872aebd4a06be77a7fb720a083acc24f6f955cd5a1b4e01b63e21bd5ef5"


def main():
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit("missing release files: " + ", ".join(missing))
    digest = hashlib.sha256((ROOT / "checkpoints/hl_repair_stb.pt").read_bytes()).hexdigest()
    if digest != CHECKPOINT_SHA256:
        raise SystemExit("released checkpoint hash mismatch")
    oversized = [p.relative_to(ROOT) for p in ROOT.rglob("*")
                 if p.is_file() and p.stat().st_size > 50_000_000]
    if oversized:
        raise SystemExit(f"files exceed 50 MB: {oversized}")
    subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, check=True)
    print("Repository preflight passed.")


if __name__ == "__main__":
    main()
