#!/usr/bin/env python3
"""Write SHA-256 hashes for release artifacts."""

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGETS = [ROOT / "checkpoints/hl_repair_stb.pt", ROOT / "paper/HL_Repair.pdf"]
TARGETS += sorted((ROOT / "results").glob("*.json"))
TARGETS += sorted((ROOT / "assets").glob("*"))
lines = [f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(ROOT)}"
         for path in TARGETS if path.is_file()]
(ROOT / "SHA256SUMS").write_text("\n".join(lines) + "\n")
