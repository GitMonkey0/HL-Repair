#!/usr/bin/env bash
set -euo pipefail
python -m pytest -q
mkdir -p outputs
python -c 'import numpy as np; x=np.random.default_rng(0).normal(size=(17,20,3)).astype("float32"); x/=np.linalg.norm(x,axis=-1,keepdims=True); np.save("outputs/smoke_input.npy",x)'
python -m hlrepair.infer outputs/smoke_input.npy outputs/smoke_output.npy
