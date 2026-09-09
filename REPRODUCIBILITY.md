# Reproducibility

## Scope

The repository separates three levels of reproduction:

1. `pytest` validates the codebook, tensor contract, released checkpoint, and
   inference path without external data.
2. `scripts/reproduce_stb.sh` retrains and evaluates HL-Repair on the official
   STB annotations using the paper's fixed split and corruption protocol.
3. `WAN.md` records the frozen-Wan generation protocol, exact revisions, model
   hashes, and the computational cost of the 1,000-video downstream study.

The generated videos and third-party model weights are not redistributed.
Their aggregate machine-readable report is included as
`results/wan_summary.json`.

## Locked STB protocol

- Train: official `STB_train.json`, sequences B2--B6, left camera only.
- Test: official `STB_test.json`, B1Counting and B1Random, left camera only.
- Window length: 17; train stride: 4; test stride: 17.
- Training seed: 2801; fixed test-corruption seed: 9917.
- Corruption curriculum: clean, isolated jitter, isolated outliers, isolated
  stale observations, mild mixed, two nominal mixed, and severe mixed copies.
- Model: width 192, four Transformer blocks, six heads.
- HL posterior: three-frame pooling, temperature 0.24.
- Loss: reconstruction + 0.2 velocity + 0.003 geometry-aware HL loss.
- Optimizer: AdamW, learning rate 3e-4, weight decay 1e-4, 40 epochs.
- Inference gate: repair when mean deviation from centered median-9 is at least
  3.75 degrees.

The annotation files used for the paper have SHA-256 digests:

```text
1cbc72f5bfcd4969f21896bf2459912b2dc38ecd7671543496e57d74d28ab533  STB_train.json
856699afce05a43c8edef68b88637ac53ef1231240eaaedc5461b6100bf74021  STB_test.json
```

## Expected outputs

The released seed-2801 checkpoint has SHA-256
`ce93d872aebd4a06be77a7fb720a083acc24f6f955cd5a1b4e01b63e21bd5ef5`.
Exact aggregate reports used by the paper are retained under `results/`.
Accelerator kernels can introduce small floating-point differences; compare
the reported metrics at the precision used in the paper rather than requiring
byte-identical checkpoints across CPU, CUDA, and Ascend.

For the released checkpoint, a full CPU verification is:

```bash
hlrepair-eval --annotations data/STB.annotations/STB_test.json \
  --checkpoint checkpoints/hl_repair_stb.pt \
  --output outputs/evaluation.json --device cpu
python scripts/verify_evaluation.py outputs/evaluation.json
```

## Hardware

The reported training runs used Python 3.11.14, PyTorch 2.9.0,
torch-npu 2.9.0, CANN 8.5.1, and an Ascend 910B2C. The core implementation is
device-agnostic and is tested on CPU. For Ascend, install the torch-npu build
matching the local PyTorch and CANN versions, then set `DEVICE=npu:0`.
