# Frozen-Wan downstream evaluation

HL-Repair changes only the tracker-side 17-frame hand-control signal. The
video generator remains frozen. The paper used:

- Wan2.1-Fun-V1.1-1.3B-Control revision
  `a116c52b08182f7a5916eae7a47fcd61a9467a04`;
- VideoX-Fun commit `968f0e2192ba4c7a12868bf36d73260d135424ca`;
- 50 non-overlapping B1 STB trajectories;
- four seeds: 43, 101, 202, and 303;
- five controls: clean, corrupted, centered median-9, continuous-only, and
  HL-Repair, totaling 1,000 videos;
- 17 frames at 256 x 256, 20 denoising steps, guidance 6.0;
- Wan denoising on Ascend and VAE encoding/decoding on CPU.

The positive prompt was:

```text
A realistic person facing the camera, clearly moving both hands, fixed camera, natural anatomy
```

The negative prompt was:

```text
deformed hands, extra fingers, fused fingers, missing fingers, extra arms, blur, text, cartoon
```

Every comparison used matched prompts, settings, and initial noise. The full
generation signature, model hashes, selected trajectory indices, inference
procedure, and aggregate statistics are in `results/wan_summary.json`. The run
used 58.14 aggregate accelerator-hours. Because Wan weights and generated
videos are large and separately licensed, they are intentionally not included
in this repository.

To repeat generation, install the pinned VideoX-Fun revision and its declared
dependencies, download the pinned Wan model revision, render each repaired
local-direction window back through the same calibrated STB carrier, and feed
the resulting skeleton video through Wan's native control interface. Verify
the component SHA-256 values against `generation_signature` in the report
before comparing results.
