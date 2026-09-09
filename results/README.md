# Released results

- `stb_summary.json`: primary control-domain table and paired intervals.
- `hl_strength_five_seeds.json`: five-seed inference-time HL audit.
- `smoothnet_five_seeds.json`: capacity-matched SmoothNet comparison.
- `cpu_latency.json`: single-thread CPU deployment benchmark.
- `reproduced_cpu.json`: clean-repository CPU reproduction of the released
  checkpoint on all 176 held-out STB windows.
- `wan_summary.json`: complete aggregate record for the 1,000-video frozen-Wan
  study, including generation signatures and per-trajectory statistics.

Paths inside the JSON files preserve the original experiment-run layout for
provenance. They are records, not expected paths in a fresh clone.
