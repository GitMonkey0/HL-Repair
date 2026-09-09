#!/usr/bin/env python3
"""Compare a reproduced STB evaluation with the released paper values."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reproduced")
    parser.add_argument("--atol", type=float, default=5e-5)
    args = parser.parse_args()
    got = json.loads(Path(args.reproduced).read_text())
    expected = {
        "corrupted": {"mean_angular_error_deg": 13.241168022155762,
                      "hl_symbol_accuracy_percent": 82.79412078857422,
                      "direction_acceleration_error": 0.6003938317298889},
        "median_9": {"mean_angular_error_deg": 4.258885383605957,
                     "hl_symbol_accuracy_percent": 91.2616958618164,
                     "direction_acceleration_error": 0.10516221821308136},
        "hl_repair": {"mean_angular_error_deg": 2.9164938926696777,
                      "hl_symbol_accuracy_percent": 92.7423095703125,
                      "direction_acceleration_error": 0.08258561789989471},
    }
    failures = []
    for method, metrics in expected.items():
        for metric, value in metrics.items():
            delta = abs(got[method][metric] - value)
            if delta > args.atol:
                failures.append(f"{method}.{metric}: |delta|={delta:g}")
    if failures:
        raise SystemExit("evaluation mismatch\n" + "\n".join(failures))
    print("STB evaluation matches the released checkpoint and protocol.")


if __name__ == "__main__":
    main()
