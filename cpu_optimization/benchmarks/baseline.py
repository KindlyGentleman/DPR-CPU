"""Warm persistent-model CPU baseline; startup and first use recorded separately."""

import csv
import time
import numpy as np
import torch
from .common import *
from ..candidates import NAMES, apply_candidate
from ..quality import compare_outputs


def main():
    p = parser(__doc__)
    p.add_argument("--candidate", choices=sorted(NAMES), default="reference")
    args = p.parse_args()
    configure_threads(args.threads)
    run = new_run(args.id, args.candidate, args.results_dir, args.checkpoint)
    cases = load_cases(args.development)
    write_json(run / "inputs.json", cases_identity(cases))
    inputs = [prepare_inputs(c["intensity"], c["valid"]) for c in cases]
    start = time.perf_counter()
    model = apply_candidate(load_reference(args.checkpoint), args.candidate)
    startup = time.perf_counter() - start
    times = []
    with torch.inference_mode():
        out, cold = timed(model, *inputs[0])
        parity = compare_outputs(cases[0]["archived_reference"], out.numpy()[0, 0])
        for i in range(args.warmups):
            model(*inputs[i % len(inputs)], false_scale=False)
        with (run / "timings.csv").open("x", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["repeat", "sample_id", "seconds", "rss_bytes"]
            )
            writer.writeheader()
            for i in range(args.repeats):
                _, elapsed = timed(model, *inputs[i % len(inputs)])
                times.append(elapsed)
                writer.writerow(
                    dict(
                        repeat=i,
                        sample_id=cases[i % len(cases)]["sample_id"],
                        seconds=elapsed,
                        rss_bytes=psutil.Process().memory_info().rss,
                    )
                )
                f.flush()
    finish(
        run,
        dict(
            status="inconclusive",
            reason="Baseline characterization; no candidate speedup claim",
            candidate=args.candidate,
            warmups=args.warmups,
            repeats=args.repeats,
            startup_s=startup,
            cold_call_s=cold,
            median_s=float(np.median(times)),
            p95_s=float(np.quantile(times, 0.95)),
            archive_parity=parity,
            peak_memory=psutil.Process().memory_info()._asdict(),
            boundary="prepared network only; excludes loading, conversion, metrics and serialization",
        ),
    )


if __name__ == "__main__":
    cli(main)
