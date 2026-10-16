"""Explicit batch throughput diagnostic, separate from the batch-one promotion track."""

import csv
import numpy as np
import torch
from .common import *
from ..quality import compare_outputs, image_metrics


def main():
    p = parser(__doc__)
    p.add_argument("--batch-size", type=int, default=2)
    args = p.parse_args()
    if args.batch_size < 1:
        p.error("--batch-size must be positive")
    configure_threads(args.threads)
    run = new_run(args.id, "batch_throughput", args.results_dir, args.checkpoint)
    cases = load_cases(args.development)
    selected = [cases[i % len(cases)] for i in range(args.batch_size)]
    write_json(run / "inputs.json", cases_identity(selected))
    tensors = [prepare_inputs(c["intensity"], c["valid"]) for c in selected]
    x = torch.cat([a[0] for a in tensors])
    m = torch.cat([a[1] for a in tensors])
    model = load_reference(args.checkpoint)
    times = []
    with torch.inference_mode():
        single = [model(*a, false_scale=False).numpy()[0, 0].copy() for a in tensors]
        batch = model(x, m, false_scale=False).numpy().copy()
        parity = [compare_outputs(a, b[0]) for a, b in zip(single, batch)]
        metrics = [
            dict(
                single=image_metrics(c["truth"], a),
                batch=image_metrics(c["truth"], b[0]),
            )
            for a, b, c in zip(single, batch, selected)
        ]
        for _ in range(args.warmups):
            model(x, m, false_scale=False)
        for i in range(args.repeats):
            _, elapsed = timed(model, x, m)
            times.append(elapsed)
    with (run / "timings.csv").open("x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["repeat", "batch_seconds"])
        writer.writerows(enumerate(times))
    finish(
        run,
        dict(
            status="inconclusive",
            reason="Batch throughput characterization; not a paired batch-one speedup",
            batch_size=args.batch_size,
            warmups=args.warmups,
            repeats=args.repeats,
            median_s=float(np.median(times)),
            p95_s=float(np.quantile(times, 0.95)),
            samples_per_second_at_median_latency=args.batch_size
            / float(np.median(times)),
            observed_samples_per_second=args.batch_size
            * len(times)
            / float(sum(times)),
            parity_against_single=parity,
            scientific_metrics=metrics,
        ),
    )


if __name__ == "__main__":
    cli(main)
