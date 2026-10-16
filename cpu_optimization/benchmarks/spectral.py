"""Full FourierUnit microbenchmarks with actual captured activations and weights."""

import copy
import csv
import random
import time
from .common import *
from ..candidates.spectral import SpectralGEMM, PackedFourierUnit, BufferFourierUnit
from ..quality import compare_outputs
from ..statistics import paired_summary


def cohort_row(shape, candidate, stats, parity):
    return dict(
        shape=shape,
        candidate=candidate,
        **stats,
        output_shape=parity["shape"],
        **{k: v for k, v in parity.items() if k != "shape"},
    )


def main():
    p = parser(__doc__)
    p.set_defaults(repeats=60)
    p.add_argument("--profile", type=Path, required=True)
    args = p.parse_args()
    configure_threads(args.threads)
    run = new_run(args.id, "spectral_micro", args.results_dir, args.checkpoint)
    model = load_reference(args.checkpoint)
    inventory = json.loads((args.profile / "fourier_inventory.json").read_text())
    modules = dict(model.named_modules())
    write_json(
        run / "inputs.json",
        dict(
            activation_archive=str(args.profile / "fourier_inputs.npz"),
            activation_sha256=digest(args.profile / "fourier_inputs.npz"),
            inventory_sha256=digest(args.profile / "fourier_inventory.json"),
        ),
    )
    rng = random.Random(20261016)
    rows = []
    raw = []
    with np.load(args.profile / "fourier_inputs.npz") as saved, torch.inference_mode():
        for key in saved.files:
            x = torch.from_numpy(saved[key])
            entry = next(e for e in inventory if str(tuple(e["shape"])) == key)
            ref = modules[entry["name"]]
            for name in ["linear", "bmm", "matmul", "packed_gemm", "static_buffers"]:
                cand = copy.deepcopy(ref)
                if name in ["linear", "bmm", "matmul"]:
                    cand.conv_layer = SpectralGEMM(cand.conv_layer, name)
                elif name == "packed_gemm":
                    cand = PackedFourierUnit(cand)
                else:
                    cand = BufferFourierUnit(cand)
                parity = compare_outputs(ref(x).numpy(), cand(x).numpy())
                for _ in range(args.warmups):
                    ref(x)
                    cand(x)
                bt, ct = [], []
                for i in range(args.repeats):
                    order = [0, 1]
                    rng.shuffle(order)
                    dt = {}
                    for arm in order:
                        start = time.perf_counter_ns()
                        (ref if arm == 0 else cand)(x)
                        dt[arm] = (time.perf_counter_ns() - start) * 1e-9
                    bt.append(dt[0])
                    ct.append(dt[1])
                    raw.append(
                        dict(
                            shape=key,
                            candidate=name,
                            pair=i,
                            order=str(order),
                            reference_s=dt[0],
                            candidate_s=dt[1],
                        )
                    )
                rows.append(cohort_row(key, name, paired_summary(bt, ct), parity))
                append_rows(run / "fourier_units.csv", rows[-1:])
                append_rows(run / "raw_timings.csv", raw[-len(bt) :])
            print(f"completed FourierUnit {key}", flush=True)
    finish(
        run,
        dict(
            status="inconclusive",
            reason="Full-unit microbenchmarks only; no network promotion",
            shapes=len(saved.files),
            rows=len(rows),
            warmups=args.warmups,
            pairs=args.repeats,
        ),
    )


if __name__ == "__main__":
    cli(main)
