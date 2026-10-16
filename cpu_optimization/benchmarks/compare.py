"""Development gate followed by randomized paired prepared-network inference."""

import csv
import random
import time
import traceback
import numpy as np
import torch
from .common import *
from ..candidates import NAMES, apply_candidate
from ..quality import compare_outputs, image_metrics, physics_metrics
from ..statistics import paired_summary, decision


def affinity_for(name):
    available = psutil.Process().cpu_affinity()
    if name == "default":
        return available
    cores = topology()["cores"]
    if not cores:
        raise RuntimeError("Verified physical core topology unavailable")
    if name == "p_only":
        ids = [v for c in cores if c["smt"] for v in c["logical_cpus"]]
    elif name == "p_physical":
        ids = [c["logical_cpus"][0] for c in cores if c["smt"]]
    elif name == "mixed_physical":
        ids = [c["logical_cpus"][0] for c in cores]
    elif name == "e_only":
        ids = [v for c in cores if not c["smt"] for v in c["logical_cpus"]]
    else:
        raise ValueError("Unknown affinity")
    ids = sorted(set(ids) & set(available))
    if not ids:
        raise RuntimeError(f"No eligible cores for {name}")
    return ids


def evaluate(reference, candidate, cases, inputs, select):
    rows = []
    predictions = {}
    passed = True
    with torch.inference_mode():
        for i, (case, inp) in enumerate(zip(cases, inputs)):
            select(False)
            b = reference(*inp, false_scale=False).numpy()[0, 0].copy()
            select(True)
            c = candidate(*inp, false_scale=False).numpy()[0, 0].copy()
            parity = compare_outputs(b, c)
            bm = image_metrics(case["truth"], b)
            cm = image_metrics(case["truth"], c)
            nrmse_increase = cm["aligned_nrmse"] - bm["aligned_nrmse"]
            ssim_loss = bm["aligned_ssim"] - cm["aligned_ssim"]
            new_failure = cm["failure"] > bm["failure"]
            ok = (
                parity["allclose"]
                and nrmse_increase <= 1e-4
                and ssim_loss <= 1e-4
                and not new_failure
            )
            passed = passed and ok
            rows.append(
                dict(
                    sample_id=case["sample_id"],
                    base_id=case["base_id"],
                    parity=parity,
                    reference=bm,
                    candidate=cm,
                    physics_reference=physics_metrics(
                        b, case["intensity"], case["valid"]
                    ),
                    physics_candidate=physics_metrics(
                        c, case["intensity"], case["valid"]
                    ),
                    nrmse_increase=nrmse_increase,
                    ssim_loss=ssim_loss,
                    psnr_loss=bm["aligned_psnr"] - cm["aligned_psnr"],
                    support_iou_loss=bm["support_iou"] - cm["support_iou"],
                    boundary_f1_loss=bm["boundary_f1"] - cm["boundary_f1"],
                    new_failure=new_failure,
                    gate_pass=ok,
                    archived_reference_parity=compare_outputs(
                        case["archived_reference"], b
                    ),
                )
            )
            predictions[f"reference_{i}"] = b
            predictions[f"candidate_{i}"] = c
    return passed, rows, predictions


def main(validate_only=False):
    p = parser(__doc__)
    p.add_argument("--candidate", choices=sorted(NAMES), required=True)
    p.add_argument("--candidate-threads", type=int, help="Default: same as --threads")
    p.add_argument(
        "--affinity",
        choices=["default", "p_only", "p_physical", "mixed_physical", "e_only"],
        default="default",
    )
    p.add_argument("--seed", type=int, default=2026101601)
    p.add_argument(
        "--time-rejected",
        action="store_true",
        help="Diagnostic timing even after failed fidelity; never promotes",
    )
    p.add_argument(
        "--archived",
        action="store_true",
        help="Final replay of five published examples after selection lock",
    )
    args = p.parse_args()
    configure_threads(args.threads)
    run = new_run(args.id, args.candidate, args.results_dir, args.checkpoint)
    original_aff = psutil.Process().cpu_affinity()
    candidate_aff = affinity_for(args.affinity)
    ct = args.candidate_threads or args.threads
    if args.archived:
        require_selection(args.candidate, ct, candidate_aff)
    cases = load_cases(args.development, args.archived)
    inputs = [prepare_inputs(c["intensity"], c["valid"]) for c in cases]
    write_json(run / "inputs.json", cases_identity(cases))
    rng = random.Random(args.seed)

    def select(is_candidate):
        requested = ct if is_candidate else args.threads
        if torch.get_num_threads() != requested:
            torch.set_num_threads(requested)
        target = candidate_aff if is_candidate else original_aff
        if psutil.Process().cpu_affinity() != target:
            psutil.Process().cpu_affinity(target)

    try:
        start = time.perf_counter()
        reference = load_reference(args.checkpoint)
        candidate = apply_candidate(load_reference(args.checkpoint), args.candidate)
        setup = time.perf_counter() - start
        with torch.inference_mode():
            select(False)
            _, cold_b = timed(reference, *inputs[0])
            select(True)
            _, cold_c = timed(candidate, *inputs[0])
        quality, rows, predictions = evaluate(
            reference, candidate, cases, inputs, select
        )
        write_json(run / "validation.json", dict(passed=quality, cases=rows))
        np.savez_compressed(run / "predictions.npz", **predictions)
        summary = dict(
            candidate=args.candidate,
            track="E",
            setup_s=setup,
            cold_reference_s=cold_b,
            cold_candidate_s=cold_c,
            candidate_threads=ct,
            reference_threads=args.threads,
            candidate_affinity=candidate_aff,
            affinity_name=args.affinity,
            warmups=args.warmups,
            quality_pass=quality,
            validation_cases=len(cases),
            independent_bases=len(set(c["base_id"] for c in cases)),
            max_abs=max(r["parity"]["max_abs"] for r in rows),
            max_relative_l2=max(r["parity"]["relative_l2"] for r in rows),
            max_nrmse_increase=max(r["nrmse_increase"] for r in rows),
            max_ssim_loss=max(r["ssim_loss"] for r in rows),
            new_failures=sum(r["new_failure"] for r in rows),
            bitwise=all(r["parity"]["bitwise"] for r in rows),
            hypothesis={
                "spectral_linear": "NHWC linear may improve spectral pointwise execution",
                "spectral_bmm": "Group-aware pre-shaped batched GEMM avoids convolution dispatch",
                "spectral_matmul": "Broadcast GEMM avoids convolution dispatch",
                "packed_gemm": "NHWC interleaved GEMM avoids inverse packing allocation",
                "static_buffers": "Instance-local static packing buffers reduce allocations",
                "local_channels_last": "Local spectral channels-last specializes layout without global transitions",
                "encoder_channels_last": "WPC-only channels-last with NCHW restoration before FFC avoids global layout penalty",
                "mkldnn_prepack": "Prepacked spatial convolutions may avoid repeated weight reorders; reflect padding retained outside oneDNN",
                "prepack_bmm": "Disjoint spatial oneDNN prepacking and spectral BMM gains may combine; joint full-model experiment required",
                "reference": "Thread/core scheduling improves batch-one latency",
            }.get(
                args.candidate,
                "Compiler may fuse pointwise operations around eager FFTs",
            ),
        )
        if validate_only or (not quality and not args.time_rejected):
            finish(
                run,
                {
                    **summary,
                    "status": "inconclusive" if quality else "rejected",
                    "reason": (
                        "Validation only"
                        if quality
                        else "Strict development numerical/scientific gate failed; no timing promotion"
                    ),
                },
            )
            return
        btimes, ctimes = [], []
        telemetry = []
        with torch.inference_mode():
            for i in range(args.warmups):
                for is_c in (False, True):
                    select(is_c)
                    (candidate if is_c else reference)(
                        *inputs[i % len(inputs)], false_scale=False
                    )
            with (run / "paired_timings.csv").open("x", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "pair",
                        "sample_id",
                        "order",
                        "reference_s",
                        "candidate_s",
                    ],
                )
                writer.writeheader()
                for i in range(args.repeats):
                    order = [False, True]
                    rng.shuffle(order)
                    elapsed = {}
                    for is_c in order:
                        select(is_c)
                        _, elapsed[is_c] = timed(
                            candidate if is_c else reference, *inputs[i % len(inputs)]
                        )
                    btimes.append(elapsed[False])
                    ctimes.append(elapsed[True])
                    writer.writerow(
                        dict(
                            pair=i,
                            sample_id=cases[i % len(cases)]["sample_id"],
                            order="CB" if order[0] else "BC",
                            reference_s=elapsed[False],
                            candidate_s=elapsed[True],
                        )
                    )
                    f.flush()
                    if i % 6 == 0:
                        telemetry.append(
                            dict(
                                pair=i,
                                cpu_frequency=psutil.cpu_freq()._asdict(),
                                rss=psutil.Process().memory_info().rss,
                            )
                        )
                        print(
                            f"pair {i+1}/{args.repeats}: reference={elapsed[False]:.4f}s candidate={elapsed[True]:.4f}s",
                            flush=True,
                        )
        stats = paired_summary(btimes, ctimes, args.seed)
        write_json(run / "telemetry.json", telemetry)
        status = decision(stats, quality, False, False)
        finish(
            run,
            {
                **summary,
                **stats,
                "status": status,
                "reason": (
                    "All fidelity checks passed; promotion requires speedup/p95 gates and isolated memory/fresh-process confirmation"
                    if quality
                    else "Rejected numerical; diagnostic timings only"
                ),
                "fresh_process_confirmed": False,
                "memory_gate": None,
                "memory_note": "Lifetime peak includes both model instances and quality caches; not isolated candidate peak",
                "peak_memory": psutil.Process().memory_info()._asdict(),
                "boundary": "prepared network only; thread/affinity switches outside clock; validation/I/O excluded",
            },
        )
    except Exception as exc:
        (run / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        finish(
            run,
            dict(
                status="blocked",
                candidate=args.candidate,
                reason=f"{type(exc).__name__}: {str(exc)[:1800]}",
                warmups=args.warmups,
                pairs=0,
            ),
        )
    finally:
        psutil.Process().cpu_affinity(original_aff)
        torch.set_num_threads(args.threads)


if __name__ == "__main__":
    cli(main)
