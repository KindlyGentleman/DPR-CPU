"""Lock a replicated fixed-worker configuration before any published-case replay."""

import argparse
import json
from datetime import datetime, timezone
from ..benchmarks.common import HOME, CHECKPOINT_SHA256, digest, write_json
from ..statistics import decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--repeat", required=True)
    args = parser.parse_args()
    path = HOME / "config/selection.json"
    if path.exists():
        raise FileExistsError(
            "Selection lock is immutable; do not retune on final replay"
        )
    if args.trial == args.repeat:
        raise ValueError("Independent process repetition is required")
    ids = [args.trial, args.repeat]
    records = []
    for ident in ids:
        directory = HOME / "results" / ident
        meta = json.loads((directory / "run.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        workers = json.loads((directory / "workers.json").read_text())
        quality = json.loads((directory / "validation.json").read_text())
        if summary.get("methodology") != "persistent_worker_processes":
            raise ValueError("Global-pool-switch scheduling screens are ineligible")
        if meta["checkpoint_sha256"] != CHECKPOINT_SHA256:
            raise ValueError("Checkpoint identity mismatch")
        if (
            decision(summary, quality["passed"], True, summary["memory_gate"])
            != "accepted"
        ):
            raise ValueError(f"{ident} does not clear all prespecified gates")
        if (
            summary["warmups"] < 10
            or summary["validation_cases"] != 18
            or summary["independent_bases"] != 7
        ):
            raise ValueError("Incomplete warmup/development validation")
        records.append(
            dict(id=ident, summary=summary, meta=meta, workers=workers, quality=quality)
        )
    first, repeat = records
    for key in ["worker_settings", "worker_thread_environment"]:
        if first["summary"][key] != repeat["summary"][key]:
            raise ValueError("Repetition used different settings")
    if (
        first["workers"][1]["environment"]["versions"]
        != repeat["workers"][1]["environment"]["versions"]
    ):
        raise ValueError("Repetition used different native runtime versions")
    if {w["pid"] for w in first["workers"]} & {w["pid"] for w in repeat["workers"]}:
        raise ValueError("Worker PID independence requires manual audit before lock")
    rows = repeat["quality"]["cases"]
    s = repeat["summary"]
    selected = s["worker_settings"][1]
    report = (
        f"**Replicated development-qualified configuration:** backend `{selected['backend']}`, "
        f"{selected['threads']} intra-op / one inter-op threads, P-only affinity {selected['affinity']}. "
        f"Fresh {s['pairs']}-pair confirmation: median **{s['candidate_median_s']:.6f}s**, "
        f"p95 **{s['candidate_p95_s']:.6f}s**, speedup **{s['speedup']:.6f}x "
        f"[{s['ci_low']:.6f}, {s['ci_high']:.6f}]**, versus nearby "
        f"{s['baseline_median_s']:.6f}s median / {s['baseline_p95_s']:.6f}s p95. "
        "The gain is narrow and machine/session-specific. No component timings or historical denominator are multiplied into it."
    )
    write_json(
        path,
        dict(
            locked_utc=datetime.now(timezone.utc).isoformat(),
            locked_before_published_replay=True,
            decision="accepted replicated development configuration; final published replay still required",
            selected=selected,
            reference=s["worker_settings"][0],
            evidence_ids=ids,
            source_artifacts={
                r["id"]: {
                    name: digest(HOME / "results" / r["id"] / name)
                    for name in [
                        "run.json",
                        "summary.json",
                        "validation.json",
                        "paired_timings.csv",
                        "workers.json",
                        "source.zip",
                    ]
                }
                for r in records
            },
            checkpoint_sha256=CHECKPOINT_SHA256,
            protocol_sha256=digest(HOME / "config/protocol.json"),
            development_bundle_sha256=digest(HOME / "config/development_inputs.npz"),
            worker_environment=s["worker_thread_environment"],
            runtime=repeat["workers"][1]["environment"],
            quality_review=dict(
                max_abs=s["max_abs"],
                max_relative_l2=s["max_relative_l2"],
                max_nrmse_increase=s["max_nrmse_increase"],
                max_ssim_loss=s["max_ssim_loss"],
                max_psnr_loss=max(r["psnr_loss"] for r in rows),
                max_support_iou_change=max(abs(r["support_iou_loss"]) for r in rows),
                max_boundary_f1_change=max(abs(r["boundary_f1_loss"]) for r in rows),
                new_failures=s["new_failures"],
                reference_failures=sum(r["reference"]["failure"] for r in rows),
                candidate_failures=sum(r["candidate"]["failure"] for r in rows),
            ),
            memory_review=s["peak_memory"],
            auto_backend="reference; portable default remains four threads and inherited affinity",
            report=report,
            limits="Same-day Windows/i7-13620H/Torch2.5.1 CPU evidence, unavailable temperature; raw570 held-out arrays unavailable; five published examples only final replay; not bitwise but all strict elementwise/scientific gates pass",
        ),
    )
    print(report)


if __name__ == "__main__":
    main()
