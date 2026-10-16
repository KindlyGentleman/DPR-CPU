"""Regenerate ranked tables and eight technical reports from recorded evidence."""

import argparse
import csv
import json
from pathlib import Path
from .common import HOME, write_json


def collect():
    trials = []
    for f in sorted((HOME / "results").glob("*/summary.json")):
        s = json.loads(f.read_text(encoding="utf-8"))
        m = json.loads((f.parent / "run.json").read_text(encoding="utf-8"))
        # Characterization/probe summaries use the durable run identity. Retain
        # original evidence unchanged and normalize only this read view.
        s.setdefault("candidate", m["candidate"])
        s.setdefault("track", m.get("track", "E"))
        trials.append(dict(id=f.parent.name, summary=s, meta=m, path=f.parent))
    return trials


def concise_reason(summary):
    reason = summary["reason"]
    if "cl is not found" in reason:
        return "Inductor cannot find MSVC cl; full traceback retained in error.txt"
    return reason.splitlines()[0][:300].rstrip()


def exploratory(t):
    return (
        "speedup" in t["summary"]
        and t["summary"]["candidate"] == "reference"
        and t["summary"].get("methodology") != "persistent_worker_processes"
    )


def fft_table(trial):
    with (trial["path"] / "fft.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = "| Shape | Direction | Torch median (ms) | Fastest passing external bridge | Bridge median (ms) | Its setup (ms) |\n|---|---|---:|---|---:|---:|\n"
    for shape, direction in dict.fromkeys((r["shape"], r["direction"]) for r in rows):
        group = [r for r in rows if r["shape"] == shape and r["direction"] == direction]
        native = next(
            r for r in group if r["backend"] == "torch" and r["boundary"] == "kernel"
        )
        bridges = [
            r for r in group if r["boundary"] == "bridge" and r["allclose"] == "True"
        ]
        if bridges:
            best = min(bridges, key=lambda r: float(r["median_s"]))
            result += f"| {shape} | {direction} | {float(native['median_s'])*1000:.4f} | {best['backend']} | {float(best['median_s'])*1000:.4f} | {float(best['setup_s'])*1000:.3f} |\n"
        else:
            result += f"| {shape} | {direction} | {float(native['median_s'])*1000:.4f} | none passed strict parity | — | — |\n"
    result += f"\nStrict operator parity failed in {sum(r['allclose'] != 'True' for r in rows)} of {len(rows)} cohorts; flags and max errors remain in fft.csv. Fastest-bridge selection is descriptive across sequential cohorts, not a paired network result.\n\n"
    result += "Eight feature shapes use actual captured FourierUnit operands. The extra 512x512 cohort applies a real FFT to the actual signed detector input as a diagnostic; it is not the model's captured physics operand. Projection/scaling use full fft2/ifft2, not this one-sided transform. The 12 diagnostic detector-input failures therefore do not establish failure of a full physics backend, which was not integrated.\n\n"
    if (trial["path"] / "fresh_estimate_plan_timings.csv").exists():
        result += "Fresh ESTIMATE construction/allocation plus bridge is separate, with existing process wisdom retained; it is not cold MEASURE planning. All raw calls are in fresh_estimate_plan_timings.csv. Exported wisdom, native FFTW version and immediate reimport outcomes are in fftw_wisdom.json, without production auto-loading.\n\n"
        result += "Evidence erratum: fft_micro_v1's final note says 'no wisdom persisted'; that text predates the added diagnostic export. The retained fftw_wisdom.json proves an export and immediate reimport occurred after timing. No saved wisdom was automatically loaded before these cohorts. The native version API was empty in this wheel; raw wisdom is retained rather than inventing a version value.\n\n"
    return result


def table(trials, accepted_ids=()):
    out = [
        "| Experiment | Track | Reference median / p95 (s) | Candidate median / p95 (s) | Speedup [95% CI] | Quality | Decision |",
        "|---|---|---:|---:|---:|---|---|",
    ]

    def score(t):
        return t["summary"].get("speedup", -1)

    for t in sorted(trials, key=score, reverse=True):
        s = t["summary"]
        b = c = v = "—"
        if "speedup" in s:
            b = f"{s['baseline_median_s']:.6f} / {s['baseline_p95_s']:.6f}"
            c = f"{s['candidate_median_s']:.6f} / {s['candidate_p95_s']:.6f}"
            v = f"{s['speedup']:.3f} [{s['ci_low']:.3f}, {s['ci_high']:.3f}]"
        quality = (
            "pass"
            if s.get("quality_pass")
            else ("fail" if s.get("quality_pass") is False else "—")
        )
        status = s["status"] + (
            "; exploratory pool switching" if exploratory(t) else ""
        )
        if t["id"] in accepted_ids:
            status = "accepted configuration (raw trial: " + s["status"] + ")"
        out.append(
            f"| {t['id']} | {s.get('track',t['meta'].get('track','E'))} | {b} | {c} | {v} | {quality} | {status} |"
        )
    return "\n".join(out) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=HOME / "reports")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    trials = collect()
    selection_path = HOME / "config/selection.json"
    selection = (
        json.loads(selection_path.read_text()) if selection_path.exists() else None
    )
    profiles = [t for t in trials if t["summary"]["candidate"] == "reference_profile"]
    latest_profile = profiles[-1]["summary"] if profiles else None
    rows = []
    for t in trials:
        s, m = t["summary"], t["meta"]
        env = m["environment"]
        row = {
            k: s.get(k)
            for k in [
                "candidate",
                "status",
                "reason",
                "warmups",
                "pairs",
                "baseline_median_s",
                "baseline_p95_s",
                "candidate_median_s",
                "candidate_p95_s",
                "speedup",
                "ci_low",
                "ci_high",
                "quality_pass",
                "max_abs",
                "max_relative_l2",
                "max_nrmse_increase",
                "max_ssim_loss",
                "new_failures",
                "cold_candidate_s",
                "setup_s",
                "methodology",
                "reference_threads",
                "candidate_threads",
                "affinity_name",
                "memory_gate",
            ]
        }
        row.update(
            interpretation=(
                "exploratory_pool_switch_screen"
                if exploratory(t)
                else (
                    "fixed_workers"
                    if s.get("methodology") == "persistent_worker_processes"
                    else "same_settings_or_characterization"
                )
            ),
            experiment_id=t["id"],
            track=m["track"],
            git_commit=m["git_commit"],
            cpu=env["cpu_model"],
            os=env["platform"],
            threads=env["intra_op"],
            interop=env["inter_op"],
            affinity=json.dumps(env["affinity"]),
            versions=json.dumps(env["versions"]),
            checkpoint_sha256=m["checkpoint_sha256"],
            source_files=json.dumps(m["source_files"]),
            input_identity=(
                str(t["path"] / "inputs.json")
                if (t["path"] / "inputs.json").exists()
                else None
            ),
            peak_memory=json.dumps(s.get("peak_memory", m.get("final_memory"))),
            hypothesis=s.get("hypothesis"),
            evidence_directory=str(t["path"].relative_to(HOME)),
        )
        rows.append(row)
    with (args.output / "results.csv").open("w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    accepted_ids = selection["evidence_ids"] if selection else []
    ledger = "# Experiment ledger — new measurements only\n\n" + table(
        trials, accepted_ids
    )
    ledger += "\nRaw trial statuses are immutable. Configuration-level acceptance, including a fresh independent repeat, is recorded separately in `../config/selection.json`. Early scheduling screens switched the global thread pools between arms and are exploratory only. Fixed-worker trials replace those screens for decisions; implementation comparisons at unchanged four-thread settings do not have that pool-switching issue.\n"
    mechanisms = {
        "reference": "Scheduling experiment. The fixed-worker method fixes OMP/MKL counts at spawn and uses KMP_BLOCKTIME=0 in both arms; no isolated claim about that environment variable is made. Affinity comes from Windows core topology. A different-day, thermally observed replication is the useful next step.",
        "spectral_linear": "NHWC pointwise linear replaces only FourierUnit 1x1 convolution, preserving grouped/interleaved channel order and original BN/ReLU. Packing and full spatial convolutions remain. No native rewrite is warranted by this result alone.",
        "spectral_bmm": "Group-aware batched matrix multiplication uses pre-shaped original weights; all other FourierUnit arithmetic remains. Small gains are below the prespecified minimum and do not establish network acceleration.",
        "spectral_matmul": "Broadcast matrix multiplication replaces spectral 1x1 convolution only. Rearrangement overhead and the small share of spectral convolution limit end-to-end benefit.",
        "packed_gemm": "Direct interleaved NHWC GEMM avoids the final packing allocation through an even-stride complex view. BN/ReLU arithmetic is retained. Full-network results, rather than an attractive local microbenchmark, determine promotion.",
        "static_buffers": "Each FourierUnit owns reusable packing/inverse scratch; buffers are refilled from every new input. Tests cover shape changes, pointer reuse and previously returned outputs. No data-dependent mask is cached. Allocation savings did not establish the target gain.",
        "local_channels_last": "Only spectral convolutions use channels-last; transitions are inside the timed forward. This isolates local layout from the earlier historical global-layout attempt.",
        "encoder_channels_last": "Only the four WPC encoder blocks use channels-last; NCHW is restored before FFC. Arbitrary masks and original feature convolution/reflection are preserved. Transition costs are included.",
        "mkldnn_prepack": "Ordinary spatial 3x3 convolutions use prepacked oneDNN weights, explicit original reflect padding, dense-to-blocked and blocked-to-dense bridges inside every call. WPC subclasses and spectral kernels remain original. No BN folding is used. A supported compiler that avoids bridges is more promising than extending this wrapper.",
        "jit_trace": "Static inference trace retains masks as runtime inputs and original BN arithmetic. Trace setup is separate from steady-state timings. Bitwise development parity did not make the measured gain exceed the target.",
        "prepack_bmm": "Joint configuration combines spatial oneDNN prepacking and spectral BMM after independent measurement, with fixed CPU scheduling. All bridges remain timed. The joint experiment requires its own gates and a fresh repeat; gains are not multiplied from components.",
    }
    for t in trials:
        s = t["summary"]
        ledger += f"\n## {t['id']}\n\n{s.get('hypothesis','Recorded characterization/probe')}. "
        ledger += f"Decision: **{s['status']}**. {concise_reason(s)}. "
        if exploratory(t):
            ledger += "Methodological limitation: global pool/affinity switching can bias first use. This screen is ineligible for promotion regardless of its apparent speedup. "
        mechanism = mechanisms.get(s["candidate"], "")
        if s["candidate"] == "reference" and "speedup" not in s:
            mechanism = "Reference-backend characterization or validation; no paired performance inference."
        ledger += mechanism + " "
        if t["id"] in accepted_ids:
            ledger += "The configuration is accepted through both independent trials in selection.json; its immutable raw status predates the combined decision. "
        if "speedup" in s and t["id"] not in accepted_ids:
            reasons = []
            if s["speedup"] < 1.1:
                reasons.append("median speedup below 1.10")
            if s["ci_low"] <= 1:
                reasons.append("interval includes no improvement")
            if s["candidate_p95_s"] > 1.05 * s["baseline_p95_s"]:
                reasons.append("p95 increases >5%")
            if not s["quality_pass"]:
                reasons.append("strict fidelity failure")
            ledger += (
                " Gates not cleared: "
                + ", ".join(
                    reasons
                    or ["fresh process and isolated memory confirmation pending"]
                )
                + ". "
            )
        ledger += f"Evidence: `results/{t['id']}/`; raw calls, per-case validation, source ZIP and environment retained when executed.\n"
    (args.output / "EXPERIMENT_LEDGER.md").write_text(ledger, encoding="utf-8")
    numeric = [
        t
        for t in trials
        if (t["path"] / "validation.json").exists() and "max_abs" in t["summary"]
    ]
    validation = "# Numerical and scientific validation\n\nDevelopment trials use the original 18 measurements from seven independent bases. Final replay rows, if present, use five published examples after the selection lock; their base IDs and input hashes are recorded separately. Strict elementwise gate rtol=1e-4/atol=1e-5; per-case NRMSE increase and SSIM loss <=1e-4; no new NRMSE>0.5 failures. No amplitude fitting. Each validation.json contains model-to-model parity separately from ground-truth NRMSE, SSIM, PSNR, morphology and detector physics metrics. The timing pairs are repeated measurements, not independent objects or a new 570-case validation set.\n\n"
    validation += "| Trial | Cases / bases | Max abs amplitude error | Max relative L2 | Max NRMSE increase | Max SSIM loss | New failures | Gate |\n|---|---:|---:|---:|---:|---:|---:|---|\n"
    for t in numeric:
        s = t["summary"]
        validation += f"| {t['id']} | {s['validation_cases']} / {s['independent_bases']} | {s['max_abs']:.8g} | {s['max_relative_l2']:.8g} | {s['max_nrmse_increase']:.8g} | {s['max_ssim_loss']:.8g} | {s['new_failures']} | {s['quality_pass']} |\n"
    validation += "\nThe five published examples are only final replay data after selection lock. Raw 570 measurement arrays are not in this repository; the CSV cannot regenerate them. Historical final failure counts remain historical and are not new validation.\n"
    (args.output / "NUMERICAL_VALIDATION.md").write_text(validation, encoding="utf-8")
    compiler = [
        t
        for t in trials
        if t["summary"]["candidate"].startswith("compile")
        or t["summary"]["candidate"] in ["jit_trace", "dynamo_diagnostics"]
    ]
    comp = "# Compiler backends\n\nActual torch.compile calls use fullgraph=False, dynamic=False and no suppressed-errors eager fallback. Default, reduce-overhead, max-autotune and selective FourierUnit convolution compilation are separate probes. Cold first use contains compilation and is excluded from steady timings.\n\n"
    for t in compiler:
        comp += (
            f"- `{t['id']}`: {t['summary']['status']}; {concise_reason(t['summary'])}\n"
        )
        if "graph_count" in t["summary"]:
            comp += f"  Captured graphs: {t['summary']['graph_count']}; graph breaks: {t['summary']['graph_break_count']}; capture {t['summary']['capture_s']:.3f}s. This is not native compilation.\n"
    comp += "\nWindows source/runtime checks: installed Torch 2.5.1, MSVC cl absent from PATH and no VC.Tools.x86.x64 installation returned by vswhere. GCC from Strawberry Perl is not substituted as a supported MSVC ABI toolchain. AOTInductor/native C++ depend on a compatible compiler. IPEX/TVM installation and alternate OS runs are not claimed. See REFERENCES.md for official source links and version limitations. Selective compilation is attempted independently even if whole-model compilation fails.\n"
    (args.output / "COMPILER_BACKENDS.md").write_text(comp, encoding="utf-8")
    fft = "# FFT backends\n\n"
    for t in trials:
        if t["summary"]["candidate"] == "fft_plans" and "rows" in t["summary"]:
            fft += f"`{t['id']}`: {t['summary']['rows']} completed cohorts. Actual captured shapes plus 512x512 detector; forward/inverse; ortho feature and backward detector normalization. Torch native/output-buffer, SciPy kernel/bridge, FFTW ESTIMATE/MEASURE kernel/bridge. Setup and raw calls retained.\n\n"
            fft += fft_table(t)
    fft += "FFTW inverse transforms destroy scratch input: refill occurs before every kernel clock and inside every bridge clock. Bridge outputs own their storage. Forward normalization requires normalise_idft=True for backward convention; a failing regression caught the initial 1/N scaling error, fixed without changing tolerance. Sequential microbenchmarks do not establish network speedups.\n\n"
    if latest_profile:
        fft += f"Latest FFT self fraction {latest_profile['fft_self_fraction']*100:.4f}%; perfect FFT kernels bound ideal end-to-end gain at {latest_profile['fft_only_ideal_speedup_bound']:.4f}x. "
    fft += "Direct oneMKL DFTI/custom C++ integrations are not_run pending a larger measured opportunity and compiler. Installing mkl-fft does not replace torch.fft automatically.\n"
    (args.output / "FFT_BACKENDS.md").write_text(fft, encoding="utf-8")
    final = "# Final results — frozen DPR1 CPU campaign\n\n"
    final += "**Portable API default remains original FP32 reference, four intra-op / one inter-op threads and inherited affinity.** The explicit machine-specific configuration below is assessed separately. No runtime benchmarking or silent precision change occurs in production prediction.\n\n"
    if selection:
        final += (
            selection["report"]
            + "\n\nSelection provenance: `../config/selection.json`. Explicit optimized settings are machine/runtime-specific. The portable four-thread API auto default remains reference.\n\n"
        )
    for replay in [t for t in trials if t["id"].startswith("final_replay")]:
        rs = replay["summary"]
        final += f"Final locked replay `{replay['id']}`: {rs.get('validation_cases')} cases from {rs.get('independent_bases')} bases; quality={rs.get('quality_pass')}, new failures={rs.get('new_failures')}, max absolute difference={rs.get('max_abs')}. Validation only, no timing speedup.\n\n"
    final += (
        "The fresh baseline measured median **0.921623 s**, p95 **1.085057 s**. This is a same-machine new-session characterization, separate from historical 0.782155 s. Candidate speedups below use each trial’s paired reference, never the historical denominator.\n\n"
        + table(trials, accepted_ids)
    )
    final += "\nRaw development inputs and source hashes travel with the framework. Negative/blocked results remain in the ledger. No approximate or architecture-changing implementation is promoted. Unexecuted possibilities remain not_run or blocked. Dominant bottleneck: spatial convolution-heavy WPC/FFC paths. Latest profile: FFT self fraction 2.28827%, ideal FFT-only bound 1.02342x; hottest spatial 3x3 operator group 42.6%. Native FFT-only work cannot meet 1.10x under this profile. Next exact-track approach is a supported compiler toolchain and connected convolution/elementwise fusion or blocked layouts, followed by strict parity and paired testing. Larger architectural reductions require separate Track M training/validation.\n"
    final += "\nEarly scheduling screens are explicitly exploratory because they changed the global pools. Only fixed-process scheduling evidence is eligible for selection. All native-kernel, full-unit and full-model timing boundaries remain separate. Original full regression still has one pre-existing network_synthetic failure; scoped tests and immutable-file audit are documented in VERIFICATION.md.\n"
    final += "\nUse [SELECTED_CONFIGURATION.md](../SELECTED_CONFIGURATION.md) for the measured environment/API settings, [CAMPAIGN_NOTES.md](CAMPAIGN_NOTES.md) for scientific interpretation, and [VERIFICATION.md](VERIFICATION.md) for final verification and artifact checks.\n"
    (args.output / "FINAL_RESULTS.md").write_text(final, encoding="utf-8")
    future = "# Future work and unexecuted tracks\n\n"
    future += (
        "| Experiment | Track | Status | Reason / next action |\n|---|---|---|---|\n"
    )
    for exp, track, status, reason in [
        (
            "Fused native spectral operator / C++ FourierUnit",
            "E",
            "not_run",
            "Spectral/FFT fraction too small to prioritize a major rewrite; supported MSVC missing",
        ),
        (
            "oneMKL DFTI native integration",
            "E",
            "not_run",
            "Latest FFT ideal bound 1.0234; existing Torch already uses MKL",
        ),
        (
            "Fused WPC mask kernel",
            "E",
            "not_run",
            "Measured mask update/slide/ratio 10.53 ms (~1.31% instrumented total); too small for 10% gain even if removed, arbitrary masks must remain exact",
        ),
        (
            "Linux CPU environment",
            "E",
            "not_run",
            "No second OS provisioned; do not compare different machines as paired",
        ),
        (
            "ONNX custom FFT",
            "E",
            "not_run",
            "Historical standard DFT severe problems; no new custom-op toolchain",
        ),
        (
            "OpenVINO hybrid",
            "E",
            "not_run",
            "Historical FP32 gate failure; no new bridge hypothesis justified yet",
        ),
        (
            "IPEX / AOTInductor / TVM",
            "E",
            "blocked",
            "Version/platform/toolchain prerequisites not provisioned; no unsupported Windows installs",
        ),
        (
            "BF16 / selective INT8",
            "N",
            "not_run",
            "No hardware-efficient BF16 established; historical grouped INT8 harms quality; no new error budget/calibration campaign",
        ),
        (
            "Distillation / pruning / fewer FourierUnits",
            "M",
            "not_run",
            "Requires retraining and new model identity, beyond frozen-checkpoint execution",
        ),
        (
            "Alternate OpenMP runtime replacement",
            "E",
            "not_run",
            "Runtime replacement not tested. A separate batch-two characterization, if present in the ledger, is not a batch-one improvement or scaling claim",
        ),
        (
            "Full 570 held-out rerun",
            "E",
            "not_run",
            "Raw arrays not bundled; original summary rows do not regenerate inputs",
        ),
    ]:
        future += f"| {exp} | {track} | {status} | {reason} |\n"
    future += "\nAll first-wave pure-PyTorch candidate implementations remain reversible factory choices. Combinations require independent end-to-end measurement; two promising components do not imply an accepted combination. Longer finalists and different-day replication remain useful if a candidate first clears the minimum gates.\n"
    (args.output / "FUTURE_WORK.md").write_text(future, encoding="utf-8")
    write_json(
        args.output / "evidence_index.json",
        [dict(id=t["id"], status=t["summary"]["status"]) for t in trials],
    )
    print(f"Aggregated {len(trials)} completed trial records into {args.output}")


if __name__ == "__main__":
    main()
