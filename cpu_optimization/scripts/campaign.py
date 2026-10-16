"""Sequential first-wave campaign: one heavy process at a time, no evidence overwrite."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import psutil
from ..benchmarks.common import ROOT, HOME, new_run, record_failed_trial


def terminate_tree(pid):
    """Stop only this campaign's owned subprocess tree, deepest children first."""
    try:
        parent = psutil.Process(pid)
        processes = [parent] + parent.children(recursive=True)
    except psutil.NoSuchProcess:
        return
    for process in reversed(processes):
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=3)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=3)


def recover_failure(ident, reason):
    directory = HOME / "results" / ident
    if not directory.exists():
        directory = new_run(ident, "process_start_failure")
    record_failed_trial(directory, reason)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--wave",
        choices=["threads", "fixed_threads", "compiler", "spectral", "micro"],
        required=True,
    )
    p.add_argument(
        "--suffix",
        default="v1",
        help="Unique run suffix; existing runs skipped without alteration",
    )
    args = p.parse_args()
    trials = []
    if args.wave == "fixed_threads":
        for t, aff in [
            (8, "default"),
            (6, "p_physical"),
            (10, "default"),
            (2, "default"),
            (6, "default"),
            (4, "p_physical"),
            (6, "p_only"),
            (10, "mixed_physical"),
        ]:
            trials.append(
                (
                    f"fixed_t{t}_{aff}",
                    "threading",
                    ["--candidate-threads", str(t), "--affinity", aff],
                )
            )
    elif args.wave == "threads":
        for t in [2, 6, 8, 10]:
            trials.append(
                (
                    f"thread_t{t}",
                    "compare",
                    [
                        "--candidate",
                        "reference",
                        "--candidate-threads",
                        str(t),
                        "--time-rejected",
                    ],
                )
            )
        for t, aff in [
            (4, "p_physical"),
            (6, "p_physical"),
            (6, "p_only"),
            (10, "mixed_physical"),
        ]:
            trials.append(
                (
                    f"affinity_t{t}_{aff}",
                    "compare",
                    [
                        "--candidate",
                        "reference",
                        "--candidate-threads",
                        str(t),
                        "--affinity",
                        aff,
                        "--time-rejected",
                    ],
                )
            )
    elif args.wave == "compiler":
        trials.append(("dynamo_graphs", "compiler_graphs", []))
        for c in [
            "compile_default",
            "compile_reduce",
            "compile_autotune",
            "compile_selective",
            "jit_trace",
        ]:
            trials.append((c, "compare", ["--candidate", c]))
    elif args.wave == "spectral":
        for c in [
            "spectral_linear",
            "spectral_bmm",
            "spectral_matmul",
            "packed_gemm",
            "static_buffers",
            "local_channels_last",
            "encoder_channels_last",
            "mkldnn_prepack",
        ]:
            trials.append((c, "compare", ["--candidate", c]))
    else:
        for c in ["spectral", "fft"]:
            trials.append(
                (c + "_micro", c, ["--profile", str(HOME / "results/profile_v2")])
            )
    for name, module, options in trials:
        ident = f"{name}_{args.suffix}"
        if (HOME / "results" / ident).exists():
            print(f"Skipping preserved trial {ident}", flush=True)
            continue
        cmd = [
            sys.executable,
            "-m",
            f"cpu_optimization.benchmarks.{module}",
            "--id",
            ident,
            *options,
        ]
        print("Running " + " ".join(cmd), flush=True)
        with (HOME / "results" / (ident + ".log")).open("x", encoding="utf-8") as log:
            try:
                child = subprocess.Popen(
                    cmd,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONUTF8": "1"},
                )
                code = child.wait(timeout=600)
                print(f"{ident}: exit={code}", flush=True)
                if code:
                    recover_failure(
                        ident, f"Child process exited {code}; see retained log"
                    )
            except subprocess.TimeoutExpired:
                terminate_tree(child.pid)
                log.write(
                    "\nProcess killed after 600s limit; partial evidence retained.\n"
                )
                print(f"{ident}: timeout, partial evidence retained", flush=True)
                recover_failure(
                    ident,
                    "Owned process tree terminated after 600s; partial evidence retained",
                )
            except OSError as exc:
                recover_failure(ident, f"Failed to start child process: {exc}")
        summary = HOME / "results" / ident / "summary.json"
        if summary.exists():
            print(summary.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
