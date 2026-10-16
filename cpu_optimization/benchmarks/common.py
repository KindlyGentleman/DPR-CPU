"""Append-only evidence, actual inputs and bounded machine provenance."""

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
import zipfile

import numpy as np
import psutil
import torch

from ..api import (
    ROOT,
    CHECKPOINT_SHA256,
    configure_threads,
    prepare_inputs,
    load_reference,
    file_sha256,
    default_checkpoint,
)

HOME = ROOT / "cpu_optimization"
_ACTIVE_RUN = None


def digest(path):
    return file_sha256(path)


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )


def append_rows(path, rows):
    """Close each completed CSV cohort to retain evidence after a later failure."""
    import csv

    exists = path.exists()
    with path.open("a" if exists else "x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def command(*args):
    try:
        return subprocess.check_output(
            args, cwd=ROOT, text=True, stderr=subprocess.STDOUT, timeout=15
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return str(exc)


def topology():
    cores = []
    if sys.platform == "win32":
        api = ctypes.windll.kernel32.GetLogicalProcessorInformationEx
        size = ctypes.c_ulong(0)
        api(0, None, ctypes.byref(size))
        buf = ctypes.create_string_buffer(size.value)
        if api(0, buf, ctypes.byref(size)):
            raw = buf.raw
            off = 0
            while off + 8 <= size.value:
                length = int.from_bytes(raw[off + 4 : off + 8], "little")
                if length < 8:
                    break
                groups = int.from_bytes(raw[off + 30 : off + 32], "little")
                cpus = []
                for i in range(groups):
                    p = off + 32 + 16 * i
                    mask = int.from_bytes(raw[p : p + 8], "little")
                    group = int.from_bytes(raw[p + 8 : p + 10], "little")
                    cpus.extend(group * 64 + j for j in range(64) if mask & (1 << j))
                cores.append(
                    dict(
                        efficiency_class=raw[off + 9],
                        smt=bool(raw[off + 8] & 1),
                        logical_cpus=cpus,
                    )
                )
                off += length
    return dict(
        logical_count=psutil.cpu_count(),
        physical_count=psutil.cpu_count(logical=False),
        cores=cores,
        interpretation="Windows RelationProcessorCore; SMT pairs identify hybrid P cores on this CPU; efficiency classes retained verbatim",
    )


def environment():
    proc = psutil.Process()
    versions = {}
    for n in [
        "torch",
        "numpy",
        "scipy",
        "scikit-image",
        "psutil",
        "pyfftw",
        "onnxruntime",
        "openvino",
        "SciencePlots",
    ]:
        try:
            versions[n] = importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:
            versions[n] = None
    try:
        freq = psutil.cpu_freq()._asdict()
    except (AttributeError, OSError):
        freq = None
    try:
        temperature = psutil.sensors_temperatures()
    except (AttributeError, OSError):
        temperature = None
    return dict(
        platform=platform.platform(),
        python=sys.version,
        executable=sys.executable,
        versions=versions,
        torch_build=torch.__config__.show(),
        topology=topology(),
        cpu_model=(
            command(
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor).Name",
            )
            if sys.platform == "win32"
            else platform.processor()
        ),
        affinity=proc.cpu_affinity() if hasattr(proc, "cpu_affinity") else None,
        intra_op=torch.get_num_threads(),
        inter_op=torch.get_num_interop_threads(),
        power_policy=(
            command("powercfg", "/getactivescheme") if sys.platform == "win32" else None
        ),
        frequency=freq,
        temperature=temperature,
        temperature_note="unavailable" if not temperature else "sensor readings",
        thread_environment={
            k: os.environ.get(k)
            for k in [
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "OMP_PROC_BIND",
                "KMP_AFFINITY",
                "KMP_BLOCKTIME",
                "MKL_DYNAMIC",
                "CXX",
                "TORCHINDUCTOR_CACHE_DIR",
                "TORCHINDUCTOR_COMPILE_THREADS",
                "TORCH_LOGS",
            ]
        },
        memory=proc.memory_info()._asdict(),
    )


def new_run(experiment_id, candidate, root=None, checkpoint=None):
    global _ACTIVE_RUN
    if Path(experiment_id).name != experiment_id or experiment_id in ("", "..", "."):
        raise ValueError("experiment_id must be a single directory name")
    run = (Path(root) if root else HOME / "results") / experiment_id
    run.mkdir(parents=True, exist_ok=False)
    paths = list((ROOT / "deeppr").glob("*.py")) + list(HOME.rglob("*.py"))
    paths += [
        p
        for p in [
            ROOT / "pyproject.toml",
            HOME / "config/protocol.json",
            HOME / "config/development_provenance.json",
            HOME / "config/external_source.json",
            HOME / "config/selection.json",
            HOME / "requirements-optional.txt",
            ROOT / "LICENSE.txt",
            HOME / "THIRD_PARTY_NOTICES.md",
            HOME / "config/user_request.txt",
            HOME / "phase2_extension/PLAN.md",
            HOME / "phase2_extension/user_request.txt",
        ]
        if p.exists()
    ]
    paths += list((ROOT / "LICENSES").glob("*.txt"))
    paths = [
        p
        for p in paths
        if not p.is_relative_to(HOME / "results")
        and "__pycache__" not in p.relative_to(ROOT).parts
    ]
    source = {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in paths}
    with zipfile.ZipFile(
        run / "source.zip", "x", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for p in paths:
            archive.write(p, p.relative_to(ROOT))
    write_json(
        run / "run.json",
        dict(
            experiment_id=experiment_id,
            candidate=candidate,
            track="E",
            created_utc=datetime.now(timezone.utc).isoformat(),
            git_commit=command("git", "rev-parse", "HEAD"),
            git_status=command("git", "status", "--short"),
            source_files=source,
            checkpoint_sha256=CHECKPOINT_SHA256,
            checkpoint_path=str(Path(checkpoint or default_checkpoint()).resolve()),
            command_line=sys.argv,
            environment=environment(),
            status="running",
        ),
    )
    _ACTIVE_RUN = run
    return run


def load_cases(development=None, archived=False):
    if archived:
        with np.load(ROOT / "cpu_study/examples.npz", allow_pickle=False) as z:
            return [
                dict(
                    sample_id=str(z["sample_ids"][i]),
                    base_id=int(str(z["sample_ids"][i]).split("__")[0][1:]),
                    split="published_final",
                    intensity=z[f"intensity_{i}"],
                    valid=z[f"valid_{i}"],
                    truth=z[f"truth_{i}"],
                    archived_reference=z[f"reference_{i}"],
                )
                for i in range(5)
            ]
    path = Path(development) if development else HOME / "config/development_inputs.npz"
    with np.load(path, allow_pickle=False) as z:
        return [
            dict(
                sample_id=str(s),
                base_id=int(str(s).split("__")[0][1:]),
                split="development",
                intensity=z[f"intensity_{i}"],
                valid=z[f"valid_{i}"],
                truth=z[f"truth_{i}"],
                archived_reference=z[f"reference_{i}"],
            )
            for i, s in enumerate(z["sample_ids"])
        ]


def require_selection(backend, threads, affinity):
    """Prevent using published cases to choose or alter a candidate configuration."""
    path = HOME / "config/selection.json"
    if not path.exists():
        raise ValueError("Published replay requires a prior selection lock")
    selected = json.loads(path.read_text(encoding="utf-8"))["selected"]
    if (
        selected["backend"] != backend
        or selected["threads"] != threads
        or selected["affinity"] != affinity
    ):
        raise ValueError("Published replay must match the locked candidate settings")


def cases_identity(cases):
    return [
        dict(
            sample_id=c["sample_id"],
            base_id=c["base_id"],
            split=c["split"],
            arrays={
                k: hashlib.sha256(np.ascontiguousarray(c[k]).tobytes()).hexdigest()
                for k in ["intensity", "valid", "truth"]
            },
        )
        for c in cases
    ]


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--id", required=True, help="New immutable experiment directory name"
    )
    p.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Reference intra-op threads; inter-op fixed to one",
    )
    p.add_argument("--warmups", type=int, default=10)
    p.add_argument(
        "--repeats",
        type=int,
        default=36,
        help="Calls or timing pairs, depending on command",
    )
    p.add_argument(
        "--development",
        type=Path,
        help="Prepared development NPZ (default: bundled original 18 cases)",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=default_checkpoint(),
        help="External original DPR1 checkpoint; SHA-256 verified before loading",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=HOME / "results",
        help="Writable parent for new immutable trial directories",
    )
    return p


def timed(model, x, m):
    start = time.perf_counter_ns()
    out = model(x, m, false_scale=False)
    return out, (time.perf_counter_ns() - start) * 1e-9


def finish(run, summary):
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    summary.setdefault("candidate", meta["candidate"])
    summary.setdefault("track", meta.get("track", "E"))
    write_json(run / "summary.json", summary)
    meta["status"] = summary["status"]
    meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
    meta["final_memory"] = psutil.Process().memory_info()._asdict()
    write_json(run / "run.json", meta)
    print(json.dumps(summary, allow_nan=False), flush=True)


def record_failed_trial(run, reason):
    """Complete a failed owned trial without replacing any terminal evidence."""
    run = Path(run)
    if not (run / "summary.json").exists():
        finish(run, dict(status="blocked", reason=reason, pairs=0))


def cli(main):
    """Persist exceptional termination for directly invoked benchmark modules."""
    try:
        main()
    except Exception as exc:
        if _ACTIVE_RUN is not None:
            error = _ACTIVE_RUN / "error.txt"
            if not error.exists():
                error.write_text(traceback.format_exc(), encoding="utf-8")
            record_failed_trial(_ACTIVE_RUN, f"{type(exc).__name__}: {exc}")
        else:
            traceback.print_exc()
        raise SystemExit(1) from exc
