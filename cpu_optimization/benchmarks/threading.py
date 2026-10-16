"""Paired fixed-thread/affinity workers, isolated before any model execution.

This avoids changing Torch thread pools between timed arms. Workers are invoked
sequentially; each owns one model, prepared inputs and its fixed CPU settings.
"""

import multiprocessing as mp
import os
import random
import traceback
import csv
import uuid
import numpy as np
import torch
from .common import *
from .compare import affinity_for, evaluate
from ..statistics import paired_summary, decision
from ..scripts.campaign import terminate_tree
from ..candidates import NAMES, apply_candidate


def memory_evidence(memory):
    """A sampled post-call RSS maximum is not a lifetime allocation peak."""
    peaks = [
        (
            max(m["peak_wset"] for m in arm)
            if arm and all("peak_wset" in m for m in arm)
            else None
        )
        for arm in memory
    ]
    return (
        dict(reference_worker_bytes=peaks[0], candidate_worker_bytes=peaks[1]),
        peaks[1] <= 1.1 * peaks[0] if all(p is not None for p in peaks) else None,
    )


def paired_worker_settings(reference_threads, candidate_threads, candidate,
                           reference_affinity='default', candidate_affinity='default'):
    """Allow a new candidate to be paired directly with the locked reference setup."""
    return [dict(threads=reference_threads, affinity=affinity_for(reference_affinity), backend='reference'),
            dict(threads=candidate_threads, affinity=affinity_for(candidate_affinity), backend=candidate)]


def compiler_profile(backend):
    if not backend.startswith('compile_'):
        return None
    try:
        from torch._inductor import metrics
        from torch._dynamo.utils import compile_times
        return {'generated_kernel_count': metrics.generated_kernel_count,
                'compilation_metrics': compile_times(), 'scope': 'compiler diagnostics outside model clock'}
    except Exception as exc:
        return {'unavailable': repr(exc)}


class WorkerProxy:
    def __init__(self, connection, input_indices):
        self.connection, self.indices = connection, input_indices
        self.last = None

    def __call__(self, x, m, false_scale=False):
        if false_scale:
            raise ValueError("Worker protocol uses false_scale=False")
        self.connection.send(dict(op="predict", index=self.indices[x.data_ptr()]))
        if not self.connection.poll(180):
            raise TimeoutError("Inference worker did not respond")
        self.last = self.connection.recv()
        if "error" in self.last:
            raise RuntimeError(self.last["error"])
        return torch.from_numpy(
            np.asarray(self.last["prediction"], dtype=np.float32)[None, None]
        )


def _worker(connection, threads, affinity, checkpoint, development, warmups, backend):
    try:
        configure_threads(threads)
        psutil.Process().cpu_affinity(affinity)
        cases = load_cases(development)
        inputs = [prepare_inputs(c["intensity"], c["valid"]) for c in cases]
        start = time.perf_counter()
        model = apply_candidate(load_reference(checkpoint), backend)
        setup = time.perf_counter() - start
        with torch.inference_mode(), torch.autocast("cpu", enabled=False):
            _, cold = timed(model, *inputs[0])
            warming = time.perf_counter()
            for i in range(warmups):
                model(*inputs[i % len(inputs)], false_scale=False)
            warmup_s = time.perf_counter()-warming
            connection.send(
                dict(
                    ready=True,
                    pid=os.getpid(),
                    instance_id=str(uuid.uuid4()),
                    process_created_s=psutil.Process().create_time(),
                    environment=environment(),
                    setup_s=setup,
                    cold_s=cold,
                    warmup_s=warmup_s,
                    compiler_profile=compiler_profile(backend),
                    inputs=cases_identity(cases),
                )
            )
            while True:
                message = connection.recv()
                if message["op"] == "close":
                    break
                if message["op"] == "warm":
                    warming = time.perf_counter()
                    for i in range(warmups):
                        model(*inputs[i % len(inputs)], false_scale=False)
                    connection.send(dict(warmed=warmups, warmup_s=time.perf_counter()-warming,
                                         compiler_profile=compiler_profile(backend)))
                    continue
                output, elapsed = timed(model, *inputs[message["index"]])
                # Output ownership, IPC and memory telemetry are outside the model clock.
                connection.send(
                    dict(
                        seconds=elapsed,
                        prediction=output.numpy()[0, 0].copy(),
                        memory=psutil.Process().memory_info()._asdict(),
                    )
                )
    except Exception:
        try:
            connection.send(dict(error=traceback.format_exc()))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        connection.close()


def start_worker(ctx, settings, checkpoint, development, warmups, startup_timeout=180):
    if not 0 < startup_timeout <= 900:
        raise ValueError('Worker startup timeout must be in (0, 900] seconds')
    parent, child = ctx.Pipe()
    process = ctx.Process(
        target=_worker,
        args=(
            child,
            settings["threads"],
            settings["affinity"],
            checkpoint,
            development,
            warmups,
            settings["backend"],
        ),
    )
    keys = {
        "OMP_NUM_THREADS": str(settings["threads"]),
        "MKL_NUM_THREADS": str(settings["threads"]),
        "KMP_BLOCKTIME": "0",
        "OPENBLAS_NUM_THREADS": "1",
    }
    previous = {k: os.environ.get(k) for k in keys}
    try:
        os.environ.update(keys)
        process.start()
    finally:
        for k, v in previous.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    child.close()
    if not parent.poll(startup_timeout):
        terminate_tree(process.pid)
        raise TimeoutError(f"Worker startup exceeded {startup_timeout}s")
    ready = parent.recv()
    if "error" in ready:
        terminate_tree(process.pid)
        raise RuntimeError(ready["error"])
    return process, parent, ready


def main():
    p = parser(__doc__)
    p.add_argument("--candidate-threads", type=int, required=True)
    p.add_argument("--candidate", choices=NAMES, default="reference")
    p.add_argument(
        "--affinity",
        choices=["default", "p_only", "p_physical", "mixed_physical", "e_only"],
        default="default",
    )
    p.add_argument("--seed", type=int, default=2026101601)
    p.add_argument('--reference-affinity', choices=['default', 'p_only', 'p_physical', 'mixed_physical', 'e_only'],
                   default='default', help='Reference worker affinity; default preserves inherited CPU set')
    p.add_argument('--worker-startup-timeout', type=float, default=180,
                   help='Bounded compiler/warmup startup allowance, at most 900 seconds')
    p.epilog = "Each worker warms up --warmups times at startup and again after quality validation. OMP/MKL threads fixed at spawn; KMP_BLOCKTIME=0 prevents idle-worker spinning."
    args = p.parse_args()
    configure_threads(args.threads)
    run = new_run(args.id, args.candidate, args.results_dir, args.checkpoint)
    cases = load_cases(args.development)
    inputs = [prepare_inputs(c["intensity"], c["valid"]) for c in cases]
    identity = cases_identity(cases)
    write_json(run / "inputs.json", identity)
    ctx = mp.get_context("spawn")
    workers = []
    configs = paired_worker_settings(args.threads, args.candidate_threads, args.candidate,
                                     args.reference_affinity, args.affinity)
    try:
        for config in configs:
            workers.append(
                start_worker(
                    ctx, config, args.checkpoint, args.development, args.warmups, args.worker_startup_timeout
                )
            )
        write_json(run / "workers.json", [w[2] for w in workers])
        if any(w[2]["inputs"] != identity for w in workers):
            raise RuntimeError("Worker input identities differ")
        proxies = [
            WorkerProxy(w[1], {x.data_ptr(): i for i, (x, m) in enumerate(inputs)})
            for w in workers
        ]
        quality, rows, predictions = evaluate(
            *proxies, cases, inputs, select=lambda _: None
        )
        write_json(run / "validation.json", dict(passed=quality, cases=rows))
        np.savez_compressed(run / "predictions.npz", **predictions)
        post_validation_warmup = []
        for process, connection, _ in workers:
            connection.send(dict(op="warm"))
            if not connection.poll(args.worker_startup_timeout):
                raise TimeoutError("Worker warmup timeout")
            warm_report = connection.recv()
            if 'error' in warm_report:
                raise RuntimeError(warm_report['error'])
            post_validation_warmup.append(warm_report)
        write_json(run/'post_validation_warmup.json', post_validation_warmup)
        rng = random.Random(args.seed)
        bt, ct = [], []
        memory = [[], []]
        raw = []
        with (run / "paired_timings.csv").open("x", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "pair",
                    "sample_id",
                    "order",
                    "reference_s",
                    "candidate_s",
                    "reference_rpc_s",
                    "candidate_rpc_s",
                ],
            )
            writer.writeheader()
            for i in range(args.repeats):
                order = [0, 1]
                rng.shuffle(order)
                elapsed = {}
                rpc = {}
                for arm in order:
                    start = time.perf_counter()
                    proxies[arm](*inputs[i % len(inputs)])
                    rpc[arm] = time.perf_counter() - start
                    elapsed[arm] = proxies[arm].last["seconds"]
                    memory[arm].append(proxies[arm].last["memory"])
                bt.append(elapsed[0])
                ct.append(elapsed[1])
                writer.writerow(
                    dict(
                        pair=i,
                        sample_id=cases[i % len(cases)]["sample_id"],
                        order="CB" if order[0] else "BC",
                        reference_s=elapsed[0],
                        candidate_s=elapsed[1],
                        reference_rpc_s=rpc[0],
                        candidate_rpc_s=rpc[1],
                    )
                )
                f.flush()
                if i % 6 == 0:
                    print(
                        f"fixed-worker pair {i+1}/{args.repeats}: {elapsed[0]:.4f}s vs {elapsed[1]:.4f}s",
                        flush=True,
                    )
        stats = paired_summary(bt, ct, args.seed)
        write_json(run / "memory_per_call.json", memory)
        peaks, memory_pass = memory_evidence(memory)
        finish(
            run,
            dict(
                **stats,
                candidate=args.candidate,
                status=decision(stats, quality, False, memory_pass),
                reason="Fixed-process paired experiment; fresh independent repetition required before selection",
                methodology="persistent_worker_processes",
                warmups=2 * args.warmups,
                reference_threads=args.threads,
                candidate_threads=args.candidate_threads,
                candidate_affinity=configs[1]["affinity"],
                affinity_name=args.affinity,
                reference_affinity_name=args.reference_affinity,
                worker_startup_timeout_s=args.worker_startup_timeout,
                quality_pass=quality,
                max_abs=max(r["parity"]["max_abs"] for r in rows),
                max_relative_l2=max(r["parity"]["relative_l2"] for r in rows),
                max_nrmse_increase=max(r["nrmse_increase"] for r in rows),
                max_ssim_loss=max(r["ssim_loss"] for r in rows),
                bitwise=all(r["parity"]["bitwise"] for r in rows),
                new_failures=sum(r["new_failure"] for r in rows),
                validation_cases=len(cases),
                independent_bases=len(set(c["base_id"] for c in cases)),
                peak_memory=peaks,
                memory_gate=memory_pass,
                memory_note="Windows lifetime peak_wset; missing lifetime peak makes the gate unknown, never substitutes sampled RSS",
                hypothesis="Fixed CPU thread/core settings and optional isolated backend reduce convolution-heavy batch-one latency",
                worker_settings=configs,
                worker_thread_environment="OMP_NUM_THREADS=MKL_NUM_THREADS=intra_op; KMP_BLOCKTIME=0; OPENBLAS_NUM_THREADS=1",
                boundary="Prepared network clocks inside workers; RPC, output ownership, loading, startup and metrics excluded",
                reference_rss_initial_bytes=memory[0][0]["rss"],
                reference_rss_final_bytes=memory[0][-1]["rss"],
                candidate_rss_initial_bytes=memory[1][0]["rss"],
                candidate_rss_final_bytes=memory[1][-1]["rss"],
            ),
        )
    finally:
        for process, connection, _ in workers:
            try:
                connection.send(dict(op="close"))
            except (BrokenPipeError, EOFError):
                pass
            process.join(timeout=10)
            if process.is_alive():
                terminate_tree(process.pid)
            connection.close()


if __name__ == "__main__":
    cli(main)
