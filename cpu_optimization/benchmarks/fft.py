"""FFT native/bridge timings on captured DPR activations; never a network speedup."""

import csv
import time
import traceback
import numpy as np
import scipy.fft
import torch
from .common import *
from ..fft.plans import FFTWPlan


def measure(fn, warmups, repeats, before=None):
    for _ in range(warmups):
        if before:
            before()
        fn()
    times = []
    for _ in range(repeats):
        if before:
            before()
        start = time.perf_counter_ns()
        fn()
        times.append((time.perf_counter_ns() - start) * 1e-9)
    return times


def main():
    p = parser(__doc__)
    p.set_defaults(repeats=60)
    p.add_argument(
        "--profile",
        type=Path,
        required=True,
        help="Profile directory containing fourier_inputs.npz",
    )
    args = p.parse_args()
    configure_threads(args.threads)
    run = new_run(args.id, "fft_plans", args.results_dir, args.checkpoint)
    with np.load(args.profile / "fourier_inputs.npz") as z:
        inputs = [z[k] for k in z.files]
    detector = load_cases(args.development)[0]["intensity"][None, None]
    write_json(
        run / "inputs.json",
        dict(
            activation_sha256=digest(args.profile / "fourier_inputs.npz"),
            inventory_sha256=digest(args.profile / "fourier_inventory.json"),
            detector=cases_identity(load_cases(args.development)[:1]),
        ),
    )
    inputs.append(detector)
    rows = []
    raw = []
    replans = []
    failures = []
    with torch.inference_mode():
        for array in inputs:
            x = torch.from_numpy(array.copy())
            shape = tuple(x.shape)
            norm = "backward" if shape[-1] == 512 else "ortho"
            spectrum = torch.fft.rfft2(x, norm=norm)
            for direction in ["forward", "inverse"]:
                a = x if direction == "forward" else spectrum
                expected = (
                    torch.fft.rfft2(a, norm=norm)
                    if direction == "forward"
                    else torch.fft.irfft2(a, s=shape[-2:], norm=norm)
                )

                def torch_fn():
                    return (
                        torch.fft.rfft2(a, norm=norm)
                        if direction == "forward"
                        else torch.fft.irfft2(a, s=shape[-2:], norm=norm)
                    )

                out_buffer = torch.empty_like(expected)

                def torch_out():
                    return (
                        torch.fft.rfft2(a, norm=norm, out=out_buffer)
                        if direction == "forward"
                        else torch.fft.irfft2(
                            a, s=shape[-2:], norm=norm, out=out_buffer
                        )
                    )

                native = a.numpy().copy()

                def scipy_fn():
                    return (
                        scipy.fft.rfft2(native, norm=norm, workers=args.threads)
                        if direction == "forward"
                        else scipy.fft.irfft2(
                            native, s=shape[-2:], norm=norm, workers=args.threads
                        )
                    )

                def scipy_bridge():
                    # Include current Torch-to-NumPy input conversion in bridge.
                    current = a.detach().numpy()
                    result = (
                        scipy.fft.rfft2(current, norm=norm, workers=args.threads)
                        if direction == "forward"
                        else scipy.fft.irfft2(
                            current, s=shape[-2:], norm=norm, workers=args.threads
                        )
                    )
                    return torch.from_numpy(result)

                implementations = [
                    ("torch", "kernel", torch_fn, None, 0.0),
                    ("torch_out", "kernel", torch_out, None, 0.0),
                    ("scipy", "kernel", scipy_fn, None, 0.0),
                    ("scipy", "bridge", scipy_bridge, None, 0.0),
                ]
                for effort in ["FFTW_ESTIMATE", "FFTW_MEASURE"]:
                    try:
                        start = time.perf_counter()
                        plan = FFTWPlan(shape, direction, norm, args.threads, effort)
                        setup = time.perf_counter() - start

                        def kernel(p=plan):
                            return p.kernel()

                        def refill(p=plan):
                            np.copyto(p.input, native)

                        def bridge(p=plan):
                            return p.bridge(a)

                        implementations.extend(
                            [
                                (effort, "kernel", kernel, refill, setup),
                                (effort, "bridge", bridge, None, setup),
                            ]
                        )
                    except ImportError as exc:
                        failures.append(
                            dict(
                                shape=shape,
                                direction=direction,
                                backend=effort,
                                status="blocked",
                                reason=str(exc),
                            )
                        )
                for backend, boundary, fn, before, setup in implementations:
                    if before:
                        before()
                    observed = fn()
                    obs = (
                        observed.numpy()
                        if isinstance(observed, torch.Tensor)
                        else observed
                    )
                    delta = np.abs(obs - expected.numpy())
                    exact = bool(
                        np.allclose(obs, expected.numpy(), rtol=1e-4, atol=1e-5)
                    )
                    times = measure(fn, args.warmups, args.repeats, before)
                    key = dict(
                        shape=str(shape),
                        direction=direction,
                        norm=norm,
                        backend=backend,
                        boundary=boundary,
                    )
                    rows.append(
                        {
                            **key,
                            "median_s": float(np.median(times)),
                            "p95_s": float(np.quantile(times, 0.95)),
                            "setup_s": setup,
                            "max_abs": float(delta.max()),
                            "relative_l2": float(
                                np.linalg.norm(delta)
                                / max(1e-30, np.linalg.norm(expected.numpy()))
                            ),
                            "allclose": exact,
                            "repeats": len(times),
                            "dtype": str(obs.dtype),
                        }
                    )
                    raw.extend(
                        {**key, "repeat": i, "seconds": t} for i, t in enumerate(times)
                    )
                    append_rows(run / "fft.csv", rows[-1:])
                    append_rows(run / "raw_timings.csv", raw[-len(times) :])
                # Bound descriptor/allocation overhead separately from reusable kernels.
                # Existing in-process wisdom is deliberately retained, not called cold.
                try:

                    def replan_bridge():
                        return FFTWPlan(
                            shape, direction, norm, args.threads, "FFTW_ESTIMATE"
                        ).bridge(a)

                    observed = replan_bridge().numpy()
                    parity = bool(
                        np.allclose(observed, expected.numpy(), rtol=1e-4, atol=1e-5)
                    )
                    times = measure(replan_bridge, args.warmups, args.repeats)
                    replans.extend(
                        dict(
                            shape=str(shape),
                            direction=direction,
                            norm=norm,
                            repeat=i,
                            seconds=t,
                            allclose=parity,
                        )
                        for i, t in enumerate(times)
                    )
                    append_rows(
                        run / "fresh_estimate_plan_timings.csv", replans[-len(times) :]
                    )
                except ImportError:
                    pass
            print(f"completed shape={shape}", flush=True)
    if replans:
        import base64
        import pyfftw

        wisdom = pyfftw.export_wisdom()
        write_json(
            run / "fftw_wisdom.json",
            dict(
                pyfftw_version=pyfftw.__version__,
                fftw_version=getattr(pyfftw, "fftw_version", None),
                simd_alignment=getattr(pyfftw, "simd_alignment", None),
                encoding="base64 of library-specific exported wisdom; not auto-loaded by inference",
                payload=[base64.b64encode(value).decode("ascii") for value in wisdom],
                immediate_reimport=list(pyfftw.import_wisdom(wisdom)),
            ),
        )
    finish(
        run,
        dict(
            status="inconclusive",
            reason="Microbenchmarks only; not a full-model speedup",
            rows=len(rows),
            blocked=failures,
            warmups=args.warmups,
            repeats=args.repeats,
            replan_cohorts=len(replans) // args.repeats if args.repeats else 0,
            replan_note="ESTIMATE construction/allocation plus bridge inside clock; current process wisdom retained; not cold MEASURE planning. Reusable setup separately in fft.csv; raw calls in fresh_estimate_plan_timings.csv",
            profile_inputs_sha256=digest(args.profile / "fourier_inputs.npz"),
            note="Sequential cohorts; FFTW planning explicit; inverse scratch refill excluded from kernel and included in bridge; outputs owned in bridge; wisdom exported diagnostically, not automatically used by inference",
        ),
    )


if __name__ == "__main__":
    cli(main)
