"""SciencePlots PNG/PDF figures from completed trial evidence; skips absent cohorts."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # registers styles
from ..benchmarks.common import HOME, load_cases, write_json, digest
from ..benchmarks.summarize import collect


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=HOME / "figures")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.style.use(["science", "ieee", "no-latex"])
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 120,
        }
    )
    trials = collect()
    paired = [t for t in trials if "speedup" in t["summary"]]
    selection_path = HOME / "config/selection.json"
    selection = (
        json.loads(selection_path.read_text()) if selection_path.exists() else None
    )
    accepted_ids = selection["evidence_ids"] if selection else []

    def exploratory(t):
        return (
            t["summary"]["candidate"] == "reference"
            and t["summary"].get("methodology") != "persistent_worker_processes"
        )

    eligible = [t for t in paired if not exploratory(t)]
    manifest = []

    def save(fig, name, caption, sources):
        for ext in ["png", "pdf"]:
            path = args.output / f"{name}.{ext}"
            fig.savefig(path, dpi=600)
            manifest.append(
                dict(
                    file=path.name,
                    sha256=digest(path),
                    width_inches=float(fig.get_size_inches()[0]),
                    height_inches=float(fig.get_size_inches()[1]),
                    dpi=600 if ext == "png" else None,
                    sources={
                        str(
                            s.relative_to(HOME)
                            if s.is_relative_to(HOME)
                            else Path("..") / s.relative_to(HOME.parent)
                        ).replace("\\", "/"): digest(s)
                        for s in sources
                    },
                    caption=caption,
                )
            )
        (args.output / f"{name}.md").write_text(caption + "\n", encoding="utf-8")
        plt.close(fig)

    profiles = [t for t in trials if t["summary"]["candidate"] == "reference_profile"]
    if profiles:
        t = profiles[-1]
        s = t["summary"]
        values = s["scope_times_us"]
        fig, axs = plt.subplots(1, 2, figsize=(7.16, 3.1), layout="constrained")
        for ax, prefix, title in zip(
            axs,
            ["stage.", "FU."],
            ["a  Disjoint stages", "b  Nested FourierUnit internals"],
        ):
            keys = [k for k in values if k.startswith(prefix)]
            ax.barh(
                np.arange(len(keys)), [values[k] / 1000 for k in keys], color="#0072B2"
            )
            ax.set_yticks(
                np.arange(len(keys)), [k[len(prefix) :].replace("_", " ") for k in keys]
            )
            ax.invert_yaxis()
            ax.set(xlabel="Inclusive CPU time per forward (ms)", title=title)
        fig.suptitle(
            f'Instrumented profile; FFT kernel self fraction {s["fft_self_fraction"]*100:.2f}%',
            fontsize=10,
        )
        save(
            fig,
            "A_profile",
            f"A. {s['repeats']} instrumented passes after {s['warmups']} warmups. Stage bars are disjoint inclusive means. FourierUnit bars are nested inclusive means and overlap stages. They are not summed. Profiler overhead means these are not deployment latencies. FFT kernel self times are distinct from inclusive FFT scopes. Instrumented output is bitwise identical.",
            [t["path"] / "summary.json", t["path"] / "operators.csv"],
        )
    if paired:
        ordered = sorted(paired, key=lambda t: t["summary"]["speedup"])
        y = np.arange(len(ordered))
        fig, ax = plt.subplots(
            figsize=(7.16, max(3, 1 + len(y) * 0.24)), layout="constrained"
        )
        for i, t in enumerate(ordered):
            s = t["summary"]
            colour = (
                "#777777"
                if exploratory(t)
                else ("#D55E00" if not s["quality_pass"] else "#0072B2")
            )
            if t["id"] in accepted_ids:
                colour = "#009E73"
            ax.plot(
                [s["ci_low"], s["ci_high"]],
                [i, i],
                color=colour,
                lw=1.2,
                ls="--" if exploratory(t) else "-",
            )
            ax.plot(
                s["speedup"],
                i,
                "x" if not s["quality_pass"] else ("+" if exploratory(t) else "o"),
                color=colour,
                ms=4,
            )
        ax.set_yticks(
            y,
            [
                t["id"].replace("_v1", "").replace("_", " ")
                + " ["
                + (
                    "accepted cfg"
                    if t["id"] in accepted_ids
                    else t["summary"]["status"]
                )
                + "]"
                for t in ordered
            ],
        )
        ax.axvline(1, color="k", lw=0.8, label="1x reference")
        ax.axvline(1.1, color="#009E73", ls="--", lw=0.8, label="1.10x target")
        ax.set(
            xlabel="Paired ratio of medians, with 95% block-bootstrap CI",
            title="B  Completed end-to-end comparisons",
        )
        ax.legend(loc="best", fontsize=7)
        save(
            fig,
            "B_speedups",
            "B. All completed paired trials. Gray plus signs are exploratory screens that changed global thread pools between arms and cannot establish promotion. Blue circles passed quality; green circles are the two trials supporting the accepted configuration in selection.json. Crosses failed strict quality. Labels show raw status except explicit accepted-configuration labels. Intervals resample circular six-pair blocks, 5,000 draws; actual pair counts are 36 or 72. Every speedup uses its own nearby reference, never the historical denominator.",
            [t["path"] / "summary.json" for t in ordered]
            + ([selection_path] if selection else []),
        )
    fft = [
        t
        for t in trials
        if t["summary"]["candidate"] == "fft_plans" and "rows" in t["summary"]
    ]
    if fft:
        t = fft[-1]
        with (t["path"] / "fft.csv").open() as f:
            rows = list(csv.DictReader(f))
        shapes = list(dict.fromkeys(r["shape"] for r in rows))
        fig, grid = plt.subplots(2, 2, figsize=(7.16, 6.4), layout="constrained")
        axs = grid[0]
        colours = ["#000000", "#009E73", "#0072B2", "#D55E00", "#CC79A7"]
        backends = list(dict.fromkeys(r["backend"] for r in rows))
        for ax, direction, title in zip(
            axs, ["forward", "inverse"], ["a  Forward real FFT", "b  Inverse real FFT"]
        ):
            for colour, backend in zip(colours, backends):
                for boundary in ["kernel", "bridge"]:
                    selected = [
                        r
                        for r in rows
                        if r["direction"] == direction
                        and r["backend"] == backend
                        and r["boundary"] == boundary
                    ]
                    if not selected:
                        continue
                    indices = [shapes.index(r["shape"]) for r in selected]
                    ax.plot(
                        indices,
                        [float(r["median_s"]) * 1000 for r in selected],
                        marker="o" if boundary == "kernel" else "s",
                        mfc="white" if boundary == "kernel" else colour,
                        color=colour,
                        ls="-" if boundary == "kernel" else "--",
                        lw=0.8,
                        ms=3,
                        label=backend.replace("FFTW_", "FFTW ") + (f" {boundary}"),
                    )
                    failed = [r for r in selected if r["allclose"] != "True"]
                    if failed:
                        ax.plot(
                            [shapes.index(r["shape"]) for r in failed],
                            [float(r["median_s"]) * 1000 for r in failed],
                            "x",
                            color="#D55E00",
                            ms=5,
                        )
            labels = []
            import ast

            for sh in shapes:
                shp = ast.literal_eval(sh)
                labels.append(f"{shp[1]}c,{shp[2]}x{shp[2]}")
            ax.set_xticks(range(len(shapes)), labels, rotation=55, ha="right")
            ax.set(yscale="log", ylabel="Median latency (ms; log scale)", title=title)
        axs[0].legend(fontsize=6.5, ncol=2, loc="best")
        units = [t for t in trials if t["summary"]["candidate"] == "spectral_micro"]
        extra_sources = []
        if units:
            unit_path = units[-1]["path"] / "fourier_units.csv"
            extra_sources.append(unit_path)
            with unit_path.open() as f:
                unit_rows = list(csv.DictReader(f))
            unit_shapes = list(dict.fromkeys(r["shape"] for r in unit_rows))
            for name, colour in zip(
                ["linear", "bmm", "matmul", "packed_gemm", "static_buffers"], colours
            ):
                selected = [r for r in unit_rows if r["candidate"] == name]
                grid[1, 0].plot(
                    range(len(selected)),
                    [float(r["candidate_median_s"]) * 1000 for r in selected],
                    "o-",
                    color=colour,
                    lw=0.8,
                    ms=3,
                    label=name.replace("_", " "),
                )
            reference_rows = [r for r in unit_rows if r["candidate"] == "linear"]
            grid[1, 0].plot(
                range(len(reference_rows)),
                [float(r["baseline_median_s"]) * 1000 for r in reference_rows],
                "x--",
                color="#777777",
                lw=0.8,
                label="reference (linear pairs)",
            )
            import ast

            grid[1, 0].set_xticks(
                range(len(unit_shapes)),
                [
                    f"{ast.literal_eval(s)[1]}c,{ast.literal_eval(s)[2]}x{ast.literal_eval(s)[2]}"
                    for s in unit_shapes
                ],
                rotation=55,
                ha="right",
            )
            grid[1, 0].set(
                yscale="log",
                ylabel="Full FourierUnit median (ms)",
                title="c  Complete FourierUnit, including packing",
            )
            grid[1, 0].legend(fontsize=6, ncol=2)
        selected = [
            u
            for u in eligible
            if u["summary"]["candidate"]
            in [
                "spectral_linear",
                "spectral_bmm",
                "spectral_matmul",
                "packed_gemm",
                "static_buffers",
            ]
        ]
        for i, u in enumerate(selected):
            us = u["summary"]
            grid[1, 1].plot(
                [us["ci_low"], us["ci_high"]], [i, i], color="#0072B2", ls="-"
            )
            grid[1, 1].plot(us["speedup"], i, "o", color="#0072B2", ms=3)
            extra_sources.append(u["path"] / "summary.json")
        grid[1, 1].set_yticks(
            range(len(selected)),
            [u["summary"]["candidate"].replace("_", " ") for u in selected],
        )
        grid[1, 1].axvline(1, color="k", lw=0.7)
        grid[1, 1].axvline(1.1, color="#009E73", ls="--", lw=0.7)
        grid[1, 1].set(
            xlabel="Full-network speedup [95% CI]",
            title="d  End-to-end translation of operator changes",
        )
        save(
            fig,
            "C_fft",
            "C. Eight actual captured FU shapes/operands plus a 512x512 real-transform diagnostic on signed detector input, FP32/complex64. The 512 diagnostic is not a captured physics operand; original physics uses full fft2/ifft2 and no native physics integration was tested. Feature norm=ortho, diagnostic norm=backward. Hollow circles: prepared FFT kernel excluding FFTW scratch refill; filled squares: bridge including required copies and owned outputs. Red crosses mark strict operator parity failures. Panels a/b are sequential cohorts, not network speedups. Panel c measures complete FUs with actual activations/original weights and randomized pairs. Gray reference curve is from linear pairs only. Panel d is separately measured full-model performance. No FFTW network integration was executed; raw setup/calls retained and inverse scratch restored every call.",
            [t["path"] / "fft.csv"] + extra_sources,
        )
    threads = [t for t in eligible if t["summary"]["candidate"] == "reference"]
    if threads:
        defaults = sorted(
            [t for t in threads if t["summary"]["affinity_name"] == "default"],
            key=lambda t: t["summary"]["candidate_threads"],
        )
        fig, axs = plt.subplots(1, 2, figsize=(7.16, 3.4), layout="constrained")
        for t in defaults:
            s = t["summary"]
            x = s["candidate_threads"]
            axs[0].plot(x, s["candidate_median_s"] * 1000, "o", color="#0072B2")
            axs[0].plot(
                x, s["candidate_p95_s"] * 1000, "s", mfc="white", color="#0072B2"
            )
            axs[0].plot(x, s["baseline_median_s"] * 1000, "x", color="#777777")
        axs[0].plot([], [], "o", color="#0072B2", label="Candidate median")
        axs[0].plot([], [], "s", mfc="white", color="#0072B2", label="Candidate p95")
        axs[0].plot([], [], "x", color="#777777", label="Nearby four-thread median")
        axs[0].set(
            xlabel="Candidate intra-op threads",
            ylabel="Prepared-network latency (ms)",
            title="a  Default scheduling",
        )
        axs[0].legend(fontsize=6)
        for i, t in enumerate(threads):
            s = t["summary"]
            axs[1].plot([s["ci_low"], s["ci_high"]], [i, i], color="#0072B2", ls="-")
            axs[1].plot(
                s["speedup"], i, "o" if s["quality_pass"] else "x", color="#0072B2"
            )
        axs[1].set_yticks(
            range(len(threads)),
            [
                f"{t['summary']['candidate_threads']}t {t['summary']['affinity_name'].replace('_',' ')} (n={t['summary']['pairs']})"
                for t in threads
            ],
        )
        axs[1].axvline(1, color="k", lw=0.7)
        axs[1].axvline(1.1, color="#009E73", ls="--", lw=0.7)
        axs[1].set(
            xlabel="Paired speedup [95% CI]", title="b  Threads and verified affinity"
        )
        save(
            fig,
            "D_threads",
            "D. Fixed persistent-worker trials only; global-pool switching screens are excluded. Each worker configures threads/affinity before loading one model and retains them throughout. OMP/MKL thread counts fixed at spawn, KMP_BLOCKTIME=0, OPENBLAS_NUM_THREADS=1. Workers execute sequentially, model-only clocks exclude RPC. Median and empirical p95 are separate statistics, not uncertainty bounds. Gray crosses show each nearby four-thread reference median. Right: six-pair block-bootstrap 95% CIs. P physical uses one logical sibling per verified P core; P only includes SMT siblings. Power and unobserved thermal variation limit generalization.",
            [t["path"] / "summary.json" for t in threads],
        )
    final = [
        t
        for t in trials
        if t["id"].startswith("final_replay")
        and (t["path"] / "predictions.npz").exists()
    ]
    if final:
        t = final[-1]
        cases = load_cases(archived=True)
        picked = [0, 1, 4]
        with np.load(t["path"] / "predictions.npz") as z:
            arrays = [
                [
                    cases[i]["truth"],
                    z[f"reference_{i}"],
                    z[f"candidate_{i}"],
                    np.abs(z[f"candidate_{i}"] - z[f"reference_{i}"]),
                ]
                for i in picked
            ]
        quality_rows = json.loads((t["path"] / "validation.json").read_text())["cases"]
        maximum = max(a.max() for row in arrays for a in row[:3])
        diffmax = max(1e-12, max(row[3].max() for row in arrays))
        fig, axs = plt.subplots(3, 4, figsize=(7.16, 5.3), layout="constrained")
        from matplotlib.colors import PowerNorm

        for ri, (i, row) in enumerate(zip(picked, arrays)):
            for ci, a in enumerate(row):
                kw = (
                    dict(cmap="viridis", norm=PowerNorm(0.5, vmin=0, vmax=maximum))
                    if ci < 3
                    else dict(cmap="magma", vmin=0, vmax=diffmax)
                )
                im = axs[ri, ci].imshow(a, interpolation="nearest", **kw)
                axs[ri, ci].set_xticks([])
                axs[ri, ci].set_yticks([])
                if ri == 0:
                    axs[ri, ci].set_title(
                        [
                            "Physical truth",
                            "Reference",
                            "Selected replay",
                            "Absolute difference",
                        ][ci]
                    )
                if ci == 0:
                    axs[ri, ci].set_ylabel(
                        cases[i]["sample_id"].replace("__", "\n"), fontsize=6
                    )
                if ci == 3:
                    fig.colorbar(im, ax=axs[ri, ci], fraction=0.046, pad=0.03)
                if ci in [1, 2]:
                    met = quality_rows[i]["reference" if ci == 1 else "candidate"]
                    axs[ri, ci].set_xlabel(
                        f"NRMSE {met['aligned_nrmse']:.3f}\nSSIM {met['aligned_ssim']:.3f}; PSNR {met['aligned_psnr']:.1f} dB",
                        fontsize=6.5,
                    )
        save(
            fig,
            "E_fidelity",
            f"E. Fixed published cases0,1,4, including the historical worst reconstruction, used only after selection lock. Truth/reference/selected share one gamma-0.5 mapping over physical amplitude0 to{maximum:.6g}; images have no registration or amplitude fit. Differences use a separate shared linear scale0 to{diffmax:.6g}, the actual largest displayed error (1e-12 fallback only for all-zero errors). Thus small errors are visible with their own explicit colorbar units. Labels report translation/inversion-aligned NRMSE, SSIM and PSNR without amplitude fitting. Small implementation errors do not remove original reconstruction failures. All five cases and metrics remain in validation.json.",
            [
                t["path"] / "validation.json",
                t["path"] / "predictions.npz",
                HOME.parent / "cpu_study/examples.npz",
            ]
            + ([selection_path] if selection else []),
        )
    if eligible:
        fig, ax = plt.subplots(figsize=(7.16, 3.5), layout="constrained")
        groups = {
            "scheduling": ("s", "#0072B2"),
            "spectral": ("o", "#009E73"),
            "layout": ("^", "#CC79A7"),
            "compiler": ("D", "#E69F00"),
            "combined": ("*", "#000000"),
        }
        used_groups = {}
        for i, t in enumerate(eligible):
            s = t["summary"]
            group = (
                "scheduling"
                if s["candidate"] == "reference"
                else (
                    "compiler"
                    if s["candidate"] == "jit_trace"
                    else (
                        "layout"
                        if s["candidate"]
                        in [
                            "mkldnn_prepack",
                            "local_channels_last",
                            "encoder_channels_last",
                        ]
                        else "spectral"
                    )
                )
            )
            if s["candidate"] == "prepack_bmm":
                group = "combined"
            used_groups.setdefault(group, []).append(s["quality_pass"])
            marker, colour = groups[group]
            ax.plot(
                [s["ci_low"], s["ci_high"]],
                [s["max_nrmse_increase"]] * 2,
                color=colour,
                lw=0.7,
                ls="-",
            )
            ax.plot(
                s["speedup"],
                s["max_nrmse_increase"],
                marker if s["quality_pass"] else "x",
                color=colour,
                ms=4,
            )
        for group, (marker, colour) in groups.items():
            if group in used_groups:
                passed = any(used_groups[group])
                ax.plot(
                    [],
                    [],
                    marker if passed else "x",
                    color=colour,
                    label=group if passed else group + " (gate failed)",
                )
        ax.legend(fontsize=7)
        ax.axvline(1, color="k", lw=0.7)
        ax.axvline(1.1, color="#009E73", ls="--", lw=0.7)
        ax.axhline(0, color="#777777", lw=0.6)
        ax.set(
            xlabel="Paired speedup [95% CI]",
            ylabel="Worst per-case aligned NRMSE increase",
            title="F  Track E performance and reconstruction fidelity",
        )
        save(
            fig,
            "F_tradeoff",
            "F. Track E comparisons eligible for interpretation; pool-switch scheduling screens excluded. Points show worst per-case ground-truth NRMSE increase, with paired speedup CIs; shape/color indicates mechanism. Overlapping points are retained and exact coordinates remain in reports/results.csv. Crosses indicate any failed strict quality gate. Passing quality does not imply promotion. Track N/M were not executed; no invented points or Pareto frontier. Trials: "
            + "; ".join(t["id"] for t in eligible),
            [t["path"] / "summary.json" for t in eligible],
        )
    write_json(args.output / "manifest.json", manifest)
    print(f"Wrote {len(manifest)//2} figures as 600-dpi PNG and vector PDF")


if __name__ == "__main__":
    main()
