"""Verify preserved history, trial source snapshots and replay every paired interval."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
from ..benchmarks.common import ROOT, HOME, digest, write_json
from ..benchmarks.summarize import collect
from ..statistics import paired_summary
from ..quality import compare_outputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=HOME / "reports/audit.json")
    args = p.parse_args()
    errors = []
    checked = 0
    preserved = json.loads((HOME / "config/preserved_sha256.json").read_text())
    for name, sha in preserved.items():
        if digest(ROOT / name) != sha:
            errors.append(f"Historical file changed: {name}")
        checked += 1
    dev = json.loads((HOME / "config/development_provenance.json").read_text())
    if digest(HOME / "config/development_inputs.npz") != dev["bundle_sha256"]:
        errors.append("Development bundle hash mismatch")
    selection_path = HOME / "config/selection.json"
    selection = (
        json.loads(selection_path.read_text()) if selection_path.exists() else None
    )
    if selection:
        for ident, artifacts in selection["source_artifacts"].items():
            for name, sha in artifacts.items():
                if digest(HOME / "results" / ident / name) != sha:
                    errors.append(f"Locked selection evidence changed: {ident}/{name}")
                checked += 1
    trials = collect()
    replayed = []
    operator_cohorts_replayed = 0
    partial_trials = []
    prediction_pairs_checked = 0
    for trial in trials:
        directory = trial["path"]
        meta = trial["meta"]
        summary = trial["summary"]
        predictions = directory / "predictions.npz"
        validation = directory / "validation.json"
        if predictions.exists() and validation.exists():
            cases = json.loads(validation.read_text())["cases"]
            with np.load(predictions, allow_pickle=False) as saved:
                for i, case in enumerate(cases):
                    parity = compare_outputs(
                        saved[f"reference_{i}"], saved[f"candidate_{i}"]
                    )
                    if parity["allclose"] != case["parity"]["allclose"]:
                        errors.append(
                            f'{trial["id"]}: reference-anchored parity changed case {i}'
                        )
                    prediction_pairs_checked += 1
        if trial["id"].startswith("final_replay") and selection:
            recorded = meta["source_files"].get(
                "cpu_optimization/config/selection.json"
            )
            if (
                recorded != digest(selection_path)
                or meta["created_utc"] <= selection["locked_utc"]
            ):
                errors.append(
                    "Final replay did not follow the immutable selection lock"
                )
        with zipfile.ZipFile(directory / "source.zip") as archive:
            for name, sha in meta["source_files"].items():
                if hashlib.sha256(archive.read(name)).hexdigest() != sha:
                    errors.append(f'{trial["id"]}: source {name}')
                checked += 1
        timings = directory / "paired_timings.csv"
        if timings.exists() and "speedup" not in summary:
            partial_trials.append(trial["id"])
        if timings.exists() and "speedup" in summary:
            with timings.open(newline="") as f:
                rows = list(csv.DictReader(f))
            actual = paired_summary(
                [float(r["reference_s"]) for r in rows],
                [float(r["candidate_s"]) for r in rows],
                summary["bootstrap_seed"],
                summary["bootstrap_draws"],
                summary["bootstrap_block"],
            )
            for key in [
                "speedup",
                "ci_low",
                "ci_high",
                "baseline_median_s",
                "baseline_p95_s",
                "candidate_median_s",
                "candidate_p95_s",
            ]:
                if not np.isclose(actual[key], summary[key], rtol=1e-12, atol=1e-15):
                    errors.append(f'{trial["id"]}: statistic {key}')
            replayed.append(trial["id"])
        units = directory / "fourier_units.csv"
        if units.exists():
            with units.open(newline="") as f:
                unit_rows = list(csv.DictReader(f))
            with (directory / "raw_timings.csv").open(newline="") as f:
                unit_raw = list(csv.DictReader(f))
            for row in unit_rows:
                group = [
                    r
                    for r in unit_raw
                    if r["shape"] == row["shape"] and r["candidate"] == row["candidate"]
                ]
                actual = paired_summary(
                    [float(r["reference_s"]) for r in group],
                    [float(r["candidate_s"]) for r in group],
                    int(row["bootstrap_seed"]),
                    int(row["bootstrap_draws"]),
                    int(row["bootstrap_block"]),
                )
                for key in [
                    "speedup",
                    "ci_low",
                    "ci_high",
                    "baseline_median_s",
                    "candidate_median_s",
                    "baseline_p95_s",
                    "candidate_p95_s",
                ]:
                    if not np.isclose(
                        actual[key], float(row[key]), rtol=1e-12, atol=1e-15
                    ):
                        errors.append(
                            f'{trial["id"]}: unit {row["shape"]}/{row["candidate"]} {key}'
                        )
                operator_cohorts_replayed += 1
        fft = directory / "fft.csv"
        if fft.exists():
            with fft.open(newline="") as f:
                fft_rows = list(csv.DictReader(f))
            with (directory / "raw_timings.csv").open(newline="") as f:
                fft_raw = list(csv.DictReader(f))
            for row in fft_rows:
                group = [
                    r
                    for r in fft_raw
                    if all(
                        r[k] == row[k]
                        for k in ["shape", "direction", "norm", "backend", "boundary"]
                    )
                ]
                values = [float(r["seconds"]) for r in group]
                for key, value in [
                    ("median_s", np.median(values)),
                    ("p95_s", np.quantile(values, 0.95)),
                ]:
                    if not np.isclose(value, float(row[key]), rtol=1e-12, atol=1e-15):
                        errors.append(f'{trial["id"]}: FFT cohort {key}')
                operator_cohorts_replayed += 1
    report = dict(
        passed=not errors,
        errors=errors,
        hashes_checked=checked,
        completed_trials=len(trials),
        paired_intervals_replayed=replayed,
        operator_cohorts_replayed=operator_cohorts_replayed,
        incomplete_paired_trials=partial_trials,
        prediction_pairs_checked=prediction_pairs_checked,
        parity_convention="abs(candidate-reference) <= atol + rtol*abs(reference); every saved end-to-end pair rechecked; no historical raw record rewritten",
        limitations="Source ZIP verifies executed Python; exact installed native libraries and OS are described by run.json, not archived binaries. No full 570 raw held-out rerun inferred.",
    )
    write_json(args.output, report)
    print(json.dumps(report))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
