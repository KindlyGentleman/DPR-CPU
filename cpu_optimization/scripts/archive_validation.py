"""Audit saved development predictions against original archived FP32 outputs.

This is an offline quality analysis, not a performance or runtime attribution.
"""

import argparse
import json
from pathlib import Path
import numpy as np
from ..benchmarks.common import load_cases, cases_identity, digest, write_json
from ..quality import compare_outputs, image_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    args = parser.parse_args()
    output = args.trial / "archive_validation.json"
    if output.exists():
        raise FileExistsError("Original archive analysis already exists")
    cases = load_cases()
    identity = json.loads((args.trial / "inputs.json").read_text())
    if identity != cases_identity(cases):
        raise ValueError("Trial does not use the original development input identities")
    rows = []
    with np.load(args.trial / "predictions.npz", allow_pickle=False) as saved:
        for i, case in enumerate(cases):
            original = case["archived_reference"]
            candidate = saved[f"candidate_{i}"]
            parity = compare_outputs(original, candidate)
            bm, cm = image_metrics(case["truth"], original), image_metrics(
                case["truth"], candidate
            )
            ni = cm["aligned_nrmse"] - bm["aligned_nrmse"]
            sl = bm["aligned_ssim"] - cm["aligned_ssim"]
            new_failure = cm["failure"] > bm["failure"]
            rows.append(
                dict(
                    sample_id=case["sample_id"],
                    base_id=case["base_id"],
                    parity=parity,
                    archived=bm,
                    candidate=cm,
                    nrmse_increase=ni,
                    ssim_loss=sl,
                    new_failure=new_failure,
                    passed=parity["allclose"]
                    and ni <= 1e-4
                    and sl <= 1e-4
                    and not new_failure,
                )
            )
    report = dict(
        passed=all(r["passed"] for r in rows),
        cases=rows,
        inputs_sha256=digest(args.trial / "inputs.json"),
        predictions_sha256=digest(args.trial / "predictions.npz"),
        method="Original archived development reference versus newly saved candidate; no timing attribution; no amplitude fit",
    )
    write_json(output, report)
    print(
        json.dumps(
            dict(
                passed=report["passed"],
                cases=len(rows),
                max_abs=max(r["parity"]["max_abs"] for r in rows),
                max_nrmse_increase=max(r["nrmse_increase"] for r in rows),
                max_ssim_loss=max(r["ssim_loss"] for r in rows),
            )
        )
    )


if __name__ == "__main__":
    main()
