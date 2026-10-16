"""Run directly from outside the checkout to verify a separately installed wheel."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.installed_root.resolve()
    sys.path.insert(0, str(root))
    import numpy as np
    import cpu_optimization
    import deeppr
    from cpu_optimization import DPRCPU
    from cpu_optimization.benchmarks.common import load_cases

    assert Path(cpu_optimization.__file__).resolve().is_relative_to(root)
    assert Path(deeppr.__file__).resolve().is_relative_to(root)
    with zipfile.ZipFile(args.wheel) as archive:
        members = archive.namelist()
        assert not any(
            n.startswith(("cpu_optimization/results/", "cpu_optimization/tests/"))
            for n in members
        )
        assert not any(n.endswith((".dll", ".pyd", ".pt")) for n in members)
        assert "cpu_optimization/config/development_inputs.npz" in members
        assert "cpu_optimization/THIRD_PARTY_NOTICES.md" in members
        assert "cpu_optimization/SELECTED_CONFIGURATION.md" in members
    case = load_cases()[0]
    model = DPRCPU.from_pretrained(
        checkpoint=args.checkpoint,
        backend="reference",
        threads=6,
        affinity=list(range(12)),
    )
    prediction = model.predict(case["intensity"], case["valid"])
    assert prediction.shape == (64, 64) and prediction.dtype == np.float32
    np.testing.assert_allclose(
        prediction, case["archived_reference"], rtol=1e-4, atol=1e-5
    )
    del model
    args.output.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "PYTHONPATH": str(root)}
    with (args.output / "installed_cli.log").open("x", encoding="utf-8") as log:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "cpu_optimization.benchmarks.baseline",
                "--id",
                "wheel_cli_smoke",
                "--checkpoint",
                str(args.checkpoint),
                "--results-dir",
                str(args.output / "cli"),
                "--warmups",
                "10",
                "--repeats",
                "2",
            ],
            env=environment,
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    report = dict(
        passed=True,
        wheel=str(args.wheel),
        wheel_sha256=hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
        cwd=str(Path.cwd()),
        cpu_optimization=str(cpu_optimization.__file__),
        deeppr=str(deeppr.__file__),
        api_shape=list(prediction.shape),
        api_dtype=str(prediction.dtype),
        max_abs=float(np.abs(prediction - case["archived_reference"]).max()),
        packaged_files=len(members),
        note="External checkpoint + packaged 18-case bundle used outside checkout; installed CLI uses 10 warmups/two diagnostic calls, never a performance promotion",
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
