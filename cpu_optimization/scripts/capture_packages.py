"""Capture allowlisted relevant package versions without importing model runtimes."""

import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    names = [
        "torch",
        "torchvision",
        "numpy",
        "scipy",
        "scikit-image",
        "pillow",
        "h5py",
        "tqdm",
        "psutil",
        "pyFFTW",
        "matplotlib",
        "SciencePlots",
        "pytest",
        "setuptools",
    ]
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    result = dict(
        captured_utc=datetime.now(timezone.utc).isoformat(),
        executable=sys.executable,
        python=sys.version,
        platform=platform.platform(),
        versions=versions,
        scope="Allowlisted relevant packages after campaign; native Torch build and per-trial environment remain in run.json",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
