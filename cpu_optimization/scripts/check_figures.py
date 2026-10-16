"""Check PNG export metadata and PDF fonts/vector content; render PDF fidelity page."""

import argparse
import hashlib
import json
import subprocess
import importlib.metadata
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures", type=Path, default=Path(__file__).resolve().parents[1] / "figures"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from PIL import Image
    from pypdf import PdfReader
    from pypdf.generic import ContentStream

    args.output.mkdir(parents=True, exist_ok=True)
    entries = json.loads((args.figures / "manifest.json").read_text())
    rows, errors = [], []
    for entry in entries:
        path = args.figures / entry["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            errors.append(f"Figure hash mismatch: {path.name}")
        if path.suffix == ".png":
            with Image.open(path) as im:
                dpi = im.info.get("dpi", (0, 0))
                rows.append(dict(file=path.name, pixels=im.size, dpi=dpi))
                if any(abs(d - 600) > 0.2 for d in dpi):
                    errors.append(f"PNG is not 600 dpi: {path.name}")
        else:
            with path.open("rb") as handle:
                doc = PdfReader(handle)
                page = doc.pages[0]
                embedded = {}
                for name, font_ref in page["/Resources"]["/Font"].items():
                    font = font_ref.get_object()
                    if "/DescendantFonts" in font:
                        font = font["/DescendantFonts"][0].get_object()
                    descriptor = font.get("/FontDescriptor")
                    descriptor = descriptor.get_object() if descriptor else {}
                    embedded[str(font.get("/BaseFont", name))] = any(
                        key in descriptor
                        and len(descriptor[key].get_object().get_data()) > 0
                        for key in ["/FontFile", "/FontFile2", "/FontFile3"]
                    )
                operations = ContentStream(page.get_contents(), doc).operations
                drawings = sum(
                    op in [b"m", b"l", b"c", b"re", b"sh"] for _, op in operations
                )
                text = page.extract_text()
                rows.append(
                    dict(
                        file=path.name,
                        pages=len(doc.pages),
                        points=[float(v) for v in page.mediabox],
                        text_characters=len(text),
                        drawing_operations=drawings,
                        raster_images=len(page.images),
                        embedded_fonts=embedded,
                    )
                )
                if (
                    len(doc.pages) != 1
                    or not text
                    or not drawings
                    or not all(embedded.values())
                ):
                    errors.append(
                        f"PDF page/text/vector/font check failed: {path.name}"
                    )
                if path.stem == "E_fidelity":
                    subprocess.run(
                        [
                            "pdftoppm",
                            "-f",
                            "1",
                            "-singlefile",
                            "-scale-to",
                            "1500",
                            "-png",
                            str(path),
                            str(args.output / "E_pdf_render"),
                        ],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                    )
    report = dict(
        passed=not errors,
        errors=errors,
        figures=rows,
        pypdf=importlib.metadata.version("pypdf"),
        renderer="Poppler pdftoppm from bundled workspace runtime",
        note="Scientific reconstruction panels contain raster data; axes/text/charts remain vector. PDF page rendered independently for visual inspection.",
    )
    (args.output / "checks.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(passed=report["passed"], errors=errors, files=len(rows))))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
