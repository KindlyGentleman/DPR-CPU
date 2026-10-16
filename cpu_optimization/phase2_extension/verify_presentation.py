"""Check final exports and identify source changes after the last model trial."""

import json
from pathlib import Path
import re

from PIL import Image

from .reporting import sha


def main():
    root = Path(__file__).resolve().parents[2]
    home = root/'cpu_optimization'
    output = home/'phase2_extension/presentation_qa.json'
    if output.exists():
        raise FileExistsError(output)
    report = home/'reports/phase2_extension_20261016_final'
    manifest = json.loads((report/'manifest.json').read_text())
    errors, exports = [], []
    for figure in manifest['figures']:
        for item in figure['files']:
            path = report/item['file']
            if sha(path) != item['sha256']:
                errors.append(f'Figure hash mismatch: {path}')
        png = report/(figure['name']+'.png')
        with Image.open(png) as image:
            metadata = {'mode': image.mode, 'pixels': list(image.size), 'dpi': image.info.get('dpi')}
            if image.mode != 'RGB' or any(abs(value-600) > .1 for value in metadata['dpi']):
                errors.append(f'PNG export metadata: {png}')
        pdf = (report/(figure['name']+'.pdf')).read_bytes()
        descriptors = re.findall(rb'/Type /FontDescriptor\b.*?>>', pdf, flags=re.S)
        embedded = bool(descriptors) and all(b'/FontFile2' in descriptor for descriptor in descriptors)
        if not embedded or b'/Subtype /Type3' in pdf:
            errors.append(f'PDF TrueType embedding check: {figure["name"]}')
        svg = (report/(figure['name']+'.svg')).read_text(encoding='utf-8')
        if '<text' not in svg:
            errors.append(f'SVG editable text missing: {figure["name"]}')
        exports.append({'figure': figure['name'], **metadata, 'pdf_truetype_embedded': embedded,
                        'svg_editable_text': '<text' in svg})
    if sha(report/'convs.csv') != sha(home/'benchmarks/convs.csv'):
        errors.append('Published benchmark CSV differs from report source CSV')
    for name, expected in manifest['source_files_sha256'].items():
        if sha(Path(name)) != expected:
            errors.append(f'Report source changed: {name}')
    executed = home/'results/phase2_compile_default_screen_20261016_v5/run.json'
    source = json.loads(executed.read_text())['source_files']
    changed = [name for name, expected in source.items() if sha(root/name) != expected]
    allowed = {'cpu_optimization/phase2_extension/PLAN.md',
               'cpu_optimization/phase2_extension/reporting.py',
               'cpu_optimization/tests/test_phase2_reporting.py'}
    if set(changed)-allowed:
        errors.append(f'Unexpected current source difference after v5: {set(changed)-allowed}')
    files = [path for path in report.iterdir() if path.is_file()]
    files += [home/'reports'/name for name in ('FINAL_RESULTS.md', 'EXPERIMENT_LEDGER.md',
                                              'COMPILER_BACKENDS.md', 'FUTURE_WORK.md', 'conv_prototypes.md')]
    files += [home/'phase2_extension'/name for name in ('RUNBOOK.md', 'CODE_REVIEW.md',
              'post_execution_audit.json', 'post_execution_integrity.json', 'release_verification.json')]
    result = {'passed': not errors, 'errors': errors, 'exports': exports,
              'reporting_script_matches_manifest': sha(Path(__file__).with_name('reporting.py')) == manifest['reporting_script_sha256'],
              'current_source_changes_after_v5': changed,
              'new_parent_verifiers_not_used_during_timing': ['verify_release.py', 'verify_presentation.py'],
              'files_sha256': {str(path): sha(path) for path in files}, 'verifier_sha256': sha(Path(__file__)),
              'visual_review': 'Parent inspected all three final PNGs: readable labels, no clipping/annotation overlap; '
                               'native hot-layer plot requires both captured input and exact 256x256x3x3 weights. '
                               'Earlier renders retained; no raw experiment edited.',
              'limits': 'PDF font dictionaries checked without a separate PDF page render; SVG text remains unembedded. '
                        'No journal-specific compliance or independent scientific audit claim.'}
    if not result['reporting_script_matches_manifest']:
        result['errors'].append('Current reporting script differs from final rendering')
        result['passed'] = False
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key != 'files_sha256'}))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
