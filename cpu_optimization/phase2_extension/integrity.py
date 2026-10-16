"""Hash-only preservation check, with no model execution or artifact rewriting."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.output.exists():
        raise FileExistsError(args.output)
    preserved = json.loads((root/'cpu_optimization/config/preserved_sha256.json').read_text())
    historical = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', '275855e64041aface027aa0be63dc9bb5b7962fe',
                                         'cpu_optimization/results', 'cpu_optimization/config/selection.json'], cwd=root, text=True).splitlines()
    mismatches, git_line_endings = [], []
    for name, expected in preserved.items():
        actual = digest(root/name) if (root/name).is_file() else None
        if actual != expected:
            mismatches.append({'path': name, 'expected_sha256': expected, 'current_sha256': actual})
    for name in historical:
        original = subprocess.check_output(['git', 'show', '275855e64041aface027aa0be63dc9bb5b7962fe:'+name], cwd=root)
        expected = hashlib.sha256(original).hexdigest()
        actual = digest(root/name) if (root/name).is_file() else None
        if actual != expected:
            record = {'path': name, 'expected_git_blob_sha256': expected, 'current_sha256': actual}
            if (root/name).is_file() and (root/name).read_bytes().replace(b'\r\n', b'\n') == original.replace(b'\r\n', b'\n'):
                git_line_endings.append(record)
            else:
                mismatches.append(record)
    result = {'passed': not mismatches, 'original_protected_files_checked': len(preserved),
              'historical_result_and_selection_files_checked': len(historical), 'mismatches': mismatches,
              'git_blob_vs_working_tree_line_ending_only': git_line_endings,
              'scope': 'Byte preservation only; not a statistical replay or independent scientific audit'}
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
