# Developer maintenance

The public showcase describes completed CPU results. Future work is recorded in
`scripts/TODO.md`; `python scripts/todo.py` prints it without launching experiments.

## Cache cleanup

Run from a Python environment with `psutil` installed (the CPU optimization extra
provides it). The PowerShell wrapper passes arguments directly to Python without
shell interpolation. It works in Windows PowerShell and PowerShell 7.

```powershell
powershell -File scripts/cleanup_cache.ps1
powershell -File scripts/cleanup_cache.ps1 -Apply
```

Dry-run is the default. Before deletion the inventory reports exact paths,
cache type, file count, bytes, tracked/ignored status, repository containment and
owned active processes. `-Python` selects an interpreter; `-Repository` selects
an exact Git root. `-Report build/cleanup.json` writes a new JSON record (create
the parent directory first). Reports refuse overwrite.

Only `__pycache__`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache`, `.hypothesis`,
`.mplconfig` and `.ipynb_checkpoints` are disposable by this policy. Virtual
environments, `.git`, `node_modules`, generic build trees and scientific result
directories are not traversed. Generic `cache`, `data`, `results`, `models` and
`artifacts` names are never deletion criteria. Tracked cache descendants refuse
the entire apply operation. Back up and review any index removal separately.

All targets must resolve strictly below the exact repository root. Links,
Windows junctions/reparse points and unexpected linked descendants are rejected.
The engine checks tracking, inventory stability and owned process state again
immediately before removal. Do not run cleanup concurrently with programs that
create or mutate caches; this is a local maintenance tool, not a filesystem
transaction against a malicious concurrent writer.

`TORCHINDUCTOR_CACHE_DIR`, `TORCH_EXTENSIONS_DIR`, default extension locations and
an explicitly supplied `-OwnedExternalCache` are inventoried. Neither `-Apply`
nor `-IncludeOwnedExternalCache` grants global Torch-cache deletion. External
deletion deliberately refuses until a separately reviewed exact ownership
policy proves experiment ownership, no active users, and preservation of compiler
logs/generated sources. The prior Phase 2 native cache is retained locally with
its 431-file / 12,792,176-byte inventory and separate 114-text-source archive.
Its exact machine path is recorded in the local audit, not a portable command.
No checkpoint, downloaded runtime dependency or virtual environment is removed.

## Release audit

```bash
python scripts/release_audit.py --verify --output build/release-audit
```

The audit checks protected upstream/showcase hashes, checkpoint, compact sources,
paired claims, plotted CSV, image/PDF metadata, relative links, documented command
targets, tracked caches, large files, obvious credentials and accidental personal
paths. `--verify` regenerates figures in a temporary copy, runs tests and original
regressions separately, and builds/installs a wheel for an outside-checkout
inference smoke check. Reports distinguish the known inherited regression from
passing scoped suites. `--require-clean` additionally fails on a dirty worktree.
Use a fresh output directory; generated audit output belongs under ignored build.

History is not rewritten to shrink clones. The release includes necessary
checkpoints and compact evidence; large historical checkpoint blobs dominate
clone size and are not Python caches. Bulky local raw research archives remain
outside the publication branch. The protected-file manifest covers all original
master files except the intentionally revised README, ignore rules and packaging.
The manifest records both original Git-content hashes and the pre-existing
Windows working-tree hashes, so native LF/CRLF checkout differences do not become
false evidence mismatches. Compact scientific sources have local Git attributes
that preserve their exact bytes on every platform.
