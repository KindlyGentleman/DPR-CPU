# Release verification — 16 October 2026

The [machine-readable audit](release_audit.json) passes all engineering checks.
It records tested source hashes and each verification outcome. This audit checks
reproduction and release integrity; it is not independent scientific validation.

| Check | Observed result |
| --- | --- |
| Integrated historical CPU/study suite | 134 passed |
| Final scoped suite including release/cleanup checks | **151 passed** (134 existing + 17 new) |
| Separate original regression suite | 14 passed, 5 skipped, 4 xfailed, 1 known inherited failure |
| Five original archived DPR1 examples | Bitwise identical under PyTorch 2.5.1+cpu |
| Installed wheel, outside-checkout CPU inference | Finite owned 64×64 FP32; archived case bitwise identical |
| Protected original master files | 95 checked, unchanged original Git/Windows checkout variants |
| Separately preserved local research/source files | 947 checked, 249,761,094 bytes, no changes |
| Compact paired statistics | All three displayed intervals replayed from raw paired rows |
| Convolution primitive statistics | 80 cohorts replayed; four prototypes slower, tile-16 alone passes strict FP32 parity |
| New figure | Exact 7.16-inch width, 600-dpi PNG, vector PDF, embedded TrueType font |
| Regeneration | Original figures/GIF/inference commands succeeded in a temporary copy; new PNG pixels and PDF streams match |
| Original GIF | Five genuine cases; 2 s/frame; infinite loop; historical bytes preserved |
| Dependencies | `pip check`: no broken requirements |
| Original-checkout cache cleanup | 1,061,641 bytes / 108 ignored, untracked files removed; protected evidence unchanged |

The unchanged inherited `network_synthetic` assertion reports DPR0 1.802e-5
and DPR1 1.866e-5 relative L2 against its original 1e-5 threshold. The audit
checks this exact failure signature and the expected 24-case collection;
unrelated errors or silently skipped float32 cases fail verification. No
tolerance or original test was modified. The separate research project's
historical 84 passing tests are not included in the CPU totals.

Independent code review identified process-inspection, manifest-coverage and
audit false-success cases. Regression tests and fixes address them; final
read-only review found no unresolved material issues. The audit rejects
optimized Python, sanitizes test-selection environment variables, verifies
both regenerated PNG pixels and PDF content, and distinguishes inherited
numeric failure from other exceptions.

Wheel verification uses the declared isolated build requirements and installs
into a fresh target outside the checkout, with explicit checkpoint/example
paths. Runtime dependencies are reused from the pinned environment; this is
not a clean-machine dependency-download test. Original source, checkpoints,
tests, license texts, historical figures/GIF and selection locks are preserved.

The first audit rejected an unresolved documentation link and a nonisolated
build using old setuptools. The link was resolved and the build now honors
declared build isolation. Failed attempts remain in the local release logs.
The final audit ran before its verification commit; the report's source hashes
bind tested Python/TOML content. Final Git/clean-checkout state is checked after
committing. [Cache cleanup details](CACHE_CLEANUP.json) use checkout-relative paths;
the exact local path inventory and byte-preserved evidence backup remain local.

A final fresh Git checkout exposed index normalization that predated archive
attributes. The corrected commit restores original archived bytes (line endings
only), adds a Git-index source-hash gate and regression test, and passes 151 scoped
tests plus all compact static/data checks. The machine report retains the earlier
full-execution source hashes and records the subsequent index/audit source checks
separately; no scientific value or plotting implementation changed.
