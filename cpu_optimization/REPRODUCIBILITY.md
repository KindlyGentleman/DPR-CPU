# Reproduction boundary

Use the root README's CPU-only recipe. `python cpu_study/plot_cpu_update.py`
checks committed compact-source hashes and replays paired statistics without
running any benchmark. `python cpu_study/reproduce.py --all` checks five genuine
archived examples against original DPR1 in a fresh process.

The source includes research implementations/tests for inspection. The full
historical raw result tree is not published. Historic benchmark/reporting/audit
entry points that consume that tree require the separately preserved local
archive; use `scripts/release_audit.py` for this compact release. Do not overwrite
immutable trial IDs or rerun long campaigns as part of routine release checks.

The seven-object development bundle is for gates, not an independent held-out
accuracy study. The 570 historical summary metric rows cannot recreate detector
arrays. Configuration provenance retains original absolute paths as historical
metadata; normal inference does not read those paths.
