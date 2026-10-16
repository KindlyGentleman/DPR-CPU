# Internal research and maintenance TODO

This file is the source of truth. Open tasks are plans, never completed evidence.
`python scripts/todo.py` prints this list and executes no experiments.

## P1 — Separate synthetic pilot (independent research project)

- [ ] Review a separately authorized budget amendment for the remaining 3,294 of 8,192 pairs; the cooperative execution cap does not reset on resume.
- [ ] Complete the synthetic pilot before any policy-selection inference. Partial run: 4,898 pairs, integrity passed, no sample failures; no completed-grid inference.
- [ ] Verify complete-grid replay, calibration-only policy lock and matched controls before interpreting results.
- [ ] Obtain independent scientific review of any future gating conclusion.

## P2 — CPU evidence and portability

- [ ] Replicate six-P-core scheduling in another controlled session with thermal/background observations.
- [ ] Evaluate another CPU or OS when resources are available; discover topology rather than reuse logical CPU IDs.
- [ ] Continue mathematically justified convolution research only when new profiling evidence supports it; retain parity and full-model gates.

## P2 — Release maintenance

- [ ] Recheck package installation and release reproducibility after dependency/runtime changes.
- [ ] Provision the inherited legacy runtime to investigate network_synthetic without changing its original tolerance or references.
- [ ] Review external native-cache ownership policy separately before deletion; preserve compiler logs/generated source first.
