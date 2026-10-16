# Frozen DPR1 CPU tools

`DPRCPU.from_pretrained()` defaults to original FP32 reference execution,
four intra-op / one inter-op thread and inherited affinity. It checks the DPR1
checkpoint SHA-256, requires finite 512×512 intensity and binary valid-mask
arrays, and returns owned finite 64×64 physical-amplitude output. `True` is valid.
Loading is CPU-only with `weights_only=True`; precision is never silently lowered.
Optional backend failures are explicit unless `allow_fallback=True` is requested.

For the tested i7-13620H topology, the development-qualified settings are
`backend='reference', threads=6, affinity=list(range(12))`. Both SMT siblings of
the verified six P cores are included. Set threads/inter-op/affinity in a fresh
process before parallel work; these are process-global settings. Do not reuse
these logical CPU IDs without topology verification on another host.

Installed wheels contain the API and config bundle, not pretrained weights.
Pass `checkpoint='/your/checkout/pretrained/param_dpr1.pt'` explicitly outside a
source checkout. Native compiler prototypes are experimental and not defaults.

See the [completed results](../docs/CPU_RESULTS.md),
[reproduction boundary](REPRODUCIBILITY.md), [selected settings](SELECTED_CONFIGURATION.md)
and [third-party notices](THIRD_PARTY_NOTICES.md).
