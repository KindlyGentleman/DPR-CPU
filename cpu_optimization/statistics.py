"""Paired circular moving-block bootstrap; no independence claim for raw calls."""

import numpy as np


def paired_summary(reference, candidate, seed=2026101601, draws=5000, block=6):
    b, c = np.asarray(reference, float), np.asarray(candidate, float)
    if (
        b.shape != c.shape
        or b.ndim != 1
        or len(b) < 2
        or not np.isfinite(b).all()
        or not np.isfinite(c).all()
        or (b <= 0).any()
        or (c <= 0).any()
    ):
        raise ValueError("At least two positive finite paired timings required")
    n = len(b)
    block = min(block, n)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, (draws, (n + block - 1) // block))
    idx = ((starts[:, :, None] + np.arange(block)) % n).reshape(draws, -1)[:, :n]
    ci = np.quantile(
        np.median(b[idx], axis=1) / np.median(c[idx], axis=1), [0.025, 0.975]
    )
    return dict(
        baseline_median_s=float(np.median(b)),
        baseline_p95_s=float(np.quantile(b, 0.95)),
        candidate_median_s=float(np.median(c)),
        candidate_p95_s=float(np.quantile(c, 0.95)),
        speedup=float(np.median(b) / np.median(c)),
        median_pair_ratio=float(np.median(b / c)),
        ci_low=float(ci[0]),
        ci_high=float(ci[1]),
        pairs=n,
        bootstrap_seed=seed,
        bootstrap_draws=draws,
        bootstrap_block=block,
        estimator="ratio_of_paired_arm_medians",
    )


def decision(summary, quality_pass, fresh_repeat, memory_pass):
    if not quality_pass:
        return "rejected"
    if (
        summary["speedup"] < 1.10
        or summary["ci_low"] <= 1
        or summary["candidate_p95_s"] > 1.05 * summary["baseline_p95_s"]
    ):
        return "inconclusive"
    if summary["pairs"] < 36 or not fresh_repeat or not memory_pass:
        return "inconclusive"
    return "accepted"
