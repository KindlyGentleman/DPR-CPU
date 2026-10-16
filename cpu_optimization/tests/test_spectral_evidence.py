from cpu_optimization.benchmarks import spectral


def test_input_identity_and_output_shape_are_both_retained():
    row = spectral.cohort_row(
        "(1, 8, 16, 16)",
        "bmm",
        {"speedup": 1.01},
        {"shape": [1, 8, 16, 16], "allclose": True, "max_abs": 0.0},
    )
    assert row["shape"] == "(1, 8, 16, 16)"
    assert row["output_shape"] == [1, 8, 16, 16]
    assert row["candidate"] == "bmm" and row["allclose"]
