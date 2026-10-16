import pytest


def test_specs_deduplicate_captured_shapes_without_losing_layer_names():
    from cpu_optimization.phase2_extension.convs import convolution_specs
    import torch
    model = torch.nn.Sequential(torch.nn.Conv2d(2, 3, 3), torch.nn.Conv2d(2, 3, 3),
                                torch.nn.Conv2d(3, 4, 1))
    entries = [dict(name=name, inputs=[dict(shape=shape, stride=stride, dtype='torch.float32',
                                           contiguous=True, channels_last=False)])
               for name, shape, stride in [('0', [1, 2, 8, 8], [128, 64, 8, 1]),
                                           ('1', [1, 2, 8, 8], [128, 64, 8, 1]),
                                           ('2', [1, 3, 6, 6], [108, 36, 6, 1])]]
    specs = convolution_specs(model, entries)
    assert len(specs) == 2
    assert specs[0]['module_names'] == ['0', '1']
    assert specs[0]['weight_shape'] == [3, 2, 3, 3]
    assert specs[0]['calls_per_forward'] == 2


def test_inventory_distinguishes_logical_and_native_padded_kernel_shapes():
    from cpu_optimization.phase2_extension.convs import convolution_specs
    import torch
    model = torch.nn.Sequential(torch.nn.Conv2d(256, 256, 3, padding=1, padding_mode='reflect'))
    specs = convolution_specs(model, [dict(name='0', inputs=[dict(shape=[1, 256, 64, 64],
                                stride=[1048576, 4096, 64, 1], dtype='torch.float32')])])
    assert specs[0]['shape'] == [1, 256, 64, 64]
    assert specs[0]['kernel_input_shape'] == [1, 256, 66, 66]


def test_functional_reference_preserves_reflection_padding():
    from cpu_optimization.phase2_extension.convs import functional_convolution
    import torch
    layer = torch.nn.Conv2d(2, 3, 3, padding=1, padding_mode='reflect').double().eval()
    x = torch.arange(2*5*7, dtype=torch.float64).reshape(1, 2, 5, 7)/100
    with torch.inference_mode():
        torch.testing.assert_close(functional_convolution(layer, x), layer(x), rtol=0, atol=0)


def test_integration_does_not_register_prototype_as_default():
    from cpu_optimization.candidates import NAMES
    assert {'phase2_winograd', 'phase2_tiled', 'phase2_unfold'} <= NAMES
    from cpu_optimization.api import DPRCPU
    import inspect
    assert inspect.signature(DPRCPU.from_pretrained).parameters['backend'].default == 'auto'
    assert DPRCPU.auto_backend() == 'reference'


def test_paired_settings_allow_direct_comparison_with_selected_reference(monkeypatch):
    from cpu_optimization.benchmarks import threading as benchmark
    monkeypatch.setattr(benchmark, 'affinity_for', lambda name: [0, 1] if name == 'p_only' else [0, 1, 2, 3])
    settings = benchmark.paired_worker_settings(6, 4, 'phase2_winograd', 'p_only', 'default')
    assert settings[0] == {'threads': 6, 'affinity': [0, 1], 'backend': 'reference'}
    assert settings[1] == {'threads': 4, 'affinity': [0, 1, 2, 3], 'backend': 'phase2_winograd'}


def test_strict_gate_records_reference_anchored_violation_count():
    from cpu_optimization.quality import compare_outputs
    import numpy as np
    result = compare_outputs(np.array([0., 1., 10000.]), np.array([2e-5, 1., 10001.00008]))
    assert result['violating_elements'] == 2
    assert result['elements'] == 3


@pytest.mark.parametrize('wrong', ['nonfinite', 'shape'])
def test_primitive_failure_retains_serializable_negative_diagnostics(wrong):
    from cpu_optimization.phase2_extension.convs import _parity
    import json
    import torch
    reference = torch.ones(1, 2, 3, 3)
    actual = torch.full_like(reference, float('nan')) if wrong == 'nonfinite' else torch.ones(1, 2, 2, 2)
    result = _parity(reference, actual)
    assert result['allclose'] is False
    assert result['max_abs'] is None
    assert result['relative_l2'] is None
    assert result['violating_elements'] == actual.numel()
    json.dumps(result, allow_nan=False)
