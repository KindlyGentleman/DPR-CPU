"""Mathematical and dispatch contracts for the new convolution prototypes."""

import pytest


@pytest.mark.parametrize('shape', [(1, 2, 10, 12), (2, 3, 9, 11), (1, 1, 5, 6)])
@pytest.mark.parametrize('padding', [0, 1])
@pytest.mark.parametrize('bias', [False, True])
def test_winograd_f2_preserves_dense_convolution_in_float64(shape, padding, bias):
    from cpu_optimization.phase2_extension.prototypes import WinogradF2Conv
    import torch
    generator = torch.Generator().manual_seed(20261016)
    layer = torch.nn.Conv2d(shape[1], 4, 3, padding=padding, bias=bias).double().eval()
    with torch.no_grad():
        layer.weight.copy_(torch.randn(layer.weight.shape, generator=generator, dtype=torch.float64))
        if bias:
            layer.bias.copy_(torch.randn(layer.bias.shape, generator=generator, dtype=torch.float64))
    x = torch.randn(shape, generator=generator, dtype=torch.float64)
    prototype = WinogradF2Conv(layer)
    with torch.inference_mode():
        torch.testing.assert_close(prototype(x), layer(x), rtol=1e-11, atol=1e-11)
    assert prototype.transform_algorithm == 'F(2x2,3x3)'


@pytest.mark.parametrize('kind', ['tiled', 'tiled_small', 'unfold'])
@pytest.mark.parametrize('shape', [(2, 3, 9, 12), (1, 2, 10, 11)])
def test_direct_prototypes_cover_partial_tiles_and_bias(kind, shape):
    from cpu_optimization.phase2_extension.prototypes import prototype_conv
    import torch
    layer = torch.nn.Conv2d(shape[1], 5, 3, padding=1, bias=True).double().eval()
    generator = torch.Generator().manual_seed(182)
    x = torch.randn(shape, generator=generator, dtype=torch.float64)
    prototype = prototype_conv(layer, kind, tile=4)
    with torch.inference_mode():
        torch.testing.assert_close(prototype(x), layer(x), rtol=1e-12, atol=1e-12)


def test_unsupported_convolution_is_rejected_and_input_is_not_cached():
    from cpu_optimization.phase2_extension.prototypes import prototype_conv
    import torch
    with pytest.raises(ValueError, match='groups|stride|dilation'):
        prototype_conv(torch.nn.Conv2d(4, 4, 3, groups=2), 'winograd')
    layer = torch.nn.Conv2d(2, 3, 3, bias=False).double().eval()
    candidate = prototype_conv(layer, 'winograd')
    with torch.inference_mode():
        x = torch.ones(1, 2, 7, 7, dtype=torch.float64)
        a = candidate(x).clone()
        b = candidate(2*x)
        torch.testing.assert_close(b, 2*a, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(candidate(x), a, rtol=0, atol=0)


def test_candidate_replaces_only_eligible_vanilla_spatial_layers():
    from cpu_optimization.phase2_extension.prototypes import replace_hot_convolutions
    import torch
    class Subclass(torch.nn.Conv2d):
        pass
    model = torch.nn.Sequential(torch.nn.ReflectionPad2d(1), torch.nn.Conv2d(256, 256, 3),
                                torch.nn.Conv2d(256, 256, 1), Subclass(256, 256, 3))
    original = model[1].weight.detach().clone()
    replaced = replace_hot_convolutions(model, 'tiled', tile=16)
    assert replaced == ['1']
    assert type(model[0]) is torch.nn.ReflectionPad2d
    assert type(model[2]) is torch.nn.Conv2d
    assert type(model[3]) is Subclass
    torch.testing.assert_close(model[1].weight, original, rtol=0, atol=0)


@pytest.mark.parametrize('kind', ['winograd', 'tiled', 'tiled_small', 'unfold'])
def test_reflect_padding_bridge_preserves_real_hot_layer_boundary(kind):
    from cpu_optimization.phase2_extension.prototypes import prototype_conv
    import torch
    layer = torch.nn.Conv2d(3, 5, 3, padding=1, padding_mode='reflect', bias=True).double().eval()
    x = torch.randn(2, 3, 9, 11, generator=torch.Generator().manual_seed(20261016), dtype=torch.float64)
    prototype = prototype_conv(layer, kind)
    with torch.inference_mode():
        torch.testing.assert_close(prototype(x), layer(x), rtol=1e-11, atol=1e-11)
    assert layer.padding == (1, 1) and layer.padding_mode == 'reflect'


def test_actual_hot_reflect_configuration_is_eligible_without_subclass_replacement():
    from cpu_optimization.phase2_extension.prototypes import replace_hot_convolutions
    import torch
    class Subclass(torch.nn.Conv2d):
        pass
    model = torch.nn.Sequential(torch.nn.Conv2d(256, 256, 3, padding=1, padding_mode='reflect'),
                                Subclass(256, 256, 3, padding=1, padding_mode='reflect'))
    assert replace_hot_convolutions(model, 'tiled') == ['0']
    assert type(model[1]) is Subclass
