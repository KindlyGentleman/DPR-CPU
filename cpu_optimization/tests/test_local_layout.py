import copy
import pytest
import torch
from cpu_optimization.candidates.layout import PrepackedConv, EncoderLayout


@pytest.mark.parametrize(
    "groups,padding", [(1, "zeros"), (2, "reflect"), (4, "reflect")]
)
def test_prepacked_conv_preserves_padding_groups_and_weights(groups, padding):
    torch.manual_seed(54)
    conv = torch.nn.Conv2d(
        4, 8, 3, padding=1, padding_mode=padding, groups=groups
    ).eval()
    candidate = PrepackedConv(copy.deepcopy(conv))
    x = torch.randn(1, 4, 16, 16)
    with torch.inference_mode():
        torch.testing.assert_close(candidate(x), conv(x), rtol=1e-4, atol=1e-5)


def test_encoder_local_layout_preserves_masks_and_boundary():
    from deeppr.weightedpartialconv2d import WeightedPartialConv2d_BN_ACT

    conv = WeightedPartialConv2d_BN_ACT(
        1, 8, 3, padding=1, padding_mode="reflect", return_mask=False
    ).eval()
    wrapped = EncoderLayout(copy.deepcopy(conv), first=True, last=True).eval()
    x = torch.rand(1, 1, 32, 32)
    m = torch.ones_like(x)
    m[:, :, 12:20, 12:20] = 0
    with torch.inference_mode():
        actual = wrapped((x, m))
        expected = conv((x, m))
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
