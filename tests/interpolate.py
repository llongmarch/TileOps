"""Correctness tests for ``tileops.interpolate``."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tileops import interpolate


@pytest.fixture(scope="module")
def x_4d(platform):
    _, dev = platform
    return torch.randn(2, 3, 32, 32, dtype=torch.float32, device=dev)


def test_interpolate_nearest_size(x_4d):
    out = interpolate(x_4d, size=(16, 16), mode="nearest")
    ref = F.interpolate(x_4d, size=(16, 16), mode="nearest")
    torch.testing.assert_close(out, ref)


def test_interpolate_bilinear_size(x_4d):
    out = interpolate(x_4d, size=(8, 8), mode="bilinear", align_corners=False)
    ref = F.interpolate(x_4d, size=(8, 8), mode="bilinear", align_corners=False)
    torch.testing.assert_close(out, ref)


def test_interpolate_bilinear_align_corners(x_4d):
    out = interpolate(x_4d, size=(24, 24), mode="bilinear", align_corners=True)
    ref = F.interpolate(x_4d, size=(24, 24), mode="bilinear", align_corners=True)
    torch.testing.assert_close(out, ref)


def test_interpolate_scale_factor(x_4d):
    out = interpolate(x_4d, scale_factor=2.0, mode="nearest")
    ref = F.interpolate(x_4d, scale_factor=2.0, mode="nearest")
    torch.testing.assert_close(out, ref)


def test_interpolate_bicubic(x_4d):
    out = interpolate(x_4d, size=(48, 48), mode="bicubic", align_corners=False)
    ref = F.interpolate(x_4d, size=(48, 48), mode="bicubic", align_corners=False)
    torch.testing.assert_close(out, ref)


def test_interpolate_area_downsample(x_4d):
    out = interpolate(x_4d, size=(8, 8), mode="area")
    ref = F.interpolate(x_4d, size=(8, 8), mode="area")
    torch.testing.assert_close(out, ref)


def test_interpolate_out_param(x_4d):
    ref = F.interpolate(x_4d, size=(16, 16), mode="bilinear", align_corners=False)
    out_tensor = torch.empty_like(ref)
    result = interpolate(x_4d, size=(16, 16), mode="bilinear",
                         align_corners=False, out=out_tensor)
    assert result.data_ptr() == out_tensor.data_ptr()
    torch.testing.assert_close(result, ref)


def test_interpolate_no_size_or_scale():
    x = torch.randn(1, 3, 16, 16)
    with pytest.raises(ValueError, match="either size or scale_factor"):
        interpolate(x)


def test_interpolate_both_size_and_scale():
    x = torch.randn(1, 3, 16, 16)
    with pytest.raises(ValueError, match="only one of size or scale_factor"):
        interpolate(x, size=(8, 8), scale_factor=2.0)
