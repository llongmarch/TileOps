"""Correctness tests for ``tileops.shape.pad``."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tileops import pad


@pytest.fixture(scope="module")
def x_3d(platform):
    _, dev = platform
    return torch.randn(4, 8, 16, dtype=torch.float32, device=dev)


def test_pad_constant_symmetric(x_3d):
    out = pad(x_3d, (2, 2, 1, 1, 3, 3))
    ref = F.pad(x_3d, (2, 2, 1, 1, 3, 3), mode="constant", value=0.0)
    torch.testing.assert_close(out, ref)


def test_pad_constant_with_value(x_3d):
    out = pad(x_3d, (1, 2, 3, 4), value=-42.0)
    ref = F.pad(x_3d, (1, 2, 3, 4), mode="constant", value=-42.0)
    torch.testing.assert_close(out, ref)


def test_pad_reflect(x_3d):
    out = pad(x_3d, (1, 2, 3, 4), mode="reflect")
    ref = F.pad(x_3d, (1, 2, 3, 4), mode="reflect")
    torch.testing.assert_close(out, ref)


def test_pad_replicate(x_3d):
    out = pad(x_3d, (1, 1, 2, 2), mode="replicate")
    ref = F.pad(x_3d, (1, 1, 2, 2), mode="replicate")
    torch.testing.assert_close(out, ref)


def test_pad_circular(x_3d):
    out = pad(x_3d, (3, 3, 2, 2), mode="circular")
    ref = F.pad(x_3d, (3, 3, 2, 2), mode="circular")
    torch.testing.assert_close(out, ref)


def test_pad_noop(x_3d):
    """Zero padding on all dims returns the same tensor."""
    pad_spec = (0,) * (x_3d.ndim * 2)
    out = pad(x_3d, pad_spec)
    torch.testing.assert_close(out, x_3d)


def test_pad_out_param(x_3d):
    pad_spec = (1, 2, 3, 4)
    expected = F.pad(x_3d, pad_spec, mode="constant", value=0.0)
    out_tensor = torch.empty_like(expected)
    result = pad(x_3d, pad_spec, out=out_tensor)
    assert result.data_ptr() == out_tensor.data_ptr()
    torch.testing.assert_close(result, expected)


def test_pad_2d_tensor(platform):
    _, dev = platform
    x = torch.randn(8, 64, dtype=torch.float32, device=dev)
    out = pad(x, (2, 3, 1, 4))
    ref = F.pad(x, (2, 3, 1, 4), mode="constant", value=0.0)
    torch.testing.assert_close(out, ref)
