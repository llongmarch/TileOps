"""Tests for ``tileops.fused``."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tileops import gelu_and_mul, silu_and_mul

RTOL, ATOL = 1e-2, 1e-2


@pytest.fixture(scope="module")
def ab_pair(platform):
    _, dev = platform
    a = torch.randn(32, 128, dtype=torch.float32, device=dev)
    b = torch.randn(32, 128, dtype=torch.float32, device=dev)
    return a, b


def test_silu_and_mul_matches_ref(platform, ab_pair):
    a, b = ab_pair
    out = silu_and_mul(a, b)
    ref = F.silu(a) * b
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_gelu_and_mul_matches_ref(platform, ab_pair):
    a, b = ab_pair
    out = gelu_and_mul(a, b)
    ref = F.gelu(a, approximate="tanh") * b
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_silu_and_mul_out(platform, ab_pair):
    a, b = ab_pair
    ref = F.silu(a) * b
    o = torch.empty_like(a)
    r = silu_and_mul(a, b, out=o)
    assert r is o
    torch.testing.assert_close(o, ref, rtol=RTOL, atol=ATOL)


def test_silu_and_mul_noncontiguous_out(platform):
    _, dev = platform
    a = torch.randn(4, 3, dtype=torch.float32, device=dev)
    b = torch.randn(4, 3, dtype=torch.float32, device=dev)
    o = torch.empty(3, 4, dtype=torch.float32, device=dev).t()
    assert not o.is_contiguous()
    r = silu_and_mul(a, b, out=o)
    assert r is o
    torch.testing.assert_close(o, F.silu(a) * b, rtol=RTOL, atol=ATOL)


def test_fused_shape_error(platform):
    _, dev = platform
    a = torch.randn(2, 3, device=dev)
    b = torch.randn(2, 4, device=dev)
    with pytest.raises(ValueError):
        silu_and_mul(a, b)
