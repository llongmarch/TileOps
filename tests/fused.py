"""Tests for ``tileops.fused``."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tileops import (
    gelu_and_mul,
    rms_norm,
    silu_and_mul,
    skip_layer_norm,
    skip_rms_norm,
)

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


def test_skip_rms_norm_matches_sequential(platform):
    _, dev = platform
    hidden = (64,)
    batch = 4
    seq = 8
    torch.manual_seed(0)
    x = torch.randn(batch, seq, *hidden, dtype=torch.float32, device=dev)
    delta = torch.randn_like(x)
    w = torch.ones(hidden, dtype=torch.float32, device=dev)
    eps = 1e-6
    ref = rms_norm(x + delta, hidden, weight=w, eps=eps)
    y = skip_rms_norm(x, delta, hidden, weight=w, eps=eps)
    torch.testing.assert_close(y, ref, rtol=RTOL * 10, atol=ATOL)


def test_skip_layer_norm_matches_sequential(platform):
    _, dev = platform
    ns = (32,)
    b, s = 3, 5
    x = torch.randn(b, s, *ns, dtype=torch.float32, device=dev)
    delta = torch.randn_like(x)
    w = torch.ones(ns, dtype=torch.float32, device=dev)
    bias_t = torch.zeros(ns, dtype=torch.float32, device=dev)
    eps = 1e-6
    ref = torch.nn.functional.layer_norm(
        x + delta, ns, weight=w.reshape(-1), bias=bias_t.reshape(-1), eps=eps,
    )
    y = skip_layer_norm(x, delta, ns, weight=w, bias=bias_t, eps=eps)
    torch.testing.assert_close(y, ref, rtol=RTOL * 10, atol=ATOL * 10)


def test_skip_rms_from_fused_submodule(platform):
    """Skip norms are re-exported from ``tileops.fused`` for discoverability."""
    from tileops.fused import skip_rms_norm as sk

    assert sk is skip_rms_norm
    _, dev = platform
    a = torch.randn(2, 3, device=dev)
    b = torch.randn(2, 4, device=dev)
    with pytest.raises(ValueError):
        silu_and_mul(a, b)
