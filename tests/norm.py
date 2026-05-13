"""Correctness tests for ``tileops.norm`` (softmax, layer norm, RMS norm)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tileops import (
    layer_norm,
    log_softmax,
    online_softmax,
    rms_norm,
    safe_softmax,
    softmax,
)

RTOL_MATH, ATOL_MATH = 1e-3, 1e-3


@pytest.fixture(scope="module")
def x_3d(platform):
    _, dev = platform
    return torch.randn(4, 8, 16, dtype=torch.float32, device=dev)


def test_softmax_last_dim(platform, x_3d):
    out = softmax(x_3d, dim=-1)
    ref = torch.softmax(x_3d, dim=-1)
    torch.testing.assert_close(out, ref, rtol=RTOL_MATH, atol=ATOL_MATH)


def test_softmax_inner_dim(platform, x_3d):
    out = softmax(x_3d, dim=1)
    ref = torch.softmax(x_3d, dim=1)
    torch.testing.assert_close(out, ref, rtol=RTOL_MATH, atol=ATOL_MATH)


def test_safe_softmax_matches_softmax(platform, x_3d):
    ref = softmax(x_3d, dim=-1)
    out = safe_softmax(x_3d, dim=-1)
    torch.testing.assert_close(out, ref, rtol=0.0, atol=0.0)


def test_online_softmax_matches_torch(platform, x_3d):
    out = online_softmax(x_3d, dim=-1)
    ref = torch.softmax(x_3d, dim=-1)
    torch.testing.assert_close(out, ref, rtol=RTOL_MATH, atol=ATOL_MATH)
    out2 = online_softmax(x_3d, dim=1)
    ref2 = torch.softmax(x_3d, dim=1)
    torch.testing.assert_close(out2, ref2, rtol=RTOL_MATH, atol=ATOL_MATH)


def test_log_softmax_last_dim(platform, x_3d):
    out = log_softmax(x_3d, dim=-1)
    ref = torch.log_softmax(x_3d, dim=-1)
    torch.testing.assert_close(out, ref, rtol=RTOL_MATH, atol=ATOL_MATH)


def test_layer_norm_affine(platform, x_3d):
    _, dev = platform
    n = x_3d.shape[-1]
    w = torch.randn(n, dtype=torch.float32, device=dev)
    b = torch.randn(n, dtype=torch.float32, device=dev)
    out = layer_norm(x_3d, (n,), weight=w, bias=b, eps=1e-5)
    ref = F.layer_norm(x_3d, (n,), weight=w, bias=b, eps=1e-5)
    torch.testing.assert_close(out, ref, rtol=RTOL_MATH, atol=ATOL_MATH)


def test_layer_norm_no_affine(platform, x_3d):
    n = x_3d.shape[-1]
    out = layer_norm(x_3d, (n,), weight=None, bias=None, eps=1e-5)
    ref = F.layer_norm(x_3d, (n,), weight=None, bias=None, eps=1e-5)
    torch.testing.assert_close(out, ref, rtol=RTOL_MATH, atol=ATOL_MATH)


def test_rms_norm(platform, x_3d):
    _, dev = platform
    n = x_3d.shape[-1]
    w = torch.ones(n, dtype=torch.float32, device=dev)
    out = rms_norm(x_3d, (n,), weight=w, eps=1e-5)
    ref = x_3d * torch.rsqrt(x_3d.pow(2).mean(-1, keepdim=True) + 1e-5) * w
    torch.testing.assert_close(out, ref, rtol=RTOL_MATH, atol=ATOL_MATH)


def test_layer_norm_multidim_tail(platform):
    _, dev = platform
    x = torch.randn(2, 3, 4, 5, dtype=torch.float32, device=dev)
    ns = (4, 5)
    w = torch.randn(4, 5, dtype=torch.float32, device=dev)
    b = torch.randn(4, 5, dtype=torch.float32, device=dev)
    out = layer_norm(x, ns, weight=w, bias=b)
    ref = F.layer_norm(x, ns, weight=w, bias=b)
    torch.testing.assert_close(out, ref, rtol=RTOL_MATH, atol=ATOL_MATH)


def test_rms_norm_multidim_weight(platform):
    _, dev = platform
    x = torch.randn(2, 3, 4, 5, dtype=torch.float32, device=dev)
    w = torch.randn(4, 5, dtype=torch.float32, device=dev)
    out = rms_norm(x, (4, 5), weight=w)
    ref = x * torch.rsqrt(x.pow(2).mean((-2, -1), keepdim=True) + 1e-5) * w
    torch.testing.assert_close(out, ref, rtol=RTOL_MATH, atol=ATOL_MATH)
