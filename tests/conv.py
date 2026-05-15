"""Tests for ``tileops.conv`` — conv1d and conv2d correctness.

Compares TileLang-backed convolution against ``torch.nn.functional.conv1d`` /
``torch.nn.functional.conv2d``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tileops import conv1d, conv2d

RTOL, ATOL = 1e-3, 1e-3


# ── conv1d ──────────────────────────────────────────────────────────────────


def test_conv1d_basic(platform):
    _, dev = platform
    N, C_in, C_out, L, K = 2, 3, 4, 16, 3
    x = torch.randn(N, C_in, L, dtype=torch.float32, device=dev)
    w = torch.randn(C_out, C_in, K, dtype=torch.float32, device=dev)
    b = torch.randn(C_out, dtype=torch.float32, device=dev)
    out = conv1d(x, w, b, stride=1)
    ref = F.conv1d(x, w, b, stride=1)
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_conv1d_no_bias(platform):
    _, dev = platform
    N, C_in, C_out, L, K = 1, 2, 3, 8, 2
    x = torch.randn(N, C_in, L, dtype=torch.float32, device=dev)
    w = torch.randn(C_out, C_in, K, dtype=torch.float32, device=dev)
    out = conv1d(x, w, stride=1)
    ref = F.conv1d(x, w, stride=1)
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_conv1d_stride2(platform):
    _, dev = platform
    N, C_in, C_out, L, K = 2, 2, 4, 20, 3
    x = torch.randn(N, C_in, L, dtype=torch.float32, device=dev)
    w = torch.randn(C_out, C_in, K, dtype=torch.float32, device=dev)
    b = torch.randn(C_out, dtype=torch.float32, device=dev)
    out = conv1d(x, w, b, stride=2)
    ref = F.conv1d(x, w, b, stride=2)
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_conv1d_groups(platform):
    _, dev = platform
    N, C_in, C_out, L, K, G = 1, 4, 6, 12, 3, 2
    x = torch.randn(N, C_in, L, dtype=torch.float32, device=dev)
    w = torch.randn(C_out, C_in // G, K, dtype=torch.float32, device=dev)
    b = torch.randn(C_out, dtype=torch.float32, device=dev)
    out = conv1d(x, w, b, stride=1, groups=G)
    ref = F.conv1d(x, w, b, stride=1, groups=G)
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_conv1d_out(platform):
    _, dev = platform
    N, C_in, C_out, L, K = 1, 1, 2, 8, 3
    x = torch.randn(N, C_in, L, dtype=torch.float32, device=dev)
    w = torch.randn(C_out, C_in, K, dtype=torch.float32, device=dev)
    b = torch.randn(C_out, dtype=torch.float32, device=dev)
    L_out = (L - K) // 1 + 1
    out = torch.empty(N, C_out, L_out, dtype=torch.float32, device=dev)
    result = conv1d(x, w, b, stride=1, out=out)
    assert result is out
    ref = F.conv1d(x, w, b, stride=1)
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


# ── conv2d ──────────────────────────────────────────────────────────────────


def test_conv2d_basic(platform):
    _, dev = platform
    N, C_in, C_out, H, W, KH, KW = 2, 3, 4, 12, 12, 3, 3
    x = torch.randn(N, C_in, H, W, dtype=torch.float32, device=dev)
    w = torch.randn(C_out, C_in, KH, KW, dtype=torch.float32, device=dev)
    b = torch.randn(C_out, dtype=torch.float32, device=dev)
    out = conv2d(x, w, b, stride=1)
    ref = F.conv2d(x, w, b, stride=1)
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_conv2d_no_bias(platform):
    _, dev = platform
    N, C_in, C_out, H, W, KH, KW = 1, 2, 3, 8, 8, 2, 2
    x = torch.randn(N, C_in, H, W, dtype=torch.float32, device=dev)
    w = torch.randn(C_out, C_in, KH, KW, dtype=torch.float32, device=dev)
    out = conv2d(x, w, stride=1)
    ref = F.conv2d(x, w, stride=1)
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_conv2d_stride2(platform):
    _, dev = platform
    N, C_in, C_out, H, W, KH, KW = 2, 2, 4, 16, 16, 3, 3
    x = torch.randn(N, C_in, H, W, dtype=torch.float32, device=dev)
    w = torch.randn(C_out, C_in, KH, KW, dtype=torch.float32, device=dev)
    b = torch.randn(C_out, dtype=torch.float32, device=dev)
    out = conv2d(x, w, b, stride=2)
    ref = F.conv2d(x, w, b, stride=2)
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_conv2d_groups(platform):
    _, dev = platform
    N, C_in, C_out, H, W, KH, KW, G = 1, 6, 8, 10, 10, 3, 3, 2
    x = torch.randn(N, C_in, H, W, dtype=torch.float32, device=dev)
    w = torch.randn(C_out, C_in // G, KH, KW, dtype=torch.float32, device=dev)
    b = torch.randn(C_out, dtype=torch.float32, device=dev)
    out = conv2d(x, w, b, stride=1, groups=G)
    ref = F.conv2d(x, w, b, stride=1, groups=G)
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_conv2d_out(platform):
    _, dev = platform
    N, C_in, C_out, H, W, KH, KW = 1, 1, 2, 8, 8, 3, 3
    x = torch.randn(N, C_in, H, W, dtype=torch.float32, device=dev)
    w = torch.randn(C_out, C_in, KH, KW, dtype=torch.float32, device=dev)
    b = torch.randn(C_out, dtype=torch.float32, device=dev)
    H_out = (H - KH) // 1 + 1
    W_out = (W - KW) // 1 + 1
    out = torch.empty(N, C_out, H_out, W_out, dtype=torch.float32, device=dev)
    result = conv2d(x, w, b, stride=1, out=out)
    assert result is out
    ref = F.conv2d(x, w, b, stride=1)
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


# ── Error cases ─────────────────────────────────────────────────────────────


def test_conv1d_shape_errors(platform):
    _, dev = platform
    with pytest.raises(ValueError):
        x = torch.randn(2, 3, 10, device=dev)
        w = torch.randn(4, 3, 3, device=dev)
        b = torch.randn(5, device=dev)  # wrong bias length
        conv1d(x, w, b)


def test_conv1d_wrong_ndim(platform):
    _, dev = platform
    with pytest.raises(ValueError):
        x = torch.randn(2, 3, 10, 10, device=dev)  # 4-D
        w = torch.randn(4, 3, 3, device=dev)
        conv1d(x, w)


def test_conv2d_wrong_ndim(platform):
    _, dev = platform
    with pytest.raises(ValueError):
        x = torch.randn(2, 3, 10, device=dev)  # 3-D
        w = torch.randn(4, 3, 3, 3, device=dev)
        conv2d(x, w)
