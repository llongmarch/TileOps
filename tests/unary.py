"""Tests for ``tileops.unary`` — mathematical unary operators.

Compares TileLang-backed unary kernels against ``torch`` reference
implementations.
"""

from __future__ import annotations

import pytest
import torch

from tileops import (
    exp, log,
    sqrt, rsqrt, square,
    abs, sign, neg,
    round, floor, ceil,
    reciprocal,
    clamp,
)

RTOL, ATOL = 1e-4, 1e-4


# ── Exponential / logarithmic ──────────────────────────────────────────────


def test_exp(platform):
    _, dev = platform
    x = torch.randn(64, 64, dtype=torch.float32, device=dev)
    torch.testing.assert_close(exp(x), torch.exp(x), rtol=RTOL, atol=ATOL)


def test_log(platform):
    _, dev = platform
    x = torch.rand(64, 64, dtype=torch.float32, device=dev).add_(0.1)  # positive
    torch.testing.assert_close(log(x), torch.log(x), rtol=RTOL, atol=ATOL)


# ── Power / root ───────────────────────────────────────────────────────────


def test_sqrt(platform):
    _, dev = platform
    x = torch.rand(64, 64, dtype=torch.float32, device=dev).add_(0.1)
    torch.testing.assert_close(sqrt(x), torch.sqrt(x), rtol=RTOL, atol=ATOL)


def test_rsqrt(platform):
    _, dev = platform
    x = torch.rand(64, 64, dtype=torch.float32, device=dev).add_(0.1)
    torch.testing.assert_close(rsqrt(x), torch.rsqrt(x), rtol=RTOL, atol=ATOL)


def test_square(platform):
    _, dev = platform
    x = torch.randn(64, 64, dtype=torch.float32, device=dev)
    torch.testing.assert_close(square(x), x * x, rtol=RTOL, atol=ATOL)


# ── Sign / absolute ────────────────────────────────────────────────────────


def test_abs(platform):
    _, dev = platform
    x = torch.randn(64, 64, dtype=torch.float32, device=dev)
    torch.testing.assert_close(abs(x), torch.abs(x), rtol=RTOL, atol=ATOL)


def test_sign(platform):
    _, dev = platform
    x = torch.randn(64, 64, dtype=torch.float32, device=dev)
    torch.testing.assert_close(sign(x), torch.sign(x), rtol=RTOL, atol=ATOL)


def test_neg(platform):
    _, dev = platform
    x = torch.randn(64, 64, dtype=torch.float32, device=dev)
    torch.testing.assert_close(neg(x), -x, rtol=RTOL, atol=ATOL)


# ── Rounding ───────────────────────────────────────────────────────────────


def test_round(platform):
    _, dev = platform
    x = torch.randn(64, 64, dtype=torch.float32, device=dev) * 10
    torch.testing.assert_close(round(x), torch.round(x), rtol=RTOL, atol=ATOL)


def test_floor(platform):
    _, dev = platform
    x = torch.randn(64, 64, dtype=torch.float32, device=dev) * 10
    torch.testing.assert_close(floor(x), torch.floor(x), rtol=RTOL, atol=ATOL)


def test_ceil(platform):
    _, dev = platform
    x = torch.randn(64, 64, dtype=torch.float32, device=dev) * 10
    torch.testing.assert_close(ceil(x), torch.ceil(x), rtol=RTOL, atol=ATOL)


# ── Reciprocal ─────────────────────────────────────────────────────────────


def test_reciprocal(platform):
    _, dev = platform
    x = torch.randn(64, 64, dtype=torch.float32, device=dev).abs_().add_(0.1)
    torch.testing.assert_close(reciprocal(x), torch.reciprocal(x), rtol=RTOL, atol=ATOL)


# ── Clamp ──────────────────────────────────────────────────────────────────


def test_clamp_default(platform):
    _, dev = platform
    x = torch.randn(64, 64, dtype=torch.float32, device=dev)
    torch.testing.assert_close(clamp(x), torch.clamp(x, 0.0, 1.0), rtol=RTOL, atol=ATOL)


def test_clamp_custom(platform):
    _, dev = platform
    x = torch.randn(64, 64, dtype=torch.float32, device=dev) * 5
    torch.testing.assert_close(
        clamp(x, min_val=-1.0, max_val=2.0),
        torch.clamp(x, -1.0, 2.0),
        rtol=RTOL, atol=ATOL,
    )


def test_clamp_out(platform):
    _, dev = platform
    x = torch.randn(64, 64, dtype=torch.float32, device=dev)
    out = torch.empty_like(x)
    result = clamp(x, min_val=-0.5, max_val=0.5, out=out)
    assert result is out
    torch.testing.assert_close(out, torch.clamp(x, -0.5, 0.5), rtol=RTOL, atol=ATOL)


# ── Out-parameter variants (representative sample) ─────────────────────────


def test_exp_out(platform):
    _, dev = platform
    x = torch.randn(32, dtype=torch.float32, device=dev)
    out = torch.empty_like(x)
    result = exp(x, out=out)
    assert result is out
    torch.testing.assert_close(out, torch.exp(x), rtol=RTOL, atol=ATOL)


def test_neg_out(platform):
    _, dev = platform
    x = torch.randn(32, dtype=torch.float32, device=dev)
    out = torch.empty_like(x)
    result = neg(x, out=out)
    assert result is out
    torch.testing.assert_close(out, -x, rtol=RTOL, atol=ATOL)


def test_sqrt_out(platform):
    _, dev = platform
    x = torch.rand(32, dtype=torch.float32, device=dev).add_(0.1)
    out = torch.empty_like(x)
    result = sqrt(x, out=out)
    assert result is out
    torch.testing.assert_close(out, torch.sqrt(x), rtol=RTOL, atol=ATOL)


# ── Empty-input error (representative) ─────────────────────────────────────


def test_empty_input(platform):
    _, dev = platform
    x = torch.empty(0, dtype=torch.float32, device=dev)
    with pytest.raises(ValueError):
        exp(x)


def test_out_shape_mismatch(platform):
    _, dev = platform
    x = torch.randn(10, dtype=torch.float32, device=dev)
    out = torch.empty(5, dtype=torch.float32, device=dev)
    with pytest.raises(ValueError):
        exp(x, out=out)


def test_out_dtype_mismatch(platform):
    _, dev = platform
    x = torch.randn(10, dtype=torch.float32, device=dev)
    out = torch.empty(10, dtype=torch.float64, device=dev)
    with pytest.raises(TypeError):
        exp(x, out=out)
