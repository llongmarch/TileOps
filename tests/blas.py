"""Tests for ``tileops.blas``."""

from __future__ import annotations

import pytest
import torch

from tileops import (
    addmm,
    baddbmm,
    bmm,
    gemm,
    gemv,
    mm,
    mv,
    outer,
)

RTOL, ATOL = 1e-3, 1e-3


def test_gemm_small(platform):
    _, dev = platform
    m, n, k = 7, 11, 5
    a = torch.randn(m, k, dtype=torch.float32, device=dev)
    b = torch.randn(k, n, dtype=torch.float32, device=dev)
    c = gemm(a, b)
    ref = a @ b
    torch.testing.assert_close(c, ref, rtol=RTOL, atol=ATOL)


def test_mm_matches_gemm(platform):
    _, dev = platform
    a = torch.randn(3, 4, dtype=torch.float32, device=dev)
    b = torch.randn(4, 2, dtype=torch.float32, device=dev)
    torch.testing.assert_close(mm(a, b), gemm(a, b), rtol=RTOL, atol=ATOL)


def test_gemm_out(platform):
    _, dev = platform
    m, n, k = 4, 6, 3
    a = torch.randn(m, k, dtype=torch.float32, device=dev)
    b = torch.randn(k, n, dtype=torch.float32, device=dev)
    out = torch.empty(m, n, dtype=torch.float32, device=dev)
    c = gemm(a, b, out=out)
    assert c is out
    torch.testing.assert_close(out, a @ b, rtol=RTOL, atol=ATOL)


def test_gemv_small(platform):
    _, dev = platform
    m, k = 9, 4
    a = torch.randn(m, k, dtype=torch.float32, device=dev)
    x = torch.randn(k, dtype=torch.float32, device=dev)
    y = gemv(a, x)
    ref = a @ x
    torch.testing.assert_close(y, ref, rtol=RTOL, atol=ATOL)


def test_mv_matches_gemv(platform):
    _, dev = platform
    a = torch.randn(4, 3, dtype=torch.float32, device=dev)
    x = torch.randn(3, dtype=torch.float32, device=dev)
    torch.testing.assert_close(mv(a, x), gemv(a, x), rtol=RTOL, atol=ATOL)


def test_gemv_out(platform):
    _, dev = platform
    m, k = 5, 2
    a = torch.randn(m, k, dtype=torch.float32, device=dev)
    x = torch.randn(k, dtype=torch.float32, device=dev)
    out = torch.empty(m, dtype=torch.float32, device=dev)
    y = gemv(a, x, out=out)
    assert y is out
    torch.testing.assert_close(out, a @ x, rtol=RTOL, atol=ATOL)


def test_outer(platform):
    _, dev = platform
    u = torch.randn(5, dtype=torch.float32, device=dev)
    v = torch.randn(7, dtype=torch.float32, device=dev)
    o = outer(u, v)
    torch.testing.assert_close(o, torch.outer(u, v), rtol=RTOL, atol=ATOL)


def test_bmm(platform):
    _, dev = platform
    bsz, m, n, k = 3, 4, 5, 2
    x = torch.randn(bsz, m, k, dtype=torch.float32, device=dev)
    y = torch.randn(bsz, k, n, dtype=torch.float32, device=dev)
    z = bmm(x, y)
    ref = torch.bmm(x, y)
    torch.testing.assert_close(z, ref, rtol=RTOL, atol=ATOL)


def test_bmm_out(platform):
    _, dev = platform
    bsz, m, n, k = 2, 3, 3, 4
    x = torch.randn(bsz, m, k, dtype=torch.float32, device=dev)
    y = torch.randn(bsz, k, n, dtype=torch.float32, device=dev)
    out = torch.empty(bsz, m, n, dtype=torch.float32, device=dev)
    assert bmm(x, y, out=out) is out
    torch.testing.assert_close(out, torch.bmm(x, y), rtol=RTOL, atol=ATOL)


def test_addmm(platform):
    _, dev = platform
    inp = torch.randn(2, 3, dtype=torch.float32, device=dev)
    m1 = torch.randn(2, 4, dtype=torch.float32, device=dev)
    m2 = torch.randn(4, 3, dtype=torch.float32, device=dev)
    beta, alpha = 0.25, 0.5
    ref = torch.addmm(inp, m1, m2, beta=beta, alpha=alpha)
    got = addmm(inp, m1, m2, beta=beta, alpha=alpha)
    torch.testing.assert_close(got, ref, rtol=RTOL, atol=ATOL)


def test_addmm_alpha_zero_broadcast(platform):
    _, dev = platform
    inp = torch.ones(1, 3, device=dev)
    m1 = torch.randn(2, 5, dtype=torch.float32, device=dev)
    m2 = torch.randn(5, 3, dtype=torch.float32, device=dev)
    got = addmm(inp, m1, m2, beta=3.0, alpha=0.0)
    ref = torch.addmm(inp, m1, m2, beta=3.0, alpha=0.0)
    torch.testing.assert_close(got, ref, rtol=RTOL, atol=ATOL)


def test_baddbmm(platform):
    _, dev = platform
    bsz, m, n, k = 2, 3, 4, 5
    inp = torch.randn(bsz, m, n, dtype=torch.float32, device=dev)
    b1 = torch.randn(bsz, m, k, dtype=torch.float32, device=dev)
    b2 = torch.randn(bsz, k, n, dtype=torch.float32, device=dev)
    beta, alpha = 0.75, -0.25
    ref = torch.baddbmm(inp, b1, b2, beta=beta, alpha=alpha)
    got = baddbmm(inp, b1, b2, beta=beta, alpha=alpha)
    torch.testing.assert_close(got, ref, rtol=RTOL, atol=ATOL)


def test_baddbmm_broadcast_input(platform):
    _, dev = platform
    bsz, m, n, k = 2, 2, 2, 2
    inp = torch.randn(1, m, n, dtype=torch.float32, device=dev)
    b1 = torch.randn(bsz, m, k, dtype=torch.float32, device=dev)
    b2 = torch.randn(bsz, k, n, dtype=torch.float32, device=dev)
    ref = torch.baddbmm(inp, b1, b2, beta=-1.0, alpha=1.0)
    got = baddbmm(inp, b1, b2, beta=-1.0, alpha=1.0)
    torch.testing.assert_close(got, ref, rtol=RTOL, atol=ATOL)


def test_gemm_shape_errors(platform):
    _, dev = platform
    a = torch.randn(2, 3, device=dev)
    b = torch.randn(4, 5, device=dev)
    with pytest.raises(ValueError):
        gemm(a, b)


def test_gemv_shape_errors(platform):
    _, dev = platform
    a = torch.randn(3, 4, device=dev)
    x = torch.randn(3, device=dev)
    with pytest.raises(ValueError):
        gemv(a, x)
