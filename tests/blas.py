"""Tests for ``tileops.blas`` (``gemm``, ``gemv``)."""

from __future__ import annotations

import pytest
import torch

from tileops import gemm, gemv

RTOL, ATOL = 1e-3, 1e-3


def test_gemm_small(platform):
    _, dev = platform
    m, n, k = 7, 11, 5
    a = torch.randn(m, k, dtype=torch.float32, device=dev)
    b = torch.randn(k, n, dtype=torch.float32, device=dev)
    c = gemm(a, b)
    ref = a @ b
    torch.testing.assert_close(c, ref, rtol=RTOL, atol=ATOL)


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


def test_gemv_out(platform):
    _, dev = platform
    m, k = 5, 2
    a = torch.randn(m, k, dtype=torch.float32, device=dev)
    x = torch.randn(k, dtype=torch.float32, device=dev)
    out = torch.empty(m, dtype=torch.float32, device=dev)
    y = gemv(a, x, out=out)
    assert y is out
    torch.testing.assert_close(out, a @ x, rtol=RTOL, atol=ATOL)


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
