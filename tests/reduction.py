"""Tests for ``tileops.reduction``."""

from __future__ import annotations

import pytest
import torch

from tileops import reduce_amax, reduce_amin, reduce_mean, reduce_prod, reduce_sum

RTOL, ATOL = 1e-3, 1e-3


@pytest.fixture(scope="module")
def x_3d(platform):
    _, dev = platform
    return torch.randn(4, 8, 16, dtype=torch.float32, device=dev)


def test_reduce_sum_last(x_3d):
    out = reduce_sum(x_3d, dim=-1)
    ref = torch.sum(x_3d, dim=-1)
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_reduce_sum_inner_keepdim(x_3d):
    out = reduce_sum(x_3d, dim=1, keepdim=True)
    ref = torch.sum(x_3d, dim=1, keepdim=True)
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_reduce_mean(x_3d):
    out = reduce_mean(x_3d, dim=0)
    ref = torch.mean(x_3d, dim=0)
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_reduce_prod_small(platform):
    _, dev = platform
    x = torch.tensor([[0.5, 2.0], [1.0, 3.0]], dtype=torch.float32, device=dev)
    out = reduce_prod(x, dim=-1)
    ref = torch.prod(x, dim=-1)
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_reduce_amax_amin(x_3d):
    oa = reduce_amax(x_3d, dim=2)
    refa = torch.amax(x_3d, dim=2)
    torch.testing.assert_close(oa, refa, rtol=RTOL, atol=ATOL)
    oi = reduce_amin(x_3d, dim=2)
    refi = torch.amin(x_3d, dim=2)
    torch.testing.assert_close(oi, refi, rtol=RTOL, atol=ATOL)


def test_reduce_sum_out(x_3d):
    ref = torch.sum(x_3d, dim=1)
    out = torch.empty_like(ref)
    r = reduce_sum(x_3d, dim=1, out=out)
    assert r is out
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)
