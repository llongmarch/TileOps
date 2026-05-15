"""Tests for ``tileops.reduction``."""

from __future__ import annotations

import pytest
import torch

from tileops import (
    reduce_all,
    reduce_amax,
    reduce_amin,
    reduce_any,
    reduce_argmax,
    reduce_argmin,
    reduce_cumsum,
    reduce_mean,
    reduce_prod,
    reduce_sum,
    topk,
)

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


def test_reduce_argmax_argmin(x_3d):
    xa = reduce_argmax(x_3d, dim=-1)
    assert xa.dtype == torch.int64
    torch.testing.assert_close(xa.float(), torch.argmax(x_3d, dim=-1).float())

    xm = torch.stack([torch.arange(x_3d.shape[-1], device=x_3d.device).float()])
    xm = xm.expand_as(x_3d)
    idx = reduce_argmin(xm, dim=-1)
    assert idx.dtype == torch.int64
    assert (idx == 0).all()


def test_reduce_argmax_keepdim(platform):
    _, dev = platform
    x = torch.randn(2, 3, 5, device=dev)
    y = reduce_argmax(x, dim=1, keepdim=True)
    assert y.shape == (2, 1, 5)
    torch.testing.assert_close(y.float(), torch.argmax(x, dim=1, keepdim=True).float())


def test_reduce_all_any_bool(platform):
    _, dev = platform
    x = torch.tensor([[True, True], [True, False]], device=dev)
    assert (reduce_all(x, dim=-1) == torch.all(x, dim=-1)).all()
    assert (reduce_any(x, dim=-1) == torch.any(x, dim=-1)).all()


def test_reduce_all_any_float(x_3d):
    xf = x_3d.clone()
    xf[xf <= 1.0] = 0
    ya = reduce_all(xf, dim=-1)
    za = torch.all(xf, dim=-1)
    assert ya.dtype == torch.bool and (ya == za).all()
    yan = reduce_any(xf, dim=-1)
    zn = torch.any(xf, dim=-1)
    assert (yan == zn).all()


def test_reduce_cumsum(x_3d):
    y = reduce_cumsum(x_3d, dim=1)
    ref = torch.cumsum(x_3d, dim=1)
    torch.testing.assert_close(y, ref, rtol=RTOL, atol=ATOL)


def test_reduce_cumsum_out(x_3d):
    ref = torch.cumsum(x_3d, dim=-1)
    out = torch.empty_like(ref)
    r = reduce_cumsum(x_3d, dim=-1, out=out)
    assert r is out
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_topk_last_dim(x_3d):
    v, i = topk(x_3d, 4, dim=-1)
    rv, ri = torch.topk(x_3d, 4, dim=-1, sorted=True)
    assert v.shape == rv.shape == (4, 8, 4)
    assert i.shape == ri.shape
    assert i.dtype == torch.int64
    torch.testing.assert_close(v, rv, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(i, ri)


def test_topk_inner_dim_smallest(x_3d):
    v, i = topk(x_3d, 3, dim=1, largest=False)
    rv, ri = torch.topk(x_3d, 3, dim=1, largest=False, sorted=True)
    assert v.shape == (4, 3, 16)
    torch.testing.assert_close(v, rv, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(i, ri)


def test_topk_1d(platform):
    _, dev = platform
    x = torch.randn(12, device=dev)
    v, i = topk(x, 5, dim=0)
    rv, ri = torch.topk(x, 5, dim=0, sorted=True)
    torch.testing.assert_close(v, rv, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(i, ri)


def test_topk_k_too_large(x_3d):
    with pytest.raises(ValueError, match="cannot exceed"):
        topk(x_3d, 100, dim=-1)
