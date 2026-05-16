"""Tests for ``tileops.indexing`` — gather, index_select, nonzero."""

from __future__ import annotations

import pytest
import torch

from tileops import gather, index_select, nonzero

RTOL, ATOL = 1e-4, 1e-4


def test_gather_basic(platform):
    _, dev = platform
    x = torch.randn(3, 5, dtype=torch.float32, device=dev)
    idx = torch.tensor([[0, 2], [1, 3]], dtype=torch.int64, device=dev)
    out = gather(x, dim=1, index=idx)
    ref = torch.gather(x, 1, idx)
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_gather_dim0(platform):
    _, dev = platform
    x = torch.randn(4, 3, dtype=torch.float32, device=dev)
    idx = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64, device=dev)
    out = gather(x, dim=0, index=idx)
    ref = torch.gather(x, 0, idx)
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_index_select(platform):
    _, dev = platform
    x = torch.randn(5, 3, dtype=torch.float32, device=dev)
    idx = torch.tensor([0, 2, 4], dtype=torch.long, device=dev)
    out = index_select(x, dim=0, index=idx)
    ref = torch.index_select(x, 0, idx)
    torch.testing.assert_close(out, ref)


def test_nonzero(platform):
    _, dev = platform
    x = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 2.0]], device=dev)
    out = nonzero(x)
    ref = torch.nonzero(x)
    torch.testing.assert_close(out, ref)
