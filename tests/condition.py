"""Tests for ``tileops.condition`` — where, masked_fill."""

from __future__ import annotations

import pytest
import torch

from tileops import where, masked_fill

RTOL, ATOL = 1e-4, 1e-4


def test_where_basic(platform):
    _, dev = platform
    cond = torch.tensor([[True, False], [False, True]], device=dev)
    x = torch.randn(2, 2, dtype=torch.float32, device=dev)
    y = torch.randn(2, 2, dtype=torch.float32, device=dev)
    out = where(cond, x, y)
    ref = torch.where(cond, x, y)
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_masked_fill(platform):
    _, dev = platform
    x = torch.randn(3, 4, dtype=torch.float32, device=dev)
    mask = torch.tensor([[True, False], [False, True]], dtype=torch.bool, device=dev)
    out = masked_fill(x, mask, -1.0)
    ref = x.masked_fill(mask, -1.0)
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)
