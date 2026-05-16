"""Tests for ``tileops.matrix`` — tril, triu."""

from __future__ import annotations

import pytest
import torch

from tileops import tril, triu

RTOL, ATOL = 1e-4, 1e-4


def test_tril(platform):
    _, dev = platform
    x = torch.randn(4, 4, dtype=torch.float32, device=dev)
    torch.testing.assert_close(tril(x), torch.tril(x), rtol=RTOL, atol=ATOL)


def test_triu(platform):
    _, dev = platform
    x = torch.randn(4, 4, dtype=torch.float32, device=dev)
    torch.testing.assert_close(triu(x), torch.triu(x), rtol=RTOL, atol=ATOL)


def test_tril_diagonal(platform):
    _, dev = platform
    x = torch.randn(5, 5, dtype=torch.float32, device=dev)
    torch.testing.assert_close(tril(x, diagonal=1), torch.tril(x, diagonal=1), rtol=RTOL, atol=ATOL)


def test_triu_diagonal(platform):
    _, dev = platform
    x = torch.randn(5, 5, dtype=torch.float32, device=dev)
    torch.testing.assert_close(triu(x, diagonal=-1), torch.triu(x, diagonal=-1), rtol=RTOL, atol=ATOL)
