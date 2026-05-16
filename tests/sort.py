"""Tests for ``tileops.sort`` — sort, argsort."""

from __future__ import annotations

import pytest
import torch

from tileops import sort, argsort

RTOL, ATOL = 1e-4, 1e-4


def test_sort_1d(platform):
    _, dev = platform
    x = torch.randn(10, dtype=torch.float32, device=dev)
    torch.testing.assert_close(sort(x), torch.sort(x).values, rtol=RTOL, atol=ATOL)


def test_sort_2d_dim1(platform):
    _, dev = platform
    x = torch.randn(3, 5, dtype=torch.float32, device=dev)
    torch.testing.assert_close(sort(x, dim=1), torch.sort(x, dim=1).values, rtol=RTOL, atol=ATOL)


def test_sort_descending(platform):
    _, dev = platform
    x = torch.randn(3, 5, dtype=torch.float32, device=dev)
    torch.testing.assert_close(sort(x, descending=True), torch.sort(x, descending=True).values, rtol=RTOL, atol=ATOL)


def test_argsort_1d(platform):
    _, dev = platform
    x = torch.randn(10, dtype=torch.float32, device=dev)
    torch.testing.assert_close(argsort(x), torch.argsort(x), rtol=RTOL, atol=ATOL)


def test_argsort_2d(platform):
    _, dev = platform
    x = torch.randn(3, 5, dtype=torch.float32, device=dev)
    torch.testing.assert_close(argsort(x, dim=1), torch.argsort(x, dim=1), rtol=RTOL, atol=ATOL)
