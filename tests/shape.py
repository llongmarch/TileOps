"""Tests for ``tileops.shape`` — cat, stack, split, chunk, permute, transpose, flip, repeat, expand."""

from __future__ import annotations

import pytest
import torch

from tileops import cat, stack, split, chunk, permute, transpose, flip, repeat, expand


def test_cat(platform):
    _, dev = platform
    a = torch.randn(2, 3, device=dev)
    b = torch.randn(2, 3, device=dev)
    torch.testing.assert_close(cat([a, b], dim=0), torch.cat([a, b], dim=0))


def test_stack(platform):
    _, dev = platform
    a = torch.randn(2, 3, device=dev)
    b = torch.randn(2, 3, device=dev)
    torch.testing.assert_close(stack([a, b]), torch.stack([a, b]))


def test_split(platform):
    _, dev = platform
    x = torch.randn(6, 4, device=dev)
    pieces = split(x, 2, dim=0)
    ref = torch.split(x, 2, dim=0)
    for p, r in zip(pieces, ref):
        torch.testing.assert_close(p, r)


def test_chunk(platform):
    _, dev = platform
    x = torch.randn(6, 4, device=dev)
    pieces = chunk(x, 3, dim=0)
    ref = torch.chunk(x, 3, dim=0)
    for p, r in zip(pieces, ref):
        torch.testing.assert_close(p, r)


def test_permute(platform):
    _, dev = platform
    x = torch.randn(2, 3, 4, device=dev)
    torch.testing.assert_close(permute(x, 2, 0, 1), x.permute(2, 0, 1))


def test_transpose(platform):
    _, dev = platform
    x = torch.randn(2, 3, device=dev)
    torch.testing.assert_close(transpose(x, 0, 1), x.transpose(0, 1))


def test_flip(platform):
    _, dev = platform
    x = torch.randn(2, 3, device=dev)
    torch.testing.assert_close(flip(x, [0, 1]), torch.flip(x, [0, 1]))


def test_repeat(platform):
    _, dev = platform
    x = torch.randn(2, 3, device=dev)
    torch.testing.assert_close(repeat(x, 2, 3), x.repeat(2, 3))


def test_expand(platform):
    _, dev = platform
    x = torch.randn(1, 3, device=dev)
    torch.testing.assert_close(expand(x, 4, 3), x.expand(4, 3))
