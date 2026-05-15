"""Tests for ``tileops.quant`` (per-tensor / per-channel quantize & dequantize)."""

from __future__ import annotations

import pytest
import torch

from tileops import (
    dequantize_per_channel,
    dequantize_per_tensor,
    quantize_per_channel,
    quantize_per_tensor,
)

RTOL, ATOL = 0.0, 0.0  # integer quant should match exactly


def _ref_quantize_per_tensor(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    s = scale.reshape(())
    return torch.clamp(torch.round(x / s), -128, 127).to(torch.int8)


def _ref_dequantize_per_tensor(
    x_q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype,
) -> torch.Tensor:
    return x_q.to(dtype) * scale.reshape(())


def _ref_quantize_per_channel(
    x: torch.Tensor, scale: torch.Tensor, axis: int,
) -> torch.Tensor:
    dim = axis if axis >= 0 else axis + x.ndim
    shape = [1] * x.ndim
    shape[dim] = -1
    return torch.clamp(torch.round(x / scale.view(*shape)), -128, 127).to(torch.int8)


def _ref_dequantize_per_channel(
    x_q: torch.Tensor, scale: torch.Tensor, axis: int, dtype: torch.dtype,
) -> torch.Tensor:
    dim = axis if axis >= 0 else axis + x_q.ndim
    shape = [1] * x_q.ndim
    shape[dim] = -1
    return x_q.to(dtype) * scale.view(*shape)


def test_quantize_per_tensor(platform):
    _, dev = platform
    x = torch.tensor([-0.25, 0.0, 0.19, 1.05], dtype=torch.float32, device=dev)
    scale = torch.tensor([0.1], dtype=torch.float32, device=dev)
    q = quantize_per_tensor(x, scale)
    ref = _ref_quantize_per_tensor(x, scale)
    assert q.dtype == torch.int8
    torch.testing.assert_close(q, ref)


def test_dequantize_per_tensor(platform):
    _, dev = platform
    x_q = torch.tensor([-3, 0, 2, 10], dtype=torch.int8, device=dev)
    scale = torch.tensor([0.05], dtype=torch.float32, device=dev)
    y = dequantize_per_tensor(x_q, scale)
    ref = _ref_dequantize_per_tensor(x_q, scale, torch.float32)
    torch.testing.assert_close(y, ref, rtol=RTOL, atol=ATOL)


def test_quant_dequant_roundtrip_per_tensor(platform):
    _, dev = platform
    x = torch.randn(32, 64, dtype=torch.float32, device=dev) * 0.3
    scale = torch.tensor([x.abs().max().item() / 127.0 + 1e-6], device=dev)
    q = quantize_per_tensor(x, scale)
    xr = dequantize_per_tensor(q, scale)
    ref_q = _ref_quantize_per_tensor(x, scale)
    ref_xr = _ref_dequantize_per_tensor(ref_q, scale, torch.float32)
    torch.testing.assert_close(q, ref_q)
    torch.testing.assert_close(xr, ref_xr, rtol=1e-5, atol=1e-5)


def test_quantize_per_channel_last_dim(platform):
    _, dev = platform
    x = torch.randn(4, 8, dtype=torch.float32, device=dev)
    scale = torch.linspace(0.05, 0.2, 8, device=dev)
    q = quantize_per_channel(x, scale, dim=-1)
    ref = _ref_quantize_per_channel(x, scale, -1)
    torch.testing.assert_close(q, ref)


def test_quantize_per_channel_inner_dim(platform):
    _, dev = platform
    x = torch.randn(3, 5, 7, dtype=torch.float32, device=dev)
    scale = torch.linspace(0.1, 0.3, 5, device=dev)
    q = quantize_per_channel(x, scale, dim=1)
    ref = _ref_quantize_per_channel(x, scale, 1)
    torch.testing.assert_close(q, ref)


def test_dequantize_per_channel(platform):
    _, dev = platform
    scale = torch.tensor([0.1, 0.2, 0.15], dtype=torch.float32, device=dev)
    x = torch.randn(2, 3, dtype=torch.float32, device=dev)
    q = _ref_quantize_per_channel(x, scale, -1)
    y = dequantize_per_channel(q, scale, dim=-1)
    ref = _ref_dequantize_per_channel(q, scale, -1, torch.float32)
    torch.testing.assert_close(y, ref, rtol=1e-5, atol=1e-5)


def test_quantize_per_tensor_out(platform):
    _, dev = platform
    x = torch.randn(10, device=dev)
    scale = torch.tensor([0.2], device=dev)
    out = torch.empty_like(x, dtype=torch.int8)
    r = quantize_per_tensor(x, scale, out=out)
    assert r is out
    torch.testing.assert_close(out, _ref_quantize_per_tensor(x, scale))


def test_scale_must_be_positive(platform):
    _, dev = platform
    x = torch.randn(4, device=dev)
    bad = torch.tensor([0.0], device=dev)
    with pytest.raises(ValueError):
        quantize_per_tensor(x, bad)
