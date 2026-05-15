"""Tests for vLLM-compatible ``tileops.quant_awq``."""

from __future__ import annotations

import torch

import tileops.quant_awq as awq_mod


def _ref_dequant(qweight, scales, zeros):
    w = awq_mod.unpack_awq_int4(qweight).to(scales.dtype)
    z = awq_mod.unpack_awq_int4(zeros).to(scales.dtype)
    k = qweight.shape[0]
    g = scales.shape[0]
    gs = k // g
    gidx = torch.arange(k, device=qweight.device) // gs
    return (w - z[gidx]) * scales[gidx]


def _make_awq_weights(k, m, group_size, dev, dtype=torch.float16):
    gs = group_size
    g = k // gs
    packed_m = m // 8
    w_int = torch.randint(0, 16, (k, m), device=dev, dtype=torch.int64)
    z_int = torch.randint(0, 16, (g, m), device=dev, dtype=torch.int64)
    s = torch.rand(g, m, device=dev, dtype=dtype) * 0.1 + 0.01
    qweight = awq_mod.pack_awq_int4(w_int)
    zeros = awq_mod.pack_awq_int4(z_int.view(g, m))
    return qweight, s, zeros, w_int, z_int


def test_unpack_pack_roundtrip():
    x = torch.randint(0, 16, (4, 16), dtype=torch.int64)
    packed = awq_mod.pack_awq_int4(x)
    back = awq_mod.unpack_awq_int4(packed)
    torch.testing.assert_close(back, x)


def test_awq_dequantize_triton(platform):
    _, dev = platform
    k, m, gs = 64, 128, 32
    qweight, scales, zeros, w_int, z_int = _make_awq_weights(k, m, gs, dev)
    out = awq_mod.awq_dequantize_triton(qweight, scales, zeros)
    ref = _ref_dequant(qweight, scales, zeros)
    assert out.shape == (k, m)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)

    # full-K group
    q2, s2, z2, _, _ = _make_awq_weights(32, 64, 32, dev)
    out2 = awq_mod.awq_dequantize_triton(q2, s2, z2)
    torch.testing.assert_close(out2, _ref_dequant(q2, s2, z2), rtol=1e-3, atol=1e-3)


def test_awq_gemm_triton(platform):
    _, dev = platform
    m, k, n, gs = 8, 64, 128, 32
    qweight, scales, zeros, _, _ = _make_awq_weights(k, n, gs, dev)
    x = torch.randn(m, k, device=dev, dtype=torch.float16)
    y = awq_mod.awq_gemm_triton(x, qweight, scales, zeros, split_k_iters=1)
    w = awq_mod.awq_dequantize_triton(qweight, scales, zeros)
    ref = x @ w
    assert y.shape == (m, n)
    torch.testing.assert_close(y, ref, rtol=1e-3, atol=1e-3)


def test_awq_gemm_split_k(platform):
    _, dev = platform
    m, k, n, gs = 4, 32, 64, 32
    qweight, scales, zeros, _, _ = _make_awq_weights(k, n, gs, dev)
    x = torch.randn(m, k, device=dev, dtype=torch.float16)
    y1 = awq_mod.awq_gemm_triton(x, qweight, scales, zeros, split_k_iters=1)
    y4 = awq_mod.awq_gemm_triton(x, qweight, scales, zeros, split_k_iters=4)
    torch.testing.assert_close(y1, y4, rtol=1e-5, atol=1e-5)
