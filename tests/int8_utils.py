"""Tests for vLLM-compatible ``tileops.quant_int8``."""

from __future__ import annotations

import math

import torch

from tileops import (
    apply_w8a8_block_int8_linear,
    block_dequant,
    input_to_int8,
    per_token_group_quant_int8,
    per_token_quant_int8,
    w8a8_block_int8_matmul,
)


def _ref_input_to_int8(x, dtype=torch.int8):
    iinfo = torch.iinfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = iinfo.max / amax
    x_scl_sat = (x * scale).clamp(min=iinfo.min, max=iinfo.max)
    return x_scl_sat.to(dtype).contiguous(), scale.float().reciprocal()


def _ref_per_token_quant_int8(x, eps=1e-10):
    x_f = x.to(torch.float32)
    if x_f.dim() == 1:
        x_f = x_f.unsqueeze(0)
        squeeze = True
    elif x_f.dim() > 2:
        orig = x.shape
        x_f = x_f.view(-1, orig[-1])
        squeeze = False
    else:
        squeeze = False
    absmax = x_f.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    scale = absmax / 127.0
    q = torch.clamp(torch.round(x_f / scale), -128, 127).to(torch.int8)
    if squeeze:
        return q.squeeze(0), scale.squeeze(-1).unsqueeze(-1)
    if x.dim() > 2:
        return q.view(*orig), scale.view(*orig[:-1], 1)
    return q, scale


def _ref_per_token_group_quant_int8(x, group_size, eps=1e-10):
    lead = x.shape[:-1]
    k = x.shape[-1]
    flat = x.contiguous().view(-1, group_size)
    q_flat, s_flat = _ref_per_token_quant_int8(flat, eps=eps)
    return q_flat.view(*lead, k), s_flat.view(*lead, k // group_size)


def _ref_block_dequant(x_q, x_s, block_size):
    block_n, block_k = block_size
    n, k = x_q.shape
    n_tiles = (n + block_n - 1) // block_n
    k_tiles = (k + block_k - 1) // block_k
    x_dq = x_q.to(torch.float32)
    for i in range(k_tiles):
        for j in range(n_tiles):
            x_dq[
                j * block_n : min((j + 1) * block_n, n),
                i * block_k : min((i + 1) * block_k, k),
            ] *= x_s[j, i]
    return x_dq


def test_input_to_int8(platform):
    _, dev = platform
    x = torch.tensor([-1.5, 0.0, 0.5, 2.0], dtype=torch.float32, device=dev)
    q, s = input_to_int8(x)
    rq, rs = _ref_input_to_int8(x)
    assert q.dtype == torch.int8
    torch.testing.assert_close(q, rq)
    torch.testing.assert_close(s, rs)


def test_per_token_quant_int8_2d(platform):
    _, dev = platform
    x = torch.randn(4, 16, dtype=torch.float32, device=dev) * 0.5
    q, s = per_token_quant_int8(x)
    rq, rs = _ref_per_token_quant_int8(x)
    assert q.dtype == torch.int8
    assert s.shape == (4, 1)
    torch.testing.assert_close(q, rq)
    torch.testing.assert_close(s, rs)


def test_per_token_quant_int8_nd(platform):
    _, dev = platform
    x = torch.randn(2, 3, 8, dtype=torch.float16, device=dev)
    q, s = per_token_quant_int8(x)
    rq, rs = _ref_per_token_quant_int8(x)
    torch.testing.assert_close(q, rq)
    torch.testing.assert_close(s, rs)


def test_per_token_group_quant_int8(platform):
    _, dev = platform
    x = torch.randn(3, 32, dtype=torch.float32, device=dev)
    gs = 8
    q, s = per_token_group_quant_int8(x, gs)
    rq, rs = _ref_per_token_group_quant_int8(x, gs)
    assert q.shape == x.shape
    assert s.shape == (3, 4)
    torch.testing.assert_close(q, rq)
    torch.testing.assert_close(s, rs)


def test_block_dequant(platform):
    _, dev = platform
    block_size = [4, 8]
    n, k = 10, 24
    w_q = torch.randint(-128, 127, (n, k), dtype=torch.int8, device=dev)
    n_tiles = math.ceil(n / block_size[0])
    k_tiles = math.ceil(k / block_size[1])
    w_s = torch.rand(n_tiles, k_tiles, dtype=torch.float32, device=dev) * 0.1 + 0.01
    y = block_dequant(w_q, w_s, block_size)
    ref = _ref_block_dequant(w_q, w_s, block_size)
    torch.testing.assert_close(y, ref)


def test_w8a8_block_int8_matmul(platform):
    _, dev = platform
    block_size = [4, 8]
    block_n, block_k = block_size
    m, n, k = 6, 10, 24
    a = torch.randn(m, k, dtype=torch.float32, device=dev)
    b = torch.randn(n, k, dtype=torch.float32, device=dev)
    q_a, s_a = per_token_group_quant_int8(a, block_k)
    q_b = torch.clamp(torch.round(b / 0.05), -128, 127).to(torch.int8)
    n_tiles = math.ceil(n / block_n)
    k_tiles = math.ceil(k / block_k)
    s_b = torch.rand(n_tiles, k_tiles, dtype=torch.float32, device=dev) * 0.05 + 0.01
    # overwrite block scales from actual block absmax for a sane reference
    for j in range(n_tiles):
        for i in range(k_tiles):
            blk = q_b[
                j * block_n : min((j + 1) * block_n, n),
                i * block_k : min((i + 1) * block_k, k),
            ].float()
            s_b[j, i] = blk.abs().max().clamp(min=1e-10) / 127.0

    from tileops.quant_int8 import _dequant_per_token_group_lastdim

    out = w8a8_block_int8_matmul(q_a, q_b, s_a, s_b, block_size, output_dtype=torch.float32)
    a_f = _dequant_per_token_group_lastdim(q_a, s_a, block_k)
    b_f = block_dequant(q_b, s_b, block_size)
    ref = (a_f @ b_f.T).to(torch.float32)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_apply_w8a8_block_int8_linear(platform):
    _, dev = platform
    block_size = [4, 8]
    batch, k, n = 3, 24, 10
    x = torch.randn(batch, k, dtype=torch.float16, device=dev)
    w = torch.randn(n, k, dtype=torch.float32, device=dev)
    w_q = torch.clamp(torch.round(w / 0.04), -128, 127).to(torch.int8)
    bn, bk = block_size
    s_w = torch.empty(
        math.ceil(n / bn), math.ceil(k / bk), dtype=torch.float32, device=dev,
    )
    for j in range(s_w.shape[0]):
        for i in range(s_w.shape[1]):
            blk = w_q[
                j * bn : min((j + 1) * bn, n),
                i * bk : min((i + 1) * bk, k),
            ].float()
            s_w[j, i] = blk.abs().max().clamp(min=1e-10) / 127.0
    bias = torch.randn(n, dtype=torch.float16, device=dev)
    y = apply_w8a8_block_int8_linear(x, w_q, block_size, s_w, bias=bias)
    q_x, s_x = per_token_group_quant_int8(x.view(-1, k), block_size[1])
    ref = w8a8_block_int8_matmul(
        q_x, w_q, s_x, s_w, block_size, output_dtype=torch.float16,
    ) + bias
    torch.testing.assert_close(y, ref.to(torch.float16).view(batch, n), rtol=1e-3, atol=1e-3)
