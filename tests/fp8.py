"""Tests for vLLM-compatible ``tileops.quant_fp8``."""

from __future__ import annotations

import math

import pytest
import torch

import tileops.quant_fp8 as fp8_mod

pytestmark = pytest.mark.skipif(
    not fp8_mod._FP8_DTYPES,
    reason="PyTorch build has no float8 support",
)

FP8 = fp8_mod.default_fp8_dtype()


@pytest.fixture
def fp8_dev():
    """FP8 ops run on CPU (MPS/CUDA float8 support varies by PyTorch build)."""
    return torch.device("cpu")


def _ref_input_to_float8(x, dtype=FP8):
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype).contiguous(), scale.float().reciprocal()


def _ref_per_token_group_quant_fp8(x, group_size, eps=1e-10, use_ue8m0=False):
    fp8_min, fp8_max = fp8_mod.get_fp8_min_max(FP8)
    flat = x.contiguous().view(-1, group_size).float()
    absmax = flat.abs().amax(dim=-1).clamp(min=eps)
    scale_raw = absmax / fp8_max
    if use_ue8m0:
        scale = torch.exp2(torch.ceil(torch.log2(scale_raw.clamp(min=1e-30))))
    else:
        scale = scale_raw
    q = (flat / scale.unsqueeze(-1)).clamp(min=fp8_min, max=fp8_max).to(FP8)
    lead = x.shape[:-1]
    k = x.shape[-1]
    return q.view(*lead, k), scale.view(*lead, k // group_size)


def test_is_fp8():
    assert fp8_mod.is_fp8(FP8)
    assert fp8_mod.is_fp8(torch.zeros(1, dtype=FP8))


def test_input_to_float8(fp8_dev):
    x = torch.tensor([-1.5, 0.0, 0.5, 2.0], dtype=torch.float32, device=fp8_dev)
    q, s = fp8_mod.input_to_float8(x, dtype=FP8)
    rq, rs = _ref_input_to_float8(x)
    assert fp8_mod.is_fp8(q)
    torch.testing.assert_close(q.float(), rq.float(), rtol=0, atol=0)
    torch.testing.assert_close(s, rs)


def test_per_token_group_quant_fp8(fp8_dev):
    x = torch.randn(3, 32, dtype=torch.float32, device=fp8_dev)
    gs = 8
    q, s = fp8_mod.per_token_group_quant_fp8(x, gs)
    rq, rs = _ref_per_token_group_quant_fp8(x, gs)
    assert q.shape == x.shape
    assert s.shape == (3, 4)
    torch.testing.assert_close(q.float(), rq.float(), rtol=0, atol=0)
    torch.testing.assert_close(s, rs, rtol=1e-5, atol=1e-5)


def test_per_token_group_quant_fp8_ue8m0(fp8_dev):
    x = torch.randn(2, 16, dtype=torch.float32, device=fp8_dev)
    q, s = fp8_mod.per_token_group_quant_fp8(x, 8, use_ue8m0=True)
    rq, rs = _ref_per_token_group_quant_fp8(x, 8, use_ue8m0=True)
    torch.testing.assert_close(s, rs, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(q.float(), rq.float(), rtol=0, atol=0)


def test_block_dequant_fp8(fp8_dev):
    block_size = [4, 8]
    block_n, block_k = block_size
    n, k = 10, 24
    w_q = torch.randn(n, k, dtype=torch.float32, device=fp8_dev).clamp(-1, 1).to(FP8)
    n_tiles = math.ceil(n / block_n)
    k_tiles = math.ceil(k / block_k)
    w_s = torch.rand(n_tiles, k_tiles, dtype=torch.float32, device=fp8_dev) * 0.1 + 0.01
    y = fp8_mod.block_dequant(w_q, w_s, block_size)
    ref = w_q.to(torch.float32)
    for i in range(k_tiles):
        for j in range(n_tiles):
            ref[
                j * block_n : min((j + 1) * block_n, n),
                i * block_k : min((i + 1) * block_k, k),
            ] *= w_s[j, i]
    torch.testing.assert_close(y, ref)


def test_w8a8_block_fp8_matmul(fp8_dev):
    block_size = [4, 8]
    block_n, block_k = block_size
    m, n, k = 6, 10, 24
    a = torch.randn(m, k, dtype=torch.float32, device=fp8_dev)
    b = torch.randn(n, k, dtype=torch.float32, device=fp8_dev)
    q_a, s_a = fp8_mod.per_token_group_quant_fp8(a, block_k)
    q_b = b.clamp(-1, 1).to(FP8)
    n_tiles = math.ceil(n / block_n)
    k_tiles = math.ceil(k / block_k)
    s_b = torch.empty(n_tiles, k_tiles, dtype=torch.float32, device=fp8_dev)
    for j in range(n_tiles):
        for i in range(k_tiles):
            blk = q_b[
                j * block_n : min((j + 1) * block_n, n),
                i * block_k : min((i + 1) * block_k, k),
            ].float()
            s_b[j, i] = blk.abs().max().clamp(min=1e-10) / fp8_mod.get_fp8_min_max(FP8)[1]

    out = fp8_mod.w8a8_block_fp8_matmul(
        q_a, q_b, s_a, s_b, block_size, output_dtype=torch.float32,
    )
    alias = fp8_mod.w8a8_triton_block_scaled_mm(
        q_a, q_b, s_a, s_b, block_size, output_dtype=torch.float32,
    )
    torch.testing.assert_close(out, alias)
    a_f = fp8_mod._dequant_per_token_group_lastdim(q_a, s_a, block_k)
    b_f = fp8_mod.block_dequant(q_b, s_b, block_size)
    ref = (a_f @ b_f.T).to(torch.float32)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_apply_w8a8_block_fp8_linear(fp8_dev):
    block_size = [4, 8]
    batch, k, n = 3, 24, 10
    x = torch.randn(batch, k, dtype=torch.float16, device=fp8_dev)
    w = torch.randn(n, k, dtype=torch.float32, device=fp8_dev)
    w_q = w.clamp(-1, 1).to(FP8)
    bn, bk = block_size
    s_w = torch.empty(
        math.ceil(n / bn), math.ceil(k / bk), dtype=torch.float32, device=fp8_dev,
    )
    fp8_max = fp8_mod.get_fp8_min_max(FP8)[1]
    for j in range(s_w.shape[0]):
        for i in range(s_w.shape[1]):
            blk = w_q[
                j * bn : min((j + 1) * bn, n),
                i * bk : min((i + 1) * bk, k),
            ].float()
            s_w[j, i] = blk.abs().max().clamp(min=1e-10) / fp8_max
    bias = torch.randn(n, dtype=torch.float16, device=fp8_dev)
    y = fp8_mod.apply_w8a8_block_fp8_linear(x, w_q, block_size, s_w, bias=bias)
    q_x, s_x = fp8_mod.per_token_group_quant_fp8(x.view(-1, k), block_size[1])
    ref = fp8_mod.w8a8_block_fp8_matmul(
        q_x, w_q, s_x, s_w, block_size, output_dtype=torch.float16,
    ) + bias
    torch.testing.assert_close(y, ref.to(torch.float16).view(batch, n), rtol=1e-3, atol=1e-3)
