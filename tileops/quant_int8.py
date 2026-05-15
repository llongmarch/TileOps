"""vLLM ``int8_utils.py``-compatible dynamic quant and W8A8 block helpers.

Semantics follow vLLM v0.21.0
``vllm.model_executor.layers.quantization.utils.int8_utils``:

* Dynamic quant returns **dequant** scales ``absmax / 127`` (multiply to recover).
* ``per_token_group_quant_int8`` uses one scale per contiguous group on the last dim.
* ``w8a8_block_int8_matmul`` / ``apply_w8a8_block_int8_linear`` use a PyTorch
  reference path (dequant + ``matmul``); a fused TileLang kernel can replace this later.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from tileops._infra import dispatch_compile_quant
from tileops.runtime import (
    default_execution_backend,
    default_tilelang_target,
    default_torch_device,
    invoke_per_token_quant_int8_kernel,
    torch_to_tl_dtype,
)


def input_to_int8(
    x: torch.Tensor,
    dtype: torch.dtype = torch.int8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tensor-wide dynamic quant → ``(q, dequant_scale)``."""
    iinfo = torch.iinfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    int8_min, int8_max = iinfo.min, iinfo.max
    scale = int8_max / amax
    x_scl_sat = (x * scale).clamp(min=int8_min, max=int8_max)
    return x_scl_sat.to(dtype).contiguous(), scale.float().reciprocal()


def _ref_per_token_quant_int8(
    x: torch.Tensor,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_f = x.to(torch.float32)
    absmax = x_f.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    scale = absmax / 127.0
    q = torch.clamp(torch.round(x_f / scale), -128, 127).to(torch.int8)
    return q, scale


def per_token_quant_int8(
    x: torch.Tensor,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row dynamic quant on the last dimension; scales have trailing size 1."""
    original_shape = x.shape
    if x.dim() == 1:
        x = x.unsqueeze(0)
        squeezed = True
    elif x.dim() > 2:
        x = x.view(-1, original_shape[-1])
        squeezed = False
    else:
        squeezed = False

    m = x.shape[0]
    n = x.shape[-1]
    x = x.contiguous()

    tgt = default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = default_execution_backend(tgt, dev)
    try:
        kernel = dispatch_compile_quant(
            op_name="per_token_quant_int8",
            target=tgt,
            shape=x.shape,
            block_size=1,
            threads=1,
            dtype=torch_to_tl_dtype(x.dtype),
            execution_backend=eb,
            rows=m,
            cols=n,
            eps=eps,
        )
        work = torch.empty((m, n), dtype=torch.float32, device=x.device)
        scales = torch.empty((m,), dtype=torch.float32, device=x.device)
        invoke_per_token_quant_int8_kernel(
            kernel, x, out_work=work, out_scale=scales,
            tilelang_target=tgt, execution_backend=eb,
        )
        x_q = work.to(torch.int8)
        scales = scales.unsqueeze(-1)
    except (ImportError, NotImplementedError, ValueError):
        x_q, scales = _ref_per_token_quant_int8(x, eps=eps)

    if squeezed:
        x_q = x_q.squeeze(0)
        scales = scales.squeeze(0)
    else:
        x_q = x_q.view(*original_shape)
        scales = scales.view(*original_shape[:-1], 1)
    return x_q, scales


def per_token_group_quant_int8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype = torch.int8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token-group dynamic quant along the last dimension."""
    if dtype != torch.int8:
        raise TypeError(
            "per_token_group_quant_int8: only torch.int8 output is supported"
        )
    if x.shape[-1] % group_size != 0:
        raise ValueError(
            "the last dimension of `x` cannot be divisible by `group_size`"
        )
    if not x.is_contiguous():
        x = x.contiguous()

    lead = x.shape[:-1]
    k = x.shape[-1]
    flat = x.view(-1, group_size)
    q_flat, s_flat = per_token_quant_int8(flat, eps=eps)
    x_q = q_flat.view(*lead, k).to(dtype)
    x_s = s_flat.view(*lead, k // group_size)
    return x_q, x_s


def block_dequant(
    x_q_block: torch.Tensor,
    x_s: torch.Tensor,
    block_size: list[int],
) -> torch.Tensor:
    """Block-wise dequantization of a 2-D ``int8`` weight matrix."""
    block_n, block_k = block_size[0], block_size[1]
    n, k = x_q_block.shape
    n_tiles = (n + block_n - 1) // block_n
    k_tiles = (k + block_k - 1) // block_k
    if n_tiles != x_s.shape[0] or k_tiles != x_s.shape[1]:
        raise ValueError(
            f"block_dequant: scale shape {tuple(x_s.shape)} does not match "
            f"block grid ({n_tiles}, {k_tiles}) for {tuple(x_q_block.shape)}"
        )

    x_dq = x_q_block.to(torch.float32)
    for i in range(k_tiles):
        for j in range(n_tiles):
            x_dq[
                j * block_n : min((j + 1) * block_n, n),
                i * block_k : min((i + 1) * block_k, k),
            ] *= x_s[j, i]
    return x_dq


def _dequant_per_token_group_lastdim(
    a_q: torch.Tensor,
    a_s: torch.Tensor,
    group_k: int,
) -> torch.Tensor:
    """Dequantize activation groups along the last dimension."""
    k = a_q.shape[-1]
    g = k // group_k
    out = a_q.to(torch.float32)
    for gi in range(g):
        scale = a_s[..., gi]
        while scale.ndim < out.ndim:
            scale = scale.unsqueeze(-1)
        sl = slice(gi * group_k, (gi + 1) * group_k)
        out[..., sl] = out[..., sl] * scale
    return out


def w8a8_block_int8_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    a_s: torch.Tensor,
    b_s: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """W8A8 block matmul reference: ``C = dequant(A) @ dequant(B).T``."""
    if len(block_size) != 2:
        raise ValueError("block_size must have length 2")
    block_n, block_k = block_size[0], block_size[1]

    if a.shape[-1] != b.shape[-1]:
        raise ValueError(
            f"w8a8_block_int8_matmul: K mismatch {a.shape[-1]} vs {b.shape[-1]}"
        )
    if a.dtype != torch.int8 or b.dtype != torch.int8:
        raise TypeError("w8a8_block_int8_matmul: A and B must be int8")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("w8a8_block_int8_matmul: A and B must be contiguous")
    if b.ndim != 2 or b_s.ndim != 2:
        raise ValueError("w8a8_block_int8_matmul: B and Bs must be 2-D")

    k = a.shape[-1]
    m = a.numel() // k
    n, k_b = b.shape
    if k != k_b:
        raise ValueError(f"inner dim mismatch: A K={k}, B K={k_b}")

    n_tiles = math.ceil(n / block_n)
    k_tiles = math.ceil(k / block_k)
    if a_s.shape[:-1] != a.shape[:-1] or a_s.shape[-1] != k_tiles:
        raise ValueError(
            f"w8a8_block_int8_matmul: As shape {tuple(a_s.shape)} incompatible "
            f"with A {tuple(a.shape)} and block_k={block_k}"
        )
    if b_s.shape != (n_tiles, k_tiles):
        raise ValueError(
            f"w8a8_block_int8_matmul: Bs shape {tuple(b_s.shape)} expected "
            f"({n_tiles}, {k_tiles})"
        )

    a_2d = a.reshape(m, k)
    a_s_2d = a_s.reshape(m, k_tiles)
    a_f = _dequant_per_token_group_lastdim(a_2d, a_s_2d, block_k)
    b_f = block_dequant(b, b_s, block_size)
    c = a_f @ b_f.T
    return c.to(output_dtype).reshape(*a.shape[:-1], n)


def apply_w8a8_block_int8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: list[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Quantized linear: group-quant input + block W8A8 matmul (+ optional bias)."""
    if input_scale is not None:
        raise ValueError("apply_w8a8_block_int8_linear: input_scale must be None")

    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[0]]
    q_input, x_scale = per_token_group_quant_int8(input_2d, block_size[1])
    output = w8a8_block_int8_matmul(
        q_input,
        weight,
        x_scale,
        weight_scale,
        block_size,
        output_dtype=input.dtype,
    )
    if bias is not None:
        output = output + bias
    return output.to(dtype=input.dtype).view(*output_shape)


__all__ = [
    "apply_w8a8_block_int8_linear",
    "block_dequant",
    "input_to_int8",
    "per_token_group_quant_int8",
    "per_token_quant_int8",
    "w8a8_block_int8_matmul",
]
