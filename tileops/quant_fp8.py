"""vLLM ``fp8_utils.py``-compatible dynamic FP8 quant and W8A8 block helpers.

Semantics follow vLLM v0.21.0
``vllm.model_executor.layers.quantization.utils.fp8_utils``:

* Per-group scales are **dequant** scales (multiply to recover float values).
* ``use_ue8m0`` rounds scales up to the next power of two (DeepGEMM layout).
* ``w8a8_block_fp8_matmul`` / ``apply_w8a8_block_fp8_linear`` use a PyTorch
  reference path (dequant + ``matmul``).
"""

from __future__ import annotations

import math
from typing import Optional, Union

import torch

_FP8_DTYPES = tuple(
    d for d in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e4m3fnuz", None))
    if d is not None
)


def is_fp8(x: Union[torch.dtype, torch.Tensor]) -> bool:
    """Return whether *x* is an FP8 dtype (or tensor stored in FP8)."""
    if isinstance(x, torch.Tensor):
        x = x.dtype
    return x in _FP8_DTYPES


def default_fp8_dtype() -> torch.dtype:
    """Default FP8 dtype (``float8_e4m3fn`` when available)."""
    if not _FP8_DTYPES:
        raise RuntimeError("PyTorch build has no float8_e4m3fn support")
    return _FP8_DTYPES[0]


def get_fp8_min_max(dtype: Optional[torch.dtype] = None) -> tuple[float, float]:
    """FP8 clamp bounds (uses vLLM FNUZ override when applicable)."""
    dtype = dtype if dtype is not None else default_fp8_dtype()
    if dtype == getattr(torch, "float8_e4m3fnuz", None):
        return -224.0, 224.0
    finfo = torch.finfo(dtype)
    return finfo.min, finfo.max


def _ue8m0_scale(scale_raw: torch.Tensor) -> torch.Tensor:
    return torch.exp2(torch.ceil(torch.log2(scale_raw.clamp(min=1e-30))))


def input_to_float8(
    x: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tensor-wide dynamic quant → ``(q, dequant_scale)``."""
    dtype = dtype if dtype is not None else default_fp8_dtype()
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype).contiguous(), scale.float().reciprocal()


def _quantize_groups_fp8(
    x_groups: torch.Tensor,
    *,
    eps: float,
    dtype: torch.dtype,
    use_ue8m0: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``(..., group_size)`` groups along the last dimension."""
    fp8_min, fp8_max = get_fp8_min_max(dtype)
    x_f = x_groups.to(torch.float32)
    absmax = x_f.abs().amax(dim=-1).clamp(min=eps)
    scale_raw = absmax / fp8_max
    scale = _ue8m0_scale(scale_raw) if use_ue8m0 else scale_raw
    q = (x_f / scale.unsqueeze(-1)).clamp(min=fp8_min, max=fp8_max).to(dtype)
    return q, scale


def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: Optional[torch.dtype] = None,
    column_major_scales: bool = False,
    tma_aligned_scales: bool = False,
    out_q: Optional[torch.Tensor] = None,
    use_ue8m0: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token-group dynamic FP8 quant along the last dimension."""
    if tma_aligned_scales:
        raise NotImplementedError(
            "per_token_group_quant_fp8: tma_aligned_scales is not implemented "
            "(use vLLM CUDA packed path for DeepGEMM TMA layouts)"
        )
    dtype = dtype if dtype is not None else default_fp8_dtype()
    if not is_fp8(dtype):
        raise TypeError(f"per_token_group_quant_fp8: unsupported dtype {dtype}")
    if x.shape[-1] % group_size != 0:
        raise ValueError(
            f"the last dimension of `x` {x.shape[-1]} must be divisible "
            f"by `group_size` {group_size}"
        )
    if x.stride(-1) != 1:
        raise ValueError("`x` groups must be contiguous along the last dimension")
    if not x.is_contiguous():
        x = x.contiguous()

    lead = x.shape[:-1]
    k = x.shape[-1]
    num_groups = k // group_size
    flat = x.view(-1, group_size)
    q_flat, s_flat = _quantize_groups_fp8(
        flat, eps=eps, dtype=dtype, use_ue8m0=use_ue8m0,
    )
    x_q = q_flat.view(*lead, k)
    if out_q is not None:
        if out_q.shape != x.shape or out_q.dtype != dtype:
            raise ValueError(
                f"per_token_group_quant_fp8: out_q must be {dtype} with shape "
                f"{tuple(x.shape)}"
            )
        out_q.copy_(x_q)
        x_q = out_q

    if column_major_scales:
        if x.dim() < 2:
            raise ValueError(
                "per_token_group_quant_fp8: column_major_scales requires ndim >= 2"
            )
        m = x.shape[-2]
        shape = x.shape[:-2] + (num_groups, m)
        x_s = s_flat.reshape(shape).transpose(-2, -1).contiguous()
    else:
        x_s = s_flat.view(*lead, num_groups)

    return x_q, x_s


def block_dequant(
    x_q_block: torch.Tensor,
    x_s: torch.Tensor,
    block_size: list[int],
) -> torch.Tensor:
    """Block-wise dequantization of a 2-D FP8 weight matrix."""
    if not is_fp8(x_q_block):
        raise TypeError(f"block_dequant: expected FP8 weights, got {x_q_block.dtype}")
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


def w8a8_block_fp8_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    a_s: torch.Tensor,
    b_s: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """W8A8 FP8 block matmul reference: ``C = dequant(A) @ dequant(B).T``."""
    if len(block_size) != 2:
        raise ValueError("block_size must have length 2")
    block_n, block_k = block_size[0], block_size[1]

    if a.shape[-1] != b.shape[-1]:
        raise ValueError(
            f"w8a8_block_fp8_matmul: K mismatch {a.shape[-1]} vs {b.shape[-1]}"
        )
    if not is_fp8(a) or not is_fp8(b):
        raise TypeError("w8a8_block_fp8_matmul: A and B must be FP8")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("w8a8_block_fp8_matmul: A and B must be contiguous")
    if b.ndim != 2 or b_s.ndim != 2:
        raise ValueError("w8a8_block_fp8_matmul: B and Bs must be 2-D")

    k = a.shape[-1]
    m = a.numel() // k
    n, k_b = b.shape
    if k != k_b:
        raise ValueError(f"inner dim mismatch: A K={k}, B K={k_b}")

    n_tiles = math.ceil(n / block_n)
    k_tiles = math.ceil(k / block_k)
    if a_s.shape[:-1] != a.shape[:-1] or a_s.shape[-1] != k_tiles:
        raise ValueError(
            f"w8a8_block_fp8_matmul: As shape {tuple(a_s.shape)} incompatible "
            f"with A {tuple(a.shape)} and block_k={block_k}"
        )
    if b_s.shape != (n_tiles, k_tiles):
        raise ValueError(
            f"w8a8_block_fp8_matmul: Bs shape {tuple(b_s.shape)} expected "
            f"({n_tiles}, {k_tiles})"
        )

    a_2d = a.reshape(m, k)
    a_s_2d = a_s.reshape(m, k_tiles)
    a_f = _dequant_per_token_group_lastdim(a_2d, a_s_2d, block_k)
    b_f = block_dequant(b, b_s, block_size)
    c = a_f @ b_f.T
    return c.to(output_dtype).reshape(*a.shape[:-1], n)


def w8a8_triton_block_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_s: torch.Tensor,
    b_s: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """vLLM name alias for :func:`w8a8_block_fp8_matmul`."""
    return w8a8_block_fp8_matmul(
        a, b, a_s, b_s, block_size, output_dtype=output_dtype,
    )


def apply_w8a8_block_fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: list[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Quantized linear: group-quant input + block W8A8 FP8 matmul (+ optional bias)."""
    if input_scale is not None:
        raise ValueError("apply_w8a8_block_fp8_linear: input_scale must be None")

    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[0]]
    q_input, x_scale = per_token_group_quant_fp8(input_2d, block_size[1])
    output = w8a8_block_fp8_matmul(
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
    "apply_w8a8_block_fp8_linear",
    "block_dequant",
    "default_fp8_dtype",
    "get_fp8_min_max",
    "input_to_float8",
    "is_fp8",
    "per_token_group_quant_fp8",
    "w8a8_block_fp8_matmul",
    "w8a8_triton_block_scaled_mm",
]
