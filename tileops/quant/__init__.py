"""Symmetric INT8 quantize / dequantize (per-tensor and per-channel).

Mapping (no zero-point, ``scale > 0``)::

    q = clamp(round(x / scale), -128, 127)
    x_hat = float(q) * scale

* :func:`quantize_per_tensor` / :func:`dequantize_per_tensor` — one scalar
  *scale* for the whole tensor.
* :func:`quantize_per_channel` / :func:`dequantize_per_channel` — one scale
  per index along *dim* (channel axis merged last, like reductions).

Quantize kernels write clamped float32 intermediates; the facade casts to
``int8`` (avoids unreliable direct ``int8`` stores on some Metal paths).

Input activations: ``float32``, ``float16``, ``bfloat16``.  *scale* is always
``float32``.  Quantized values are ``torch.int8``.
"""

from __future__ import annotations

from typing import Optional

import torch

from tileops._infra import dispatch_compile_quant
from tileops._shape import merge_on_dim, unmerge
from tileops.runtime import (
    default_execution_backend,
    default_tilelang_target,
    default_torch_device,
    invoke_quant_kernel,
    suggest_tile_config,
    torch_to_tl_dtype,
)

def _validate_scale_tensor(
    scale: torch.Tensor,
    *,
    op: str,
    numel: int = 1,
) -> torch.Tensor:
    if scale.dtype != torch.float32:
        raise TypeError(f"{op}: scale must be float32 (got {scale.dtype})")
    if scale.numel() != numel:
        raise ValueError(
            f"{op}: scale must have {numel} element(s), got {scale.numel()}"
        )
    if torch.any(scale <= 0):
        raise ValueError(f"{op}: scale values must be positive")
    return scale.contiguous().reshape(numel)


def quantize_per_tensor(
    x: torch.Tensor,
    scale: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Quantize *x* with a single scalar *scale* → ``int8``."""
    op = "quantize_per_tensor"
    if not x.dtype.is_floating_point:
        raise TypeError(f"{op}: input must be floating-point (got {x.dtype})")
    sf = _validate_scale_tensor(scale, op=op, numel=1)
    if sf.device != x.device:
        raise TypeError(f"{op}: scale device must match input")

    tgt = default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = default_execution_backend(tgt, dev)
    bs, threads = suggest_tile_config(x.numel())
    kernel = dispatch_compile_quant(
        op_name=op,
        target=tgt,
        shape=x.shape,
        block_size=bs,
        threads=threads,
        dtype=torch_to_tl_dtype(x.dtype),
        execution_backend=eb,
    )
    work = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    invoke_quant_kernel(
        kernel, x, sf, out=work,
        tilelang_target=tgt, execution_backend=eb,
    )
    result = work.to(torch.int8)
    if out is not None:
        if out.shape != x.shape or out.dtype != torch.int8:
            raise ValueError(f"{op}: out must be int8 with shape {tuple(x.shape)}")
        if out.device != x.device:
            raise TypeError(f"{op}: out device must match input")
        out.copy_(result)
        return out
    return result


def dequantize_per_tensor(
    x_q: torch.Tensor,
    scale: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.float32,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dequantize ``int8`` *x_q* with scalar *scale*."""
    op = "dequantize_per_tensor"
    if x_q.dtype != torch.int8:
        raise TypeError(f"{op}: input must be torch.int8 (got {x_q.dtype})")
    if out_dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError(
            f"{op}: out_dtype must be float32, float16, or bfloat16"
        )
    sf = _validate_scale_tensor(scale, op=op, numel=1)
    if sf.device != x_q.device:
        raise TypeError(f"{op}: scale device must match input")

    tgt = default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = default_execution_backend(tgt, dev)
    bs, threads = suggest_tile_config(x_q.numel())
    kernel = dispatch_compile_quant(
        op_name=op,
        target=tgt,
        shape=x_q.shape,
        block_size=bs,
        threads=threads,
        dtype=torch_to_tl_dtype(out_dtype),
        execution_backend=eb,
    )
    if out is None:
        out_f = torch.empty(x_q.shape, dtype=out_dtype, device=x_q.device)
    else:
        if out.shape != x_q.shape or out.dtype != out_dtype:
            raise ValueError(
                f"{op}: out must be {out_dtype} with shape {tuple(x_q.shape)}"
            )
        if out.device != x_q.device:
            raise TypeError(f"{op}: out device must match input")
        out_f = out
    return invoke_quant_kernel(
        kernel, x_q, sf, out=out_f,
        tilelang_target=tgt, execution_backend=eb,
    )


def quantize_per_channel(
    x: torch.Tensor,
    scale: torch.Tensor,
    dim: int,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Quantize *x* with one *scale* per index along *dim*."""
    op = "quantize_per_channel"
    if not x.dtype.is_floating_point:
        raise TypeError(f"{op}: input must be floating-point (got {x.dtype})")
    dim = dim if dim >= 0 else dim + x.ndim
    n = x.shape[dim]
    sf = _validate_scale_tensor(scale, op=op, numel=n)
    if sf.device != x.device:
        raise TypeError(f"{op}: scale device must match input")

    x_2d, m, cols, perm, orig = merge_on_dim(x, dim, op_name=op)
    tgt = default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = default_execution_backend(tgt, dev)
    bs, threads = suggest_tile_config(m * cols)
    kernel = dispatch_compile_quant(
        op_name=op,
        target=tgt,
        shape=x.shape,
        block_size=bs,
        threads=threads,
        dtype=torch_to_tl_dtype(x.dtype),
        execution_backend=eb,
        rows=m,
        cols=cols,
    )
    work = torch.empty(m, cols, dtype=torch.float32, device=x.device)
    invoke_quant_kernel(
        kernel, x_2d, sf, out=work,
        tilelang_target=tgt, execution_backend=eb,
    )
    result = unmerge(work.to(torch.int8), orig, perm)
    if out is not None:
        if out.shape != x.shape or out.dtype != torch.int8:
            raise ValueError(f"{op}: out must be int8 with shape {tuple(x.shape)}")
        if out.device != x.device:
            raise TypeError(f"{op}: out device must match input")
        out.copy_(result)
        return out
    return result


def dequantize_per_channel(
    x_q: torch.Tensor,
    scale: torch.Tensor,
    dim: int,
    *,
    out_dtype: torch.dtype = torch.float32,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dequantize ``int8`` *x_q* with per-channel *scale* along *dim*."""
    op = "dequantize_per_channel"
    if x_q.dtype != torch.int8:
        raise TypeError(f"{op}: input must be torch.int8 (got {x_q.dtype})")
    if out_dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError(
            f"{op}: out_dtype must be float32, float16, or bfloat16"
        )
    dim = dim if dim >= 0 else dim + x_q.ndim
    n = x_q.shape[dim]
    sf = _validate_scale_tensor(scale, op=op, numel=n)
    if sf.device != x_q.device:
        raise TypeError(f"{op}: scale device must match input")

    x_2d, m, cols, perm, orig = merge_on_dim(x_q, dim, op_name=op)
    tgt = default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = default_execution_backend(tgt, dev)
    bs, threads = suggest_tile_config(m * cols)
    kernel = dispatch_compile_quant(
        op_name=op,
        target=tgt,
        shape=x_q.shape,
        block_size=bs,
        threads=threads,
        dtype=torch_to_tl_dtype(out_dtype),
        execution_backend=eb,
        rows=m,
        cols=cols,
    )
    if out is None:
        out_2d = torch.empty(m, cols, dtype=out_dtype, device=x_q.device)
    else:
        if out.shape != x_q.shape or out.dtype != out_dtype:
            raise ValueError(
                f"{op}: out must be {out_dtype} with shape {tuple(x_q.shape)}"
            )
        if out.device != x_q.device:
            raise TypeError(f"{op}: out device must match input")
        out_2d = out.reshape(m, cols)
    invoke_quant_kernel(
        kernel, x_2d, sf, out=out_2d,
        tilelang_target=tgt, execution_backend=eb,
    )
    result = unmerge(out_2d, orig, perm)
    if out is not None:
        return out
    return result


from tileops.quant.awq import (
    AWQ_TRITON_SUPPORTED_GROUP_SIZES,
    REVERSE_AWQ_ORDER,
    awq_dequantize,
    awq_dequantize_triton,
    awq_gemm,
    awq_gemm_triton,
    pack_awq_int4,
    unpack_awq_int4,
)
from tileops.quant.fp8 import (
    apply_w8a8_block_fp8_linear,
    block_dequant as block_dequant_fp8,
    default_fp8_dtype,
    get_fp8_min_max,
    input_to_float8,
    is_fp8,
    per_token_group_quant_fp8,
    w8a8_block_fp8_matmul,
    w8a8_triton_block_scaled_mm,
)
from tileops.quant.int8 import (
    apply_w8a8_block_int8_linear,
    block_dequant,
    input_to_int8,
    per_token_group_quant_int8,
    per_token_quant_int8,
    w8a8_block_int8_matmul,
)

__all__ = [
    "AWQ_TRITON_SUPPORTED_GROUP_SIZES",
    "REVERSE_AWQ_ORDER",
    "apply_w8a8_block_fp8_linear",
    "apply_w8a8_block_int8_linear",
    "awq_dequantize",
    "awq_dequantize_triton",
    "awq_gemm",
    "awq_gemm_triton",
    "block_dequant",
    "block_dequant_fp8",
    "dequantize_per_channel",
    "dequantize_per_tensor",
    "default_fp8_dtype",
    "get_fp8_min_max",
    "input_to_float8",
    "input_to_int8",
    "is_fp8",
    "per_token_group_quant_fp8",
    "per_token_group_quant_int8",
    "per_token_quant_int8",
    "quantize_per_channel",
    "quantize_per_tensor",
    "w8a8_block_fp8_matmul",
    "w8a8_block_int8_matmul",
    "w8a8_triton_block_scaled_mm",
    "pack_awq_int4",
    "unpack_awq_int4",
]
