"""CUDA / HIP TileLang kernels for :mod:`tileops.quant` (compile-only backend).

Symmetric mapping (no zero-point)::

    q = clamp(round(x / scale), -128, 127)   # int8
    x = float(q) * scale

*scale* buffers are ``float32``.  Input activations may be ``float32``,
``float16``, or ``bfloat16``.  Per-channel kernels use a row-major
``(rows, cols)`` view with one scale per column (channel index).
"""

import math
from typing import Any, Optional

import tilelang.language as T
from tilelang.language import cast as tl_cast
from tilelang.language.tir import op as tir_op

from tileops.runtime import compile_prim, default_tilelang_target

_INT8_MIN = -128.0
_INT8_MAX = 127.0


def _clamp_int8(x: Any) -> Any:
    """Clamp float expression *x* to ``[-128, 127]``."""
    hi = tir_op.if_then_else(x > _INT8_MAX, _INT8_MAX, x)
    return tir_op.if_then_else(hi < _INT8_MIN, _INT8_MIN, hi)


def _quantize_per_tensor_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(
        arg0_X: T.Tensor((n,), dtype),
        arg1_Scale: T.Tensor((1,), T.float32),
        arg2_Y: T.Tensor((n,), T.float32),
    ):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    qf = _clamp_int8(tir_op.round(arg0_X[idx] / arg1_Scale[0]))
                    arg2_Y[idx] = qf

    return main


def _dequantize_per_tensor_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(
        arg0_X: T.Tensor((n,), T.int8),
        arg1_Scale: T.Tensor((1,), T.float32),
        arg2_Y: T.Tensor((n,), dtype),
    ):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    arg2_Y[idx] = tl_cast(arg0_X[idx], dtype) * arg1_Scale[0]

    return main


def _quantize_per_channel_prim(
    m: int, n: int, bs: int, t: int, dtype: Any,
) -> Any:
    total = m * n

    @T.prim_func
    def main(
        arg0_X: T.Tensor((total,), dtype),
        arg1_Scale: T.Tensor((n,), T.float32),
        arg2_Y: T.Tensor((total,), T.float32),
    ):
        with T.Kernel(T.ceildiv(total, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < total:
                    ch = idx - (idx // n) * n
                    qf = _clamp_int8(tir_op.round(arg0_X[idx] / arg1_Scale[ch]))
                    arg2_Y[idx] = qf

    return main


def _dequantize_per_channel_prim(
    m: int, n: int, bs: int, t: int, dtype: Any,
) -> Any:
    total = m * n

    @T.prim_func
    def main(
        arg0_X: T.Tensor((total,), T.int8),
        arg1_Scale: T.Tensor((n,), T.float32),
        arg2_Y: T.Tensor((total,), dtype),
    ):
        with T.Kernel(T.ceildiv(total, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < total:
                    ch = idx - (idx // n) * n
                    arg2_Y[idx] = tl_cast(arg0_X[idx], dtype) * arg1_Scale[ch]

    return main


_POINTWISE_PRIM: dict[str, Any] = {
    "quantize_per_tensor": _quantize_per_tensor_prim,
    "dequantize_per_tensor": _dequantize_per_tensor_prim,
}

def _per_token_quant_int8_prim(
    m: int, n: int, _: int, __: int, dtype: Any, eps: float,
) -> Any:
    """Per-row dynamic quant: ``scale = absmax/127``, ``q = round(x/scale)``."""
    @T.prim_func
    def main(
        arg0_X: T.Tensor((m * n,), dtype),
        arg1_Y: T.Tensor((m * n,), T.float32),
        arg2_Scale: T.Tensor((m,), T.float32),
    ):
        with T.Kernel(m, threads=1) as (row,):
            base = row * n
            mx = T.alloc_local((1,), T.float32)
            mx[0] = 0.0
            for j in range(n):
                v = tir_op.abs(arg0_X[base + j])
                mx[0] = tir_op.if_then_else(v > mx[0], v, mx[0])
            absmax = tir_op.if_then_else(mx[0] > eps, mx[0], eps)
            scale_row = absmax / 127.0
            arg2_Scale[row] = scale_row
            for j in range(n):
                qf = _clamp_int8(tir_op.round(arg0_X[base + j] / scale_row))
                arg1_Y[base + j] = qf

    return main


_CHANNEL_PRIM: dict[str, Any] = {
    "quantize_per_channel": _quantize_per_channel_prim,
    "dequantize_per_channel": _dequantize_per_channel_prim,
}

_ROW_PRIM: dict[str, Any] = {
    "per_token_quant_int8": _per_token_quant_int8_prim,
}

_cache: dict[tuple, Any] = {}


def _compile_pointwise(
    op_name: str,
    num_elements: int,
    block_size: int,
    threads: int,
    dtype: Any,
    *,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> Any:
    tgt = target if target is not None else default_tilelang_target()
    key = (
        "cuda_quant_v6", op_name, num_elements, block_size, threads,
        dtype, tgt, execution_backend,
    )
    hit = _cache.get(key)
    if hit is not None:
        return hit
    kernel = compile_prim(
        _POINTWISE_PRIM[op_name](num_elements, block_size, threads, dtype),
        target=tgt,
        execution_backend=execution_backend,
    )
    _cache[key] = kernel
    return kernel


def _compile_row(
    op_name: str,
    rows: int,
    cols: int,
    block_size: int,
    threads: int,
    dtype: Any,
    *,
    eps: float = 1e-10,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> Any:
    tgt = target if target is not None else default_tilelang_target()
    key = (
        "cuda_quant_v7", op_name, rows, cols, block_size, threads,
        dtype, eps, tgt, execution_backend,
    )
    hit = _cache.get(key)
    if hit is not None:
        return hit
    kernel = compile_prim(
        _ROW_PRIM[op_name](rows, cols, block_size, threads, dtype, eps),
        target=tgt,
        execution_backend=execution_backend,
    )
    _cache[key] = kernel
    return kernel


def _compile_channel(
    op_name: str,
    rows: int,
    cols: int,
    block_size: int,
    threads: int,
    dtype: Any,
    *,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> Any:
    tgt = target if target is not None else default_tilelang_target()
    total = rows * cols
    key = (
        "cuda_quant_v6", op_name, rows, cols, block_size, threads,
        dtype, tgt, execution_backend,
    )
    hit = _cache.get(key)
    if hit is not None:
        return hit
    kernel = compile_prim(
        _CHANNEL_PRIM[op_name](rows, cols, block_size, threads, dtype),
        target=tgt,
        execution_backend=execution_backend,
    )
    _cache[key] = kernel
    return kernel


def quantize_per_tensor(
    shape: tuple[int, ...],
    block_size: int = 1024,
    threads: int = 128,
    *,
    dtype: Any,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> Any:
    n = math.prod(shape)
    return _compile_pointwise(
        "quantize_per_tensor", n, block_size, threads, dtype,
        target=target, execution_backend=execution_backend,
    )


def dequantize_per_tensor(
    shape: tuple[int, ...],
    block_size: int = 1024,
    threads: int = 128,
    *,
    dtype: Any,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> Any:
    n = math.prod(shape)
    return _compile_pointwise(
        "dequantize_per_tensor", n, block_size, threads, dtype,
        target=target, execution_backend=execution_backend,
    )


def quantize_per_channel(
    rows: int,
    cols: int,
    block_size: int = 1024,
    threads: int = 128,
    *,
    dtype: Any,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> Any:
    return _compile_channel(
        "quantize_per_channel", rows, cols, block_size, threads, dtype,
        target=target, execution_backend=execution_backend,
    )


def dequantize_per_channel(
    rows: int,
    cols: int,
    block_size: int = 1024,
    threads: int = 128,
    *,
    dtype: Any,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> Any:
    return _compile_channel(
        "dequantize_per_channel", rows, cols, block_size, threads, dtype,
        target=target, execution_backend=execution_backend,
    )


def per_token_quant_int8(
    rows: int,
    cols: int,
    threads: int = 1,
    *,
    dtype: Any,
    eps: float = 1e-10,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> Any:
    return _compile_row(
        "per_token_quant_int8", rows, cols, 1, threads, dtype,
        eps=eps,
        target=target, execution_backend=execution_backend,
    )


__all__ = [
    "quantize_per_tensor",
    "dequantize_per_tensor",
    "quantize_per_channel",
    "dequantize_per_channel",
    "per_token_quant_int8",
]
