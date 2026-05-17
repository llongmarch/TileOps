"""CUDA / HIP row-wise softmax and log-softmax.

Each kernel uses one thread block per logical row (*M* rows, *N* columns
after merging leading dimensions).  Inputs are flat buffers of length
``M * N``.

The facade :mod:`tileops.softmax` dispatches here for ``cuda`` / ``hip`` /
``cutedsl`` targets.
"""

from typing import Any, Optional

import tilelang.language as T
from tilelang.language.tir import op as tir_op

from tileops.runtime import make_generic_cached_compiler


def softmax_kernel(m: int, n: int, _: int, dtype: Any, __eps: float) -> Any:
    @T.prim_func
    def _kernel(
        arg0_X: T.Tensor((m * n,), dtype),
        arg1_Y: T.Tensor((m * n,), dtype),
    ):
        with T.Kernel(m, threads=1) as (row,):
            base = row * n
            max_buf = T.alloc_local((1,), T.float32)
            max_buf[0] = arg0_X[base + 0]
            for j in range(1, n):
                a = max_buf[0]
                b = arg0_X[base + j]
                max_buf[0] = 0.5 * (a + b + tir_op.abs(a - b))
            sum_buf = T.alloc_local((1,), T.float32)
            sum_buf[0] = 0.0
            for j in range(n):
                ev = tir_op.exp(arg0_X[base + j] - max_buf[0])
                arg1_Y[base + j] = ev
                sum_buf[0] = sum_buf[0] + ev
            for j in range(n):
                arg1_Y[base + j] = arg1_Y[base + j] / sum_buf[0]

    return _kernel


def online_softmax_kernel(m: int, n: int, _: int, dtype: Any, __eps: float) -> Any:
    """Row softmax via one-pass online max / denominator (Milo-style).

    Maintains running maximum *m* and denominator *d* = Σ exp(x−m) over the
    prefix; numerically matches the two-pass max-subtraction kernel.
    """
    @T.prim_func
    def _kernel(
        arg0_X: T.Tensor((m * n,), dtype),
        arg1_Y: T.Tensor((m * n,), dtype),
    ):
        with T.Kernel(m, threads=1) as (row,):
            base = row * n
            max_buf = T.alloc_local((1,), T.float32)
            max_buf[0] = arg0_X[base + 0]
            sum_buf = T.alloc_local((1,), T.float32)
            sum_buf[0] = 1.0
            for j in range(1, n):
                xj = arg0_X[base + j]
                a = max_buf[0]
                b = xj
                m_new = 0.5 * (a + b + tir_op.abs(a - b))
                sum_buf[0] = (
                    sum_buf[0] * tir_op.exp(a - m_new)
                    + tir_op.exp(b - m_new)
                )
                max_buf[0] = m_new
            for j in range(n):
                arg1_Y[base + j] = (
                    tir_op.exp(arg0_X[base + j] - max_buf[0]) / sum_buf[0]
                )

    return _kernel


def log_softmax_kernel(m: int, n: int, _: int, dtype: Any, __eps: float) -> Any:
    @T.prim_func
    def _kernel(
        arg0_X: T.Tensor((m * n,), dtype),
        arg1_Y: T.Tensor((m * n,), dtype),
    ):
        with T.Kernel(m, threads=1) as (row,):
            base = row * n
            max_buf = T.alloc_local((1,), T.float32)
            max_buf[0] = arg0_X[base + 0]
            for j in range(1, n):
                a = max_buf[0]
                b = arg0_X[base + j]
                max_buf[0] = 0.5 * (a + b + tir_op.abs(a - b))
            sum_buf = T.alloc_local((1,), T.float32)
            sum_buf[0] = 0.0
            for j in range(n):
                sum_buf[0] = sum_buf[0] + tir_op.exp(arg0_X[base + j] - max_buf[0])
            log_z = tir_op.log(sum_buf[0])
            for j in range(n):
                arg1_Y[base + j] = arg0_X[base + j] - max_buf[0] - log_z

    return _kernel


_SOFTMAX_PRIM: dict[str, Any] = {
    "softmax": softmax_kernel,
    "online_softmax": online_softmax_kernel,
    "log_softmax": log_softmax_kernel,
}

_compile_cache = make_generic_cached_compiler(cache_prefix="cuda_softmax_v1")


def _compile(
    op_name: str,
    rows: int,
    cols: int,
    threads: int,
    dtype: Any,
    *,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
    eps: Optional[float] = None,
) -> Any:
    pe = float(eps) if eps is not None else 0.0
    return _compile_cache(
        (op_name, rows, cols, threads, dtype, pe),
        _SOFTMAX_PRIM[op_name],
        rows, cols, threads, dtype, pe,
        target=target,
        execution_backend=execution_backend,
    )


def softmax(
    rows: int,
    cols: int,
    threads: int = 1,
    *,
    dtype: Any,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
    eps: Optional[float] = None,
) -> Any:
    return _compile(
        "softmax", rows, cols, threads, dtype,
        target=target, execution_backend=execution_backend, eps=eps,
    )


def online_softmax(
    rows: int,
    cols: int,
    threads: int = 1,
    *,
    dtype: Any,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
    eps: Optional[float] = None,
) -> Any:
    return _compile(
        "online_softmax", rows, cols, threads, dtype,
        target=target, execution_backend=execution_backend, eps=eps,
    )


def log_softmax(
    rows: int,
    cols: int,
    threads: int = 1,
    *,
    dtype: Any,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
    eps: Optional[float] = None,
) -> Any:
    return _compile(
        "log_softmax", rows, cols, threads, dtype,
        target=target, execution_backend=execution_backend, eps=eps,
    )


__all__ = [
    "softmax",
    "online_softmax",
    "log_softmax",
]
