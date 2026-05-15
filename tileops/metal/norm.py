"""Metal row-wise softmax, log-softmax, layer norm, RMS norm.

Mirrors :mod:`tileops.cuda.norm` with a separate in-process cache prefix.

Metal codegen assigns ``buffer(i)`` in **lexicographic parameter-name order**,
so multi-buffer kernels use ``arg0_*``, ``arg1_*``, … names so host call order
``(X, Gamma, Beta, out)`` matches the generated shader bindings.
"""

from typing import Any, Optional

import tilelang.language as T
from tilelang.language.tir import op as tir_op

from tileops.runtime import compile_prim, default_tilelang_target


def _softmax_prim(m: int, n: int, _: int, dtype: Any, __eps: float) -> Any:
    # Parameter names use arg0_/arg1_/… so Metal buffer order (lexicographic)
    # matches the host call order (see layer_norm / rms_norm).
    @T.prim_func
    def main(
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

    return main


def _online_softmax_prim(m: int, n: int, _: int, dtype: Any, __eps: float) -> Any:
    """One-pass online softmax (same numerics as :func:`_softmax_prim`)."""
    @T.prim_func
    def main(
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

    return main


def _log_softmax_prim(m: int, n: int, _: int, dtype: Any, __eps: float) -> Any:
    @T.prim_func
    def main(
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

    return main


def _layer_norm_prim(m: int, n: int, _: int, dtype: Any, eps: float) -> Any:
    @T.prim_func
    def main(
        arg0_X: T.Tensor((m * n,), dtype),
        arg1_Gamma: T.Tensor((n,), dtype),
        arg2_Beta: T.Tensor((n,), dtype),
        arg3_Y: T.Tensor((m * n,), dtype),
    ):
        with T.Kernel(m, threads=1) as (row,):
            base = row * n
            mean_buf = T.alloc_local((1,), T.float32)
            mean_buf[0] = 0.0
            for j in range(n):
                mean_buf[0] = mean_buf[0] + arg0_X[base + j]
            mean_val = mean_buf[0] / float(n)
            var_buf = T.alloc_local((1,), T.float32)
            var_buf[0] = 0.0
            for j in range(n):
                d = arg0_X[base + j] - mean_val
                var_buf[0] = var_buf[0] + d * d
            var_val = var_buf[0] / float(n)
            inv_std = tir_op.rsqrt(var_val + eps)
            for j in range(n):
                normed = (arg0_X[base + j] - mean_val) * inv_std
                arg3_Y[base + j] = normed * arg1_Gamma[j] + arg2_Beta[j]

    return main


def _rms_norm_prim(m: int, n: int, _: int, dtype: Any, eps: float) -> Any:
    @T.prim_func
    def main(
        arg0_X: T.Tensor((m * n,), dtype),
        arg1_W: T.Tensor((n,), dtype),
        arg2_Y: T.Tensor((m * n,), dtype),
    ):
        with T.Kernel(m, threads=1) as (row,):
            base = row * n
            ms_buf = T.alloc_local((1,), T.float32)
            ms_buf[0] = 0.0
            for j in range(n):
                xv = arg0_X[base + j]
                ms_buf[0] = ms_buf[0] + xv * xv
            mean_sq = ms_buf[0] / float(n)
            inv_rms = tir_op.rsqrt(mean_sq + eps)
            for j in range(n):
                arg2_Y[base + j] = arg0_X[base + j] * inv_rms * arg1_W[j]

    return main


def _skip_rms_norm_prim(m: int, n: int, _: int, dtype: Any, eps: float) -> Any:
    @T.prim_func
    def main(
        arg0_X: T.Tensor((m * n,), dtype),
        arg1_Residual: T.Tensor((m * n,), dtype),
        arg2_W: T.Tensor((n,), dtype),
        arg3_Y: T.Tensor((m * n,), dtype),
    ):
        with T.Kernel(m, threads=1) as (row,):
            base = row * n
            ms_buf = T.alloc_local((1,), T.float32)
            ms_buf[0] = 0.0
            for j in range(n):
                t = arg0_X[base + j] + arg1_Residual[base + j]
                ms_buf[0] = ms_buf[0] + t * t
            mean_sq = ms_buf[0] / float(n)
            inv_rms = tir_op.rsqrt(mean_sq + eps)
            for j in range(n):
                t = arg0_X[base + j] + arg1_Residual[base + j]
                arg3_Y[base + j] = t * inv_rms * arg2_W[j]

    return main


def _skip_layer_norm_prim(m: int, n: int, _: int, dtype: Any, eps: float) -> Any:
    @T.prim_func
    def main(
        arg0_X: T.Tensor((m * n,), dtype),
        arg1_Residual: T.Tensor((m * n,), dtype),
        arg2_Gamma: T.Tensor((n,), dtype),
        arg3_Beta: T.Tensor((n,), dtype),
        arg4_Y: T.Tensor((m * n,), dtype),
    ):
        with T.Kernel(m, threads=1) as (row,):
            base = row * n
            mean_buf = T.alloc_local((1,), T.float32)
            mean_buf[0] = 0.0
            for j in range(n):
                t = arg0_X[base + j] + arg1_Residual[base + j]
                mean_buf[0] = mean_buf[0] + t
            mean_val = mean_buf[0] / float(n)
            var_buf = T.alloc_local((1,), T.float32)
            var_buf[0] = 0.0
            for j in range(n):
                t = arg0_X[base + j] + arg1_Residual[base + j]
                d = t - mean_val
                var_buf[0] = var_buf[0] + d * d
            var_val = var_buf[0] / float(n)
            inv_std = tir_op.rsqrt(var_val + eps)
            for j in range(n):
                t = arg0_X[base + j] + arg1_Residual[base + j]
                normed = (t - mean_val) * inv_std
                arg4_Y[base + j] = normed * arg2_Gamma[j] + arg3_Beta[j]

    return main


_NORM_PRIM: dict[str, Any] = {
    "softmax": _softmax_prim,
    "online_softmax": _online_softmax_prim,
    "log_softmax": _log_softmax_prim,
    "layer_norm": _layer_norm_prim,
    "rms_norm": _rms_norm_prim,
    "skip_rms_norm": _skip_rms_norm_prim,
    "skip_layer_norm": _skip_layer_norm_prim,
}

_cache: dict[tuple, Any] = {}


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
    tgt = target if target is not None else default_tilelang_target()
    # Bump first tuple element when IR/signature changes so in-process cache
    # does not reuse stale Metal JIT after edits.
    key = ("metal_norm_v3", op_name, rows, cols, threads, dtype, tgt,
           execution_backend, pe)
    hit = _cache.get(key)
    if hit is not None:
        return hit
    kernel = compile_prim(
        _NORM_PRIM[op_name](rows, cols, threads, dtype, pe),
        target=tgt,
        execution_backend=execution_backend,
    )
    _cache[key] = kernel
    return kernel


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


def layer_norm(
    rows: int,
    cols: int,
    threads: int = 1,
    *,
    dtype: Any,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
    eps: Optional[float] = None,
) -> Any:
    e = 1e-5 if eps is None else float(eps)
    return _compile(
        "layer_norm", rows, cols, threads, dtype,
        target=target, execution_backend=execution_backend, eps=e,
    )


def rms_norm(
    rows: int,
    cols: int,
    threads: int = 1,
    *,
    dtype: Any,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
    eps: Optional[float] = None,
) -> Any:
    e = 1e-5 if eps is None else float(eps)
    return _compile(
        "rms_norm", rows, cols, threads, dtype,
        target=target, execution_backend=execution_backend, eps=e,
    )


def skip_rms_norm(
    rows: int,
    cols: int,
    threads: int = 1,
    *,
    dtype: Any,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
    eps: Optional[float] = None,
) -> Any:
    e = 1e-5 if eps is None else float(eps)
    return _compile(
        "skip_rms_norm", rows, cols, threads, dtype,
        target=target, execution_backend=execution_backend, eps=e,
    )


def skip_layer_norm(
    rows: int,
    cols: int,
    threads: int = 1,
    *,
    dtype: Any,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
    eps: Optional[float] = None,
) -> Any:
    e = 1e-5 if eps is None else float(eps)
    return _compile(
        "skip_layer_norm", rows, cols, threads, dtype,
        target=target, execution_backend=execution_backend, eps=e,
    )


__all__ = [
    "softmax",
    "online_softmax",
    "log_softmax",
    "layer_norm",
    "rms_norm",
    "skip_layer_norm",
    "skip_rms_norm",
]
