"""Metal row-wise reductions (see :mod:`tileops.cuda.reduce`)."""

from typing import Any, Optional

import tilelang.language as T
from tilelang.language.tir import op as tir_op

from tileops.runtime import compile_prim, default_tilelang_target


def _sum_prim(m: int, n: int, _: int, dtype: Any, __unused: float) -> Any:
    @T.prim_func
    def main(
        arg0_X: T.Tensor((m * n,), dtype),
        arg1_Y: T.Tensor((m,), dtype),
    ):
        with T.Kernel(m, threads=1) as (row,):
            base = row * n
            acc = T.alloc_local((1,), T.float32)
            acc[0] = 0.0
            for j in range(n):
                acc[0] = acc[0] + arg0_X[base + j]
            arg1_Y[row] = acc[0]

    return main


def _mean_prim(m: int, n: int, _: int, dtype: Any, __unused: float) -> Any:
    @T.prim_func
    def main(
        arg0_X: T.Tensor((m * n,), dtype),
        arg1_Y: T.Tensor((m,), dtype),
    ):
        with T.Kernel(m, threads=1) as (row,):
            base = row * n
            acc = T.alloc_local((1,), T.float32)
            acc[0] = 0.0
            for j in range(n):
                acc[0] = acc[0] + arg0_X[base + j]
            arg1_Y[row] = acc[0] / float(n)

    return main


def _prod_prim(m: int, n: int, _: int, dtype: Any, __unused: float) -> Any:
    @T.prim_func
    def main(
        arg0_X: T.Tensor((m * n,), dtype),
        arg1_Y: T.Tensor((m,), dtype),
    ):
        with T.Kernel(m, threads=1) as (row,):
            base = row * n
            acc = T.alloc_local((1,), T.float32)
            acc[0] = 1.0
            for j in range(n):
                acc[0] = acc[0] * arg0_X[base + j]
            arg1_Y[row] = acc[0]

    return main


def _amax_prim(m: int, n: int, _: int, dtype: Any, __unused: float) -> Any:
    @T.prim_func
    def main(
        arg0_X: T.Tensor((m * n,), dtype),
        arg1_Y: T.Tensor((m,), dtype),
    ):
        with T.Kernel(m, threads=1) as (row,):
            base = row * n
            mx = T.alloc_local((1,), T.float32)
            mx[0] = arg0_X[base + 0]
            for j in range(1, n):
                a = mx[0]
                b = arg0_X[base + j]
                mx[0] = 0.5 * (a + b + tir_op.abs(a - b))
            arg1_Y[row] = mx[0]

    return main


def _amin_prim(m: int, n: int, _: int, dtype: Any, __unused: float) -> Any:
    @T.prim_func
    def main(
        arg0_X: T.Tensor((m * n,), dtype),
        arg1_Y: T.Tensor((m,), dtype),
    ):
        with T.Kernel(m, threads=1) as (row,):
            base = row * n
            mn = T.alloc_local((1,), T.float32)
            mn[0] = arg0_X[base + 0]
            for j in range(1, n):
                a = mn[0]
                b = arg0_X[base + j]
                mn[0] = 0.5 * (a + b - tir_op.abs(a - b))
            arg1_Y[row] = mn[0]

    return main


_REDUCE_PRIM: dict[str, Any] = {
    "sum": _sum_prim,
    "mean": _mean_prim,
    "prod": _prod_prim,
    "amax": _amax_prim,
    "amin": _amin_prim,
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
) -> Any:
    tgt = target if target is not None else default_tilelang_target()
    key = ("metal_reduce_v1", op_name, rows, cols, threads, dtype, tgt,
           execution_backend)
    hit = _cache.get(key)
    if hit is not None:
        return hit
    kernel = compile_prim(
        _REDUCE_PRIM[op_name](rows, cols, threads, dtype, 0.0),
        target=tgt,
        execution_backend=execution_backend,
    )
    _cache[key] = kernel
    return kernel


def row_sum(
    rows: int,
    cols: int,
    threads: int = 1,
    *,
    dtype: Any,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> Any:
    return _compile(
        "sum", rows, cols, threads, dtype,
        target=target, execution_backend=execution_backend,
    )


def row_mean(
    rows: int,
    cols: int,
    threads: int = 1,
    *,
    dtype: Any,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> Any:
    return _compile(
        "mean", rows, cols, threads, dtype,
        target=target, execution_backend=execution_backend,
    )


def row_prod(
    rows: int,
    cols: int,
    threads: int = 1,
    *,
    dtype: Any,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> Any:
    return _compile(
        "prod", rows, cols, threads, dtype,
        target=target, execution_backend=execution_backend,
    )


def row_amax(
    rows: int,
    cols: int,
    threads: int = 1,
    *,
    dtype: Any,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> Any:
    return _compile(
        "amax", rows, cols, threads, dtype,
        target=target, execution_backend=execution_backend,
    )


def row_amin(
    rows: int,
    cols: int,
    threads: int = 1,
    *,
    dtype: Any,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> Any:
    return _compile(
        "amin", rows, cols, threads, dtype,
        target=target, execution_backend=execution_backend,
    )


__all__ = ["row_sum", "row_mean", "row_prod", "row_amax", "row_amin"]
