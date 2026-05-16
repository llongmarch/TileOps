"""CUDA / HIP row-wise reductions over the last axis of a 2-D view.

Each kernel maps a flat row of length *N* (``rows * cols`` flattened as *rows*
blocks of *cols* elements) to one output value per row (length *rows*), except
*cumsum* which writes prefix sums inside each row (output length ``rows * cols``).
The facade merges ``dim`` to the last dimension before calling here.
"""

from typing import Any, Optional

import tilelang.language as T
from tilelang.language.tir import op as tir_op

from tileops.runtime import compile_prim, default_tilelang_target, make_generic_cached_compiler


def sum_kernel(m: int, n: int, _: int, dtype: Any, __unused: float) -> Any:
    @T.prim_func
    def _kernel(
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

    return _kernel


def mean_kernel(m: int, n: int, _: int, dtype: Any, __unused: float) -> Any:
    @T.prim_func
    def _kernel(
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

    return _kernel


def prod_kernel(m: int, n: int, _: int, dtype: Any, __unused: float) -> Any:
    @T.prim_func
    def _kernel(
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

    return _kernel


def amax_kernel(m: int, n: int, _: int, dtype: Any, __unused: float) -> Any:
    @T.prim_func
    def _kernel(
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

    return _kernel


def amin_kernel(m: int, n: int, _: int, dtype: Any, __unused: float) -> Any:
    @T.prim_func
    def _kernel(
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

    return _kernel


def argmax_kernel(m: int, n: int, _: int, dtype: Any, __unused: float) -> Any:
    @T.prim_func
    def _kernel(
        arg0_X: T.Tensor((m * n,), dtype),
        arg1_Index: T.Tensor((m,), T.int64),
    ):
        with T.Kernel(m, threads=1) as (row,):
            base = row * n
            best_val = T.alloc_local((1,), T.float32)
            best_idx = T.alloc_local((1,), T.int64)
            best_val[0] = arg0_X[base + 0]
            best_idx[0] = 0
            for j in range(1, n):
                v = arg0_X[base + j]
                if v > best_val[0]:
                    best_val[0] = v
                    best_idx[0] = j
            arg1_Index[row] = best_idx[0]

    return _kernel


def make_topk_kernel(largest: bool):
    def topk_kernel(m: int, n: int, k: int, _: int, dtype: Any, __unused: float) -> Any:
        @T.prim_func
        def _kernel(
            arg0_X: T.Tensor((m * n,), dtype),
            arg1_Values: T.Tensor((m * k,), dtype),
            arg2_Index: T.Tensor((m * k,), T.int64),
        ):
            with T.Kernel(m, threads=1) as (row,):
                base = row * n
                val = T.alloc_local((k,), T.float32)
                idx = T.alloc_local((k,), T.int64)
                for i in range(k):
                    val[i] = arg0_X[base + i]
                    idx[i] = i
                for i in range(k):
                    for t in range(i + 1, k):
                        if largest:
                            if val[t] > val[i]:
                                val[i], val[t] = val[t], val[i]
                                idx[i], idx[t] = idx[t], idx[i]
                        else:
                            if val[t] < val[i]:
                                val[i], val[t] = val[t], val[i]
                                idx[i], idx[t] = idx[t], idx[i]
                for j in range(k, n):
                    v = arg0_X[base + j]
                    for t in range(k):
                        if largest:
                            if v > val[t]:
                                for s in range(k - 1, t, -1):
                                    val[s] = val[s - 1]
                                    idx[s] = idx[s - 1]
                                val[t] = v
                                idx[t] = j
                                break
                        else:
                            if v < val[t]:
                                for s in range(k - 1, t, -1):
                                    val[s] = val[s - 1]
                                    idx[s] = idx[s - 1]
                                val[t] = v
                                idx[t] = j
                                break
                for i in range(k):
                    arg1_Values[row * k + i] = val[i]
                    arg2_Index[row * k + i] = idx[i]

        return _kernel

    return topk_kernel


def argmin_kernel(m: int, n: int, _: int, dtype: Any, __unused: float) -> Any:
    @T.prim_func
    def _kernel(
        arg0_X: T.Tensor((m * n,), dtype),
        arg1_Index: T.Tensor((m,), T.int64),
    ):
        with T.Kernel(m, threads=1) as (row,):
            base = row * n
            best_val = T.alloc_local((1,), T.float32)
            best_idx = T.alloc_local((1,), T.int64)
            best_val[0] = arg0_X[base + 0]
            best_idx[0] = 0
            for j in range(1, n):
                v = arg0_X[base + j]
                if v < best_val[0]:
                    best_val[0] = v
                    best_idx[0] = j
            arg1_Index[row] = best_idx[0]

    return _kernel


def all_kernel(m: int, n: int, _: int, dtype: Any, __unused: float) -> Any:
    @T.prim_func
    def _kernel(
        arg0_X: T.Tensor((m * n,), dtype),
        arg1_Y: T.Tensor((m,), dtype),
    ):
        with T.Kernel(m, threads=1) as (row,):
            base = row * n
            acc = T.alloc_local((1,), T.float32)
            acc[0] = 1.0
            for j in range(n):
                v = arg0_X[base + j]
                bit = tir_op.if_then_else(v != 0.0, 1.0, 0.0)
                acc[0] = acc[0] * bit
            arg1_Y[row] = acc[0]

    return _kernel


def any_kernel(m: int, n: int, _: int, dtype: Any, __unused: float) -> Any:
    @T.prim_func
    def _kernel(
        arg0_X: T.Tensor((m * n,), dtype),
        arg1_Y: T.Tensor((m,), dtype),
    ):
        with T.Kernel(m, threads=1) as (row,):
            base = row * n
            acc = T.alloc_local((1,), T.float32)
            acc[0] = 0.0
            for j in range(n):
                v = arg0_X[base + j]
                bit = tir_op.if_then_else(v != 0.0, 1.0, 0.0)
                a = acc[0]
                acc[0] = 0.5 * (a + bit + tir_op.abs(a - bit))
            arg1_Y[row] = acc[0]

    return _kernel


def cumsum_kernel(m: int, n: int, _: int, dtype: Any, __unused: float) -> Any:
    @T.prim_func
    def _kernel(
        arg0_X: T.Tensor((m * n,), dtype),
        arg1_Y: T.Tensor((m * n,), dtype),
    ):
        with T.Kernel(m, threads=1) as (row,):
            base = row * n
            acc = T.alloc_local((1,), T.float32)
            acc[0] = 0.0
            for j in range(n):
                acc[0] = acc[0] + arg0_X[base + j]
                arg1_Y[base + j] = acc[0]

    return _kernel


_REDUCE_PRIM: dict[str, Any] = {
    "sum": sum_kernel,
    "mean": mean_kernel,
    "prod": prod_kernel,
    "amax": amax_kernel,
    "amin": amin_kernel,
    "argmax": argmax_kernel,
    "argmin": argmin_kernel,
    "all": all_kernel,
    "any": any_kernel,
    "cumsum": cumsum_kernel,
}

_compile_cache = make_generic_cached_compiler(cache_prefix="cuda_reduce_v3")
_topk_cache: dict[tuple, Any] = {}


def _compile_topk(
    rows: int,
    cols: int,
    k: int,
    threads: int,
    dtype: Any,
    *,
    largest: bool,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> Any:
    tgt = target if target is not None else default_tilelang_target()
    key = (
        "cuda_reduce_topk_v4", rows, cols, k, threads, dtype, largest, tgt,
        execution_backend,
    )
    hit = _topk_cache.get(key)
    if hit is not None:
        return hit
    kernel = compile_prim(
        make_topk_kernel(largest)(rows, cols, k, threads, dtype, 0.0),
        target=tgt,
        execution_backend=execution_backend,
        out_idx=[1, 2],
    )
    _topk_cache[key] = kernel
    return kernel


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
    return _compile_cache(
        (op_name, rows, cols, threads, dtype),
        _REDUCE_PRIM[op_name],
        rows, cols, threads, dtype, 0.0,
        target=target,
        execution_backend=execution_backend,
    )


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


def row_argmax(
    rows: int,
    cols: int,
    threads: int = 1,
    *,
    dtype: Any,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> Any:
    return _compile(
        "argmax", rows, cols, threads, dtype,
        target=target, execution_backend=execution_backend,
    )


def row_argmin(
    rows: int,
    cols: int,
    threads: int = 1,
    *,
    dtype: Any,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> Any:
    return _compile(
        "argmin", rows, cols, threads, dtype,
        target=target, execution_backend=execution_backend,
    )


def row_topk(
    rows: int,
    cols: int,
    k: int,
    threads: int = 1,
    *,
    dtype: Any,
    largest: bool = True,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> Any:
    return _compile_topk(
        rows, cols, k, threads, dtype,
        largest=largest,
        target=target, execution_backend=execution_backend,
    )


def row_all(
    rows: int,
    cols: int,
    threads: int = 1,
    *,
    dtype: Any,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> Any:
    return _compile(
        "all", rows, cols, threads, dtype,
        target=target, execution_backend=execution_backend,
    )


def row_any(
    rows: int,
    cols: int,
    threads: int = 1,
    *,
    dtype: Any,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> Any:
    return _compile(
        "any", rows, cols, threads, dtype,
        target=target, execution_backend=execution_backend,
    )


def row_cumsum(
    rows: int,
    cols: int,
    threads: int = 1,
    *,
    dtype: Any,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> Any:
    return _compile(
        "cumsum", rows, cols, threads, dtype,
        target=target, execution_backend=execution_backend,
    )


__all__ = [
    "row_sum",
    "row_mean",
    "row_prod",
    "row_amax",
    "row_amin",
    "row_argmax",
    "row_argmin",
    "row_topk",
    "row_all",
    "row_any",
    "row_cumsum",
]
