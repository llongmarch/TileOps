"""CUDA / HIP row-wise sort / argsort over the last axis of a 2-D view."""

from typing import Any, Optional

import tilelang.language as T

from tileops.runtime import make_generic_cached_compiler

_compile = make_generic_cached_compiler(cache_prefix="cuda_sort")


def sort_kernel(m: int, n: int, descending: bool, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(
        arg0_X: T.Tensor((m * n,), dtype),
        arg1_Values: T.Tensor((m * n,), dtype),
    ):
        with T.Kernel(m, threads=1) as (row,):
            base = row * n
            buf = T.alloc_local((n,), T.float32)
            for i in range(n):
                buf[i] = arg0_X[base + i]
            for i in range(n):
                for j in range(i + 1, n):
                    if descending:
                        if buf[j] > buf[i]:
                            buf[i], buf[j] = buf[j], buf[i]
                    else:
                        if buf[j] < buf[i]:
                            buf[i], buf[j] = buf[j], buf[i]
            for i in range(n):
                arg1_Values[base + i] = buf[i]
    return _kernel


def sort_row(
    m: int, n: int, descending: bool, *,
    dtype: Any,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> Any:
    return _compile(
        ("sort", m, n, descending, dtype),
        sort_kernel,
        m, n, descending, dtype,
        target=target,
        execution_backend=execution_backend,
    )


def argsort_kernel(m: int, n: int, descending: bool, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(
        arg0_X: T.Tensor((m * n,), dtype),
        arg1_Values: T.Tensor((m * n,), T.int64),
    ):
        with T.Kernel(m, threads=1) as (row,):
            base = row * n
            val = T.alloc_local((n,), T.float32)
            idx = T.alloc_local((n,), T.int64)
            for i in range(n):
                val[i] = arg0_X[base + i]
                idx[i] = i
            for i in range(n):
                for j in range(i + 1, n):
                    if descending:
                        if val[j] > val[i]:
                            val[i], val[j] = val[j], val[i]
                            idx[i], idx[j] = idx[j], idx[i]
                    else:
                        if val[j] < val[i]:
                            val[i], val[j] = val[j], val[i]
                            idx[i], idx[j] = idx[j], idx[i]
            for i in range(n):
                arg1_Values[base + i] = idx[i]
    return _kernel


def argsort_row(
    m: int, n: int, descending: bool, *,
    dtype: Any,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> Any:
    return _compile(
        ("argsort", m, n, descending, dtype),
        argsort_kernel,
        m, n, descending, dtype,
        target=target,
        execution_backend=execution_backend,
    )


__all__ = ["sort_row", "argsort_row"]
