"""Metal reference GEMM and GEMV (scalar accumulation loops).

Mirrors :mod:`tileops.backend.cuda.blas` with a separate in-process cache prefix.
Parameter names ``arg0_*`` … preserve lexicographic Metal buffer order.
"""

from typing import Any, Optional

import tilelang.language as T

from tileops.runtime import make_generic_cached_compiler


def gemm_kernel(m: int, n: int, k: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(
        arg0_A: T.Tensor((m, k), dtype),
        arg1_B: T.Tensor((k, n), dtype),
        arg2_C: T.Tensor((m, n), dtype),
    ):
        with T.Kernel(m, n, threads=1) as (i, j):
            acc = T.alloc_local((1,), T.float32)
            acc[0] = 0.0
            for kk in range(k):
                acc[0] = acc[0] + arg0_A[i, kk] * arg1_B[kk, j]
            arg2_C[i, j] = acc[0]

    return _kernel


def gemv_kernel(m: int, k: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(
        arg0_A: T.Tensor((m, k), dtype),
        arg1_x: T.Tensor((k,), dtype),
        arg2_y: T.Tensor((m,), dtype),
    ):
        with T.Kernel(m, threads=1) as (row,):
            acc = T.alloc_local((1,), T.float32)
            acc[0] = 0.0
            for p in range(k):
                acc[0] = acc[0] + arg0_A[row, p] * arg1_x[p]
            arg2_y[row] = acc[0]

    return _kernel


_compile = make_generic_cached_compiler(cache_prefix="metal_blas_v1")


def gemm(
    m: int,
    n: int,
    k: int,
    *,
    dtype: Any,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> Any:
    return _compile(
        ("gemm", m, n, k, dtype),
        gemm_kernel,
        m, n, k, dtype,
        target=target,
        execution_backend=execution_backend,
    )


def gemv(
    m: int,
    k: int,
    *,
    dtype: Any,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> Any:
    return _compile(
        ("gemv", m, k, dtype),
        gemv_kernel,
        m, k, dtype,
        target=target,
        execution_backend=execution_backend,
    )


__all__ = ["gemm", "gemv"]
