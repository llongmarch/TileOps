"""CUDA / HIP reference GEMM and GEMV (scalar accumulation loops).

``gemm`` computes ``C = A @ B`` with *A* ``(M, K)``, *B* ``(K, N)``, *C*
``(M, N)``.  ``gemv`` computes ``y = A @ x`` with *A* ``(M, K)``, *x* ``(K,)``,
*y* ``(M,)``.

Each output element (or row for GEMV) is handled by a separate kernel instance
with ``threads=1`` — suitable for correctness and modest problem sizes, not a
cuBLAS replacement.
"""

from typing import Any, Optional

import tilelang.language as T

from tileops.runtime import compile_prim, default_tilelang_target


def _gemm_prim(m: int, n: int, k: int, dtype: Any) -> Any:
    @T.prim_func
    def main(
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

    return main


def _gemv_prim(m: int, k: int, dtype: Any) -> Any:
    @T.prim_func
    def main(
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

    return main


_cache: dict[tuple, Any] = {}


def _compile_key_prefix() -> str:
    return "cuda_blas_v1"


def gemm(
    m: int,
    n: int,
    k: int,
    *,
    dtype: Any,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> Any:
    tgt = target if target is not None else default_tilelang_target()
    key = (_compile_key_prefix(), "gemm", m, n, k, dtype, tgt, execution_backend)
    hit = _cache.get(key)
    if hit is not None:
        return hit
    kernel = compile_prim(
        _gemm_prim(m, n, k, dtype),
        target=tgt,
        execution_backend=execution_backend,
    )
    _cache[key] = kernel
    return kernel


def gemv(
    m: int,
    k: int,
    *,
    dtype: Any,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> Any:
    tgt = target if target is not None else default_tilelang_target()
    key = (_compile_key_prefix(), "gemv", m, k, dtype, tgt, execution_backend)
    hit = _cache.get(key)
    if hit is not None:
        return hit
    kernel = compile_prim(
        _gemv_prim(m, k, dtype),
        target=tgt,
        execution_backend=execution_backend,
    )
    _cache[key] = kernel
    return kernel


__all__ = ["gemm", "gemv"]
