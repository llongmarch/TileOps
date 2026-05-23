"""Metal tiled GEMM and GEMV with shared-memory pipelining.

``gemm`` computes ``C = A @ B`` with *A* ``(M, K)``, *B* ``(K, N)``, *C*
``(M, N)`` using explicit accumulation loops (no Tensor Cores on Apple Silicon).
``gemv`` computes ``y = A @ x`` with *A* ``(M, K)``, *x* ``(K,)``, *y* ``(M,)``
using ``T.alloc_reducer`` for efficient block-level reduction.

Tile sizes are chosen heuristically from problem dimensions.
"""

from __future__ import annotations

from typing import Any, Optional

import tilelang.language as T

from tileops.runtime import make_generic_cached_compiler


# ── Heuristics ────────────────────────────────────────────────────────────

def _heuristic_gemm_config(m: int, n: int, k: int) -> dict:
    """Tile sizes tuned for Apple Silicon GPU (no Tensor Cores)."""
    total = m * n
    if total <= 256 * 256 or k <= 256:
        return {"block_M": 32, "block_N": 32, "block_K": 16, "num_stages": 2, "threads": 256}
    if total <= 1024 * 1024:
        return {"block_M": 64, "block_N": 64, "block_K": 32, "num_stages": 2, "threads": 256}
    return {"block_M": 64, "block_N": 128, "block_K": 32, "num_stages": 2, "threads": 256}


def _heuristic_gemv_config(m: int, k: int) -> dict:
    if m <= 1024 and k <= 1024:
        return {"block_M": 64, "block_K": 64, "num_stages": 2, "threads": 256}
    return {"block_M": 128, "block_K": 128, "num_stages": 2, "threads": 256}


# ── GEMM kernel builder ──────────────────────────────────────────────────

def gemm_kernel(
    m: int,
    n: int,
    k: int,
    block_M: int,
    block_N: int,
    block_K: int,
    num_stages: int,
    threads: int,
    dtype: Any,
    accum_dtype: Any = T.float32,
) -> Any:
    """Build a tiled GEMM PrimFunc with explicit accumulation for Metal."""

    @T.prim_func
    def _kernel(
        arg0_A: T.Tensor((m, k), dtype),
        arg1_B: T.Tensor((k, n), dtype),
        arg2_C: T.Tensor((m, n), dtype),
    ):
        with T.Kernel(T.ceildiv(n, block_N), T.ceildiv(m, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_local((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(k, block_K), num_stages=num_stages):
                T.copy(arg0_A[by * block_M, ko * block_K], A_shared)
                T.copy(arg1_B[ko * block_K, bx * block_N], B_shared)
                for i, j in T.Parallel(block_M, block_N):
                    for kk in T.serial(block_K):
                        C_local[i, j] += (
                            A_shared[i, kk].astype(accum_dtype)
                            * B_shared[kk, j].astype(accum_dtype)
                        )

            for i, j in T.Parallel(block_M, block_N):
                arg2_C[by * block_M + i, bx * block_N + j] = C_local[i, j]

    return _kernel


# ── GEMV kernel builder ──────────────────────────────────────────────────

def gemv_kernel(
    m: int,
    k: int,
    block_M: int,
    block_K: int,
    num_stages: int,
    threads: int,
    dtype: Any,
    accum_dtype: Any = T.float32,
) -> Any:
    """Build a tiled GEMV PrimFunc with ``T.alloc_reducer`` for block-level reduction."""

    @T.prim_func
    def _kernel(
        arg0_A: T.Tensor((m, k), dtype),
        arg1_x: T.Tensor((k,), dtype),
        arg2_y: T.Tensor((m,), dtype),
    ):
        with T.Kernel(T.ceildiv(m, block_M), threads=threads) as i0_m:
            y_reducer = T.alloc_reducer(block_M, accum_dtype, replication="all")
            T.clear(y_reducer)
            for i0_k in T.Pipelined(T.ceildiv(k, block_K), num_stages=num_stages):
                A_smem = T.alloc_shared((block_M, block_K), dtype)
                T.copy(arg0_A[i0_m * block_M, i0_k * block_K], A_smem)
                A_frag = T.alloc_fragment((block_M, block_K), dtype)
                T.copy(A_smem, A_frag)
                x_frag = T.alloc_fragment(block_K, dtype)
                T.copy(arg1_x[i0_k * block_K], x_frag)
                for i1_m, i1_k in T.Parallel(block_M, block_K):
                    y_reducer[i1_m] += (
                        A_frag[i1_m, i1_k].astype(accum_dtype)
                        * x_frag[i1_k].astype(accum_dtype)
                    )
            T.finalize_reducer(y_reducer)
            T.copy(y_reducer, arg2_y[i0_m * block_M])

    return _kernel


# ── Compilation & caching ────────────────────────────────────────────────

_compile = make_generic_cached_compiler(cache_prefix="metal_blas_v2")


def gemm(
    m: int,
    n: int,
    k: int,
    *,
    dtype: Any,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> Any:
    cfg = _heuristic_gemm_config(m, n, k)
    return _compile(
        ("gemm", m, n, k, dtype, cfg["block_M"], cfg["block_N"], cfg["block_K"], cfg["num_stages"], cfg["threads"]),
        gemm_kernel,
        m, n, k,
        cfg["block_M"], cfg["block_N"], cfg["block_K"], cfg["num_stages"], cfg["threads"], dtype,
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
    cfg = _heuristic_gemv_config(m, k)
    return _compile(
        ("gemv", m, k, dtype, cfg["block_M"], cfg["block_K"], cfg["num_stages"], cfg["threads"]),
        gemv_kernel,
        m, k,
        cfg["block_M"], cfg["block_K"], cfg["num_stages"], cfg["threads"], dtype,
        target=target,
        execution_backend=execution_backend,
    )


__all__ = ["gemm", "gemv"]
