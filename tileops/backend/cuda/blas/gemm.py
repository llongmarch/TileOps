"""CUDA standard GEMM kernel — ``T.gemm`` with shared-memory pipelining.

Supports float16, bfloat16, float32.  Uses Tensor Core acceleration via
``T.gemm`` when the target architecture supports it.
"""

from __future__ import annotations

from typing import Any

import tilelang.language as T


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
    """Build a tiled GEMM PrimFunc with ``T.gemm`` for Tensor Core acceleration."""

    @T.prim_func
    def _kernel(
        arg0_A: T.Tensor((m, k), dtype),
        arg1_B: T.Tensor((k, n), dtype),
        arg2_C: T.Tensor((m, n), dtype),
    ):
        with T.Kernel(T.ceildiv(n, block_N), T.ceildiv(m, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(k, block_K), num_stages=num_stages):
                T.copy(arg0_A[by * block_M, ko * block_K], A_shared)
                T.copy(arg1_B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, arg2_C[by * block_M, bx * block_N])

    return _kernel
