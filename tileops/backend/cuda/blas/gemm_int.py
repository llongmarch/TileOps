"""CUDA INT4 / INT8 GEMM kernel — ``T.gemm`` with ``transpose_B=True``.

B is stored in (K, N) layout and transposed during load into B_shared as
(block_N, block_K) so that ``T.gemm(transpose_B=True)`` matches the standard
BLAS convention.

Reference: ``env/blas/example_tilelang_gemm_int4.py``
"""

from __future__ import annotations

from typing import Any

import tilelang.language as T


def gemm_int_kernel(
    m: int,
    n: int,
    k: int,
    block_M: int,
    block_N: int,
    block_K: int,
    num_stages: int,
    threads: int,
    dtype: Any,
    accum_dtype: Any = T.int32,
) -> Any:
    """Build an INT4 or INT8 GEMM PrimFunc.

    A is ``(M, K)``, B is ``(K, N)`` in global memory (standard layout).
    For INT4, the K dimension refers to the *packed* size (K // 2 logical elements).
    ``accum_dtype`` defaults to ``T.int32`` for integer accumulation.
    """

    @T.prim_func
    def _kernel(
        arg0_A: T.Tensor((m, k), dtype),
        arg1_B: T.Tensor((k, n), dtype),
        arg2_C: T.Tensor((m, n), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(n, block_N), T.ceildiv(m, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(k, block_K), num_stages=num_stages):
                T.copy(arg0_A[by * block_M, ko * block_K], A_shared)
                for ni, ki in T.Parallel(block_N, block_K):
                    B_shared[ni, ki] = arg1_B[ko * block_K + ki, bx * block_N + ni]
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            T.copy(C_local, arg2_C[by * block_M, bx * block_N])

    return _kernel
