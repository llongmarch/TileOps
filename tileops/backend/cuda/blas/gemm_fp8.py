"""CUDA FP8 GEMM kernel — ``T.gemm`` with ``transpose_B=True``, k-pack, warp policy.

B is stored in (K, N) layout in global memory (standard BLAS convention);
inside the kernel it is loaded element-wise into B_shared as (block_N, block_K)
so that ``T.gemm(..., transpose_B=True)`` can treat it as transposed.

References:
- ``env/blas/example_tilelang_gemm_fp8_sm89.py``
- ``env/blas/example_tilelang_gemm_amd_fp8.py``
"""

from __future__ import annotations

from typing import Any

import tilelang.language as T


def gemm_fp8_kernel(
    m: int,
    n: int,
    k: int,
    block_M: int,
    block_N: int,
    block_K: int,
    num_stages: int,
    threads: int,
    k_pack: int,
    dtype: Any,
    accum_dtype: Any = T.float32,
) -> Any:
    """Build an FP8 GEMM PrimFunc.

    A is ``(M, K)``, B is ``(K, N)`` in global memory.
    B is transposed during load so ``T.gemm(transpose_B=True)`` works correctly.
    ``k_pack`` controls the K-dimension packing for Tensor Core MMA instructions
    (k_pack=2 doubles throughput on SM89+ for FP8).  ``T.GemmWarpPolicy.FullRow``
    schedules full-row accumulations per warp.
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
                T.gemm(
                    A_shared,
                    B_shared,
                    C_local,
                    transpose_B=True,
                    k_pack=k_pack,
                    policy=T.GemmWarpPolicy.FullRow,
                )

            T.copy(C_local, arg2_C[by * block_M, bx * block_N])

    return _kernel
