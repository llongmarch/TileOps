"""CUDA SM100 (Blackwell) GEMM kernel — ``T.tcgen05_gemm`` + TMA + tmem + mbarrier.

Uses the Blackwell warp-group MMA instruction ``tcgen05`` with Tensor Memory
Accelerator (TMA) for async copies and mbarrier-based synchronization.

Reference: ``env/blas/example_tilelang_gemm_fp8_sm100.py``
"""

from __future__ import annotations

from typing import Any

import tilelang.language as T


def gemm_sm100_kernel(
    m: int,
    n: int,
    k: int,
    block_M: int,
    block_N: int,
    block_K: int,
    num_stages: int,
    threads: int,
    in_dtype: Any,
    out_dtype: Any,
    accum_dtype: Any,
    trans_A: bool = False,
    trans_B: bool = True,
) -> Any:
    """Build an SM100 GEMM PrimFunc with tcgen05_gemm + TMA.

    A is ``(M, K)``, B is ``(K, N)`` in global memory (standard layout).
    Inside the kernel B is transposed during shared-memory load when ``trans_B=True``,
    and ``tcgen05_gemm`` is used for warp-group MMA on Blackwell tensor cores.
    """

    A_shape = (k, m) if trans_A else (m, k)
    B_shape = (k, n)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    @T.prim_func
    def _kernel(
        arg0_A: T.Tensor(A_shape, in_dtype),
        arg1_B: T.Tensor(B_shape, in_dtype),
        arg2_C: T.Tensor((m, n), out_dtype),
    ):
        with T.Kernel(T.ceildiv(n, block_N), T.ceildiv(m, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            C_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
            mbar = T.alloc_barrier(1)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)

            for ko in T.Pipelined(T.ceildiv(k, block_K), num_stages=num_stages):
                T.copy(arg0_A[by * block_M, ko * block_K], A_shared)
                if trans_B:
                    # Element-wise transposed load: B (K, N) → B_shared (block_N, block_K)
                    for ni, ki in T.Parallel(block_N, block_K):
                        B_shared[ni, ki] = arg1_B[ko * block_K + ki, bx * block_N + ni]
                else:
                    T.copy(arg1_B[ko * block_K, bx * block_N], B_shared)
                T.tcgen05_gemm(
                    A_shared,
                    B_shared,
                    C_tmem,
                    trans_A=trans_A,
                    trans_B=trans_B,
                    mbar=mbar,
                    clear_accum=(ko == 0),
                )
                T.mbarrier_wait_parity(mbar, ko % 2)

            T.copy(C_tmem, C_local)
            T.copy(C_local, C_shared)
            T.copy(C_shared, arg2_C[by * block_M, bx * block_N])

    return _kernel
