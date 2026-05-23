"""CUDA GEMV kernel — block-reduction via ``T.alloc_reducer``."""

from __future__ import annotations

from typing import Any

import tilelang.language as T


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
    """Build a tiled GEMV PrimFunc with ``T.alloc_reducer``."""

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
