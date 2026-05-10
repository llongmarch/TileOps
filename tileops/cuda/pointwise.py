"""CUDA-specific pointwise kernels (shared-memory variants).

These kernels use ``T.alloc_shared`` / ``T.copy`` and therefore require
a CUDA or HIP target.  They are discovered automatically by the dispatch
layer in :mod:`tileops.pointwise` and can also be imported directly::

    from tileops.cuda.pointwise import pointwise_add_shared
"""

import math
from typing import Any, Optional, Sequence

import tilelang.language as T

from tileops.runtime import (
    compile_prim,
    default_tilelang_target,
    require_float32_prim,
    target_kind,
)


# ── PrimFunc builder ────────────────────────────────────────────────────

def _pointwise_add_shared_prim(
    num_elements: int, block_size: int, threads: int,
) -> Any:
    """Shared-memory tiled add: stages inputs through SMEM, writes GMEM.

    Useful for benchmarking SMEM staging overhead vs direct GMEM access.
    Only compiles on CUDA / HIP (``T.alloc_shared`` emits ``shared.dyn``
    which the Metal / LLVM backends do not support).
    """

    @T.prim_func
    def main(
        A: T.Tensor((num_elements,), T.float32),
        B: T.Tensor((num_elements,), T.float32),
        C: T.Tensor((num_elements,), T.float32),
    ):
        with T.Kernel(T.ceildiv(num_elements, block_size), threads=threads) as (bx,):
            A_shared = T.alloc_shared((block_size,), T.float32)
            B_shared = T.alloc_shared((block_size,), T.float32)

            T.copy(A[bx * block_size], A_shared)
            T.copy(B[bx * block_size], B_shared)

            for i in T.Parallel(block_size):
                idx = bx * block_size + i
                if idx < num_elements:
                    C[idx] = A_shared[i] + B_shared[i]

    return main


# ── Public API ──────────────────────────────────────────────────────────

def pointwise_add_shared(
    shape: Sequence[int],
    block_size: int = 1024,
    threads: int = 128,
    *,
    in_dtype: Any = "float32",
    out_dtype: Any = "float32",
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> Any:
    """Compile an add kernel that stages inputs through shared memory.

    Only supported on CUDA / HIP targets.  The top-level
    ``tileops.pointwise.pointwise_add`` uses the GMEM-direct path by
    default; call this function explicitly when you want the shared-tile
    variant for benchmarking.

    Raises ``NotImplementedError`` on Metal or LLVM.
    """
    require_float32_prim(in_dtype, out_dtype, what="pointwise_add_shared")
    tgt = target if target is not None else default_tilelang_target()
    if target_kind(tgt) not in ("cuda", "hip"):
        raise NotImplementedError(
            "pointwise_add_shared uses GPU shared memory and is only "
            f"supported for cuda / hip targets; got {tgt!r}.  "
            "Use pointwise_add for Metal / CPU."
        )
    return compile_prim(
        _pointwise_add_shared_prim(math.prod(shape), block_size, threads),
        target=tgt,
        execution_backend=execution_backend,
    )
