"""CUDA / HIP where and masked_fill kernels (element-wise conditionals).

``where(condition, x, y)`` selects from *x* or *y* per element based on
*condition*.
``masked_fill(x, mask, value)`` writes *value* wherever *mask* is non-zero.
"""

from typing import Any, Optional

import tilelang.language as T
from tilelang.language.tir import op as tir_op

from tileops.runtime import (
    build_op,
    make_cached_compiler,
    make_generic_cached_compiler,
)


def where_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(
        arg0_cond: T.Tensor((n,), T.float32),
        arg1_x: T.Tensor((n,), dtype),
        arg2_y: T.Tensor((n,), dtype),
        arg3_out: T.Tensor((n,), dtype),
    ):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    c = arg0_cond[idx]
                    if c != 0.0:
                        arg3_out[idx] = arg1_x[idx]
                    else:
                        arg3_out[idx] = arg2_y[idx]
    return _kernel


_COND_PRIM = {"where": where_kernel}
_cond_compile = make_cached_compiler(_COND_PRIM, cache_prefix="cuda_condition")
where = build_op("where", compile_fn=_cond_compile)


# ── masked_fill (needs extra ``value`` compile-time parameter) ──────────

_mf_compile = make_generic_cached_compiler(cache_prefix="cuda_condition_mf")


def masked_fill_kernel(n: int, bs: int, t: int, dtype: Any, *, value: float) -> Any:
    val = float(value)

    @T.prim_func
    def _kernel(
        arg0_x: T.Tensor((n,), dtype),
        arg1_mask: T.Tensor((n,), T.float32),
        arg2_out: T.Tensor((n,), dtype),
    ):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    m = arg1_mask[idx]
                    if m != 0.0:
                        arg2_out[idx] = val
                    else:
                        arg2_out[idx] = arg0_x[idx]
    return _kernel


def masked_fill(
    shape: tuple[int, ...],
    block_size: int,
    threads: int,
    *,
    dtype: Any,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
    value: float = 0.0,
) -> Any:
    n = 1
    for s in shape:
        n *= s
    val = float(value)
    return _mf_compile(
        ("masked_fill", n, block_size, threads, dtype, val),
        masked_fill_kernel,
        n, block_size, threads, dtype,
        target=target,
        execution_backend=execution_backend,
        value=val,
    )


__all__ = ["where", "masked_fill"]
