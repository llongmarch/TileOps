"""Metal fused activation–multiply kernels (mirrors :mod:`tileops.backend.cuda.fused`)."""

from typing import Any

import tilelang.language as T
from tilelang.language.tir import op as tir_op

from tileops.runtime import build_op, make_cached_compiler


def silu_and_mul_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    x = A[idx]
                    s = 1.0 / (1.0 + tir_op.exp(-x))
                    C[idx] = (x * s) * B[idx]

    return _kernel


def gelu_and_mul_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    x = A[idx]
                    xc = x * x * x
                    inner = 0.7978845608028654 * (x + 0.044715 * xc)
                    g = 0.5 * x * (1.0 + tir_op.tanh(inner))
                    C[idx] = g * B[idx]

    return _kernel


_FUSED_PRIM: dict[str, Any] = {
    "silu_and_mul": silu_and_mul_kernel,
    "gelu_and_mul": gelu_and_mul_kernel,
}

_compile = make_cached_compiler(_FUSED_PRIM, cache_prefix="metal_fused")

silu_and_mul = build_op(
    "silu_and_mul",
    compile_fn=_compile,
    doc="Metal: ``C = silu(A) * B``.",
)
gelu_and_mul = build_op(
    "gelu_and_mul",
    compile_fn=_compile,
    doc="Metal: ``C = gelu(A, tanh) * B``.",
)

__all__ = ["silu_and_mul", "gelu_and_mul"]
