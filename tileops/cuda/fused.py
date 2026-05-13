"""CUDA / HIP fused activation–multiply kernels.

Each kernel takes two same-shaped buffers *A*, *B* and writes *C* with
``C[i] = act(A[i]) * B[i]`` where *act* is SiLU or GELU (tanh approximation).
The facade :mod:`tileops.fused` dispatches here for ``cuda`` / ``hip`` /
``cutedsl`` targets.
"""

from typing import Any

import tilelang.language as T
from tilelang.language.tir import op as tir_op

from tileops._infra import make_backend_op, make_cached_compiler


def _silu_and_mul_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    """``C = silu(A) * B`` with ``silu(x) = x * sigmoid(x)``."""
    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    x = A[idx]
                    s = 1.0 / (1.0 + tir_op.exp(-x))
                    C[idx] = (x * s) * B[idx]

    return main


def _gelu_and_mul_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    """``C = gelu_tanh(A) * B`` — same tanh GELU as :mod:`tileops.cuda.activation`."""
    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    x = A[idx]
                    xc = x * x * x
                    inner = 0.7978845608028654 * (x + 0.044715 * xc)
                    g = 0.5 * x * (1.0 + tir_op.tanh(inner))
                    C[idx] = g * B[idx]

    return main


_FUSED_PRIM: dict[str, Any] = {
    "silu_and_mul": _silu_and_mul_prim,
    "gelu_and_mul": _gelu_and_mul_prim,
}

_compile = make_cached_compiler(_FUSED_PRIM, cache_prefix="cuda_fused")

silu_and_mul = make_backend_op(
    "silu_and_mul",
    compile_fn=_compile,
    doc="CUDA/HIP: ``C = silu(A) * B``.",
)
gelu_and_mul = make_backend_op(
    "gelu_and_mul",
    compile_fn=_compile,
    doc="CUDA/HIP: ``C = gelu(A, tanh) * B`` (PyTorch tanh approx).",
)

__all__ = ["silu_and_mul", "gelu_and_mul"]
