"""CUDA / HIP binary element-wise kernels (TileLang ``@T.prim_func``).

Full set of binary ops over a 1-D flat view of ``prod(shape)`` elements
with tail guard.  The facade :mod:`tileops.binary` dispatches here for
``cuda`` / ``hip`` / ``cutedsl`` targets.

Categories:
  * **Arithmetic** — add, sub, mul, div, pow, fmod, remainder, floor_div
  * **Extrema** — maximum, minimum
  * **Comparison** — eq, ne, gt, ge, lt, le  (output 0.0 / 1.0)
  * **Math** — atan2, copysign, hypot, xlogy
  * **Logical** — logical_and, logical_or, logical_xor  (output 0.0 / 1.0)

Also hosts :func:`add_shared` — a shared-memory tiled add variant that
only works on CUDA / HIP.
"""

import math
from typing import Any, Optional, Sequence

import tilelang.language as T
from tilelang.language.tir import op as tir_op

from tileops._infra import make_backend_op, make_cached_compiler
from tileops.runtime import (
    compile_prim,
    default_tilelang_target,
    target_kind,
)

# ═══════════════════════════════════════════════════════════════════════════
# PrimFunc builders — Arithmetic
# ═══════════════════════════════════════════════════════════════════════════


def _add_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    C[idx] = A[idx] + B[idx]
    return main


def _sub_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    C[idx] = A[idx] - B[idx]
    return main


def _mul_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    C[idx] = A[idx] * B[idx]
    return main


def _div_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    C[idx] = A[idx] / B[idx]
    return main


def _pow_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    C[idx] = tir_op.pow(A[idx], B[idx])
    return main


def _fmod_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    """C-style fmod: sign of result matches the dividend."""
    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    C[idx] = tir_op.fmod(A[idx], B[idx])
    return main


def _remainder_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    """Python-style remainder: sign of result matches the divisor."""
    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    a = A[idx]
                    b = B[idx]
                    C[idx] = a - tir_op.floor(a / b) * b
    return main


def _floor_div_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    C[idx] = tir_op.floor(A[idx] / B[idx])
    return main


# ═══════════════════════════════════════════════════════════════════════════
# PrimFunc builders — Extrema (branchless)
# ═══════════════════════════════════════════════════════════════════════════


def _maximum_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    a = A[idx]
                    b = B[idx]
                    C[idx] = 0.5 * (a + b + tir_op.abs(a - b))
    return main


def _minimum_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    a = A[idx]
                    b = B[idx]
                    C[idx] = 0.5 * (a + b - tir_op.abs(a - b))
    return main


# ═══════════════════════════════════════════════════════════════════════════
# PrimFunc builders — Comparison  (output 0.0 / 1.0 in same dtype)
# ═══════════════════════════════════════════════════════════════════════════


def _eq_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    if A[idx] == B[idx]:
                        C[idx] = 1.0
                    else:
                        C[idx] = 0.0
    return main


def _ne_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    if A[idx] != B[idx]:
                        C[idx] = 1.0
                    else:
                        C[idx] = 0.0
    return main


def _gt_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    if A[idx] > B[idx]:
                        C[idx] = 1.0
                    else:
                        C[idx] = 0.0
    return main


def _ge_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    if A[idx] >= B[idx]:
                        C[idx] = 1.0
                    else:
                        C[idx] = 0.0
    return main


def _lt_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    if A[idx] < B[idx]:
                        C[idx] = 1.0
                    else:
                        C[idx] = 0.0
    return main


def _le_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    if A[idx] <= B[idx]:
                        C[idx] = 1.0
                    else:
                        C[idx] = 0.0
    return main


# ═══════════════════════════════════════════════════════════════════════════
# PrimFunc builders — Math
# ═══════════════════════════════════════════════════════════════════════════


def _atan2_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    C[idx] = tir_op.atan2(A[idx], B[idx])
    return main


def _copysign_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    if B[idx] >= 0.0:
                        C[idx] = tir_op.abs(A[idx])
                    else:
                        C[idx] = -tir_op.abs(A[idx])
    return main


def _hypot_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    """Overflow-safe: ``|mx| * sqrt(1 + (mn/mx)^2)``."""
    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    ax = tir_op.abs(A[idx])
                    bx = tir_op.abs(B[idx])
                    if ax >= bx:
                        mx = ax
                        mn = bx
                    else:
                        mx = bx
                        mn = ax
                    if mx == 0.0:
                        C[idx] = 0.0
                    else:
                        r = mn / mx
                        C[idx] = mx * tir_op.sqrt(1.0 + r * r)
    return main


def _xlogy_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    """x * log(y), defined as 0 when x == 0."""
    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    if A[idx] == 0.0:
                        C[idx] = 0.0
                    else:
                        C[idx] = A[idx] * tir_op.log(B[idx])
    return main


# ═══════════════════════════════════════════════════════════════════════════
# PrimFunc builders — Logical  (float → 0.0 / 1.0; non-zero = True)
# ═══════════════════════════════════════════════════════════════════════════


def _logical_and_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    if A[idx] != 0.0:
                        if B[idx] != 0.0:
                            C[idx] = 1.0
                        else:
                            C[idx] = 0.0
                    else:
                        C[idx] = 0.0
    return main


def _logical_or_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    if A[idx] != 0.0:
                        C[idx] = 1.0
                    else:
                        if B[idx] != 0.0:
                            C[idx] = 1.0
                        else:
                            C[idx] = 0.0
    return main


def _logical_xor_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    if A[idx] != 0.0:
                        if B[idx] != 0.0:
                            C[idx] = 0.0
                        else:
                            C[idx] = 1.0
                    else:
                        if B[idx] != 0.0:
                            C[idx] = 1.0
                        else:
                            C[idx] = 0.0
    return main


# ═══════════════════════════════════════════════════════════════════════════
# Cached compiler + public API
# ═══════════════════════════════════════════════════════════════════════════

_BINARY_PRIM: dict[str, Any] = {
    # arithmetic
    "add": _add_prim, "sub": _sub_prim, "mul": _mul_prim,
    "div": _div_prim, "pow": _pow_prim, "fmod": _fmod_prim,
    "remainder": _remainder_prim, "floor_div": _floor_div_prim,
    # extrema
    "maximum": _maximum_prim, "minimum": _minimum_prim,
    # comparison
    "eq": _eq_prim, "ne": _ne_prim, "gt": _gt_prim,
    "ge": _ge_prim, "lt": _lt_prim, "le": _le_prim,
    # math
    "atan2": _atan2_prim, "copysign": _copysign_prim,
    "hypot": _hypot_prim, "xlogy": _xlogy_prim,
    # logical
    "logical_and": _logical_and_prim, "logical_or": _logical_or_prim,
    "logical_xor": _logical_xor_prim,
}

_compile = make_cached_compiler(_BINARY_PRIM, cache_prefix="cuda_binary")

# -- arithmetic --------------------------------------------------------
add = make_backend_op("add", compile_fn=_compile,
                      doc="CUDA/HIP: ``C[i] = A[i] + B[i]``.")
sub = make_backend_op("sub", compile_fn=_compile,
                      doc="CUDA/HIP: ``C[i] = A[i] - B[i]``.")
mul = make_backend_op("mul", compile_fn=_compile,
                      doc="CUDA/HIP: ``C[i] = A[i] * B[i]``.")
div = make_backend_op("div", compile_fn=_compile,
                      doc="CUDA/HIP: ``C[i] = A[i] / B[i]``.")
pow = make_backend_op("pow", compile_fn=_compile,
                      doc="CUDA/HIP: ``C[i] = A[i] ** B[i]``.")
fmod = make_backend_op("fmod", compile_fn=_compile,
                       doc="CUDA/HIP: C-style fmod (sign of dividend).")
remainder = make_backend_op("remainder", compile_fn=_compile,
                            doc="CUDA/HIP: Python-style remainder (sign of divisor).")
floor_div = make_backend_op("floor_div", compile_fn=_compile,
                            doc="CUDA/HIP: ``C[i] = floor(A[i] / B[i])``.")

# -- extrema -----------------------------------------------------------
maximum = make_backend_op("maximum", compile_fn=_compile,
                          doc="CUDA/HIP: ``C[i] = max(A[i], B[i])``.")
minimum = make_backend_op("minimum", compile_fn=_compile,
                          doc="CUDA/HIP: ``C[i] = min(A[i], B[i])``.")

# -- comparison  (output 0.0 / 1.0) ------------------------------------
eq = make_backend_op("eq", compile_fn=_compile, doc="CUDA/HIP: ``A[i] == B[i]``.")
ne = make_backend_op("ne", compile_fn=_compile, doc="CUDA/HIP: ``A[i] != B[i]``.")
gt = make_backend_op("gt", compile_fn=_compile, doc="CUDA/HIP: ``A[i] > B[i]``.")
ge = make_backend_op("ge", compile_fn=_compile, doc="CUDA/HIP: ``A[i] >= B[i]``.")
lt = make_backend_op("lt", compile_fn=_compile, doc="CUDA/HIP: ``A[i] < B[i]``.")
le = make_backend_op("le", compile_fn=_compile, doc="CUDA/HIP: ``A[i] <= B[i]``.")

# -- math ---------------------------------------------------------------
atan2 = make_backend_op("atan2", compile_fn=_compile,
                        doc="CUDA/HIP: ``C[i] = atan2(A[i], B[i])``.")
copysign = make_backend_op("copysign", compile_fn=_compile,
                           doc="CUDA/HIP: ``C[i] = |A[i]| * sign(B[i])``.")
hypot = make_backend_op("hypot", compile_fn=_compile,
                        doc="CUDA/HIP: ``C[i] = sqrt(A[i]² + B[i]²)``.")
xlogy = make_backend_op("xlogy", compile_fn=_compile,
                        doc="CUDA/HIP: ``C[i] = A[i]*log(B[i])``  (0 when A[i]==0).")

# -- logical  (output 0.0 / 1.0) ----------------------------------------
logical_and = make_backend_op("logical_and", compile_fn=_compile,
                              doc="CUDA/HIP: logical AND (non-zero = True).")
logical_or = make_backend_op("logical_or", compile_fn=_compile,
                             doc="CUDA/HIP: logical OR (non-zero = True).")
logical_xor = make_backend_op("logical_xor", compile_fn=_compile,
                              doc="CUDA/HIP: logical XOR (non-zero = True).")

# ── Shared-memory tiled add (CUDA / HIP only) ───────────────────────────

_SHARED_CACHE: dict[tuple, Any] = {}


def _add_shared_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            A_shared = T.alloc_shared((bs,), dtype)
            B_shared = T.alloc_shared((bs,), dtype)
            T.copy(A[bx * bs], A_shared)
            T.copy(B[bx * bs], B_shared)
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    C[idx] = A_shared[i] + B_shared[i]
    return main


def add_shared(
    shape: Sequence[int],
    block_size: int = 1024,
    threads: int = 128,
    *,
    dtype: Any = T.float32,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> Any:
    """Compile shared-memory tiled add (CUDA / HIP / cutedsl only).

    ``T.copy`` loads a full ``block_size``-wide slice without bounds
    checking, so *num_elements* **must** be divisible by *block_size*.
    """
    tgt = target if target is not None else default_tilelang_target()
    if target_kind(tgt) not in ("cuda", "hip", "cutedsl"):
        raise NotImplementedError(
            "add_shared uses GPU shared memory and is only "
            f"supported for cuda / hip / cutedsl targets; got {tgt!r}.  "
            "Use add() for other backends."
        )
    num_elements = math.prod(shape)
    if num_elements % block_size != 0:
        raise ValueError(
            f"add_shared requires num_elements ({num_elements}) to be "
            f"divisible by block_size ({block_size}) because T.copy "
            f"does not guard the tail block.  Adjust block_size or pad "
            f"the input."
        )
    key = ("shared_add", num_elements, block_size, threads, dtype, tgt, execution_backend)
    cached = _SHARED_CACHE.get(key)
    if cached is not None:
        return cached
    kernel = compile_prim(
        _add_shared_prim(num_elements, block_size, threads, dtype),
        target=tgt,
        execution_backend=execution_backend,
    )
    _SHARED_CACHE[key] = kernel
    return kernel


__all__ = [
    # arithmetic
    "add", "sub", "mul", "div", "pow", "fmod", "remainder", "floor_div",
    # extrema
    "maximum", "minimum",
    # comparison
    "eq", "ne", "gt", "ge", "lt", "le",
    # math
    "atan2", "copysign", "hypot", "xlogy",
    # logical
    "logical_and", "logical_or", "logical_xor",
    # special
    "add_shared",
]
