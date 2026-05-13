"""Metal binary element-wise kernels (TileLang ``@T.prim_func``).

Scheduling mirrors :mod:`tileops.cuda.binary` but is a **separate** source
file so Metal-specific tuning can diverge.  The facade :mod:`tileops.binary`
dispatches here for ``metal`` targets.

Categories:
  * **Arithmetic** — add, sub, mul, div, pow, fmod, remainder, floor_div
  * **Extrema** — maximum, minimum
  * **Comparison** — eq, ne, gt, ge, lt, le  (output 0.0 / 1.0)
  * **Math** — atan2, copysign, hypot, xlogy
  * **Logical** — logical_and, logical_or, logical_xor  (output 0.0 / 1.0)
"""

from typing import Any

import tilelang.language as T
from tilelang.language.tir import op as tir_op

from tileops._infra import make_backend_op, make_cached_compiler

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
    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    C[idx] = tir_op.fmod(A[idx], B[idx])
    return main


def _remainder_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
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
                    y = A[idx]
                    x = B[idx]
                    if x > 0.0:
                        C[idx] = tir_op.call_pure_extern("float32", "atan", y / x)
                    else:
                        if x < 0.0:
                            if y >= 0.0:
                                C[idx] = (
                                    tir_op.call_pure_extern("float32", "atan", y / x)
                                    + 3.141592653589793
                                )
                            else:
                                C[idx] = (
                                    tir_op.call_pure_extern("float32", "atan", y / x)
                                    - 3.141592653589793
                                )
                        else:
                            if y > 0.0:
                                C[idx] = 1.5707963267948966
                            else:
                                if y < 0.0:
                                    C[idx] = -1.5707963267948966
                                else:
                                    C[idx] = 0.0
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
    """``sqrt(a² + b²)``; branchless on Metal to avoid scoped immutable vars."""
    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    a = A[idx]
                    b = B[idx]
                    C[idx] = tir_op.sqrt(a * a + b * b)
    return main


def _xlogy_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
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
    "add": _add_prim, "sub": _sub_prim, "mul": _mul_prim,
    "div": _div_prim, "pow": _pow_prim, "fmod": _fmod_prim,
    "remainder": _remainder_prim, "floor_div": _floor_div_prim,
    "maximum": _maximum_prim, "minimum": _minimum_prim,
    "eq": _eq_prim, "ne": _ne_prim, "gt": _gt_prim,
    "ge": _ge_prim, "lt": _lt_prim, "le": _le_prim,
    "atan2": _atan2_prim, "copysign": _copysign_prim,
    "hypot": _hypot_prim, "xlogy": _xlogy_prim,
    "logical_and": _logical_and_prim, "logical_or": _logical_or_prim,
    "logical_xor": _logical_xor_prim,
}

_compile = make_cached_compiler(_BINARY_PRIM, cache_prefix="metal_binary")

# -- arithmetic --------------------------------------------------------
add = make_backend_op("add", compile_fn=_compile, doc="Metal: ``C[i] = A[i] + B[i]``.")
sub = make_backend_op("sub", compile_fn=_compile, doc="Metal: ``C[i] = A[i] - B[i]``.")
mul = make_backend_op("mul", compile_fn=_compile, doc="Metal: ``C[i] = A[i] * B[i]``.")
div = make_backend_op("div", compile_fn=_compile, doc="Metal: ``C[i] = A[i] / B[i]``.")
pow = make_backend_op("pow", compile_fn=_compile, doc="Metal: ``C[i] = A[i] ** B[i]``.")
fmod = make_backend_op("fmod", compile_fn=_compile, doc="Metal: C-style fmod.")
remainder = make_backend_op("remainder", compile_fn=_compile, doc="Metal: Python-style remainder.")
floor_div = make_backend_op("floor_div", compile_fn=_compile, doc="Metal: ``floor(A[i]/B[i])``.")

# -- extrema -----------------------------------------------------------
maximum = make_backend_op("maximum", compile_fn=_compile, doc="Metal: ``max(A[i], B[i])``.")
minimum = make_backend_op("minimum", compile_fn=_compile, doc="Metal: ``min(A[i], B[i])``.")

# -- comparison --------------------------------------------------------
eq = make_backend_op("eq", compile_fn=_compile, doc="Metal: ``A[i] == B[i]``.")
ne = make_backend_op("ne", compile_fn=_compile, doc="Metal: ``A[i] != B[i]``.")
gt = make_backend_op("gt", compile_fn=_compile, doc="Metal: ``A[i] > B[i]``.")
ge = make_backend_op("ge", compile_fn=_compile, doc="Metal: ``A[i] >= B[i]``.")
lt = make_backend_op("lt", compile_fn=_compile, doc="Metal: ``A[i] < B[i]``.")
le = make_backend_op("le", compile_fn=_compile, doc="Metal: ``A[i] <= B[i]``.")

# -- math ---------------------------------------------------------------
atan2 = make_backend_op("atan2", compile_fn=_compile, doc="Metal: ``atan2(A[i], B[i])``.")
copysign = make_backend_op("copysign", compile_fn=_compile, doc="Metal: ``|A[i]|*sign(B[i])``.")
hypot = make_backend_op("hypot", compile_fn=_compile, doc="Metal: ``sqrt(A²+B²)``.")
xlogy = make_backend_op("xlogy", compile_fn=_compile, doc="Metal: ``A*log(B)`` (0 when A==0).")

# -- logical ------------------------------------------------------------
logical_and = make_backend_op("logical_and", compile_fn=_compile, doc="Metal: logical AND.")
logical_or = make_backend_op("logical_or", compile_fn=_compile, doc="Metal: logical OR.")
logical_xor = make_backend_op("logical_xor", compile_fn=_compile, doc="Metal: logical XOR.")

__all__ = [
    "add", "sub", "mul", "div", "pow", "fmod", "remainder", "floor_div",
    "maximum", "minimum",
    "eq", "ne", "gt", "ge", "lt", "le",
    "atan2", "copysign", "hypot", "xlogy",
    "logical_and", "logical_or", "logical_xor",
]
