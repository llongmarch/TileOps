"""Metal binary element-wise kernels (TileLang ``@T.prim_func``).

Scheduling mirrors :mod:`tileops.backend.cuda.binary` but is a **separate** source
file so Metal-specific tuning can diverge.  The facade :mod:`tileops.binary`
dispatches here for ``metal`` targets.

Categories:
  * **Arithmetic** — add, sub, mul, div, pow, fmod, remainder, floor_div
  * **Extrema** — maximum, minimum
  * **Comparison** — eq, ne, gt, ge, lt, le  (output 0.0 / 1.0)
  * **Math** — atan2, copysign, hypot, xlogy, logaddexp
  * **Logical** — logical_and, logical_or, logical_xor  (output 0.0 / 1.0)
  * **Bitwise** — bitwise_and, bitwise_or, bitwise_xor, shift_left, shift_right
"""

from typing import Any

import tilelang.language as T
from tilelang.language.tir import op as tir_op

from tileops.runtime import build_op, make_cached_compiler

# ═══════════════════════════════════════════════════════════════════════════
# PrimFunc builders — Arithmetic
# ═══════════════════════════════════════════════════════════════════════════


def add_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    C[idx] = A[idx] + B[idx]
    return _kernel


def sub_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    C[idx] = A[idx] - B[idx]
    return _kernel


def mul_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    C[idx] = A[idx] * B[idx]
    return _kernel


def div_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    C[idx] = A[idx] / B[idx]
    return _kernel


def pow_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    C[idx] = tir_op.pow(A[idx], B[idx])
    return _kernel


def fmod_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    C[idx] = tir_op.fmod(A[idx], B[idx])
    return _kernel


def remainder_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    a = A[idx]
                    b = B[idx]
                    C[idx] = a - tir_op.floor(a / b) * b
    return _kernel


def floor_div_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    C[idx] = tir_op.floor(A[idx] / B[idx])
    return _kernel


# ═══════════════════════════════════════════════════════════════════════════
# PrimFunc builders — Extrema (branchless)
# ═══════════════════════════════════════════════════════════════════════════


def maximum_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    a = A[idx]
                    b = B[idx]
                    C[idx] = 0.5 * (a + b + tir_op.abs(a - b))
    return _kernel


def minimum_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    a = A[idx]
                    b = B[idx]
                    C[idx] = 0.5 * (a + b - tir_op.abs(a - b))
    return _kernel


# ═══════════════════════════════════════════════════════════════════════════
# PrimFunc builders — Comparison  (output 0.0 / 1.0 in same dtype)
# ═══════════════════════════════════════════════════════════════════════════


def eq_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    if A[idx] == B[idx]:
                        C[idx] = 1.0
                    else:
                        C[idx] = 0.0
    return _kernel


def ne_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    if A[idx] != B[idx]:
                        C[idx] = 1.0
                    else:
                        C[idx] = 0.0
    return _kernel


def gt_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    if A[idx] > B[idx]:
                        C[idx] = 1.0
                    else:
                        C[idx] = 0.0
    return _kernel


def ge_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    if A[idx] >= B[idx]:
                        C[idx] = 1.0
                    else:
                        C[idx] = 0.0
    return _kernel


def lt_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    if A[idx] < B[idx]:
                        C[idx] = 1.0
                    else:
                        C[idx] = 0.0
    return _kernel


def le_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    if A[idx] <= B[idx]:
                        C[idx] = 1.0
                    else:
                        C[idx] = 0.0
    return _kernel


# ═══════════════════════════════════════════════════════════════════════════
# PrimFunc builders — Math
# ═══════════════════════════════════════════════════════════════════════════


def atan2_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
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
    return _kernel


def copysign_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    if B[idx] >= 0.0:
                        C[idx] = tir_op.abs(A[idx])
                    else:
                        C[idx] = -tir_op.abs(A[idx])
    return _kernel


def hypot_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    """``sqrt(a² + b²)``; branchless on Metal to avoid scoped immutable vars."""
    @T.prim_func
    def _kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    a = A[idx]
                    b = B[idx]
                    C[idx] = tir_op.sqrt(a * a + b * b)
    return _kernel


def xlogy_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    if A[idx] == 0.0:
                        C[idx] = 0.0
                    else:
                        C[idx] = A[idx] * tir_op.log(B[idx])
    return _kernel


# ═══════════════════════════════════════════════════════════════════════════
# PrimFunc builders — Logical  (float → 0.0 / 1.0; non-zero = True)
# ═══════════════════════════════════════════════════════════════════════════


def logical_and_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
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
    return _kernel


def logical_or_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
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
    return _kernel


def logical_xor_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
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
    return _kernel


# ═══════════════════════════════════════════════════════════════════════════
# PrimFunc builders — Bitwise
# ═══════════════════════════════════════════════════════════════════════════


def bitwise_and_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    C[idx] = tir_op.bitwise_and(A[idx], B[idx])
    return _kernel


def bitwise_or_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    C[idx] = tir_op.bitwise_or(A[idx], B[idx])
    return _kernel


def bitwise_xor_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    C[idx] = tir_op.bitwise_xor(A[idx], B[idx])
    return _kernel


def shift_left_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    C[idx] = tir_op.shift_left(A[idx], B[idx])
    return _kernel


def shift_right_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    C[idx] = tir_op.shift_right(A[idx], B[idx])
    return _kernel


# ═══════════════════════════════════════════════════════════════════════════
# PrimFunc builders — LogAddExp
# ═══════════════════════════════════════════════════════════════════════════


def logaddexp_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    C[idx] = tir_op.log(tir_op.exp(A[idx]) + tir_op.exp(B[idx]))
    return _kernel


# ═══════════════════════════════════════════════════════════════════════════
# Cached compiler + public API
# ═══════════════════════════════════════════════════════════════════════════

_BINARY_PRIM: dict[str, Any] = {
    "add": add_kernel, "sub": sub_kernel, "mul": mul_kernel,
    "div": div_kernel, "pow": pow_kernel, "fmod": fmod_kernel,
    "remainder": remainder_kernel, "floor_div": floor_div_kernel,
    "maximum": maximum_kernel, "minimum": minimum_kernel,
    "eq": eq_kernel, "ne": ne_kernel, "gt": gt_kernel,
    "ge": ge_kernel, "lt": lt_kernel, "le": le_kernel,
    "atan2": atan2_kernel, "copysign": copysign_kernel,
    "hypot": hypot_kernel, "xlogy": xlogy_kernel,
    "logaddexp": logaddexp_kernel,
    "logical_and": logical_and_kernel, "logical_or": logical_or_kernel,
    "logical_xor": logical_xor_kernel,
    "bitwise_and": bitwise_and_kernel, "bitwise_or": bitwise_or_kernel,
    "bitwise_xor": bitwise_xor_kernel,
    "shift_left": shift_left_kernel, "shift_right": shift_right_kernel,
}

_compile = make_cached_compiler(_BINARY_PRIM, cache_prefix="metal_binary")

# -- arithmetic --------------------------------------------------------
add = build_op("add", compile_fn=_compile, doc="Metal: ``C[i] = A[i] + B[i]``.")
sub = build_op("sub", compile_fn=_compile, doc="Metal: ``C[i] = A[i] - B[i]``.")
mul = build_op("mul", compile_fn=_compile, doc="Metal: ``C[i] = A[i] * B[i]``.")
div = build_op("div", compile_fn=_compile, doc="Metal: ``C[i] = A[i] / B[i]``.")
pow = build_op("pow", compile_fn=_compile, doc="Metal: ``C[i] = A[i] ** B[i]``.")
fmod = build_op("fmod", compile_fn=_compile, doc="Metal: C-style fmod.")
remainder = build_op("remainder", compile_fn=_compile, doc="Metal: Python-style remainder.")
floor_div = build_op("floor_div", compile_fn=_compile, doc="Metal: ``floor(A[i]/B[i])``.")

# -- extrema -----------------------------------------------------------
maximum = build_op("maximum", compile_fn=_compile, doc="Metal: ``max(A[i], B[i])``.")
minimum = build_op("minimum", compile_fn=_compile, doc="Metal: ``min(A[i], B[i])``.")

# -- comparison --------------------------------------------------------
eq = build_op("eq", compile_fn=_compile, doc="Metal: ``A[i] == B[i]``.")
ne = build_op("ne", compile_fn=_compile, doc="Metal: ``A[i] != B[i]``.")
gt = build_op("gt", compile_fn=_compile, doc="Metal: ``A[i] > B[i]``.")
ge = build_op("ge", compile_fn=_compile, doc="Metal: ``A[i] >= B[i]``.")
lt = build_op("lt", compile_fn=_compile, doc="Metal: ``A[i] < B[i]``.")
le = build_op("le", compile_fn=_compile, doc="Metal: ``A[i] <= B[i]``.")

# -- math ---------------------------------------------------------------
atan2 = build_op("atan2", compile_fn=_compile, doc="Metal: ``atan2(A[i], B[i])``.")
copysign = build_op("copysign", compile_fn=_compile, doc="Metal: ``|A[i]|*sign(B[i])``.")
hypot = build_op("hypot", compile_fn=_compile, doc="Metal: ``sqrt(A²+B²)``.")
xlogy = build_op("xlogy", compile_fn=_compile, doc="Metal: ``A*log(B)`` (0 when A==0).")

# -- logical ------------------------------------------------------------
logical_and = build_op("logical_and", compile_fn=_compile, doc="Metal: logical AND.")
logical_or = build_op("logical_or", compile_fn=_compile, doc="Metal: logical OR.")
logical_xor = build_op("logical_xor", compile_fn=_compile, doc="Metal: logical XOR.")

# -- bitwise --------------------------------------------------------------
bitwise_and = build_op("bitwise_and", compile_fn=_compile, doc="Metal: ``bitwise_and(A[i], B[i])``.")
bitwise_or = build_op("bitwise_or", compile_fn=_compile, doc="Metal: ``bitwise_or(A[i], B[i])``.")
bitwise_xor = build_op("bitwise_xor", compile_fn=_compile, doc="Metal: ``bitwise_xor(A[i], B[i])``.")
shift_left = build_op("shift_left", compile_fn=_compile, doc="Metal: ``A[i] << B[i]``.")
shift_right = build_op("shift_right", compile_fn=_compile, doc="Metal: ``A[i] >> B[i]``.")

# -- math (continued) -----------------------------------------------------
logaddexp = build_op("logaddexp", compile_fn=_compile, doc="Metal: ``log(exp(A[i]) + exp(B[i]))``.")

__all__ = [
    "add", "sub", "mul", "div", "pow", "fmod", "remainder", "floor_div",
    "maximum", "minimum",
    "eq", "ne", "gt", "ge", "lt", "le",
    "atan2", "copysign", "hypot", "xlogy", "logaddexp",
    "logical_and", "logical_or", "logical_xor",
    "bitwise_and", "bitwise_or", "bitwise_xor", "shift_left", "shift_right",
]
