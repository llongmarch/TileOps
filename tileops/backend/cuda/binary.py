"""CUDA / HIP binary element-wise kernels (TileLang ``@T.prim_func``).

Full set of binary ops over a 1-D flat view of ``prod(shape)`` elements
with tail guard.  The facade :mod:`tileops.binary` dispatches here for
``cuda`` / ``hip`` / ``cutedsl`` targets.

Categories:
  * **Arithmetic** — add, sub, mul, div, pow, fmod, remainder, floor_div
  * **Extrema** — maximum, minimum
  * **Comparison** — eq, ne, gt, ge, lt, le  (output 0.0 / 1.0)
  * **Math** — atan2, copysign, hypot, xlogy, logaddexp
  * **Logical** — logical_and, logical_or, logical_xor  (output 0.0 / 1.0)
  * **Bitwise** — bitwise_and, bitwise_or, bitwise_xor, shift_left, shift_right

Also hosts :func:`add_shared` — a shared-memory tiled add variant that
only works on CUDA / HIP.
"""

import math
from typing import Any, Optional, Sequence

import tilelang.language as T
from tilelang.language.tir import op as tir_op

from tileops.runtime import build_op, make_cached_compiler
from tileops.runtime import (
    compile_prim,
    default_tilelang_target,
    target_kind,
)

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
    """C-style fmod: sign of result matches the dividend."""
    @T.prim_func
    def _kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    C[idx] = tir_op.fmod(A[idx], B[idx])
    return _kernel


def remainder_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    """Python-style remainder: sign of result matches the divisor."""
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
                    C[idx] = tir_op.atan2(A[idx], B[idx])
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
    """Overflow-safe: ``|mx| * sqrt(1 + (mn/mx)^2)``."""
    @T.prim_func
    def _kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
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
    return _kernel


def xlogy_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    """x * log(y), defined as 0 when x == 0."""
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
    # arithmetic
    "add": add_kernel, "sub": sub_kernel, "mul": mul_kernel,
    "div": div_kernel, "pow": pow_kernel, "fmod": fmod_kernel,
    "remainder": remainder_kernel, "floor_div": floor_div_kernel,
    # extrema
    "maximum": maximum_kernel, "minimum": minimum_kernel,
    # comparison
    "eq": eq_kernel, "ne": ne_kernel, "gt": gt_kernel,
    "ge": ge_kernel, "lt": lt_kernel, "le": le_kernel,
    # math
    "atan2": atan2_kernel, "copysign": copysign_kernel,
    "hypot": hypot_kernel, "xlogy": xlogy_kernel,
    "logaddexp": logaddexp_kernel,
    # logical
    "logical_and": logical_and_kernel, "logical_or": logical_or_kernel,
    "logical_xor": logical_xor_kernel,
    # bitwise
    "bitwise_and": bitwise_and_kernel, "bitwise_or": bitwise_or_kernel,
    "bitwise_xor": bitwise_xor_kernel,
    "shift_left": shift_left_kernel, "shift_right": shift_right_kernel,
}

_compile = make_cached_compiler(_BINARY_PRIM, cache_prefix="cuda_binary")

# -- arithmetic --------------------------------------------------------
add = build_op("add", compile_fn=_compile,
                      doc="CUDA/HIP: ``C[i] = A[i] + B[i]``.")
sub = build_op("sub", compile_fn=_compile,
                      doc="CUDA/HIP: ``C[i] = A[i] - B[i]``.")
mul = build_op("mul", compile_fn=_compile,
                      doc="CUDA/HIP: ``C[i] = A[i] * B[i]``.")
div = build_op("div", compile_fn=_compile,
                      doc="CUDA/HIP: ``C[i] = A[i] / B[i]``.")
pow = build_op("pow", compile_fn=_compile,
                      doc="CUDA/HIP: ``C[i] = A[i] ** B[i]``.")
fmod = build_op("fmod", compile_fn=_compile,
                       doc="CUDA/HIP: C-style fmod (sign of dividend).")
remainder = build_op("remainder", compile_fn=_compile,
                            doc="CUDA/HIP: Python-style remainder (sign of divisor).")
floor_div = build_op("floor_div", compile_fn=_compile,
                            doc="CUDA/HIP: ``C[i] = floor(A[i] / B[i])``.")

# -- extrema -----------------------------------------------------------
maximum = build_op("maximum", compile_fn=_compile,
                          doc="CUDA/HIP: ``C[i] = max(A[i], B[i])``.")
minimum = build_op("minimum", compile_fn=_compile,
                          doc="CUDA/HIP: ``C[i] = min(A[i], B[i])``.")

# -- comparison  (output 0.0 / 1.0) ------------------------------------
eq = build_op("eq", compile_fn=_compile, doc="CUDA/HIP: ``A[i] == B[i]``.")
ne = build_op("ne", compile_fn=_compile, doc="CUDA/HIP: ``A[i] != B[i]``.")
gt = build_op("gt", compile_fn=_compile, doc="CUDA/HIP: ``A[i] > B[i]``.")
ge = build_op("ge", compile_fn=_compile, doc="CUDA/HIP: ``A[i] >= B[i]``.")
lt = build_op("lt", compile_fn=_compile, doc="CUDA/HIP: ``A[i] < B[i]``.")
le = build_op("le", compile_fn=_compile, doc="CUDA/HIP: ``A[i] <= B[i]``.")

# -- math ---------------------------------------------------------------
atan2 = build_op("atan2", compile_fn=_compile,
                        doc="CUDA/HIP: ``C[i] = atan2(A[i], B[i])``.")
copysign = build_op("copysign", compile_fn=_compile,
                           doc="CUDA/HIP: ``C[i] = |A[i]| * sign(B[i])``.")
hypot = build_op("hypot", compile_fn=_compile,
                        doc="CUDA/HIP: ``C[i] = sqrt(A[i]² + B[i]²)``.")
xlogy = build_op("xlogy", compile_fn=_compile,
                        doc="CUDA/HIP: ``C[i] = A[i]*log(B[i])``  (0 when A[i]==0).")

# -- logical  (output 0.0 / 1.0) ----------------------------------------
logical_and = build_op("logical_and", compile_fn=_compile,
                              doc="CUDA/HIP: logical AND (non-zero = True).")
logical_or = build_op("logical_or", compile_fn=_compile,
                             doc="CUDA/HIP: logical OR (non-zero = True).")
logical_xor = build_op("logical_xor", compile_fn=_compile,
                              doc="CUDA/HIP: logical XOR (non-zero = True).")

# -- bitwise --------------------------------------------------------------
bitwise_and = build_op("bitwise_and", compile_fn=_compile,
                              doc="CUDA/HIP: ``bitwise_and(A[i], B[i])``.")
bitwise_or = build_op("bitwise_or", compile_fn=_compile,
                             doc="CUDA/HIP: ``bitwise_or(A[i], B[i])``.")
bitwise_xor = build_op("bitwise_xor", compile_fn=_compile,
                              doc="CUDA/HIP: ``bitwise_xor(A[i], B[i])``.")
shift_left = build_op("shift_left", compile_fn=_compile,
                             doc="CUDA/HIP: ``A[i] << B[i]``.")
shift_right = build_op("shift_right", compile_fn=_compile,
                              doc="CUDA/HIP: ``A[i] >> B[i]``.")

# -- math (continued) -----------------------------------------------------
logaddexp = build_op("logaddexp", compile_fn=_compile,
                            doc="CUDA/HIP: ``log(exp(A[i]) + exp(B[i]))``.")

# ── Shared-memory tiled add (CUDA / HIP only) ──────────────────────────

_SHARED_CACHE: dict[tuple, Any] = {}


def add_shared_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            A_shared = T.alloc_shared((bs,), dtype)
            B_shared = T.alloc_shared((bs,), dtype)
            T.copy(A[bx * bs], A_shared)
            T.copy(B[bx * bs], B_shared)
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    C[idx] = A_shared[i] + B_shared[i]
    return _kernel


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
        add_shared_kernel(num_elements, block_size, threads, dtype),
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
    "atan2", "copysign", "hypot", "xlogy", "logaddexp",
    # logical
    "logical_and", "logical_or", "logical_xor",
    # bitwise
    "bitwise_and", "bitwise_or", "bitwise_xor", "shift_left", "shift_right",
    # special
    "add_shared",
]
