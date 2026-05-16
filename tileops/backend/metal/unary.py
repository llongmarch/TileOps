"""Metal mathematical unary kernels (TileLang ``@T.prim_func``).

Mirrors :mod:`tileops.backend.cuda.unary` with a separate in-process cache prefix.
Parameter names ``arg0_*`` … preserve lexicographic Metal buffer order.

Covers:

  * **Exponential / logarithmic** — exp, log, exp2, exp10, log2, log10, log1p
  * **Trigonometry** — sin, cos, tan
  * **Inverse trigonometry** — asin, acos, atan
  * **Hyperbolic** — sinh, cosh
  * **Inverse hyperbolic** — asinh, acosh, atanh
  * **Power / root** — sqrt, rsqrt, square
  * **Error function** — erf
  * **Sign / absolute** — abs, sign, neg
  * **Special-value detection** — isnan, isinf, isfinite
  * **Rounding** — round, floor, ceil, trunc
  * **Reciprocal** — reciprocal
  * **Bitwise** — bitwise_not
  * **Clamping** — clamp
"""

from typing import Any

import tilelang.language as T
from tilelang.language.tir import op as tir_op

from tileops.runtime import (
    build_op,
    make_cached_compiler,
    make_generic_cached_compiler,
)

# ═══════════════════════════════════════════════════════════════════════════
# PrimFunc builders — Exponential / logarithmic
# ═══════════════════════════════════════════════════════════════════════════


def exp_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    Y[idx] = tir_op.exp(X[idx])
    return _kernel


def log_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    Y[idx] = tir_op.log(X[idx])
    return _kernel


def exp2_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    Y[idx] = tir_op.exp2(X[idx])
    return _kernel


def exp10_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    Y[idx] = tir_op.exp10(X[idx])
    return _kernel


def log2_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    Y[idx] = tir_op.log2(X[idx])
    return _kernel


def log10_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    Y[idx] = tir_op.log10(X[idx])
    return _kernel


def log1p_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    Y[idx] = tir_op.log(1.0 + X[idx])
    return _kernel


# ═══════════════════════════════════════════════════════════════════════════
# PrimFunc builders — Trigonometry
# ═══════════════════════════════════════════════════════════════════════════


def sin_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    Y[idx] = tir_op.sin(X[idx])
    return _kernel


def cos_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    Y[idx] = tir_op.cos(X[idx])
    return _kernel


def tan_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    Y[idx] = tir_op.call_pure_extern("float32", "tan", X[idx])
    return _kernel


# ═══════════════════════════════════════════════════════════════════════════
# PrimFunc builders — Inverse trigonometry
# ═══════════════════════════════════════════════════════════════════════════


def asin_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    Y[idx] = tir_op.call_pure_extern("float32", "asin", X[idx])
    return _kernel


def acos_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    Y[idx] = tir_op.call_pure_extern("float32", "acos", X[idx])
    return _kernel


def atan_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    Y[idx] = tir_op.call_pure_extern("float32", "atan", X[idx])
    return _kernel


# ═══════════════════════════════════════════════════════════════════════════
# PrimFunc builders — Hyperbolic
# ═══════════════════════════════════════════════════════════════════════════


def sinh_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    Y[idx] = tir_op.call_pure_extern("float32", "sinh", X[idx])
    return _kernel


def cosh_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    Y[idx] = tir_op.call_pure_extern("float32", "cosh", X[idx])
    return _kernel


# ═══════════════════════════════════════════════════════════════════════════
# PrimFunc builders — Inverse hyperbolic
# ═══════════════════════════════════════════════════════════════════════════


def asinh_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    Y[idx] = tir_op.call_pure_extern("float32", "asinh", X[idx])
    return _kernel


def acosh_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    Y[idx] = tir_op.call_pure_extern("float32", "acosh", X[idx])
    return _kernel


def atanh_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    Y[idx] = tir_op.call_pure_extern("float32", "atanh", X[idx])
    return _kernel


# ═══════════════════════════════════════════════════════════════════════════
# PrimFunc builders — Power / root
# ═══════════════════════════════════════════════════════════════════════════


def sqrt_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    Y[idx] = tir_op.sqrt(X[idx])
    return _kernel


def rsqrt_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    Y[idx] = 1.0 / tir_op.sqrt(X[idx])
    return _kernel


def square_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    x = X[idx]
                    Y[idx] = x * x
    return _kernel


# ═══════════════════════════════════════════════════════════════════════════
# PrimFunc builders — Error function
# ═══════════════════════════════════════════════════════════════════════════


def erf_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    x = X[idx]
                    # Metal lacks native erf. Use tanh-based approximation:
                    #   erf(x) ≈ tanh(2*x/sqrt(pi) * (1 + 0.044715*x^2))
                    # (same approximation as GELU's erf variant)
                    Y[idx] = tir_op.tanh(1.1283791670955126 * x * (1.0 + 0.044715 * x * x))
    return _kernel


# ═══════════════════════════════════════════════════════════════════════════
# PrimFunc builders — Sign / absolute
# ═══════════════════════════════════════════════════════════════════════════


def abs_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    Y[idx] = tir_op.abs(X[idx])
    return _kernel


def sign_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    x = X[idx]
                    s = 0.0
                    if x > 0.0:
                        s = 1.0
                    elif x < 0.0:
                        s = -1.0
                    Y[idx] = s
    return _kernel


def neg_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    Y[idx] = -X[idx]
    return _kernel


# ═══════════════════════════════════════════════════════════════════════════
# PrimFunc builders — Special-value detection
# ═══════════════════════════════════════════════════════════════════════════


def isnan_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    if tir_op.isnan(X[idx]):
                        Y[idx] = 1.0
                    else:
                        Y[idx] = 0.0
    return _kernel


def isinf_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    if tir_op.isinf(X[idx]):
                        Y[idx] = 1.0
                    else:
                        Y[idx] = 0.0
    return _kernel


def isfinite_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    if tir_op.isfinite(X[idx]):
                        Y[idx] = 1.0
                    else:
                        Y[idx] = 0.0
    return _kernel


# ═══════════════════════════════════════════════════════════════════════════
# PrimFunc builders — Rounding
# ═══════════════════════════════════════════════════════════════════════════


def round_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    Y[idx] = tir_op.round(X[idx])
    return _kernel


def floor_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    Y[idx] = tir_op.floor(X[idx])
    return _kernel


def ceil_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    Y[idx] = tir_op.ceil(X[idx])
    return _kernel


def trunc_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    Y[idx] = tir_op.call_pure_extern("float32", "trunc", X[idx])
    return _kernel


# ═══════════════════════════════════════════════════════════════════════════
# PrimFunc builders — Reciprocal
# ═══════════════════════════════════════════════════════════════════════════


def reciprocal_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    Y[idx] = 1.0 / X[idx]
    return _kernel


# ═══════════════════════════════════════════════════════════════════════════
# PrimFunc builders — Bitwise
# ═══════════════════════════════════════════════════════════════════════════


def bitwise_not_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    Y[idx] = tir_op.bitwise_not(X[idx])
    return _kernel


# ═══════════════════════════════════════════════════════════════════════════
# PrimFunc builders — Clamp
# ═══════════════════════════════════════════════════════════════════════════


def clamp_kernel(n: int, bs: int, t: int, dtype: Any, *, min_val: float, max_val: float) -> Any:
    lo = float(min_val)
    hi = float(max_val)

    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    x = X[idx]
                    Y[idx] = 0.5 * (
                        tir_op.abs(x - lo) - tir_op.abs(x - hi)
                    ) + 0.5 * (lo + hi)
    return _kernel


# ═══════════════════════════════════════════════════════════════════════════
# Cached compiler + public API
# ═══════════════════════════════════════════════════════════════════════════

_UNARY_PRIM: dict[str, Any] = {
    # exponential / logarithmic
    "exp": exp_kernel, "log": log_kernel,
    "exp2": exp2_kernel, "exp10": exp10_kernel,
    "log2": log2_kernel, "log10": log10_kernel, "log1p": log1p_kernel,
    # trigonometry
    "sin": sin_kernel, "cos": cos_kernel, "tan": tan_kernel,
    # inverse trigonometry
    "asin": asin_kernel, "acos": acos_kernel, "atan": atan_kernel,
    # hyperbolic
    "sinh": sinh_kernel, "cosh": cosh_kernel,
    # inverse hyperbolic
    "asinh": asinh_kernel, "acosh": acosh_kernel, "atanh": atanh_kernel,
    # power / root
    "sqrt": sqrt_kernel, "rsqrt": rsqrt_kernel, "square": square_kernel,
    # error function
    "erf": erf_kernel,
    # sign / absolute
    "abs": abs_kernel, "sign": sign_kernel, "neg": neg_kernel,
    # special-value detection
    "isnan": isnan_kernel, "isinf": isinf_kernel, "isfinite": isfinite_kernel,
    # rounding
    "round": round_kernel, "floor": floor_kernel, "ceil": ceil_kernel,
    "trunc": trunc_kernel,
    # reciprocal
    "reciprocal": reciprocal_kernel,
    # bitwise
    "bitwise_not": bitwise_not_kernel,
}

_compile = make_cached_compiler(_UNARY_PRIM, cache_prefix="metal_unary")

# -- exponential / logarithmic -------------------------------------------
exp = build_op("exp", compile_fn=_compile, doc="Metal: ``exp(x)``.")
log = build_op("log", compile_fn=_compile, doc="Metal: ``log(x)`` (natural log).")
exp2 = build_op("exp2", compile_fn=_compile, doc="Metal: ``exp2(x)`` (2**x).")
exp10 = build_op("exp10", compile_fn=_compile, doc="Metal: ``exp10(x)`` (10**x).")
log2 = build_op("log2", compile_fn=_compile, doc="Metal: ``log2(x)``.")
log10 = build_op("log10", compile_fn=_compile, doc="Metal: ``log10(x)``.")
log1p = build_op("log1p", compile_fn=_compile, doc="Metal: ``log1p(x)`` (log(1+x)).")

# -- trigonometry ---------------------------------------------------------
sin = build_op("sin", compile_fn=_compile, doc="Metal: ``sin(x)``.")
cos = build_op("cos", compile_fn=_compile, doc="Metal: ``cos(x)``.")
tan = build_op("tan", compile_fn=_compile, doc="Metal: ``tan(x)``.")

# -- inverse trigonometry -------------------------------------------------
asin = build_op("asin", compile_fn=_compile, doc="Metal: ``asin(x)``.")
acos = build_op("acos", compile_fn=_compile, doc="Metal: ``acos(x)``.")
atan = build_op("atan", compile_fn=_compile, doc="Metal: ``atan(x)``.")

# -- hyperbolic -----------------------------------------------------------
sinh = build_op("sinh", compile_fn=_compile, doc="Metal: ``sinh(x)``.")
cosh = build_op("cosh", compile_fn=_compile, doc="Metal: ``cosh(x)``.")

# -- inverse hyperbolic ---------------------------------------------------
asinh = build_op("asinh", compile_fn=_compile, doc="Metal: ``asinh(x)``.")
acosh = build_op("acosh", compile_fn=_compile, doc="Metal: ``acosh(x)``.")
atanh = build_op("atanh", compile_fn=_compile, doc="Metal: ``atanh(x)``.")

# -- power / root --------------------------------------------------------
sqrt = build_op("sqrt", compile_fn=_compile, doc="Metal: ``sqrt(x)``.")
rsqrt = build_op("rsqrt", compile_fn=_compile, doc="Metal: ``rsqrt(x)`` (1/sqrt(x)).")
square = build_op("square", compile_fn=_compile, doc="Metal: ``square(x)`` (x*x).")

# -- error function ------------------------------------------------------
erf = build_op("erf", compile_fn=_compile, doc="Metal: ``erf(x)``.")

# -- sign / absolute -----------------------------------------------------
abs = build_op("abs", compile_fn=_compile, doc="Metal: ``abs(x)``.")
sign = build_op("sign", compile_fn=_compile, doc="Metal: ``sign(x)`` (-1, 0, or 1).")
neg = build_op("neg", compile_fn=_compile, doc="Metal: ``neg(x)`` (-x).")

# -- special-value detection ----------------------------------------------
isnan = build_op("isnan", compile_fn=_compile, doc="Metal: ``isnan(x)`` (1.0 if NaN).")
isinf = build_op("isinf", compile_fn=_compile, doc="Metal: ``isinf(x)`` (1.0 if ±inf).")
isfinite = build_op("isfinite", compile_fn=_compile, doc="Metal: ``isfinite(x)`` (1.0 if finite).")

# -- rounding ------------------------------------------------------------
round = build_op("round", compile_fn=_compile, doc="Metal: ``round(x)`` (nearest integer).")
floor = build_op("floor", compile_fn=_compile, doc="Metal: ``floor(x)``.")
ceil = build_op("ceil", compile_fn=_compile, doc="Metal: ``ceil(x)``.")
trunc = build_op("trunc", compile_fn=_compile, doc="Metal: ``trunc(x)`` (toward zero).")

# -- reciprocal ----------------------------------------------------------
reciprocal = build_op("reciprocal", compile_fn=_compile, doc="Metal: ``reciprocal(x)`` (1/x).")

# -- bitwise --------------------------------------------------------------
bitwise_not = build_op("bitwise_not", compile_fn=_compile, doc="Metal: ``bitwise_not(x)``.")

# -- clamp (special: compile-time min/max) -------------------------------

_clamp_compile = make_generic_cached_compiler(cache_prefix="metal_unary_clamp")


def clamp(
    shape: tuple[int, ...],
    block_size: int,
    threads: int,
    *,
    dtype: Any,
    target: str = None,
    execution_backend: str = None,
    min_val: float = 0.0,
    max_val: float = 1.0,
) -> Any:
    """Compile a clamp kernel (bakes *min_val* and *max_val* at compile time)."""
    n = 1
    for s in shape:
        n *= s
    lo, hi = float(min_val), float(max_val)
    return _clamp_compile(
        ("clamp", n, block_size, threads, dtype, lo, hi),
        clamp_kernel,
        n, block_size, threads, dtype,
        target=target,
        execution_backend=execution_backend,
        min_val=lo, max_val=hi,
    )


__all__ = [
    # exponential / logarithmic
    "exp", "log", "exp2", "exp10", "log2", "log10", "log1p",
    # trigonometry
    "sin", "cos", "tan",
    # inverse trigonometry
    "asin", "acos", "atan",
    # hyperbolic
    "sinh", "cosh",
    # inverse hyperbolic
    "asinh", "acosh", "atanh",
    # power / root
    "sqrt", "rsqrt", "square",
    # error function
    "erf",
    # sign / absolute
    "abs", "sign", "neg",
    # special-value detection
    "isnan", "isinf", "isfinite",
    # rounding
    "round", "floor", "ceil", "trunc",
    # reciprocal
    "reciprocal",
    # bitwise
    "bitwise_not",
    # clamp
    "clamp",
]
