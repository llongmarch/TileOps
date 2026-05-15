"""Metal mathematical unary kernels (TileLang ``@T.prim_func``).

Mirrors :mod:`tileops.cuda.unary` with a separate in-process cache prefix.
Parameter names ``arg0_*`` … preserve lexicographic Metal buffer order.

Covers:

  * **Exponential / logarithmic** — exp, log
  * **Power / root** — sqrt, rsqrt, square
  * **Sign / absolute** — abs, sign, neg
  * **Rounding** — round, floor, ceil
  * **Reciprocal** — reciprocal
  * **Clamping** — clamp
"""

from typing import Any

import tilelang.language as T
from tilelang.language.tir import op as tir_op

from tileops._infra import make_backend_op, make_cached_compiler

# ═══════════════════════════════════════════════════════════════════════════
# PrimFunc builders — Exponential / logarithmic
# ═══════════════════════════════════════════════════════════════════════════


def _exp_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    Y[idx] = tir_op.exp(X[idx])
    return main


def _log_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    Y[idx] = tir_op.log(X[idx])
    return main


# ═══════════════════════════════════════════════════════════════════════════
# PrimFunc builders — Power / root
# ═══════════════════════════════════════════════════════════════════════════


def _sqrt_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    Y[idx] = tir_op.sqrt(X[idx])
    return main


def _rsqrt_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    Y[idx] = 1.0 / tir_op.sqrt(X[idx])
    return main


def _square_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    x = X[idx]
                    Y[idx] = x * x
    return main


# ═══════════════════════════════════════════════════════════════════════════
# PrimFunc builders — Sign / absolute
# ═══════════════════════════════════════════════════════════════════════════


def _abs_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    Y[idx] = tir_op.abs(X[idx])
    return main


def _sign_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
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
    return main


def _neg_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    Y[idx] = -X[idx]
    return main


# ═══════════════════════════════════════════════════════════════════════════
# PrimFunc builders — Rounding
# ═══════════════════════════════════════════════════════════════════════════


def _round_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    Y[idx] = tir_op.round(X[idx])
    return main


def _floor_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    Y[idx] = tir_op.floor(X[idx])
    return main


def _ceil_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    Y[idx] = tir_op.ceil(X[idx])
    return main


# ═══════════════════════════════════════════════════════════════════════════
# PrimFunc builders — Reciprocal
# ═══════════════════════════════════════════════════════════════════════════


def _reciprocal_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    Y[idx] = 1.0 / X[idx]
    return main


# ═══════════════════════════════════════════════════════════════════════════
# PrimFunc builders — Clamp
# ═══════════════════════════════════════════════════════════════════════════


def _clamp_prim(n: int, bs: int, t: int, dtype: Any, *, min_val: float, max_val: float) -> Any:
    lo = float(min_val)
    hi = float(max_val)

    @T.prim_func
    def main(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    x = X[idx]
                    Y[idx] = 0.5 * (
                        tir_op.abs(x - lo) - tir_op.abs(x - hi)
                    ) + 0.5 * (lo + hi)
    return main


# ═══════════════════════════════════════════════════════════════════════════
# Cached compiler + public API
# ═══════════════════════════════════════════════════════════════════════════

_UNARY_PRIM: dict[str, Any] = {
    "exp": _exp_prim, "log": _log_prim,
    "sqrt": _sqrt_prim, "rsqrt": _rsqrt_prim, "square": _square_prim,
    "abs": _abs_prim, "sign": _sign_prim, "neg": _neg_prim,
    "round": _round_prim, "floor": _floor_prim, "ceil": _ceil_prim,
    "reciprocal": _reciprocal_prim,
}

_compile = make_cached_compiler(_UNARY_PRIM, cache_prefix="metal_unary")

# -- exponential / logarithmic -------------------------------------------
exp = make_backend_op("exp", compile_fn=_compile, doc="Metal: ``exp(x)``.")
log = make_backend_op("log", compile_fn=_compile, doc="Metal: ``log(x)`` (natural log).")

# -- power / root --------------------------------------------------------
sqrt = make_backend_op("sqrt", compile_fn=_compile, doc="Metal: ``sqrt(x)``.")
rsqrt = make_backend_op("rsqrt", compile_fn=_compile, doc="Metal: ``rsqrt(x)`` (1/sqrt(x)).")
square = make_backend_op("square", compile_fn=_compile, doc="Metal: ``square(x)`` (x*x).")

# -- sign / absolute -----------------------------------------------------
abs = make_backend_op("abs", compile_fn=_compile, doc="Metal: ``abs(x)``.")
sign = make_backend_op("sign", compile_fn=_compile, doc="Metal: ``sign(x)`` (-1, 0, or 1).")
neg = make_backend_op("neg", compile_fn=_compile, doc="Metal: ``neg(x)`` (-x).")

# -- rounding ------------------------------------------------------------
round = make_backend_op("round", compile_fn=_compile, doc="Metal: ``round(x)`` (nearest integer).")
floor = make_backend_op("floor", compile_fn=_compile, doc="Metal: ``floor(x)``.")
ceil = make_backend_op("ceil", compile_fn=_compile, doc="Metal: ``ceil(x)``.")

# -- reciprocal ----------------------------------------------------------
reciprocal = make_backend_op("reciprocal", compile_fn=_compile, doc="Metal: ``reciprocal(x)`` (1/x).")

# -- clamp (special: compile-time min/max) -------------------------------

_clamp_cache: dict[tuple, Any] = {}
from tileops.runtime import compile_prim, default_tilelang_target


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
    tgt = target if target is not None else default_tilelang_target()
    lo, hi = float(min_val), float(max_val)
    key = ("metal_clamp_v1", n, block_size, threads, dtype, tgt, execution_backend, lo, hi)
    hit = _clamp_cache.get(key)
    if hit is not None:
        return hit
    kernel = compile_prim(
        _clamp_prim(n, block_size, threads, dtype, min_val=lo, max_val=hi),
        target=tgt,
        execution_backend=execution_backend,
    )
    _clamp_cache[key] = kernel
    return kernel


__all__ = [
    "exp", "log",
    "sqrt", "rsqrt", "square",
    "abs", "sign", "neg",
    "round", "floor", "ceil",
    "reciprocal",
    "clamp",
]
