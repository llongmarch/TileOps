"""Mathematical unary operators.

These are pure-math element-wise operators beyond the activation family
(:mod:`tileops.activation`):

  * **Exponential / logarithmic** — :func:`exp`, :func:`log`
  * **Power / root** — :func:`sqrt`, :func:`rsqrt`, :func:`square`
  * **Sign / absolute** — :func:`abs`, :func:`sign`, :func:`neg`
  * **Rounding** — :func:`round`, :func:`floor`, :func:`ceil`
  * **Arithmetic** — :func:`reciprocal`
  * **Clamping** — :func:`clamp`

Each function takes one :class:`torch.Tensor` input and returns a
:class:`torch.Tensor`.  Kernel compilation, caching, and backend dispatch
are transparent.  The dtype is inferred from the input tensor.

Usage::

    from tileops import exp, log, sqrt, rsqrt, abs, neg, clamp
    y = exp(x)
    z = clamp(x, min_val=-1.0, max_val=1.0)
"""

from __future__ import annotations

import importlib
import math
from typing import Any, Optional

import torch

from tileops.runtime import (
    BACKEND_PACKAGES,
    dispatch_compile,
    default_execution_backend,
    default_search_space,
    default_tilelang_target,
    default_torch_device,
    invoke_unary_kernel,
    search_best_config,
    suggest_tile_config,
    target_kind,
    torch_to_tl_dtype,
    validate_single_input,
)


def _get_backend_prim_builder(target: str, module_name: str, op_name: str):
    """Resolve and return the kernel builder for *op_name* from the backend module."""
    kind = target_kind(target)
    pkg = BACKEND_PACKAGES.get(kind)
    if pkg is None:
        raise NotImplementedError(
            f"No {module_name} backend for target kind {kind!r} ({target!r})."
        )
    mod = importlib.import_module(f"{pkg}.{module_name}")
    builder = mod._UNARY_PRIM.get(op_name)  # type: ignore[attr-defined]
    if builder is None:
        raise NotImplementedError(
            f"{pkg}.{module_name} has no _UNARY_PRIM entry for {op_name!r}"
        )
    return builder


def _dispatch_unary(
    op_name: str,
    x: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    autotune: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    validate_single_input(x, out, op_name)
    tl_dtype = torch_to_tl_dtype(x.dtype)
    tgt = default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = default_execution_backend(tgt, dev)
    shape = x.shape
    n = math.prod(shape)

    if autotune and op_name != "clamp":
        prim_builder = _get_backend_prim_builder(tgt, "unary", op_name)
        config_space = default_search_space(n)

        def builder_args_fn(config: dict[str, int]) -> tuple:
            return (n, config["block_size"], config["threads"], tl_dtype)

        best = search_best_config(
            op_name=op_name,
            prim_builder=prim_builder,
            builder_args_fn=builder_args_fn,
            config_space=config_space,
            num_elements=n,
            dtype=tl_dtype,
            target=tgt,
            execution_backend=eb,
        )
        bs, threads = best["block_size"], best["threads"]
    elif autotune and op_name == "clamp":
        raise NotImplementedError(
            "Autotune for clamp is not yet supported (requires baking min_val/max_val "
            "into the search).  Use autotune=False and a heuristic config."
        )
    else:
        bs, threads = suggest_tile_config(n)

    kernel = dispatch_compile(
        module_name="unary", op_name=op_name, target=tgt,
        shape=shape, block_size=bs, threads=threads,
        dtype=tl_dtype, execution_backend=eb,
        **kwargs,
    )
    return invoke_unary_kernel(kernel, x, out=out,
                               tilelang_target=tgt, execution_backend=eb)


# ── Exponential / logarithmic ─────────────────────────────────────────────


def exp(x: torch.Tensor, *, out: Optional[torch.Tensor] = None, autotune: bool = False) -> torch.Tensor:
    """``exp(x)`` — base-*e* exponential."""
    return _dispatch_unary("exp", x, out, autotune=autotune)


def log(x: torch.Tensor, *, out: Optional[torch.Tensor] = None, autotune: bool = False) -> torch.Tensor:
    """``log(x)`` — natural logarithm."""
    return _dispatch_unary("log", x, out, autotune=autotune)


def exp2(x: torch.Tensor, *, out: Optional[torch.Tensor] = None, autotune: bool = False) -> torch.Tensor:
    """``exp2(x)`` — base-2 exponential (2**x)."""
    return _dispatch_unary("exp2", x, out, autotune=autotune)


def exp10(x: torch.Tensor, *, out: Optional[torch.Tensor] = None, autotune: bool = False) -> torch.Tensor:
    """``exp10(x)`` — base-10 exponential (10**x)."""
    return _dispatch_unary("exp10", x, out, autotune=autotune)


def log2(x: torch.Tensor, *, out: Optional[torch.Tensor] = None, autotune: bool = False) -> torch.Tensor:
    """``log2(x)`` — base-2 logarithm."""
    return _dispatch_unary("log2", x, out, autotune=autotune)


def log10(x: torch.Tensor, *, out: Optional[torch.Tensor] = None, autotune: bool = False) -> torch.Tensor:
    """``log10(x)`` — base-10 logarithm."""
    return _dispatch_unary("log10", x, out, autotune=autotune)


def log1p(x: torch.Tensor, *, out: Optional[torch.Tensor] = None, autotune: bool = False) -> torch.Tensor:
    """``log1p(x)`` — log(1 + x), numerically stable for small x."""
    return _dispatch_unary("log1p", x, out, autotune=autotune)


# ── Trigonometry ────────────────────────────────────────────────────────────


def sin(x: torch.Tensor, *, out: Optional[torch.Tensor] = None, autotune: bool = False) -> torch.Tensor:
    """``sin(x)`` — element-wise sine."""
    return _dispatch_unary("sin", x, out, autotune=autotune)


def cos(x: torch.Tensor, *, out: Optional[torch.Tensor] = None, autotune: bool = False) -> torch.Tensor:
    """``cos(x)`` — element-wise cosine."""
    return _dispatch_unary("cos", x, out, autotune=autotune)


def tan(x: torch.Tensor, *, out: Optional[torch.Tensor] = None, autotune: bool = False) -> torch.Tensor:
    """``tan(x)`` — element-wise tangent."""
    return _dispatch_unary("tan", x, out, autotune=autotune)


# ── Inverse trigonometry ────────────────────────────────────────────────────


def asin(x: torch.Tensor, *, out: Optional[torch.Tensor] = None, autotune: bool = False) -> torch.Tensor:
    """``asin(x)`` — element-wise arc-sine."""
    return _dispatch_unary("asin", x, out, autotune=autotune)


def acos(x: torch.Tensor, *, out: Optional[torch.Tensor] = None, autotune: bool = False) -> torch.Tensor:
    """``acos(x)`` — element-wise arc-cosine."""
    return _dispatch_unary("acos", x, out, autotune=autotune)


def atan(x: torch.Tensor, *, out: Optional[torch.Tensor] = None, autotune: bool = False) -> torch.Tensor:
    """``atan(x)`` — element-wise arc-tangent."""
    return _dispatch_unary("atan", x, out, autotune=autotune)


# ── Hyperbolic ───────────────────────────────────────────────────────────────


def sinh(x: torch.Tensor, *, out: Optional[torch.Tensor] = None, autotune: bool = False) -> torch.Tensor:
    """``sinh(x)`` — element-wise hyperbolic sine."""
    return _dispatch_unary("sinh", x, out, autotune=autotune)


def cosh(x: torch.Tensor, *, out: Optional[torch.Tensor] = None, autotune: bool = False) -> torch.Tensor:
    """``cosh(x)`` — element-wise hyperbolic cosine."""
    return _dispatch_unary("cosh", x, out, autotune=autotune)


# ── Inverse hyperbolic ───────────────────────────────────────────────────────


def asinh(x: torch.Tensor, *, out: Optional[torch.Tensor] = None, autotune: bool = False) -> torch.Tensor:
    """``asinh(x)`` — element-wise inverse hyperbolic sine."""
    return _dispatch_unary("asinh", x, out, autotune=autotune)


def acosh(x: torch.Tensor, *, out: Optional[torch.Tensor] = None, autotune: bool = False) -> torch.Tensor:
    """``acosh(x)`` — element-wise inverse hyperbolic cosine."""
    return _dispatch_unary("acosh", x, out, autotune=autotune)


def atanh(x: torch.Tensor, *, out: Optional[torch.Tensor] = None, autotune: bool = False) -> torch.Tensor:
    """``atanh(x)`` — element-wise inverse hyperbolic tangent."""
    return _dispatch_unary("atanh", x, out, autotune=autotune)


# ── Power / root ──────────────────────────────────────────────────────────


def sqrt(x: torch.Tensor, *, out: Optional[torch.Tensor] = None, autotune: bool = False) -> torch.Tensor:
    """``sqrt(x)`` — square root."""
    return _dispatch_unary("sqrt", x, out, autotune=autotune)


def rsqrt(x: torch.Tensor, *, out: Optional[torch.Tensor] = None, autotune: bool = False) -> torch.Tensor:
    """``rsqrt(x)`` — reciprocal square root (1 / sqrt(x))."""
    return _dispatch_unary("rsqrt", x, out, autotune=autotune)


def square(x: torch.Tensor, *, out: Optional[torch.Tensor] = None, autotune: bool = False) -> torch.Tensor:
    """``square(x)`` — element-wise square (x * x)."""
    return _dispatch_unary("square", x, out, autotune=autotune)


# ── Error function ───────────────────────────────────────────────────────────


def erf(x: torch.Tensor, *, out: Optional[torch.Tensor] = None, autotune: bool = False) -> torch.Tensor:
    """``erf(x)`` — error function (Gauss)."""
    return _dispatch_unary("erf", x, out, autotune=autotune)


# ── Sign / absolute ───────────────────────────────────────────────────────


def abs(x: torch.Tensor, *, out: Optional[torch.Tensor] = None, autotune: bool = False) -> torch.Tensor:
    """``abs(x)`` — element-wise absolute value."""
    return _dispatch_unary("abs", x, out, autotune=autotune)


def sign(x: torch.Tensor, *, out: Optional[torch.Tensor] = None, autotune: bool = False) -> torch.Tensor:
    """``sign(x)`` — sign function (-1, 0, or 1)."""
    return _dispatch_unary("sign", x, out, autotune=autotune)


def neg(x: torch.Tensor, *, out: Optional[torch.Tensor] = None, autotune: bool = False) -> torch.Tensor:
    """``neg(x)`` — element-wise negation (-x)."""
    return _dispatch_unary("neg", x, out, autotune=autotune)


# ── Special-value detection ─────────────────────────────────────────────────


def isnan(x: torch.Tensor, *, out: Optional[torch.Tensor] = None, autotune: bool = False) -> torch.Tensor:
    """``isnan(x)`` — 1.0 where NaN, else 0.0."""
    return _dispatch_unary("isnan", x, out, autotune=autotune)


def isinf(x: torch.Tensor, *, out: Optional[torch.Tensor] = None, autotune: bool = False) -> torch.Tensor:
    """``isinf(x)`` — 1.0 where ±inf, else 0.0."""
    return _dispatch_unary("isinf", x, out, autotune=autotune)


def isfinite(x: torch.Tensor, *, out: Optional[torch.Tensor] = None, autotune: bool = False) -> torch.Tensor:
    """``isfinite(x)`` — 1.0 where finite, else 0.0."""
    return _dispatch_unary("isfinite", x, out, autotune=autotune)


# ── Rounding ──────────────────────────────────────────────────────────────


def round(x: torch.Tensor, *, out: Optional[torch.Tensor] = None, autotune: bool = False) -> torch.Tensor:
    """``round(x)`` — round to nearest integer."""
    return _dispatch_unary("round", x, out, autotune=autotune)


def floor(x: torch.Tensor, *, out: Optional[torch.Tensor] = None, autotune: bool = False) -> torch.Tensor:
    """``floor(x)`` — round down."""
    return _dispatch_unary("floor", x, out, autotune=autotune)


def ceil(x: torch.Tensor, *, out: Optional[torch.Tensor] = None, autotune: bool = False) -> torch.Tensor:
    """``ceil(x)`` — round up."""
    return _dispatch_unary("ceil", x, out, autotune=autotune)


def trunc(x: torch.Tensor, *, out: Optional[torch.Tensor] = None, autotune: bool = False) -> torch.Tensor:
    """``trunc(x)`` — truncate toward zero."""
    return _dispatch_unary("trunc", x, out, autotune=autotune)


# ── Reciprocal ────────────────────────────────────────────────────────────


def reciprocal(x: torch.Tensor, *, out: Optional[torch.Tensor] = None, autotune: bool = False) -> torch.Tensor:
    """``reciprocal(x)`` — element-wise inverse (1/x)."""
    return _dispatch_unary("reciprocal", x, out, autotune=autotune)


# ── Bitwise ─────────────────────────────────────────────────────────────────


def bitwise_not(x: torch.Tensor, *, out: Optional[torch.Tensor] = None, autotune: bool = False) -> torch.Tensor:
    """``bitwise_not(x)`` — element-wise bitwise NOT."""
    return _dispatch_unary("bitwise_not", x, out, autotune=autotune)


# ── Clamp ─────────────────────────────────────────────────────────────────


def clamp(
    x: torch.Tensor,
    min_val: float = 0.0,
    max_val: float = 1.0,
    *,
    out: Optional[torch.Tensor] = None,
    autotune: bool = False,
) -> torch.Tensor:
    """``clamp(x, min_val, max_val)`` — clamp values to ``[min_val, max_val]``.

    *min_val* and *max_val* are baked into the compiled kernel at compile time,
    so calling ``clamp`` with new threshold values triggers a re-compile.
    """
    return _dispatch_unary("clamp", x, out, autotune=autotune, min_val=float(min_val), max_val=float(max_val))


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
