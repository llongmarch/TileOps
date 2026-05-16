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

import math
from typing import Any, Optional

import torch

from tileops.runtime import dispatch_compile
from tileops.runtime import (
    default_execution_backend,
    default_tilelang_target,
    default_torch_device,
    invoke_unary_kernel,
    suggest_tile_config,
    torch_to_tl_dtype,
    validate_single_input,
)


def _dispatch_unary(
    op_name: str,
    x: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> torch.Tensor:
    validate_single_input(x, out, op_name)
    tl_dtype = torch_to_tl_dtype(x.dtype)
    tgt = default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = default_execution_backend(tgt, dev)
    shape = x.shape
    bs, threads = suggest_tile_config(math.prod(shape))
    kernel = dispatch_compile(
        module_name="unary", op_name=op_name, target=tgt,
        shape=shape, block_size=bs, threads=threads,
        dtype=tl_dtype, execution_backend=eb,
        **kwargs,
    )
    return invoke_unary_kernel(kernel, x, out=out,
                               tilelang_target=tgt, execution_backend=eb)


# ── Exponential / logarithmic ─────────────────────────────────────────────


def exp(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``exp(x)`` — base-*e* exponential."""
    return _dispatch_unary("exp", x, out)


def log(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``log(x)`` — natural logarithm."""
    return _dispatch_unary("log", x, out)


def exp2(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``exp2(x)`` — base-2 exponential (2**x)."""
    return _dispatch_unary("exp2", x, out)


def exp10(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``exp10(x)`` — base-10 exponential (10**x)."""
    return _dispatch_unary("exp10", x, out)


def log2(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``log2(x)`` — base-2 logarithm."""
    return _dispatch_unary("log2", x, out)


def log10(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``log10(x)`` — base-10 logarithm."""
    return _dispatch_unary("log10", x, out)


def log1p(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``log1p(x)`` — log(1 + x), numerically stable for small x."""
    return _dispatch_unary("log1p", x, out)


# ── Trigonometry ────────────────────────────────────────────────────────────


def sin(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``sin(x)`` — element-wise sine."""
    return _dispatch_unary("sin", x, out)


def cos(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``cos(x)`` — element-wise cosine."""
    return _dispatch_unary("cos", x, out)


def tan(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``tan(x)`` — element-wise tangent."""
    return _dispatch_unary("tan", x, out)


# ── Inverse trigonometry ────────────────────────────────────────────────────


def asin(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``asin(x)`` — element-wise arc-sine."""
    return _dispatch_unary("asin", x, out)


def acos(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``acos(x)`` — element-wise arc-cosine."""
    return _dispatch_unary("acos", x, out)


def atan(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``atan(x)`` — element-wise arc-tangent."""
    return _dispatch_unary("atan", x, out)


# ── Hyperbolic ───────────────────────────────────────────────────────────────


def sinh(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``sinh(x)`` — element-wise hyperbolic sine."""
    return _dispatch_unary("sinh", x, out)


def cosh(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``cosh(x)`` — element-wise hyperbolic cosine."""
    return _dispatch_unary("cosh", x, out)


# ── Inverse hyperbolic ───────────────────────────────────────────────────────


def asinh(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``asinh(x)`` — element-wise inverse hyperbolic sine."""
    return _dispatch_unary("asinh", x, out)


def acosh(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``acosh(x)`` — element-wise inverse hyperbolic cosine."""
    return _dispatch_unary("acosh", x, out)


def atanh(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``atanh(x)`` — element-wise inverse hyperbolic tangent."""
    return _dispatch_unary("atanh", x, out)


# ── Power / root ──────────────────────────────────────────────────────────


def sqrt(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``sqrt(x)`` — square root."""
    return _dispatch_unary("sqrt", x, out)


def rsqrt(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``rsqrt(x)`` — reciprocal square root (1 / sqrt(x))."""
    return _dispatch_unary("rsqrt", x, out)


def square(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``square(x)`` — element-wise square (x * x)."""
    return _dispatch_unary("square", x, out)


# ── Error function ───────────────────────────────────────────────────────────


def erf(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``erf(x)`` — error function (Gauss)."""
    return _dispatch_unary("erf", x, out)


# ── Sign / absolute ───────────────────────────────────────────────────────


def abs(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``abs(x)`` — element-wise absolute value."""
    return _dispatch_unary("abs", x, out)


def sign(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``sign(x)`` — sign function (-1, 0, or 1)."""
    return _dispatch_unary("sign", x, out)


def neg(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``neg(x)`` — element-wise negation (-x)."""
    return _dispatch_unary("neg", x, out)


# ── Special-value detection ─────────────────────────────────────────────────


def isnan(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``isnan(x)`` — 1.0 where NaN, else 0.0."""
    return _dispatch_unary("isnan", x, out)


def isinf(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``isinf(x)`` — 1.0 where ±inf, else 0.0."""
    return _dispatch_unary("isinf", x, out)


def isfinite(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``isfinite(x)`` — 1.0 where finite, else 0.0."""
    return _dispatch_unary("isfinite", x, out)


# ── Rounding ──────────────────────────────────────────────────────────────


def round(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``round(x)`` — round to nearest integer."""
    return _dispatch_unary("round", x, out)


def floor(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``floor(x)`` — round down."""
    return _dispatch_unary("floor", x, out)


def ceil(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``ceil(x)`` — round up."""
    return _dispatch_unary("ceil", x, out)


def trunc(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``trunc(x)`` — truncate toward zero."""
    return _dispatch_unary("trunc", x, out)


# ── Reciprocal ────────────────────────────────────────────────────────────


def reciprocal(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``reciprocal(x)`` — element-wise inverse (1/x)."""
    return _dispatch_unary("reciprocal", x, out)


# ── Bitwise ─────────────────────────────────────────────────────────────────


def bitwise_not(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``bitwise_not(x)`` — element-wise bitwise NOT."""
    return _dispatch_unary("bitwise_not", x, out)


# ── Clamp ─────────────────────────────────────────────────────────────────


def clamp(
    x: torch.Tensor,
    min_val: float = 0.0,
    max_val: float = 1.0,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``clamp(x, min_val, max_val)`` — clamp values to ``[min_val, max_val]``.

    *min_val* and *max_val* are baked into the compiled kernel at compile time,
    so calling ``clamp`` with new threshold values triggers a re-compile.
    """
    return _dispatch_unary("clamp", x, out, min_val=float(min_val), max_val=float(max_val))


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
