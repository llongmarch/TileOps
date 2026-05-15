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
from typing import Optional

import torch

from tileops._infra import dispatch_compile
from tileops.runtime import (
    default_execution_backend,
    default_tilelang_target,
    default_torch_device,
    invoke_unary_kernel,
    suggest_tile_config,
    target_kind,
    torch_to_tl_dtype,
)


def _validate_unary(
    x: torch.Tensor, out: Optional[torch.Tensor], op: str,
) -> None:
    if x.numel() == 0:
        raise ValueError(f"{op}: input tensor is empty")
    if out is not None:
        if out.shape != x.shape:
            raise ValueError(
                f"{op}: out.shape={out.shape} != input shape {x.shape}"
            )
        if out.dtype != x.dtype:
            raise TypeError(
                f"{op}: out.dtype={out.dtype} != input dtype {x.dtype}"
            )
        if out.device != x.device:
            raise ValueError(
                f"{op}: out.device={out.device} != input device {x.device}"
            )


def _run_unary(
    op_name: str,
    x: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    _validate_unary(x, out, op_name)
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
    )
    return invoke_unary_kernel(kernel, x, out=out,
                               tilelang_target=tgt, execution_backend=eb)


# ── Exponential / logarithmic ─────────────────────────────────────────────


def exp(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``exp(x)`` — base-*e* exponential."""
    return _run_unary("exp", x, out)


def log(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``log(x)`` — natural logarithm."""
    return _run_unary("log", x, out)


# ── Power / root ──────────────────────────────────────────────────────────


def sqrt(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``sqrt(x)`` — square root."""
    return _run_unary("sqrt", x, out)


def rsqrt(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``rsqrt(x)`` — reciprocal square root (1 / sqrt(x))."""
    return _run_unary("rsqrt", x, out)


def square(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``square(x)`` — element-wise square (x * x)."""
    return _run_unary("square", x, out)


# ── Sign / absolute ───────────────────────────────────────────────────────


def abs(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``abs(x)`` — element-wise absolute value."""
    return _run_unary("abs", x, out)


def sign(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``sign(x)`` — sign function (-1, 0, or 1)."""
    return _run_unary("sign", x, out)


def neg(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``neg(x)`` — element-wise negation (-x)."""
    return _run_unary("neg", x, out)


# ── Rounding ──────────────────────────────────────────────────────────────


def round(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``round(x)`` — round to nearest integer."""
    return _run_unary("round", x, out)


def floor(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``floor(x)`` — round down."""
    return _run_unary("floor", x, out)


def ceil(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``ceil(x)`` — round up."""
    return _run_unary("ceil", x, out)


# ── Reciprocal ────────────────────────────────────────────────────────────


def reciprocal(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``reciprocal(x)`` — element-wise inverse (1/x)."""
    return _run_unary("reciprocal", x, out)


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
    _validate_unary(x, out, "clamp")
    tl_dtype = torch_to_tl_dtype(x.dtype)
    tgt = default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = default_execution_backend(tgt, dev)
    shape = x.shape
    n = math.prod(shape)
    bs, threads = suggest_tile_config(n)

    # Clamp has a special backend signature (extra min_val / max_val).
    kind = target_kind(tgt)
    from tileops._infra import BACKEND_PACKAGES
    pkg = BACKEND_PACKAGES.get(kind)
    if pkg is None:
        raise NotImplementedError(
            f"No unary backend for target kind {kind!r} ({tgt!r})"
        )
    import importlib
    try:
        mod = importlib.import_module(f"{pkg}.unary")
    except ImportError as exc:
        raise ImportError(
            f"Failed to import {pkg}.unary for target {tgt!r}"
        ) from exc

    clamp_fn = getattr(mod, "clamp", None)
    if clamp_fn is None:
        raise NotImplementedError(
            f"{pkg}.unary has no clamp() for {tgt!r}"
        )

    kernel = clamp_fn(
        shape, bs, threads,
        dtype=tl_dtype,
        target=tgt,
        execution_backend=eb,
        min_val=float(min_val),
        max_val=float(max_val),
    )
    return invoke_unary_kernel(kernel, x, out=out,
                               tilelang_target=tgt, execution_backend=eb)


__all__ = [
    "exp", "log",
    "sqrt", "rsqrt", "square",
    "abs", "sign", "neg",
    "round", "floor", "ceil",
    "reciprocal",
    "clamp",
]
