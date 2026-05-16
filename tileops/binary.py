"""Binary element-wise operators.

Each function takes two :class:`torch.Tensor` inputs and returns a
:class:`torch.Tensor` — just like ``torch.add``, ``torch.sub``, etc.
Kernel compilation, caching, and backend dispatch are handled
transparently.  The dtype is inferred from the input tensors.

**Arithmetic** — :func:`add`, :func:`sub`, :func:`mul`, :func:`div`,
:func:`pow`, :func:`fmod`, :func:`remainder`, :func:`floor_div`

**Extrema** — :func:`maximum`, :func:`minimum`

**Comparison** (output 0.0 / 1.0 in input dtype) — :func:`eq`, :func:`ne`,
:func:`gt`, :func:`ge`, :func:`lt`, :func:`le`

**Math** — :func:`atan2`, :func:`copysign`, :func:`hypot`, :func:`xlogy`

**Logical** (non-zero = True; output 0.0 / 1.0) — :func:`logical_and`,
:func:`logical_or`, :func:`logical_xor`

::

    from tileops import add, maximum, gt
    out = add(x, y)
    mx  = maximum(x, y)
    mask = gt(x, y)        # float tensor of 0s and 1s
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from tileops.runtime import dispatch_compile
from tileops.runtime import (
    default_execution_backend,
    default_tilelang_target,
    default_torch_device,
    invoke_kernel,
    suggest_tile_config,
    torch_to_tl_dtype,
)


def _validate_binary(
    x: torch.Tensor, y: torch.Tensor, out: Optional[torch.Tensor], op: str,
) -> None:
    if x.shape != y.shape:
        raise ValueError(
            f"{op}: shape mismatch — x.shape={x.shape}, y.shape={y.shape}"
        )
    if x.dtype != y.dtype:
        raise TypeError(
            f"{op}: dtype mismatch — x.dtype={x.dtype}, y.dtype={y.dtype}"
        )
    if x.device != y.device:
        raise ValueError(
            f"{op}: device mismatch — x.device={x.device}, y.device={y.device}"
        )
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


def _dispatch_binary(
    op_name: str,
    x: torch.Tensor,
    y: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    _validate_binary(x, y, out, op_name)
    tl_dtype = torch_to_tl_dtype(x.dtype)
    tgt = default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = default_execution_backend(tgt, dev)
    shape = x.shape
    bs, threads = suggest_tile_config(math.prod(shape))
    kernel = dispatch_compile(
        module_name="binary", op_name=op_name, target=tgt,
        shape=shape, block_size=bs, threads=threads,
        dtype=tl_dtype, execution_backend=eb,
    )
    return invoke_kernel(kernel, x, y, out=out,
                         tilelang_target=tgt, execution_backend=eb)


# ── Arithmetic ────────────────────────────────────────────────────────────

def add(x: torch.Tensor, y: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Element-wise addition: ``out[i] = x[i] + y[i]``."""
    return _dispatch_binary("add", x, y, out)


def sub(x: torch.Tensor, y: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Element-wise subtraction: ``out[i] = x[i] - y[i]``."""
    return _dispatch_binary("sub", x, y, out)


def mul(x: torch.Tensor, y: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Element-wise multiplication: ``out[i] = x[i] * y[i]``."""
    return _dispatch_binary("mul", x, y, out)


def div(x: torch.Tensor, y: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Element-wise division: ``out[i] = x[i] / y[i]``."""
    return _dispatch_binary("div", x, y, out)


def pow(x: torch.Tensor, y: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Element-wise power: ``out[i] = x[i] ** y[i]``."""
    return _dispatch_binary("pow", x, y, out)


def fmod(x: torch.Tensor, y: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """C-style float modulo (sign matches dividend ``x``)."""
    return _dispatch_binary("fmod", x, y, out)


def remainder(x: torch.Tensor, y: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Python-style remainder (sign matches divisor ``y``)."""
    return _dispatch_binary("remainder", x, y, out)


def floor_div(x: torch.Tensor, y: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Floor division: ``out[i] = floor(x[i] / y[i])``."""
    return _dispatch_binary("floor_div", x, y, out)


# ── Extrema ───────────────────────────────────────────────────────────────

def maximum(x: torch.Tensor, y: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Element-wise maximum: ``out[i] = max(x[i], y[i])``."""
    return _dispatch_binary("maximum", x, y, out)


def minimum(x: torch.Tensor, y: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Element-wise minimum: ``out[i] = min(x[i], y[i])``."""
    return _dispatch_binary("minimum", x, y, out)


# ── Comparison (output 0.0 / 1.0) ────────────────────────────────────────

def eq(x: torch.Tensor, y: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Element-wise equality: 1.0 where ``x[i] == y[i]``, else 0.0."""
    return _dispatch_binary("eq", x, y, out)


def ne(x: torch.Tensor, y: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Element-wise inequality: 1.0 where ``x[i] != y[i]``, else 0.0."""
    return _dispatch_binary("ne", x, y, out)


def gt(x: torch.Tensor, y: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Element-wise greater-than: 1.0 where ``x[i] > y[i]``, else 0.0."""
    return _dispatch_binary("gt", x, y, out)


def ge(x: torch.Tensor, y: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Element-wise greater-or-equal: 1.0 where ``x[i] >= y[i]``."""
    return _dispatch_binary("ge", x, y, out)


def lt(x: torch.Tensor, y: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Element-wise less-than: 1.0 where ``x[i] < y[i]``, else 0.0."""
    return _dispatch_binary("lt", x, y, out)


def le(x: torch.Tensor, y: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Element-wise less-or-equal: 1.0 where ``x[i] <= y[i]``."""
    return _dispatch_binary("le", x, y, out)


# ── Math ──────────────────────────────────────────────────────────────────

def atan2(x: torch.Tensor, y: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Element-wise ``atan2(x, y)`` (four-quadrant arc-tangent)."""
    return _dispatch_binary("atan2", x, y, out)


def copysign(x: torch.Tensor, y: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Element-wise copysign: ``|x[i]|`` with sign of ``y[i]``."""
    return _dispatch_binary("copysign", x, y, out)


def hypot(x: torch.Tensor, y: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Element-wise hypotenuse: ``sqrt(x[i]² + y[i]²)``."""
    return _dispatch_binary("hypot", x, y, out)


def xlogy(x: torch.Tensor, y: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Element-wise ``x * log(y)``, defined as 0 when ``x == 0``."""
    return _dispatch_binary("xlogy", x, y, out)


# ── Logical (non-zero = True; output 0.0 / 1.0) ─────────────────────────

def logical_and(x: torch.Tensor, y: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Element-wise logical AND (non-zero is True)."""
    return _dispatch_binary("logical_and", x, y, out)


def logical_or(x: torch.Tensor, y: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Element-wise logical OR (non-zero is True)."""
    return _dispatch_binary("logical_or", x, y, out)


def logical_xor(x: torch.Tensor, y: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Element-wise logical XOR (non-zero is True)."""
    return _dispatch_binary("logical_xor", x, y, out)


# ── Bitwise ─────────────────────────────────────────────────────────────────


def bitwise_and(x: torch.Tensor, y: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Element-wise bitwise AND."""
    return _dispatch_binary("bitwise_and", x, y, out)


def bitwise_or(x: torch.Tensor, y: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Element-wise bitwise OR."""
    return _dispatch_binary("bitwise_or", x, y, out)


def bitwise_xor(x: torch.Tensor, y: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Element-wise bitwise XOR."""
    return _dispatch_binary("bitwise_xor", x, y, out)


def shift_left(x: torch.Tensor, y: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Element-wise left shift: ``x[i] << y[i]``."""
    return _dispatch_binary("shift_left", x, y, out)


def shift_right(x: torch.Tensor, y: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Element-wise right shift: ``x[i] >> y[i]``."""
    return _dispatch_binary("shift_right", x, y, out)


# ── Math (continued) ────────────────────────────────────────────────────────


def logaddexp(x: torch.Tensor, y: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Element-wise ``log(exp(x) + exp(y))``, numerically stable."""
    return _dispatch_binary("logaddexp", x, y, out)
