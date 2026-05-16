"""Unary activation operators.

Each function takes one :class:`torch.Tensor` input and returns a
:class:`torch.Tensor` — just like ``torch.relu``, ``torch.sigmoid``, etc.
Kernel compilation, caching, and backend dispatch are handled
transparently.  The dtype is inferred from the input tensor.

**Classic** — :func:`relu`, :func:`sigmoid`, :func:`tanh`

**GELU** — :func:`gelu` (tanh approx), :func:`gelu_exact` (erf)

**Swish family** — :func:`silu`, :func:`hardswish`, :func:`hardsigmoid`

**ReLU variants** — :func:`leaky_relu`, :func:`relu6`, :func:`elu`,
:func:`selu`, :func:`celu`, :func:`hardtanh`

**Smooth** — :func:`softplus`, :func:`mish`, :func:`softsign`

**Log** — :func:`log_sigmoid`

::

    from tileops import relu, mish, gelu_exact
    out = relu(x)
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
    invoke_unary_kernel,
    suggest_tile_config,
    torch_to_tl_dtype,
    validate_single_input,
)


def _dispatch_activation(
    op_name: str,
    x: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    validate_single_input(x, out, op_name)
    tl_dtype = torch_to_tl_dtype(x.dtype)
    tgt = default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = default_execution_backend(tgt, dev)
    shape = x.shape
    bs, threads = suggest_tile_config(math.prod(shape))
    kernel = dispatch_compile(
        module_name="activation", op_name=op_name, target=tgt,
        shape=shape, block_size=bs, threads=threads,
        dtype=tl_dtype, execution_backend=eb,
    )
    return invoke_unary_kernel(kernel, x, out=out,
                               tilelang_target=tgt, execution_backend=eb)


# ── Classic ───────────────────────────────────────────────────────────────

def relu(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """ReLU: ``max(0, x)``."""
    return _dispatch_activation("relu", x, out)


def sigmoid(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Sigmoid: ``1 / (1 + exp(-x))``."""
    return _dispatch_activation("sigmoid", x, out)


def tanh(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Tanh activation."""
    return _dispatch_activation("tanh", x, out)


# ── GELU ──────────────────────────────────────────────────────────────────

def gelu(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """GELU (tanh approximation)."""
    return _dispatch_activation("gelu", x, out)


def gelu_exact(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """GELU (exact, via erf): ``0.5 * x * (1 + erf(x / sqrt(2)))``."""
    return _dispatch_activation("gelu_exact", x, out)


# ── Swish family ──────────────────────────────────────────────────────────

def silu(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """SiLU (Swish): ``x * sigmoid(x)``."""
    return _dispatch_activation("silu", x, out)


def hardswish(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """HardSwish: ``x * clamp(x+3, 0, 6) / 6``."""
    return _dispatch_activation("hardswish", x, out)


def hardsigmoid(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """HardSigmoid: ``clamp(x/6 + 0.5, 0, 1)``."""
    return _dispatch_activation("hardsigmoid", x, out)


# ── ReLU variants ─────────────────────────────────────────────────────────

def leaky_relu(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """LeakyReLU: ``max(0.01*x, x)``."""
    return _dispatch_activation("leaky_relu", x, out)


def relu6(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """ReLU6: ``min(max(0, x), 6)``."""
    return _dispatch_activation("relu6", x, out)


def elu(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """ELU (alpha=1): ``x if x > 0 else exp(x) - 1``."""
    return _dispatch_activation("elu", x, out)


def selu(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """SELU (self-normalizing ELU)."""
    return _dispatch_activation("selu", x, out)


def celu(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """CELU (alpha=1): continuous ELU."""
    return _dispatch_activation("celu", x, out)


def hardtanh(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """HardTanh: ``clamp(x, -1, 1)``."""
    return _dispatch_activation("hardtanh", x, out)


# ── Smooth ────────────────────────────────────────────────────────────────

def softplus(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Softplus: ``log(1 + exp(x))``."""
    return _dispatch_activation("softplus", x, out)


def mish(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mish: ``x * tanh(softplus(x))``."""
    return _dispatch_activation("mish", x, out)


def softsign(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Softsign: ``x / (1 + |x|)``."""
    return _dispatch_activation("softsign", x, out)


# ── Log ───────────────────────────────────────────────────────────────────

def log_sigmoid(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """LogSigmoid: ``log(sigmoid(x))``."""
    return _dispatch_activation("log_sigmoid", x, out)
