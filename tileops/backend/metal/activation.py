"""Metal unary activation kernels (TileLang ``@T.prim_func``).

Scheduling mirrors :mod:`tileops.backend.cuda.activation` but lives in a separate
file so Metal-specific tuning can diverge.  The facade
:mod:`tileops.activation` dispatches here for ``metal`` targets.

Categories:
  * **Classic** — relu, sigmoid, tanh
  * **GELU** — gelu (tanh approx), gelu_exact (erf)
  * **Swish family** — silu, hardswish, hardsigmoid
  * **ReLU variants** — leaky_relu, relu6, elu, selu, celu, hardtanh
  * **Smooth** — softplus, mish, softsign
  * **Log** — log_sigmoid
"""

from typing import Any

import tilelang.language as T
from tilelang.language.tir import op as tir_op

from tileops.runtime import build_op, make_cached_compiler

_SELU_ALPHA = 1.6732632423543772
_SELU_SCALE = 1.0507009873554805

# ═══════════════════════════════════════════════════════════════════════════
# PrimFunc builders — Classic
# ═══════════════════════════════════════════════════════════════════════════


def relu_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    xv = X[idx]
                    Y[idx] = 0.5 * (xv + tir_op.abs(xv))
    return _kernel


def sigmoid_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    x = X[idx]
                    if x >= 0.0:
                        Y[idx] = 1.0 / (1.0 + tir_op.exp(-x))
                    else:
                        e = tir_op.exp(x)
                        Y[idx] = e / (1.0 + e)
    return _kernel


def tanh_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    Y[idx] = tir_op.tanh(X[idx])
    return _kernel


# ═══════════════════════════════════════════════════════════════════════════
# PrimFunc builders — GELU
# ═══════════════════════════════════════════════════════════════════════════


def gelu_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    x = X[idx]
                    xc = x * x * x
                    inner = 0.7978845608028654 * (x + 0.044715 * xc)
                    Y[idx] = 0.5 * x * (1.0 + tir_op.tanh(inner))
    return _kernel


def gelu_exact_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    x = X[idx]
                    Y[idx] = 0.5 * x * (1.0 + tir_op.erf(x * 0.7071067811865476))
    return _kernel


# ═══════════════════════════════════════════════════════════════════════════
# PrimFunc builders — Swish family
# ═══════════════════════════════════════════════════════════════════════════


def silu_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    x = X[idx]
                    s = 1.0 / (1.0 + tir_op.exp(-x))
                    Y[idx] = x * s
    return _kernel


def hardswish_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    x = X[idx]
                    inner = x + 3.0
                    clamped = 0.5 * (tir_op.abs(inner) - tir_op.abs(inner - 6.0) + 6.0)
                    Y[idx] = x * clamped / 6.0
    return _kernel


def hardsigmoid_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    x = X[idx]
                    v = x / 6.0 + 0.5
                    Y[idx] = 0.5 * (tir_op.abs(v) - tir_op.abs(v - 1.0) + 1.0)
    return _kernel


# ═══════════════════════════════════════════════════════════════════════════
# PrimFunc builders — ReLU variants
# ═══════════════════════════════════════════════════════════════════════════


def leaky_relu_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    x = X[idx]
                    if x >= 0.0:
                        Y[idx] = x
                    else:
                        Y[idx] = 0.01 * x
    return _kernel


def relu6_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    x = X[idx]
                    Y[idx] = 0.5 * (tir_op.abs(x) - tir_op.abs(x - 6.0) + 6.0)
    return _kernel


def elu_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    x = X[idx]
                    if x > 0.0:
                        Y[idx] = x
                    else:
                        Y[idx] = tir_op.exp(x) - 1.0
    return _kernel


def selu_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    alpha = _SELU_ALPHA
    scale = _SELU_SCALE

    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    x = X[idx]
                    if x > 0.0:
                        Y[idx] = scale * x
                    else:
                        Y[idx] = scale * alpha * (tir_op.exp(x) - 1.0)
    return _kernel


def celu_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    x = X[idx]
                    if x > 0.0:
                        Y[idx] = x
                    else:
                        Y[idx] = tir_op.exp(x) - 1.0
    return _kernel


def hardtanh_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    x = X[idx]
                    Y[idx] = 0.5 * (tir_op.abs(x + 1.0) - tir_op.abs(x - 1.0))
    return _kernel


# ═══════════════════════════════════════════════════════════════════════════
# PrimFunc builders — Smooth activations
# ═══════════════════════════════════════════════════════════════════════════


def softplus_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    x = X[idx]
                    if x > 20.0:
                        Y[idx] = x
                    else:
                        Y[idx] = tir_op.log(1.0 + tir_op.exp(x))
    return _kernel


def mish_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    x = X[idx]
                    if x > 20.0:
                        Y[idx] = x * tir_op.tanh(x)
                    else:
                        Y[idx] = x * tir_op.tanh(tir_op.log(1.0 + tir_op.exp(x)))
    return _kernel


def softsign_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    x = X[idx]
                    Y[idx] = x / (1.0 + tir_op.abs(x))
    return _kernel


# ═══════════════════════════════════════════════════════════════════════════
# PrimFunc builders — Log
# ═══════════════════════════════════════════════════════════════════════════


def log_sigmoid_kernel(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    x = X[idx]
                    if x >= 0.0:
                        Y[idx] = -tir_op.log(1.0 + tir_op.exp(-x))
                    else:
                        Y[idx] = x - tir_op.log(1.0 + tir_op.exp(x))
    return _kernel


# ═══════════════════════════════════════════════════════════════════════════
# Cached compiler + public API
# ═══════════════════════════════════════════════════════════════════════════

_ACTIVATION_PRIM: dict[str, Any] = {
    "relu": relu_kernel, "sigmoid": sigmoid_kernel, "tanh": tanh_kernel,
    "gelu": gelu_kernel, "gelu_exact": gelu_exact_kernel,
    "silu": silu_kernel, "hardswish": hardswish_kernel,
    "hardsigmoid": hardsigmoid_kernel,
    "leaky_relu": leaky_relu_kernel, "relu6": relu6_kernel,
    "elu": elu_kernel, "selu": selu_kernel, "celu": celu_kernel,
    "hardtanh": hardtanh_kernel,
    "softplus": softplus_kernel, "mish": mish_kernel,
    "softsign": softsign_kernel,
    "log_sigmoid": log_sigmoid_kernel,
}

_compile = make_cached_compiler(_ACTIVATION_PRIM, cache_prefix="metal_activation")

relu = build_op("relu", compile_fn=_compile, doc="Metal: **ReLU**.")
sigmoid = build_op("sigmoid", compile_fn=_compile, doc="Metal: **sigmoid**.")
tanh = build_op("tanh", compile_fn=_compile, doc="Metal: **tanh**.")
gelu = build_op("gelu", compile_fn=_compile, doc="Metal: **GELU** (tanh approx).")
gelu_exact = build_op("gelu_exact", compile_fn=_compile, doc="Metal: **GELU** (exact, erf).")
silu = build_op("silu", compile_fn=_compile, doc="Metal: **SiLU** (Swish).")
hardswish = build_op("hardswish", compile_fn=_compile, doc="Metal: **HardSwish**.")
hardsigmoid = build_op("hardsigmoid", compile_fn=_compile, doc="Metal: **HardSigmoid**.")
leaky_relu = build_op("leaky_relu", compile_fn=_compile, doc="Metal: **LeakyReLU** (slope=0.01).")
relu6 = build_op("relu6", compile_fn=_compile, doc="Metal: **ReLU6**.")
elu = build_op("elu", compile_fn=_compile, doc="Metal: **ELU** (alpha=1).")
selu = build_op("selu", compile_fn=_compile, doc="Metal: **SELU**.")
celu = build_op("celu", compile_fn=_compile, doc="Metal: **CELU** (alpha=1).")
hardtanh = build_op("hardtanh", compile_fn=_compile, doc="Metal: **HardTanh** (clamp -1..1).")
softplus = build_op("softplus", compile_fn=_compile, doc="Metal: **Softplus**.")
mish = build_op("mish", compile_fn=_compile, doc="Metal: **Mish**.")
softsign = build_op("softsign", compile_fn=_compile, doc="Metal: **Softsign**.")
log_sigmoid = build_op("log_sigmoid", compile_fn=_compile, doc="Metal: **LogSigmoid**.")

__all__ = [
    "relu", "sigmoid", "tanh",
    "gelu", "gelu_exact",
    "silu", "hardswish", "hardsigmoid",
    "leaky_relu", "relu6", "elu", "selu", "celu", "hardtanh",
    "softplus", "mish", "softsign",
    "log_sigmoid",
]
