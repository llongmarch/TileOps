"""Metal unary activation kernels (TileLang ``@T.prim_func``).

Scheduling mirrors :mod:`tileops.cuda.activation` but lives in a separate
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

from tileops._infra import make_backend_op, make_cached_compiler

_SELU_ALPHA = 1.6732632423543772
_SELU_SCALE = 1.0507009873554805

# ═══════════════════════════════════════════════════════════════════════════
# PrimFunc builders — Classic
# ═══════════════════════════════════════════════════════════════════════════


def _relu_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    xv = X[idx]
                    Y[idx] = 0.5 * (xv + tir_op.abs(xv))
    return main


def _sigmoid_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
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
    return main


def _tanh_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    Y[idx] = tir_op.tanh(X[idx])
    return main


# ═══════════════════════════════════════════════════════════════════════════
# PrimFunc builders — GELU
# ═══════════════════════════════════════════════════════════════════════════


def _gelu_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    x = X[idx]
                    xc = x * x * x
                    inner = 0.7978845608028654 * (x + 0.044715 * xc)
                    Y[idx] = 0.5 * x * (1.0 + tir_op.tanh(inner))
    return main


def _gelu_exact_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    x = X[idx]
                    Y[idx] = 0.5 * x * (1.0 + tir_op.erf(x * 0.7071067811865476))
    return main


# ═══════════════════════════════════════════════════════════════════════════
# PrimFunc builders — Swish family
# ═══════════════════════════════════════════════════════════════════════════


def _silu_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    x = X[idx]
                    s = 1.0 / (1.0 + tir_op.exp(-x))
                    Y[idx] = x * s
    return main


def _hardswish_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    x = X[idx]
                    inner = x + 3.0
                    clamped = 0.5 * (tir_op.abs(inner) - tir_op.abs(inner - 6.0) + 6.0)
                    Y[idx] = x * clamped / 6.0
    return main


def _hardsigmoid_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    x = X[idx]
                    v = x / 6.0 + 0.5
                    Y[idx] = 0.5 * (tir_op.abs(v) - tir_op.abs(v - 1.0) + 1.0)
    return main


# ═══════════════════════════════════════════════════════════════════════════
# PrimFunc builders — ReLU variants
# ═══════════════════════════════════════════════════════════════════════════


def _leaky_relu_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    x = X[idx]
                    if x >= 0.0:
                        Y[idx] = x
                    else:
                        Y[idx] = 0.01 * x
    return main


def _relu6_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    x = X[idx]
                    Y[idx] = 0.5 * (tir_op.abs(x) - tir_op.abs(x - 6.0) + 6.0)
    return main


def _elu_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    x = X[idx]
                    if x > 0.0:
                        Y[idx] = x
                    else:
                        Y[idx] = tir_op.exp(x) - 1.0
    return main


def _selu_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    alpha = _SELU_ALPHA
    scale = _SELU_SCALE

    @T.prim_func
    def main(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    x = X[idx]
                    if x > 0.0:
                        Y[idx] = scale * x
                    else:
                        Y[idx] = scale * alpha * (tir_op.exp(x) - 1.0)
    return main


def _celu_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    x = X[idx]
                    if x > 0.0:
                        Y[idx] = x
                    else:
                        Y[idx] = tir_op.exp(x) - 1.0
    return main


def _hardtanh_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    x = X[idx]
                    Y[idx] = 0.5 * (tir_op.abs(x + 1.0) - tir_op.abs(x - 1.0))
    return main


# ═══════════════════════════════════════════════════════════════════════════
# PrimFunc builders — Smooth activations
# ═══════════════════════════════════════════════════════════════════════════


def _softplus_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    x = X[idx]
                    if x > 20.0:
                        Y[idx] = x
                    else:
                        Y[idx] = tir_op.log(1.0 + tir_op.exp(x))
    return main


def _mish_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    x = X[idx]
                    if x > 20.0:
                        Y[idx] = x * tir_op.tanh(x)
                    else:
                        Y[idx] = x * tir_op.tanh(tir_op.log(1.0 + tir_op.exp(x)))
    return main


def _softsign_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    x = X[idx]
                    Y[idx] = x / (1.0 + tir_op.abs(x))
    return main


# ═══════════════════════════════════════════════════════════════════════════
# PrimFunc builders — Log
# ═══════════════════════════════════════════════════════════════════════════


def _log_sigmoid_prim(n: int, bs: int, t: int, dtype: Any) -> Any:
    @T.prim_func
    def main(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, bs), threads=t) as (bx,):
            for i in T.Parallel(bs):
                idx = bx * bs + i
                if idx < n:
                    x = X[idx]
                    if x >= 0.0:
                        Y[idx] = -tir_op.log(1.0 + tir_op.exp(-x))
                    else:
                        Y[idx] = x - tir_op.log(1.0 + tir_op.exp(x))
    return main


# ═══════════════════════════════════════════════════════════════════════════
# Cached compiler + public API
# ═══════════════════════════════════════════════════════════════════════════

_ACTIVATION_PRIM: dict[str, Any] = {
    "relu": _relu_prim, "sigmoid": _sigmoid_prim, "tanh": _tanh_prim,
    "gelu": _gelu_prim, "gelu_exact": _gelu_exact_prim,
    "silu": _silu_prim, "hardswish": _hardswish_prim,
    "hardsigmoid": _hardsigmoid_prim,
    "leaky_relu": _leaky_relu_prim, "relu6": _relu6_prim,
    "elu": _elu_prim, "selu": _selu_prim, "celu": _celu_prim,
    "hardtanh": _hardtanh_prim,
    "softplus": _softplus_prim, "mish": _mish_prim,
    "softsign": _softsign_prim,
    "log_sigmoid": _log_sigmoid_prim,
}

_compile = make_cached_compiler(_ACTIVATION_PRIM, cache_prefix="metal_activation")

relu = make_backend_op("relu", compile_fn=_compile, doc="Metal: **ReLU**.")
sigmoid = make_backend_op("sigmoid", compile_fn=_compile, doc="Metal: **sigmoid**.")
tanh = make_backend_op("tanh", compile_fn=_compile, doc="Metal: **tanh**.")
gelu = make_backend_op("gelu", compile_fn=_compile, doc="Metal: **GELU** (tanh approx).")
gelu_exact = make_backend_op("gelu_exact", compile_fn=_compile, doc="Metal: **GELU** (exact, erf).")
silu = make_backend_op("silu", compile_fn=_compile, doc="Metal: **SiLU** (Swish).")
hardswish = make_backend_op("hardswish", compile_fn=_compile, doc="Metal: **HardSwish**.")
hardsigmoid = make_backend_op("hardsigmoid", compile_fn=_compile, doc="Metal: **HardSigmoid**.")
leaky_relu = make_backend_op("leaky_relu", compile_fn=_compile, doc="Metal: **LeakyReLU** (slope=0.01).")
relu6 = make_backend_op("relu6", compile_fn=_compile, doc="Metal: **ReLU6**.")
elu = make_backend_op("elu", compile_fn=_compile, doc="Metal: **ELU** (alpha=1).")
selu = make_backend_op("selu", compile_fn=_compile, doc="Metal: **SELU**.")
celu = make_backend_op("celu", compile_fn=_compile, doc="Metal: **CELU** (alpha=1).")
hardtanh = make_backend_op("hardtanh", compile_fn=_compile, doc="Metal: **HardTanh** (clamp -1..1).")
softplus = make_backend_op("softplus", compile_fn=_compile, doc="Metal: **Softplus**.")
mish = make_backend_op("mish", compile_fn=_compile, doc="Metal: **Mish**.")
softsign = make_backend_op("softsign", compile_fn=_compile, doc="Metal: **Softsign**.")
log_sigmoid = make_backend_op("log_sigmoid", compile_fn=_compile, doc="Metal: **LogSigmoid**.")

__all__ = [
    "relu", "sigmoid", "tanh",
    "gelu", "gelu_exact",
    "silu", "hardswish", "hardsigmoid",
    "leaky_relu", "relu6", "elu", "selu", "celu", "hardtanh",
    "softplus", "mish", "softsign",
    "log_sigmoid",
]
