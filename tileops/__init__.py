"""TileOps — high-performance deep-learning operators built on TileLang.

Usage is as simple as PyTorch::

    from tileops import add, relu, maximum, gt
    out  = add(x, y)
    act  = relu(x)
    mx   = maximum(x, y)
    mask = gt(x, y)          # float tensor of 0s and 1s

All public symbols are listed in :data:`__all__`.  Kernel IR lives in
backend subpackages (``tileops.cuda.binary``, ``tileops.metal.norm``, …);
facades ``tileops.binary``, ``tileops.activation``, ``tileops.fused``,
``tileops.norm``, ``tileops.blas``, and ``tileops.reduction`` provide the
end-user API.  Runtime helpers live in ``tileops.runtime``.
"""

# ── Re-exports: runtime ─────────────────────────────────────────────────
from tileops.runtime import (
    bench_ms,
    compile_prim,
    default_execution_backend,
    default_tilelang_target,
    default_torch_device,
    heuristic_tilelang_target,
    invoke_gemm_kernel,
    invoke_gemv_kernel,
    invoke_kernel,
    invoke_nary_kernel,
    invoke_row_reduce_kernel,
    invoke_unary_kernel,
    make_kernel_runner,
    make_unary_kernel_runner,
    normalize_elem_dtype,
    setup_metal_workarounds,
    suggest_pointwise_config,
    suggest_tile_config,
    sync_device,
    target_kind,
    torch_to_tl_dtype,
)

# ── Re-exports: binary element-wise ops ─────────────────────────────────
from tileops.binary import (
    # arithmetic
    add, sub, mul, div, pow, fmod, remainder, floor_div,
    # extrema
    maximum, minimum,
    # comparison
    eq, ne, gt, ge, lt, le,
    # math
    atan2, copysign, hypot, xlogy,
    # logical
    logical_and, logical_or, logical_xor,
)

# ── Re-exports: activation ops ──────────────────────────────────────────
from tileops.activation import (
    # classic
    relu, sigmoid, tanh,
    # gelu
    gelu, gelu_exact,
    # swish family
    silu, hardswish, hardsigmoid,
    # relu variants
    leaky_relu, relu6, elu, selu, celu, hardtanh,
    # smooth
    softplus, mish, softsign,
    # log
    log_sigmoid,
)

# ── Re-exports: fused (silu_and_mul, gelu_and_mul) ───────────────────────
from tileops.fused import gelu_and_mul, silu_and_mul

# ── Re-exports: BLAS (gemm, gemv) ────────────────────────────────────────
from tileops.blas import gemm, gemv

# ── Re-exports: reductions (sum / mean / … along dim) ─────────────────────
from tileops.reduction import (
    reduce_amax,
    reduce_amin,
    reduce_mean,
    reduce_prod,
    reduce_sum,
)

# ── Re-exports: norm (softmax, layer norm, …) ────────────────────────────
from tileops.norm import (
    layer_norm,
    log_softmax,
    online_softmax,
    rms_norm,
    safe_softmax,
    softmax,
)

__all__ = [
    # runtime / host helpers
    "bench_ms",
    "compile_prim",
    "default_execution_backend",
    "default_tilelang_target",
    "default_torch_device",
    "heuristic_tilelang_target",
    "invoke_gemm_kernel",
    "invoke_gemv_kernel",
    "invoke_kernel",
    "invoke_row_reduce_kernel",
    "invoke_nary_kernel",
    "invoke_unary_kernel",
    "make_kernel_runner",
    "make_unary_kernel_runner",
    "normalize_elem_dtype",
    "setup_metal_workarounds",
    "suggest_pointwise_config",
    "suggest_tile_config",
    "sync_device",
    "target_kind",
    "torch_to_tl_dtype",
    # binary — arithmetic
    "add", "sub", "mul", "div", "pow",
    "fmod", "remainder", "floor_div",
    # binary — extrema
    "maximum", "minimum",
    # binary — comparison
    "eq", "ne", "gt", "ge", "lt", "le",
    # binary — math
    "atan2", "copysign", "hypot", "xlogy",
    # binary — logical
    "logical_and", "logical_or", "logical_xor",
    # activation — classic
    "relu", "sigmoid", "tanh",
    # activation — gelu
    "gelu", "gelu_exact",
    # activation — swish family
    "silu", "hardswish", "hardsigmoid",
    # activation — relu variants
    "leaky_relu", "relu6", "elu", "selu", "celu", "hardtanh",
    # activation — smooth
    "softplus", "mish", "softsign",
    # activation — log
    "log_sigmoid",
    # fused
    "silu_and_mul",
    "gelu_and_mul",
    # blas
    "gemm",
    "gemv",
    # reduction
    "reduce_sum",
    "reduce_mean",
    "reduce_prod",
    "reduce_amax",
    "reduce_amin",
    # norm
    "softmax",
    "safe_softmax",
    "online_softmax",
    "log_softmax",
    "layer_norm",
    "rms_norm",
]
