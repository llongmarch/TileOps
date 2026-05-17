"""TileOps — high-performance deep-learning operators built on TileLang.

Usage is as simple as PyTorch::

    from tileops import add, relu, maximum, gt
    out  = add(x, y)
    act  = relu(x)
    mx   = maximum(x, y)
    mask = gt(x, y)          # float tensor of 0s and 1s

All public symbols are listed in :data:`__all__`.  Kernel IR lives in the
backend subpackage (``tileops.backend.cuda``, ``tileops.backend.metal``).
Facades (``tileops.binary``, ``tileops.activation``, ``tileops.norm``, …)
provide the end-user API.  Internal infrastructure lives in
``tileops.runtime``.
"""

# ── Re-exports: runtime ─────────────────────────────────────────────────
from tileops.runtime import (
    bench_ms,
    compile_prim,
    default_execution_backend,
    default_tilelang_target,
    default_torch_device,
    heuristic_tilelang_target,
    invoke_conv_kernel,
    invoke_gather_kernel,
    invoke_tril_triu_kernel,
    invoke_row_sort_kernel,
    invoke_gemm_kernel,
    invoke_gemv_kernel,
    invoke_kernel,
    invoke_nary_kernel,
    invoke_row_index_reduce_kernel,
    invoke_row_reduce_kernel,
    invoke_row_topk_kernel,
    invoke_unary_kernel,
    make_kernel_runner,
    make_unary_kernel_runner,
    normalize_elem_dtype,
    setup_metal_workarounds,
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
    atan2, copysign, hypot, xlogy, logaddexp,
    # logical
    logical_and, logical_or, logical_xor,
    # bitwise
    bitwise_and, bitwise_or, bitwise_xor, shift_left, shift_right,
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

# ── Re-exports: fused (SwiGLU / GeGLU + residual norms) ─────────────────────
from tileops.fused import (
    gelu_and_mul,
    silu_and_mul,
)

# ── Re-exports: BLAS (gemm, mm, bmm, …) ───────────────────────────────────
from tileops.blas import (
    addmm,
    baddbmm,
    bmm,
    gemm,
    gemv,
    mm,
    mv,
    outer,
)

# ── Re-exports: reductions (sum / mean / … along dim) ─────────────────────
from tileops.reduction import (
    reduce_all,
    reduce_amax,
    reduce_amin,
    reduce_any,
    reduce_argmax,
    reduce_argmin,
    reduce_cumsum,
    reduce_mean,
    reduce_prod,
    reduce_sum,
)

# ── Re-exports: topk (separate module) ──────────────────────────────────────
from tileops.topk import topk

# ── Re-exports: softmax ──────────────────────────────────────────────────
from tileops.softmax import (
    log_softmax,
    online_softmax,
    safe_softmax,
    softmax,
)

# ── Re-exports: norm (layer norm, RMS norm, …) ───────────────────────────
from tileops.norm import (
    layer_norm,
    rms_norm,
    skip_layer_norm,
    skip_rms_norm,
)

# ── Re-exports: conv ────────────────────────────────────────────────────────
from tileops.conv import (
    conv1d,
    conv2d,
)

# ── Re-exports: unary (math) ───────────────────────────────────────────────
from tileops.unary import (
    # exponential / logarithmic
    exp, log, exp2, exp10, log2, log10, log1p,
    # trigonometry
    sin, cos, tan,
    # inverse trigonometry
    asin, acos, atan,
    # hyperbolic
    sinh, cosh,
    # inverse hyperbolic
    asinh, acosh, atanh,
    # power / root
    sqrt, rsqrt, square,
    # error function
    erf,
    # sign / absolute
    abs, sign, neg,
    # special-value detection
    isnan, isinf, isfinite,
    # rounding
    round, floor, ceil, trunc,
    # reciprocal
    reciprocal,
    # bitwise
    bitwise_not,
    # clamp
    clamp,
)

# ── Re-exports: indexing ───────────────────────────────────────────────────
from tileops.indexing import (
    gather,
    index_select,
    nonzero,
)

# ── Re-exports: condition ──────────────────────────────────────────────────
from tileops.condition import (
    where,
    masked_fill,
)

# ── Re-exports: matrix ─────────────────────────────────────────────────────
from tileops.matrix import (
    tril,
    triu,
)

# ── Re-exports: shape ──────────────────────────────────────────────────────
from tileops.shape import (
    cat, stack, split, chunk,
    permute, transpose, flip,
    repeat, expand,
)

# ── Re-exports: pad ──────────────────────────────────────────────────────
from tileops.pad import pad

# ── Re-exports: interpolate ──────────────────────────────────────────────
from tileops.interpolate import interpolate

# ── Re-exports: sort ───────────────────────────────────────────────────────
from tileops.sort import (
    sort,
    argsort,
)

# ── Re-exports: quant (symmetric INT8 + vLLM-style helpers) ───────────────
from tileops.quant import (
    dequantize_per_channel,
    dequantize_per_tensor,
    quantize_per_channel,
    quantize_per_tensor,
)
from tileops.quant_awq import (
    AWQ_TRITON_SUPPORTED_GROUP_SIZES,
    REVERSE_AWQ_ORDER,
    awq_dequantize,
    awq_dequantize_triton,
    awq_gemm,
    awq_gemm_triton,
    pack_awq_int4,
    unpack_awq_int4,
)
from tileops.quant_fp8 import (
    apply_w8a8_block_fp8_linear,
    block_dequant as block_dequant_fp8,
    default_fp8_dtype,
    get_fp8_min_max,
    input_to_float8,
    is_fp8,
    per_token_group_quant_fp8,
    w8a8_block_fp8_matmul,
    w8a8_triton_block_scaled_mm,
)
from tileops.quant_int8 import (
    apply_w8a8_block_int8_linear,
    block_dequant,
    input_to_int8,
    per_token_group_quant_int8,
    per_token_quant_int8,
    w8a8_block_int8_matmul,
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
    "invoke_conv_kernel",
    "invoke_gather_kernel",
    "invoke_tril_triu_kernel",
    "invoke_row_sort_kernel",
    "invoke_kernel",
    "invoke_row_index_reduce_kernel",
    "invoke_row_reduce_kernel",
    "invoke_row_topk_kernel",
    "invoke_nary_kernel",
    "invoke_unary_kernel",
    "make_kernel_runner",
    "make_unary_kernel_runner",
    "normalize_elem_dtype",
    "setup_metal_workarounds",
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
    # binary — math (continued)
    "logaddexp",
    # binary — logical
    "logical_and", "logical_or", "logical_xor",
    # binary — bitwise
    "bitwise_and", "bitwise_or", "bitwise_xor", "shift_left", "shift_right",
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
    # fused (activation kernels)
    "silu_and_mul",
    "gelu_and_mul",
    # blas
    "addmm",
    "baddbmm",
    "bmm",
    "gemm",
    "gemv",
    "mm",
    "mv",
    "outer",
    # reduction
    "reduce_sum",
    "reduce_mean",
    "reduce_prod",
    "reduce_amax",
    "reduce_amin",
    "reduce_argmax",
    "reduce_argmin",
    "reduce_all",
    "reduce_any",
    "reduce_cumsum",
    "topk",
    # softmax
    "softmax",
    "safe_softmax",
    "online_softmax",
    "log_softmax",
    # norm
    "layer_norm",
    "rms_norm",
    "skip_layer_norm",
    "skip_rms_norm",
    # conv
    "conv1d",
    "conv2d",
    # unary — exponential / logarithmic
    "exp", "log", "exp2", "exp10", "log2", "log10", "log1p",
    # unary — trigonometry
    "sin", "cos", "tan",
    # unary — inverse trigonometry
    "asin", "acos", "atan",
    # unary — hyperbolic
    "sinh", "cosh",
    # unary — inverse hyperbolic
    "asinh", "acosh", "atanh",
    # unary — power / root
    "sqrt", "rsqrt", "square",
    # unary — error function
    "erf",
    # unary — sign / absolute
    "abs", "sign", "neg",
    # unary — special-value detection
    "isnan", "isinf", "isfinite",
    # unary — rounding
    "round", "floor", "ceil", "trunc",
    # unary — reciprocal
    "reciprocal",
    # unary — bitwise
    "bitwise_not",
    # unary — clamp
    "clamp",
    # indexing
    "gather", "index_select", "nonzero",
    # condition
    "where", "masked_fill",
    # matrix
    "tril", "triu",
    # shape
    "cat", "stack", "split", "chunk",
    "permute", "transpose", "flip",
    "repeat", "expand",
    # pad
    "pad",
    # interpolate
    "interpolate",
    # sort
    "sort", "argsort",
    # quant
    "quantize_per_tensor",
    "dequantize_per_tensor",
    "quantize_per_channel",
    "dequantize_per_channel",
    "input_to_int8",
    "input_to_float8",
    "is_fp8",
    "default_fp8_dtype",
    "get_fp8_min_max",
    "per_token_quant_int8",
    "per_token_group_quant_int8",
    "per_token_group_quant_fp8",
    "block_dequant",
    "block_dequant_fp8",
    "w8a8_block_int8_matmul",
    "w8a8_block_fp8_matmul",
    "w8a8_triton_block_scaled_mm",
    "apply_w8a8_block_int8_linear",
    "apply_w8a8_block_fp8_linear",
    "AWQ_TRITON_SUPPORTED_GROUP_SIZES",
    "REVERSE_AWQ_ORDER",
    "awq_dequantize",
    "awq_dequantize_triton",
    "awq_gemm",
    "awq_gemm_triton",
    "pack_awq_int4",
    "unpack_awq_int4",
]
