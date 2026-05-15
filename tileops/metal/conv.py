"""Metal reference 1-D and 2-D convolution (scalar accumulation loops).

Mirrors :mod:`tileops.cuda.conv` with a separate in-process cache prefix.
Parameter names ``arg0_*`` … preserve lexicographic Metal buffer order.
"""

from typing import Any, Optional

import tilelang.language as T

from tileops.runtime import compile_prim, default_tilelang_target


def _conv1d_prim(
    N: int, C_out: int, C_in: int, L: int,
    kernel_size: int, stride: int, groups: int,
    dtype: Any,
) -> Any:
    C_in_per_group = C_in // groups
    C_out_per_group = C_out // groups
    L_out = (L - kernel_size) // stride + 1

    @T.prim_func
    def main(
        arg0_input: T.Tensor((N, C_in, L), dtype),
        arg1_weight: T.Tensor((C_out, C_in_per_group, kernel_size), dtype),
        arg2_bias: T.Tensor((C_out,), dtype),
        arg3_output: T.Tensor((N, C_out, L_out), dtype),
    ):
        with T.Kernel(N, C_out, L_out, threads=1) as (n, co, l):
            acc = T.alloc_local((1,), T.float32)
            acc[0] = 0.0
            g = co // C_out_per_group
            in_c_offset = g * C_in_per_group
            for ci in range(C_in_per_group):
                for k in range(kernel_size):
                    acc[0] = acc[0] + (
                        arg0_input[n, in_c_offset + ci, l * stride + k]
                        * arg1_weight[co, ci, k]
                    )
            acc[0] = acc[0] + arg2_bias[co]
            arg3_output[n, co, l] = acc[0]

    return main


def _conv2d_prim(
    N: int, C_out: int, C_in: int, H: int, W: int,
    kernel_h: int, kernel_w: int, stride_h: int, stride_w: int,
    groups: int,
    dtype: Any,
) -> Any:
    C_in_per_group = C_in // groups
    C_out_per_group = C_out // groups
    H_out = (H - kernel_h) // stride_h + 1
    W_out = (W - kernel_w) // stride_w + 1

    @T.prim_func
    def main(
        arg0_input: T.Tensor((N, C_in, H, W), dtype),
        arg1_weight: T.Tensor((C_out, C_in_per_group, kernel_h, kernel_w), dtype),
        arg2_bias: T.Tensor((C_out,), dtype),
        arg3_output: T.Tensor((N, C_out, H_out, W_out), dtype),
    ):
        with T.Kernel(N, C_out, H_out, W_out, threads=1) as (n, co, h, w):
            acc = T.alloc_local((1,), T.float32)
            acc[0] = 0.0
            g = co // C_out_per_group
            in_c_offset = g * C_in_per_group
            for ci in range(C_in_per_group):
                for kh in range(kernel_h):
                    for kw in range(kernel_w):
                        acc[0] = acc[0] + (
                            arg0_input[
                                n, in_c_offset + ci,
                                h * stride_h + kh, w * stride_w + kw
                            ]
                            * arg1_weight[co, ci, kh, kw]
                        )
            acc[0] = acc[0] + arg2_bias[co]
            arg3_output[n, co, h, w] = acc[0]

    return main


_cache: dict[tuple, Any] = {}


def _compile_key_prefix() -> str:
    return "metal_conv_v1"


def conv1d(
    N: int, C_out: int, C_in: int, L: int,
    kernel_size: int, stride: int, groups: int,
    *,
    dtype: Any,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> Any:
    tgt = target if target is not None else default_tilelang_target()
    key = (_compile_key_prefix(), "conv1d", N, C_out, C_in, L,
           kernel_size, stride, groups, dtype, tgt, execution_backend)
    hit = _cache.get(key)
    if hit is not None:
        return hit
    kernel = compile_prim(
        _conv1d_prim(N, C_out, C_in, L, kernel_size, stride, groups, dtype),
        target=tgt,
        execution_backend=execution_backend,
    )
    _cache[key] = kernel
    return kernel


def conv2d(
    N: int, C_out: int, C_in: int, H: int, W: int,
    kernel_h: int, kernel_w: int, stride_h: int, stride_w: int,
    groups: int,
    *,
    dtype: Any,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> Any:
    tgt = target if target is not None else default_tilelang_target()
    key = (_compile_key_prefix(), "conv2d", N, C_out, C_in, H, W,
           kernel_h, kernel_w, stride_h, stride_w, groups,
           dtype, tgt, execution_backend)
    hit = _cache.get(key)
    if hit is not None:
        return hit
    kernel = compile_prim(
        _conv2d_prim(
            N, C_out, C_in, H, W,
            kernel_h, kernel_w, stride_h, stride_w, groups,
            dtype,
        ),
        target=tgt,
        execution_backend=execution_backend,
    )
    _cache[key] = kernel
    return kernel


__all__ = ["conv1d", "conv2d"]
