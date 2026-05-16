"""Convolution ops: ``conv1d``, ``conv2d`` backed by TileLang kernels.

Shapes follow PyTorch's ``F.conv1d`` / ``F.conv2d`` conventions:

- ``conv1d``: input ``(N, C_in, L)``, weight ``(C_out, C_in // groups, kernel_size)``,
  bias ``(C_out,)``, output ``(N, C_out, L_out)``.
- ``conv2d``: input ``(N, C_in, H, W)``, weight
  ``(C_out, C_in // groups, kernel_h, kernel_w)``, bias ``(C_out,)``,
  output ``(N, C_out, H_out, W_out)``.
"""

from __future__ import annotations

from typing import Optional

import torch

from tileops.runtime import dispatch_compile_conv
from tileops.runtime import (
    default_execution_backend,
    default_tilelang_target,
    default_torch_device,
    invoke_conv_kernel,
    torch_to_tl_dtype,
)


def conv1d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    stride: int = 1,
    groups: int = 1,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """1-D convolution over a signal composed of several input planes.

    Shapes: input ``(N, C_in, L)``, weight ``(C_out, C_in // groups, kernel_size)``.

    Returns
    -------
    torch.Tensor
        ``(N, C_out, L_out)`` where ``L_out = (L - kernel_size) // stride + 1``.
    """
    if input.ndim != 3:
        raise ValueError(f"conv1d expects 3-D input, got {input.ndim}D")
    if weight.ndim != 3:
        raise ValueError(f"conv1d expects 3-D weight, got {weight.ndim}D")

    N, C_in, L = input.shape
    C_out, C_in_per_group, kernel_size = weight.shape

    if C_in // groups != C_in_per_group:
        raise ValueError(
            f"conv1d: C_in={C_in}, groups={groups}, but weight has "
            f"{C_in_per_group} input channels per group"
        )
    if C_out % groups != 0:
        raise ValueError(
            f"conv1d: C_out={C_out} must be divisible by groups={groups}"
        )
    if kernel_size > L:
        raise ValueError(
            f"conv1d: kernel_size={kernel_size} > input length L={L}"
        )

    if bias is None:
        bias = torch.zeros(C_out, dtype=input.dtype, device=input.device)
    if bias.shape != (C_out,):
        raise ValueError(
            f"conv1d: bias must be 1-D of length {C_out}, got {tuple(bias.shape)}"
        )

    if input.dtype != weight.dtype or weight.dtype != bias.dtype:
        raise TypeError("conv1d: input, weight, and bias dtypes must match")

    L_out = (L - kernel_size) // stride + 1

    tl_dtype = torch_to_tl_dtype(input.dtype)
    tgt = default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = default_execution_backend(tgt, dev)

    kernel = dispatch_compile_conv(
        op_name="conv1d",
        target=tgt,
        N=N,
        C_out=C_out,
        C_in=C_in,
        L=L,
        kernel_size=kernel_size,
        stride=stride,
        groups=groups,
        dtype=tl_dtype,
        execution_backend=eb,
    )

    if out is None:
        out = torch.empty(N, C_out, L_out, dtype=input.dtype, device=input.device)
    else:
        if out.shape != (N, C_out, L_out):
            raise ValueError(
                f"conv1d: out.shape={tuple(out.shape)} != {(N, C_out, L_out)}"
            )

    return invoke_conv_kernel(
        kernel, input, weight, bias, out=out,
        tilelang_target=tgt,
        execution_backend=eb,
    )


def conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    stride: int = 1,
    groups: int = 1,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """2-D convolution over a 4-D input composed of several input planes.

    Shapes: input ``(N, C_in, H, W)``, weight
    ``(C_out, C_in // groups, kernel_h, kernel_w)``.

    Returns
    -------
    torch.Tensor
        ``(N, C_out, H_out, W_out)`` where
        ``H_out = (H - kernel_h) // stride + 1``,
        ``W_out = (W - kernel_w) // stride + 1``.
    """
    if input.ndim != 4:
        raise ValueError(f"conv2d expects 4-D input, got {input.ndim}D")
    if weight.ndim != 4:
        raise ValueError(f"conv2d expects 4-D weight, got {weight.ndim}D")

    N, C_in, H, W = input.shape
    C_out, C_in_per_group, kernel_h, kernel_w = weight.shape

    if C_in // groups != C_in_per_group:
        raise ValueError(
            f"conv2d: C_in={C_in}, groups={groups}, but weight has "
            f"{C_in_per_group} input channels per group"
        )
    if C_out % groups != 0:
        raise ValueError(
            f"conv2d: C_out={C_out} must be divisible by groups={groups}"
        )
    if kernel_h > H:
        raise ValueError(
            f"conv2d: kernel_h={kernel_h} > input height H={H}"
        )
    if kernel_w > W:
        raise ValueError(
            f"conv2d: kernel_w={kernel_w} > input width W={W}"
        )

    if bias is None:
        bias = torch.zeros(C_out, dtype=input.dtype, device=input.device)
    if bias.shape != (C_out,):
        raise ValueError(
            f"conv2d: bias must be 1-D of length {C_out}, got {tuple(bias.shape)}"
        )

    if input.dtype != weight.dtype or weight.dtype != bias.dtype:
        raise TypeError("conv2d: input, weight, and bias dtypes must match")

    H_out = (H - kernel_h) // stride + 1
    W_out = (W - kernel_w) // stride + 1

    tl_dtype = torch_to_tl_dtype(input.dtype)
    tgt = default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = default_execution_backend(tgt, dev)

    kernel = dispatch_compile_conv(
        op_name="conv2d",
        target=tgt,
        N=N,
        C_out=C_out,
        C_in=C_in,
        L=0,  # unused for conv2d
        kernel_size=0,  # unused for conv2d
        stride=0,  # unused for conv2d
        groups=groups,
        H=H,
        W=W,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        stride_h=stride,
        stride_w=stride,
        dtype=tl_dtype,
        execution_backend=eb,
    )

    if out is None:
        out = torch.empty(
            N, C_out, H_out, W_out,
            dtype=input.dtype, device=input.device,
        )
    else:
        if out.shape != (N, C_out, H_out, W_out):
            raise ValueError(
                f"conv2d: out.shape={tuple(out.shape)} != "
                f"{(N, C_out, H_out, W_out)}"
            )

    return invoke_conv_kernel(
        kernel, input, weight, bias, out=out,
        tilelang_target=tgt,
        execution_backend=eb,
    )


__all__ = ["conv1d", "conv2d"]
