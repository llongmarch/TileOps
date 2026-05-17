"""Boundary padding with constant fill.

Constant-mode padding on the last 1 or 2 dimensions is compiled through
TileLang; other modes (reflect, replicate, circular) and higher-dimensional
padding fall back to :func:`torch.nn.functional.pad`.

**API**

* :func:`pad` — pad input tensor boundaries.

::

    from tileops import pad
    y = pad(x, (1, 2))                        # pad last dim: left=1, right=2
    y = pad(x, (1, 2, 3, 4))                  # pad last 2 dims
    y = pad(x, (1, 2, 3, 4), value=-1.0)      # constant fill value
    y = pad(x, (1, 2), mode="reflect")         # reflect (falls back to PyTorch)
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from tileops.runtime import dispatch_compile_pad
from tileops.runtime import (
    default_execution_backend,
    default_tilelang_target,
    default_torch_device,
    invoke_nary_kernel,
    torch_to_tl_dtype,
)

_PAD_THREADS = 1


def pad(
    input: torch.Tensor,
    pad: Sequence[int],
    mode: str = "constant",
    value: float = 0.0,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pad *input* on its boundaries.

    *pad* follows the PyTorch convention: ``(left, right, top, bottom, …)``
    in reverse dimension order (last dimension first).  Padding amounts must
    be non-negative.

    For ``mode="constant"`` with padding on at most the last 2 dimensions,
    a TileLang kernel is used.  Other modes and arbitrary-dimensional padding
    delegate to :func:`torch.nn.functional.pad`.
    """
    import torch.nn.functional as F  # noqa: E402

    _pad = tuple(pad)
    if len(_pad) % 2 != 0:
        raise ValueError(
            f"pad: pad length must be even, got {len(_pad)}"
        )
    ndim_pad = len(_pad) // 2
    if ndim_pad > input.ndim:
        raise ValueError(
            f"pad: pad specifies {ndim_pad} dims but input has {input.ndim} dims"
        )

    # Non-constant modes fall back to PyTorch.
    if mode != "constant":
        result = F.pad(input, _pad, mode=mode, value=value)
        if out is not None:
            out.copy_(result)
            return out
        return result

    # General N-D padding where N > 2 falls back to PyTorch.
    if ndim_pad > 2:
        result = F.pad(input, _pad, mode="constant", value=value)
        if out is not None:
            out.copy_(result)
            return out
        return result

    # Build the output shape.
    out_shape_list = list(input.shape)
    for i in range(ndim_pad):
        dim = input.ndim - 1 - i
        left = _pad[i * 2]
        right = _pad[i * 2 + 1]
        out_shape_list[dim] += left + right
    out_shape = tuple(out_shape_list)

    # Pre-allocate and fill with constant value.
    if out is not None:
        if out.shape != out_shape:
            raise ValueError(
                f"pad: out.shape={tuple(out.shape)} != expected {out_shape}"
            )
        if out.dtype != input.dtype or out.device != input.device:
            raise TypeError("pad: out dtype/device must match input")
        result = out
    else:
        result = torch.empty(out_shape, dtype=input.dtype, device=input.device)
    result.fill_(value)

    if input.numel() == 0:
        return result

    # Dispatch to TileLang kernel.
    tl_dtype = torch_to_tl_dtype(input.dtype)
    tgt = default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = default_execution_backend(tgt, dev)

    if ndim_pad == 1:
        pad_left, pad_right = _pad[0], _pad[1]
        m = 1
        for s in input.shape[:-1]:
            m *= s
        n = input.shape[-1]
        input_2d = input.contiguous().view(m, n)
        out_flat = result.contiguous().view(-1)

        kernel = dispatch_compile_pad(
            op_name="pad1d", target=tgt, rows=m, cols=n,
            threads=_PAD_THREADS, dtype=tl_dtype,
            execution_backend=eb,
            pad_left=pad_left, pad_right=pad_right, value=value,
        )
        invoke_nary_kernel(
            kernel, input_2d, out=out_flat,
            tilelang_target=tgt, execution_backend=eb,
        )
    else:  # ndim_pad == 2
        pad_left, pad_right = _pad[0], _pad[1]
        pad_top, pad_bottom = _pad[2], _pad[3]
        m = 1
        for s in input.shape[:-2]:
            m *= s
        h_in, w_in = input.shape[-2], input.shape[-1]

        # Reshape input to (M, H_in * W_in) — flat columns per row.
        input_2d = input.contiguous().view(m, h_in * w_in)
        out_flat = result.contiguous().view(-1)

        kernel = dispatch_compile_pad(
            op_name="pad2d", target=tgt, rows=m, cols=h_in * w_in,
            threads=_PAD_THREADS, dtype=tl_dtype,
            execution_backend=eb,
            pad_top=pad_top, pad_bottom=pad_bottom,
            pad_left=pad_left, pad_right=pad_right,
            h_in=h_in, w_in=w_in,
            value=value,
        )
        invoke_nary_kernel(
            kernel, input_2d, out=out_flat,
            tilelang_target=tgt, execution_backend=eb,
        )

    return result
