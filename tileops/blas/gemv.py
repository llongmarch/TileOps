"""GEMV and matrix–vector alias (``mv``).

All ops are backed by the TileLang ``gemv`` kernel.
"""

from __future__ import annotations

from typing import Optional

import torch

from tileops.runtime import (
    default_execution_backend,
    default_tilelang_target,
    default_torch_device,
    dispatch_compile_blas,
    invoke_gemv_kernel,
    torch_to_tl_dtype,
)


def gemv(
    a: torch.Tensor,
    x: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Matrix-vector multiply ``y = A @ x``.

    Shapes: *A* ``(M, K)``, *x* ``(K,)``, output ``(M,)``.
    """
    if a.ndim != 2 or x.ndim != 1:
        raise ValueError("gemv expects A 2-D and x 1-D")
    m, k = a.shape
    if x.shape[0] != k:
        raise ValueError(
            f"gemv: A.shape[1]={k} != len(x)={x.shape[0]}"
        )
    if a.dtype != x.dtype:
        raise TypeError("gemv: A and x dtypes must match")
    tl_dtype = torch_to_tl_dtype(a.dtype)
    tgt = default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = default_execution_backend(tgt, dev)
    kernel = dispatch_compile_blas(
        op_name="gemv",
        target=tgt,
        m=m,
        k=k,
        dtype=tl_dtype,
        execution_backend=eb,
    )
    return invoke_gemv_kernel(
        kernel, a, x, out=out,
        tilelang_target=tgt,
        execution_backend=eb,
    )


def mv(
    input: torch.Tensor,
    vec: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``torch.mv`` — matrix–vector ``input @ vec``."""
    return gemv(input, vec, out=out)
