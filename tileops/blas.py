"""GEMM and GEMV — matrix-matrix and matrix-vector multiply.

Computes ``C = A @ B`` (``gemm``) and ``y = A @ x`` (``gemv``) in row-major
layout, same dtypes as supported elsewhere in TileOps (``float32``, ``float16``,
``bfloat16``).  Kernels are reference scalar loops; for large matrices prefer
PyTorch / cuBLAS.
"""

from __future__ import annotations

from typing import Optional

import torch

from tileops._infra import dispatch_compile_blas
from tileops.runtime import (
    default_execution_backend,
    default_tilelang_target,
    default_torch_device,
    invoke_gemm_kernel,
    invoke_gemv_kernel,
    torch_to_tl_dtype,
)


def gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Matrix multiply ``C = A @ B`` (2-D tensors, row-major).

    Shapes: *A* ``(M, K)``, *B* ``(K, N)``, output ``(M, N)``.
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("gemm expects 2-D A and B")
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError(
            f"gemm: incompatible shapes {tuple(a.shape)} @ {tuple(b.shape)}"
        )
    if a.dtype != b.dtype:
        raise TypeError("gemm: A and B dtypes must match")
    tl_dtype = torch_to_tl_dtype(a.dtype)
    tgt = default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = default_execution_backend(tgt, dev)
    kernel = dispatch_compile_blas(
        op_name="gemm",
        target=tgt,
        m=m,
        k=k,
        dtype=tl_dtype,
        execution_backend=eb,
        n=n,
    )
    return invoke_gemm_kernel(
        kernel, a, b, out=out,
        tilelang_target=tgt,
        execution_backend=eb,
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


__all__ = ["gemm", "gemv"]
