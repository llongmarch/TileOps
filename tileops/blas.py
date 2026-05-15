"""BLAS-style ops: matrix–matrix / matrix–vector multiply and PyTorch aliases.

``gemm`` / ``gemv`` are the TileLang-backed primitives.  ``mm``, ``mv``,
``outer``, ``bmm`` wrap them with :class:`torch.Tensor` shapes matching
``torch.mm``, ``torch.mv``, etc.  ``addmm`` / ``baddbmm`` compute the
matrix product with the same kernels, then apply ``beta`` / ``alpha`` with
PyTorch element-wise ops (on-device, broadcast rules match ``torch.addmm``).

Dtypes: ``float32``, ``float16``, ``bfloat16``.
"""

from __future__ import annotations

import numbers
from typing import Optional, Union

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

Scalar = Union[float, int]


def _as_float(x: Scalar, *, name: str) -> float:
    if not isinstance(x, numbers.Real) or isinstance(x, bool):
        raise TypeError(f"{name}: expected a real scalar, got {type(x).__name__}")
    return float(x)


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


def mm(
    input: torch.Tensor,
    mat2: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``torch.mm`` — 2-D matrix multiply ``input @ mat2``."""
    return gemm(input, mat2, out=out)


def mv(
    input: torch.Tensor,
    vec: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``torch.mv`` — matrix–vector ``input @ vec``."""
    return gemv(input, vec, out=out)


def outer(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``torch.outer`` — outer product ``a ⊗ b``, shape ``(len(a), len(b))``."""
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("outer expects 1-D a and b")
    if a.dtype != b.dtype:
        raise TypeError("outer: a and b dtypes must match")
    return gemm(a.unsqueeze(1), b.unsqueeze(0), out=out)


def bmm(
    input: torch.Tensor,
    mat2: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Batched matrix multiply ``input @ mat2`` (``torch.bmm``).

    Shapes: ``(B, M, K)`` and ``(B, K, N)`` → ``(B, M, N)``.
    """
    if input.ndim != 3 or mat2.ndim != 3:
        raise ValueError("bmm expects 3-D tensors")
    bsz, m, k1 = input.shape
    bsz2, k2, n = mat2.shape
    if bsz != bsz2:
        raise ValueError(
            f"bmm: batch size mismatch {bsz} vs {bsz2}"
        )
    if k1 != k2:
        raise ValueError(
            f"bmm: incompatible K dims {k1} vs {k2}"
        )
    if input.dtype != mat2.dtype:
        raise TypeError("bmm: input and mat2 dtypes must match")

    tgt = default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = default_execution_backend(tgt, dev)
    tl_dtype = torch_to_tl_dtype(input.dtype)
    kernel = dispatch_compile_blas(
        op_name="gemm",
        target=tgt,
        m=m,
        k=k1,
        dtype=tl_dtype,
        execution_backend=eb,
        n=n,
    )

    if out is None:
        out_batch = torch.empty(bsz, m, n, dtype=input.dtype, device=input.device)
    else:
        if out.shape != (bsz, m, n):
            raise ValueError(
                f"bmm: out.shape={tuple(out.shape)} != {(bsz, m, n)}"
            )
        if out.dtype != input.dtype or out.device != input.device:
            raise TypeError("bmm: out dtype/device must match input")
        out_batch = out

    for i in range(bsz):
        invoke_gemm_kernel(
            kernel, input[i], mat2[i], out=out_batch[i],
            tilelang_target=tgt,
            execution_backend=eb,
        )
    return out_batch


def addmm(
    input: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    *,
    beta: Scalar = 1.0,
    alpha: Scalar = 1.0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused ``beta * input + alpha * (mat1 @ mat2)`` (``torch.addmm``)."""
    bcoef = _as_float(beta, name="addmm.beta")
    acoef = _as_float(alpha, name="addmm.alpha")

    if mat1.ndim != 2 or mat2.ndim != 2:
        raise ValueError("addmm expects 2-D mat1 and mat2")
    m, k1 = mat1.shape
    k2, n = mat2.shape
    if k1 != k2:
        raise ValueError(
            f"addmm: inner dims {k1} vs {k2} for "
            f"{tuple(mat1.shape)} @ {tuple(mat2.shape)}"
        )
    if mat1.dtype != mat2.dtype:
        raise TypeError("addmm: mat1 and mat2 dtypes must match")

    tgt = default_tilelang_target()
    tl_dtype = torch_to_tl_dtype(mat1.dtype)

    shape_mn = (m, n)
    try:
        torch.broadcast_to(input, shape_mn)
    except RuntimeError as exc:
        raise ValueError(
            f"addmm: input with shape {tuple(input.shape)} is not broadcastable "
            f"to {shape_mn}"
        ) from exc

    # --- alpha == 0: skip matmul (match ``torch.addmm`` result shape ``(m, n)``)
    if acoef == 0.0:
        if out is None:
            return torch.mul(torch.broadcast_to(input, shape_mn), bcoef)
        torch.mul(input, bcoef, out=out)
        return out

    dev = default_torch_device(tgt)
    eb = default_execution_backend(tgt, dev)
    kernel = dispatch_compile_blas(
        op_name="gemm",
        target=tgt,
        m=m,
        k=k1,
        dtype=tl_dtype,
        execution_backend=eb,
        n=n,
    )

    if out is None:
        out_t = torch.empty(m, n, dtype=mat1.dtype, device=mat1.device)
    else:
        if out.shape != (m, n):
            raise ValueError(
                f"addmm: out.shape={tuple(out.shape)} != ({m}, {n})"
            )
        if out.dtype != mat1.dtype or out.device != mat1.device:
            raise TypeError("addmm: out dtype/device must match mat1")
        out_t = out

    invoke_gemm_kernel(
        kernel, mat1, mat2, out=out_t,
        tilelang_target=tgt,
        execution_backend=eb,
    )
    if acoef != 1.0:
        out_t.mul_(acoef)
    if bcoef != 0.0:
        out_t.add_(input, alpha=bcoef)
    return out_t


def baddbmm(
    input: torch.Tensor,
    batch1: torch.Tensor,
    batch2: torch.Tensor,
    *,
    beta: Scalar = 1.0,
    alpha: Scalar = 1.0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Batched ``torch.baddbmm`` — ``beta * input + alpha * bmm(batch1, batch2)``."""
    bcoef = _as_float(beta, name="baddbmm.beta")
    acoef = _as_float(alpha, name="baddbmm.alpha")

    if input.ndim != 3 or batch1.ndim != 3 or batch2.ndim != 3:
        raise ValueError("baddbmm expects three 3-D tensors")
    bsz, m, k1 = batch1.shape
    bsz2, k2, n = batch2.shape
    if bsz != bsz2:
        raise ValueError(f"baddbmm: batch mismatch {bsz} vs {bsz2}")
    if k1 != k2:
        raise ValueError(f"baddbmm: inner dims {k1} vs {k2}")
    if batch1.dtype != batch2.dtype or batch1.dtype != input.dtype:
        raise TypeError("baddbmm: batch1, batch2, and input dtypes must match")

    shape_bmn = (bsz, m, n)
    try:
        inp_b = torch.broadcast_to(input, shape_bmn)
    except RuntimeError as exc:
        raise ValueError(
            f"baddbmm: input shape {tuple(input.shape)} not broadcastable to "
            f"{shape_bmn}"
        ) from exc

    if acoef == 0.0:
        if out is None:
            tmp = torch.empty(
                shape_bmn, dtype=batch1.dtype, device=batch1.device,
            )
            torch.mul(input, bcoef, out=tmp)
            return tmp
        torch.mul(input, bcoef, out=out)
        return out

    if out is None:
        out_full = torch.empty(shape_bmn, dtype=batch1.dtype, device=batch1.device)
    else:
        if out.shape != shape_bmn:
            raise ValueError(
                f"baddbmm: out.shape={tuple(out.shape)} != {shape_bmn}"
            )
        if out.dtype != batch1.dtype or out.device != batch1.device:
            raise TypeError("baddbmm: out dtype/device must match batch1")
        out_full = out

    for i in range(bsz):
        addmm(
            inp_b[i],
            batch1[i],
            batch2[i],
            beta=bcoef,
            alpha=acoef,
            out=out_full[i],
        )
    return out_full


__all__ = [
    "addmm",
    "baddbmm",
    "bmm",
    "gemm",
    "gemv",
    "mm",
    "mv",
    "outer",
]
