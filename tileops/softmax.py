"""Row-wise softmax and log-softmax.

Operations follow the last reduced axis after merging all leading dimensions
(``torch.nn.functional`` semantics on a reshaped tensor).  Supported dtypes
match the rest of TileOps: ``float32``, ``float16``, ``bfloat16``.

**API**

* :func:`softmax` — two-pass stable softmax (row max, then ``exp`` / sum).
* :func:`safe_softmax` — same numerics as ``softmax`` (explicit "safe" alias).
* :func:`online_softmax` — one-pass streaming max + denominator, same result.
* :func:`log_softmax` — stable reduce over ``dim``.

::

    from tileops import softmax, safe_softmax, online_softmax, log_softmax
    y = softmax(x, dim=-1)
"""

from __future__ import annotations

from typing import Optional

import torch

from tileops.runtime import dispatch_compile_softmax
from tileops._shape import (
    merge_on_dim as _merge_on_dim,
    unmerge as _unmerge,
)
from tileops.runtime import (
    default_execution_backend,
    default_tilelang_target,
    default_torch_device,
    invoke_nary_kernel,
    torch_to_tl_dtype,
)

_SOFTMAX_THREADS = 1


def softmax(
    x: torch.Tensor,
    dim: int = -1,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Softmax along *dim* (numerically stable: subtract row max)."""
    x_2d, m, n, perm, orig = _merge_on_dim(x, dim, op_name="softmax")
    tl_dtype = torch_to_tl_dtype(x.dtype)
    tgt = default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = default_execution_backend(tgt, dev)
    kernel = dispatch_compile_softmax(
        op_name="softmax", target=tgt, rows=m, cols=n,
        threads=_SOFTMAX_THREADS, dtype=tl_dtype, execution_backend=eb, eps=None,
    )
    y_2d = invoke_nary_kernel(
        kernel, x_2d, out=None,
        tilelang_target=tgt, execution_backend=eb,
    )
    result = _unmerge(y_2d, orig, perm)
    if out is not None:
        if out.shape != x.shape:
            raise ValueError(f"softmax: out.shape={out.shape} != input {x.shape}")
        if out.dtype != x.dtype or out.device != x.device:
            raise TypeError("softmax: out dtype/device must match input")
        out.copy_(result)
        return out
    return result


def safe_softmax(
    x: torch.Tensor,
    dim: int = -1,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Stable softmax (subtract row max before ``exp``).

    Uses the same TileLang kernel as :func:`softmax`; numerically matches
    ``torch.softmax``.  Provided as an explicit *safe* name alongside
    :func:`online_softmax`.
    """
    return softmax(x, dim=dim, out=out)


def online_softmax(
    x: torch.Tensor,
    dim: int = -1,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Softmax along *dim* using a one-pass online max / denominator update.

    Mathematically equivalent to :func:`softmax` / ``torch.softmax``; useful
    when a single streaming reduction over the axis is desired.
    """
    x_2d, m, n, perm, orig = _merge_on_dim(x, dim, op_name="online_softmax")
    tl_dtype = torch_to_tl_dtype(x.dtype)
    tgt = default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = default_execution_backend(tgt, dev)
    kernel = dispatch_compile_softmax(
        op_name="online_softmax", target=tgt, rows=m, cols=n,
        threads=_SOFTMAX_THREADS, dtype=tl_dtype, execution_backend=eb, eps=None,
    )
    y_2d = invoke_nary_kernel(
        kernel, x_2d, out=None,
        tilelang_target=tgt, execution_backend=eb,
    )
    result = _unmerge(y_2d, orig, perm)
    if out is not None:
        if out.shape != x.shape:
            raise ValueError(
                f"online_softmax: out.shape={out.shape} != input {x.shape}"
            )
        if out.dtype != x.dtype or out.device != x.device:
            raise TypeError("online_softmax: out dtype/device must match input")
        out.copy_(result)
        return out
    return result


def log_softmax(
    x: torch.Tensor,
    dim: int = -1,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Log-softmax along *dim*."""
    x_2d, m, n, perm, orig = _merge_on_dim(x, dim, op_name="log_softmax")
    tl_dtype = torch_to_tl_dtype(x.dtype)
    tgt = default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = default_execution_backend(tgt, dev)
    kernel = dispatch_compile_softmax(
        op_name="log_softmax", target=tgt, rows=m, cols=n,
        threads=_SOFTMAX_THREADS, dtype=tl_dtype, execution_backend=eb, eps=None,
    )
    y_2d = invoke_nary_kernel(
        kernel, x_2d, out=None,
        tilelang_target=tgt, execution_backend=eb,
    )
    result = _unmerge(y_2d, orig, perm)
    if out is not None:
        if out.shape != x.shape:
            raise ValueError(f"log_softmax: out.shape={out.shape} != input {x.shape}")
        if out.dtype != x.dtype or out.device != x.device:
            raise TypeError("log_softmax: out dtype/device must match input")
        out.copy_(result)
        return out
    return result
