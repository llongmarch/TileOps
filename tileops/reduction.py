"""Reductions along a single dimension (``reduce_sum``, ``reduce_mean``, …).

Uses the same *merge dim to last axis* layout as :mod:`tileops.norm`.  Names use
the ``reduce_*`` prefix so ``from tileops import reduce_sum`` does not shadow
Python's built-in :func:`sum`.

Supported dtypes:

* ``reduce_sum``, ``reduce_mean``, ``reduce_prod``, ``reduce_amax``,
  ``reduce_amin``, ``reduce_cumsum`` — ``float32``, ``float16``, ``bfloat16``.
* ``reduce_argmax`` / ``reduce_argmin`` — same input dtypes;
  indices are ``torch.int64`` (like PyTorch).
* ``reduce_all`` / ``reduce_any`` — booleans cast to ``float32`` internally,
  floating dtypes use PyTorch-like *non-zero is true* semantics on the
  original values.

``reduce_cumsum`` uses prefix sums inside each merged row via
:func:`tileops.runtime.invoke_unary_kernel`.

.. seealso:: :mod:`tileops.topk` for the top-*k* operator.
"""

from __future__ import annotations

from typing import Optional

import torch

from tileops.runtime import dispatch_compile_reduce
from tileops._shape import merge_on_dim, unmerge
from tileops.runtime import (
    default_execution_backend,
    default_tilelang_target,
    default_torch_device,
    invoke_row_index_reduce_kernel,
    invoke_row_reduce_kernel,
    invoke_unary_kernel,
    torch_to_tl_dtype,
)

_REDUCE_THREADS = 1

_PUBLIC_NAMES: dict[str, str] = {
    "sum": "reduce_sum",
    "mean": "reduce_mean",
    "prod": "reduce_prod",
    "amax": "reduce_amax",
    "amin": "reduce_amin",
    "argmax": "reduce_argmax",
    "argmin": "reduce_argmin",
    "all": "reduce_all",
    "any": "reduce_any",
    "cumsum": "reduce_cumsum",
}


def _reduce_output_shape(
    orig_shape: tuple[int, ...],
    dim: int,
    *,
    keepdim: bool,
) -> tuple[int, ...]:
    if keepdim:
        return orig_shape[:dim] + (1,) + orig_shape[dim + 1:]
    return orig_shape[:dim] + orig_shape[dim + 1:]


def _float_reduce_dispatch(
    op_name: str,
    x: torch.Tensor,
    dim: int,
    *,
    keepdim: bool,
    out: Optional[torch.Tensor],
) -> torch.Tensor:
    """Scalar row reduce; output dtype matches *x*."""
    dim_n = dim if dim >= 0 else dim + x.ndim
    pub = _PUBLIC_NAMES[op_name]
    x_2d, m, n, _perm, orig = merge_on_dim(x, dim, op_name=pub)
    tl_dtype = torch_to_tl_dtype(x.dtype)
    tgt = default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = default_execution_backend(tgt, dev)
    kernel = dispatch_compile_reduce(
        op_name=op_name,
        target=tgt,
        rows=m,
        cols=n,
        threads=_REDUCE_THREADS,
        dtype=tl_dtype,
        execution_backend=eb,
    )
    out_shape = _reduce_output_shape(orig, dim_n, keepdim=keepdim)
    out_1d = torch.empty(m, dtype=x.dtype, device=x.device)
    invoke_row_reduce_kernel(
        kernel, x_2d, out_1d,
        tilelang_target=tgt,
        execution_backend=eb,
    )
    result = out_1d.reshape(out_shape)
    if out is not None:
        if out.shape != result.shape:
            raise ValueError(
                f"{pub}: out.shape={tuple(out.shape)} != "
                f"{tuple(result.shape)}"
            )
        if out.dtype != x.dtype or out.device != x.device:
            raise TypeError(f"{pub}: out dtype/device must match input")
        out.copy_(result)
        return out
    return result


def _logical_view(x: torch.Tensor, *, pub: str) -> torch.Tensor:
    if x.dtype == torch.bool:
        return x.float()
    if not x.dtype.is_floating_point:
        raise TypeError(
            f"{pub}: unsupported dtype {x.dtype}; expected bool "
            "or a floating dtype"
        )
    return x


def reduce_sum(
    x: torch.Tensor,
    dim: int,
    *,
    keepdim: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sum of *x* along *dim* (like ``torch.sum(x, dim=…)``)."""
    return _float_reduce_dispatch("sum", x, dim, keepdim=keepdim, out=out)


def reduce_mean(
    x: torch.Tensor,
    dim: int,
    *,
    keepdim: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean of *x* along *dim* (like ``torch.mean(x, dim=…)``)."""
    return _float_reduce_dispatch("mean", x, dim, keepdim=keepdim, out=out)


def reduce_prod(
    x: torch.Tensor,
    dim: int,
    *,
    keepdim: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Product of *x* along *dim* (like ``torch.prod(x, dim=…)``)."""
    return _float_reduce_dispatch("prod", x, dim, keepdim=keepdim, out=out)


def reduce_amax(
    x: torch.Tensor,
    dim: int,
    *,
    keepdim: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Maximum of *x* along *dim* (like ``torch.amax(x, dim=…)``)."""
    return _float_reduce_dispatch("amax", x, dim, keepdim=keepdim, out=out)


def reduce_amin(
    x: torch.Tensor,
    dim: int,
    *,
    keepdim: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Minimum of *x* along *dim* (like ``torch.amin(x, dim=…)``)."""
    return _float_reduce_dispatch("amin", x, dim, keepdim=keepdim, out=out)


def reduce_argmax(
    x: torch.Tensor,
    dim: int,
    *,
    keepdim: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Argmax along *dim* (``torch.argmax`` semantics for ties: first wins).
    Returned dtype is ``torch.int64``.
    """
    if not x.dtype.is_floating_point:
        raise TypeError(
            "reduce_argmax: input must have a floating dtype "
            f"(got {x.dtype})"
        )
    dim_n = dim if dim >= 0 else dim + x.ndim
    pub = "reduce_argmax"
    x_2d, m, n, _perm, orig = merge_on_dim(x, dim, op_name=pub)
    tl_dtype = torch_to_tl_dtype(x.dtype)
    tgt = default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = default_execution_backend(tgt, dev)
    kernel = dispatch_compile_reduce(
        op_name="argmax",
        target=tgt,
        rows=m,
        cols=n,
        threads=_REDUCE_THREADS,
        dtype=tl_dtype,
        execution_backend=eb,
    )
    out_shape = _reduce_output_shape(orig, dim_n, keepdim=keepdim)
    idx_1d = torch.empty(m, dtype=torch.int64, device=x.device)
    invoke_row_index_reduce_kernel(
        kernel, x_2d, idx_1d,
        tilelang_target=tgt,
        execution_backend=eb,
    )
    result = idx_1d.reshape(out_shape)
    if out is not None:
        if out.shape != result.shape:
            raise ValueError(
                f"{pub}: out.shape={tuple(out.shape)} != "
                f"{tuple(result.shape)}"
            )
        if out.dtype != torch.int64 or out.device != x.device:
            raise TypeError(
                f"{pub}: out must be torch.int64 on the input device"
            )
        out.copy_(result)
        return out
    return result


def reduce_argmin(
    x: torch.Tensor,
    dim: int,
    *,
    keepdim: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Argmin along *dim* (first minimal index wins).  ``torch.int64``."""
    if not x.dtype.is_floating_point:
        raise TypeError(
            "reduce_argmin: input must have a floating dtype "
            f"(got {x.dtype})"
        )
    dim_n = dim if dim >= 0 else dim + x.ndim
    pub = "reduce_argmin"
    x_2d, m, n, _perm, orig = merge_on_dim(x, dim, op_name=pub)
    tl_dtype = torch_to_tl_dtype(x.dtype)
    tgt = default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = default_execution_backend(tgt, dev)
    kernel = dispatch_compile_reduce(
        op_name="argmin",
        target=tgt,
        rows=m,
        cols=n,
        threads=_REDUCE_THREADS,
        dtype=tl_dtype,
        execution_backend=eb,
    )
    out_shape = _reduce_output_shape(orig, dim_n, keepdim=keepdim)
    idx_1d = torch.empty(m, dtype=torch.int64, device=x.device)
    invoke_row_index_reduce_kernel(
        kernel, x_2d, idx_1d,
        tilelang_target=tgt,
        execution_backend=eb,
    )
    result = idx_1d.reshape(out_shape)
    if out is not None:
        if out.shape != result.shape:
            raise ValueError(
                f"{pub}: out.shape={tuple(out.shape)} != "
                f"{tuple(result.shape)}"
            )
        if out.dtype != torch.int64 or out.device != x.device:
            raise TypeError(
                f"{pub}: out must be torch.int64 on the input device"
            )
        out.copy_(result)
        return out
    return result


def reduce_all(
    x: torch.Tensor,
    dim: int,
    *,
    keepdim: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """True iff every merged-row element is non-zero (``torch.all`` for float /
    logical ``and`` for bool).
    """
    dim_n = dim if dim >= 0 else dim + x.ndim
    pub = "reduce_all"
    xw = _logical_view(x, pub=pub)
    x_2d, m, n, _perm, orig = merge_on_dim(xw, dim, op_name=pub)
    tl_dtype = torch_to_tl_dtype(xw.dtype)
    tgt = default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = default_execution_backend(tgt, dev)
    kernel = dispatch_compile_reduce(
        op_name="all",
        target=tgt,
        rows=m,
        cols=n,
        threads=_REDUCE_THREADS,
        dtype=tl_dtype,
        execution_backend=eb,
    )
    out_shape = _reduce_output_shape(orig, dim_n, keepdim=keepdim)
    out_1d = torch.empty(m, dtype=xw.dtype, device=x.device)
    invoke_row_reduce_kernel(
        kernel, x_2d, out_1d,
        tilelang_target=tgt,
        execution_backend=eb,
    )
    result = out_1d.reshape(out_shape) != 0
    if out is not None:
        if out.shape != result.shape:
            raise ValueError(
                f"{pub}: out.shape={tuple(out.shape)} != "
                f"{tuple(result.shape)}"
            )
        if out.dtype != torch.bool or out.device != x.device:
            raise TypeError(
                f"{pub}: out must be torch.bool on the input device"
            )
        out.copy_(result)
        return out
    return result


def reduce_any(
    x: torch.Tensor,
    dim: int,
    *,
    keepdim: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """True iff any merged-row element is non-zero."""
    dim_n = dim if dim >= 0 else dim + x.ndim
    pub = "reduce_any"
    xw = _logical_view(x, pub=pub)
    x_2d, m, n, _perm, orig = merge_on_dim(xw, dim, op_name=pub)
    tl_dtype = torch_to_tl_dtype(xw.dtype)
    tgt = default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = default_execution_backend(tgt, dev)
    kernel = dispatch_compile_reduce(
        op_name="any",
        target=tgt,
        rows=m,
        cols=n,
        threads=_REDUCE_THREADS,
        dtype=tl_dtype,
        execution_backend=eb,
    )
    out_shape = _reduce_output_shape(orig, dim_n, keepdim=keepdim)
    out_1d = torch.empty(m, dtype=xw.dtype, device=x.device)
    invoke_row_reduce_kernel(
        kernel, x_2d, out_1d,
        tilelang_target=tgt,
        execution_backend=eb,
    )
    result = out_1d.reshape(out_shape) != 0
    if out is not None:
        if out.shape != result.shape:
            raise ValueError(
                f"{pub}: out.shape={tuple(out.shape)} != "
                f"{tuple(result.shape)}"
            )
        if out.dtype != torch.bool or out.device != x.device:
            raise TypeError(
                f"{pub}: out must be torch.bool on the input device"
            )
        out.copy_(result)
        return out
    return result


def reduce_cumsum(
    x: torch.Tensor,
    dim: int,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Inclusive cumulative sum along *dim* (like ``torch.cumsum``)."""
    pub = "reduce_cumsum"
    if not x.dtype.is_floating_point:
        raise TypeError(
            f"{pub}: unsupported dtype {x.dtype}; "
            "expected float16, float32, or bfloat16"
        )
    x_2d, m, n, perm, orig = merge_on_dim(x, dim, op_name=pub)
    tl_dtype = torch_to_tl_dtype(x.dtype)
    tgt = default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = default_execution_backend(tgt, dev)
    kernel = dispatch_compile_reduce(
        op_name="cumsum",
        target=tgt,
        rows=m,
        cols=n,
        threads=_REDUCE_THREADS,
        dtype=tl_dtype,
        execution_backend=eb,
    )
    work = torch.empty_like(x_2d)
    out_perm = invoke_unary_kernel(
        kernel, x_2d,
        out=work,
        tilelang_target=tgt,
        execution_backend=eb,
    )
    # Reshape permutation layout back to the user's axis order.
    result = unmerge(out_perm.reshape(m, n), orig, perm)
    if out is not None:
        if out.shape != result.shape:
            raise ValueError(
                f"{pub}: out.shape={tuple(out.shape)} != "
                f"{tuple(result.shape)}"
            )
        if out.dtype != x.dtype or out.device != x.device:
            raise TypeError(f"{pub}: out dtype/device must match input")
        out.copy_(result)
        return out
    return result


__all__ = [
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
]
