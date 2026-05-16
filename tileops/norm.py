"""Row-wise softmax, log-softmax, layer normalization, and RMS normalization.

Operations follow the last reduced axis after merging all leading dimensions
(``torch.nn.functional`` semantics on a reshaped tensor).  Supported dtypes
match the rest of TileOps: ``float32``, ``float16``, ``bfloat16``.

**API**

* :func:`softmax` — two-pass stable softmax (row max, then ``exp`` / sum).
* :func:`safe_softmax` — same numerics as ``softmax`` (explicit “safe” alias).
* :func:`online_softmax` — one-pass streaming max + denominator, same result.
* :func:`log_softmax` — stable reduce over ``dim``.
* :func:`layer_norm` — mean / variance over ``normalized_shape`` (trailing
  dims of ``x``), then ``gamma * x_hat + beta`` (``weight`` / ``bias`` optional).
* :func:`rms_norm` — RMS scaling over ``normalized_shape``, optional ``weight``.
* :func:`skip_rms_norm` — fused ``rms_norm(x + residual, …)`` (common pre-norm block).
* :func:`skip_layer_norm` — fused ``layer_norm(x + residual, …)``.

::

    from tileops import softmax, safe_softmax, online_softmax, layer_norm, rms_norm
    y = softmax(x, dim=-1)
    z = layer_norm(t, (hidden,), weight=w, bias=b)
    h = skip_rms_norm(x, delta, (hidden,), weight=w_rms)
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from tileops.runtime import dispatch_compile_norm
from tileops._shape import (
    merge_on_dim as _merge_on_dim,
    trailing_normalized as _trailing_normalized,
    unmerge as _unmerge,
)
from tileops.runtime import (
    default_execution_backend,
    default_tilelang_target,
    default_torch_device,
    invoke_nary_kernel,
    torch_to_tl_dtype,
)

_NORM_THREADS = 1


def _default_affine(
    n: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    need_bias: bool,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    w = weight if weight is not None else torch.ones(n, device=device, dtype=dtype)
    b = (
        bias if bias is not None else (
            torch.zeros(n, device=device, dtype=dtype) if need_bias else None
        )
    )
    return w, b


def _flatten_affine(
    op: str,
    name: str,
    tensor: torch.Tensor,
    n: int,
    normalized_shape: Sequence[int],
    x: torch.Tensor,
) -> torch.Tensor:
    """Validate affine tensor and flatten PyTorch-shaped weights if needed."""
    expected_tail = tuple(normalized_shape)
    if tensor.shape != (n,) and tuple(tensor.shape) != expected_tail:
        raise ValueError(
            f"{op}: {name}.shape {tuple(tensor.shape)} must be "
            f"({n},) or {expected_tail}"
        )
    if tensor.dtype != x.dtype or tensor.device != x.device:
        raise TypeError(f"{op}: {name} dtype/device must match input")
    return tensor.contiguous().view(n)


def _prepare_affine(
    op: str,
    w: torch.Tensor,
    b: Optional[torch.Tensor],
    n: int,
    normalized_shape: Sequence[int],
    x: torch.Tensor,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    w_flat = _flatten_affine(op, "weight", w, n, normalized_shape, x)
    b_flat = (
        None if b is None
        else _flatten_affine(op, "bias", b, n, normalized_shape, x)
    )
    return w_flat, b_flat


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
    kernel = dispatch_compile_norm(
        op_name="softmax", target=tgt, rows=m, cols=n,
        threads=_NORM_THREADS, dtype=tl_dtype, execution_backend=eb, eps=None,
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
    kernel = dispatch_compile_norm(
        op_name="online_softmax", target=tgt, rows=m, cols=n,
        threads=_NORM_THREADS, dtype=tl_dtype, execution_backend=eb, eps=None,
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
    kernel = dispatch_compile_norm(
        op_name="log_softmax", target=tgt, rows=m, cols=n,
        threads=_NORM_THREADS, dtype=tl_dtype, execution_backend=eb, eps=None,
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


def layer_norm(
    x: torch.Tensor,
    normalized_shape: Sequence[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Layer normalization over trailing ``normalized_shape`` (see ``F.layer_norm``)."""
    x_2d, m, n, orig_shape = _trailing_normalized(x, normalized_shape)
    w, b = _default_affine(n, x.device, x.dtype, weight=weight, bias=bias, need_bias=True)
    w, b = _prepare_affine("layer_norm", w, b, n, normalized_shape, x)
    tl_dtype = torch_to_tl_dtype(x.dtype)
    tgt = default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = default_execution_backend(tgt, dev)
    kernel = dispatch_compile_norm(
        op_name="layer_norm", target=tgt, rows=m, cols=n,
        threads=_NORM_THREADS, dtype=tl_dtype, execution_backend=eb, eps=eps,
    )
    y_2d = invoke_nary_kernel(
        kernel, x_2d, w, b, out=None,
        tilelang_target=tgt, execution_backend=eb,
    )
    result = y_2d.view(orig_shape)
    if out is not None:
        if out.shape != x.shape:
            raise ValueError(f"layer_norm: out.shape={out.shape} != input {x.shape}")
        if out.dtype != x.dtype or out.device != x.device:
            raise TypeError("layer_norm: out dtype/device must match input")
        out.copy_(result)
        return out
    return result


def rms_norm(
    x: torch.Tensor,
    normalized_shape: Sequence[int],
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """RMS normalization: ``x / sqrt(mean(x^2)+eps) * weight``."""
    x_2d, m, n, orig_shape = _trailing_normalized(x, normalized_shape)
    w, _ = _default_affine(n, x.device, x.dtype, weight=weight, bias=None, need_bias=False)
    w, _ = _prepare_affine("rms_norm", w, None, n, normalized_shape, x)
    tl_dtype = torch_to_tl_dtype(x.dtype)
    tgt = default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = default_execution_backend(tgt, dev)
    kernel = dispatch_compile_norm(
        op_name="rms_norm", target=tgt, rows=m, cols=n,
        threads=_NORM_THREADS, dtype=tl_dtype, execution_backend=eb, eps=eps,
    )
    y_2d = invoke_nary_kernel(
        kernel, x_2d, w, out=None,
        tilelang_target=tgt, execution_backend=eb,
    )
    result = y_2d.view(orig_shape)
    if out is not None:
        if out.shape != x.shape:
            raise ValueError(f"rms_norm: out.shape={out.shape} != input {x.shape}")
        if out.dtype != x.dtype or out.device != x.device:
            raise TypeError("rms_norm: out dtype/device must match input")
        out.copy_(result)
        return out
    return result


def skip_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    normalized_shape: Sequence[int],
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """RMS normalization of ``x + residual`` without materializing the sum.

    Matches ``rms_norm(x + residual, …)`` numerically on the fused sum.
    """
    if x.shape != residual.shape:
        raise ValueError(
            "skip_rms_norm: x and residual must have the same shape "
            f"(got {tuple(x.shape)} vs {tuple(residual.shape)})"
        )
    if residual.dtype != x.dtype or residual.device != x.device:
        raise TypeError("skip_rms_norm: residual dtype/device must match x")
    x_2d, m, n, orig_shape = _trailing_normalized(x, normalized_shape)
    r_2d = residual.contiguous().reshape(m, n)
    w, _ = _default_affine(
        n, x.device, x.dtype,
        weight=weight, bias=None, need_bias=False,
    )
    w, _ = _prepare_affine("skip_rms_norm", w, None, n, normalized_shape, x)
    tl_dtype = torch_to_tl_dtype(x.dtype)
    tgt = default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = default_execution_backend(tgt, dev)
    kernel = dispatch_compile_norm(
        op_name="skip_rms_norm", target=tgt, rows=m, cols=n,
        threads=_NORM_THREADS, dtype=tl_dtype,
        execution_backend=eb, eps=eps,
    )
    y_2d = invoke_nary_kernel(
        kernel, x_2d, r_2d, w, out=None,
        tilelang_target=tgt, execution_backend=eb,
    )
    result = y_2d.view(orig_shape)
    if out is not None:
        if out.shape != x.shape:
            raise ValueError(
                f"skip_rms_norm: out.shape={tuple(out.shape)} != input "
                f"{tuple(x.shape)}"
            )
        if out.dtype != x.dtype or out.device != x.device:
            raise TypeError(
                "skip_rms_norm: out dtype/device must match input",
            )
        out.copy_(result)
        return out
    return result


def skip_layer_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    normalized_shape: Sequence[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Layer normalization of ``x + residual`` (fused sum)."""
    if x.shape != residual.shape:
        raise ValueError(
            "skip_layer_norm: x and residual must have the same shape "
            f"(got {tuple(x.shape)} vs {tuple(residual.shape)})"
        )
    if residual.dtype != x.dtype or residual.device != x.device:
        raise TypeError("skip_layer_norm: residual dtype/device must match x")
    x_2d, m, n, orig_shape = _trailing_normalized(x, normalized_shape)
    r_2d = residual.contiguous().reshape(m, n)
    w, b = _default_affine(
        n, x.device, x.dtype,
        weight=weight, bias=bias, need_bias=True,
    )
    w, b = _prepare_affine("skip_layer_norm", w, b, n, normalized_shape, x)
    tl_dtype = torch_to_tl_dtype(x.dtype)
    tgt = default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = default_execution_backend(tgt, dev)
    kernel = dispatch_compile_norm(
        op_name="skip_layer_norm", target=tgt, rows=m, cols=n,
        threads=_NORM_THREADS, dtype=tl_dtype,
        execution_backend=eb, eps=eps,
    )
    assert b is not None
    y_2d = invoke_nary_kernel(
        kernel, x_2d, r_2d, w, b, out=None,
        tilelang_target=tgt, execution_backend=eb,
    )
    result = y_2d.view(orig_shape)
    if out is not None:
        if out.shape != x.shape:
            raise ValueError(
                f"skip_layer_norm: out.shape={tuple(out.shape)} != input "
                f"{tuple(x.shape)}"
            )
        if out.dtype != x.dtype or out.device != x.device:
            raise TypeError(
                "skip_layer_norm: out dtype/device must match input",
            )
        out.copy_(result)
        return out
    return result
