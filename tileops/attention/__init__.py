"""Flash Attention v1 and v2.

Implements Dao et al. (2022) Flash Attention v1 and Dao (2023) Flash
Attention v2 as row-wise TileLang kernels with online-softmax rescaling.
Each query token is processed by an independent thread block.

The kernel is compiled once per unique ``(seq_len_q, seq_len_k, head_dim)``
shape and reused across batch and head dimensions, following the same pattern
as :func:`~tileops.bmm`.

Supported dtypes: ``float32``, ``float16``, ``bfloat16``.

**API**

* :func:`flash_attn_v1` — per-element online-softmax over K/V (v1 algorithm).
* :func:`flash_attn_v2` — tile-based correction, fewer exp evaluations (v2 algorithm).

::

    from tileops.attention import flash_attn_v1, flash_attn_v2
    y = flash_attn_v1(q, k, v)
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from tileops.runtime import (
    default_execution_backend,
    default_tilelang_target,
    default_torch_device,
    invoke_attention_kernel,
    torch_to_tl_dtype,
)


def _validate_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    op: str,
) -> tuple[int, int, int, int]:
    """Validate Q/K/V shapes and return (batch, num_heads, seq_len_q, head_dim)."""
    if q.ndim != 4:
        raise ValueError(
            f"{op}: expected 4-D Q, K, V (batch, num_heads, seq_len, head_dim), "
            f"got Q.ndim={q.ndim}"
        )
    if k.ndim != 4 or v.ndim != 4:
        raise ValueError(f"{op}: K and V must be 4-D")
    batch, num_heads, sq, d = q.shape
    _, _, sk, dk = k.shape
    _, _, sv, dv = v.shape
    if sk != sv:
        raise ValueError(
            f"{op}: K seq_len={sk} != V seq_len={sv}"
        )
    if dk != d or dv != d:
        raise ValueError(
            f"{op}: head dim mismatch Q={d}, K={dk}, V={dv}"
        )
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError(f"{op}: batch size mismatch")
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise ValueError(f"{op}: num_heads mismatch")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError(f"{op}: Q, K, V dtypes must match")
    if q.device != k.device or q.device != v.device:
        raise TypeError(f"{op}: Q, K, V devices must match")
    return batch, num_heads, sq, d


def _attention_impl(
    op_name: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: Optional[float],
    out: Optional[torch.Tensor],
) -> torch.Tensor:
    """Shared implementation for flash_attn_v1 / flash_attn_v2.

    Compiles one kernel per ``(seq_len_q, seq_len_k, head_dim)`` shape
    and iterates over batch and head dimensions, reusing the compiled
    kernel like :func:`~tileops.bmm`.
    """
    batch, num_heads, sq, d = _validate_qkv(q, k, v, op=op_name)
    n = k.shape[2]  # seq_len_k
    scale = softmax_scale if softmax_scale is not None else (1.0 / math.sqrt(float(d)))

    tl_dtype = torch_to_tl_dtype(q.dtype)
    tgt = default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = default_execution_backend(tgt, dev)

    from tileops.runtime import dispatch_compile_attention

    kernel = dispatch_compile_attention(
        op_name=op_name,
        target=tgt,
        m=sq,   # per-head query rows
        n=n,    # per-head key/value rows
        d=d,
        scale=scale,
        dtype=tl_dtype,
        execution_backend=eb,
    )

    if out is None:
        out_buf = torch.empty_like(q)
    else:
        if out.shape != q.shape:
            raise ValueError(
                f"{op_name}: out.shape={tuple(out.shape)} != input "
                f"{tuple(q.shape)}"
            )
        if out.dtype != q.dtype or out.device != q.device:
            raise TypeError(f"{op_name}: out dtype/device must match Q")
        out_buf = out

    for bi in range(batch):
        for hi in range(num_heads):
            q_bh = q[bi, hi].contiguous()  # (sq, d)
            k_bh = k[bi, hi].contiguous()  # (n,  d)
            v_bh = v[bi, hi].contiguous()  # (n,  d)
            invoke_attention_kernel(
                kernel, q_bh, k_bh, v_bh, out=out_buf[bi, hi],
                tilelang_target=tgt, execution_backend=eb,
            )

    return out_buf


def flash_attn_v1(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Flash Attention v1 (Dao et al., 2022).

    Per-element online-softmax over the K/V sequence.  Each query token
    processes all key/value tokens sequentially, maintaining a running
    maximum and denominator with online rescaling.

    Parameters
    ----------
    q : torch.Tensor
        Queries of shape ``(batch, num_heads, seq_len_q, head_dim)``.
    k : torch.Tensor
        Keys of shape ``(batch, num_heads, seq_len_k, head_dim)``.
    v : torch.Tensor
        Values of shape ``(batch, num_heads, seq_len_k, head_dim)``.
    softmax_scale : float, optional
        Scale applied to QK^T before softmax (default: ``1 / sqrt(head_dim)``).
    out : torch.Tensor, optional
        Pre-allocated output of shape ``(batch, num_heads, seq_len_q, head_dim)``.

    Returns
    -------
    torch.Tensor
        Attention output, same shape as *q*.
    """
    return _attention_impl(
        "flash_attn_v1", q, k, v,
        softmax_scale=softmax_scale, out=out,
    )


def flash_attn_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Flash Attention v2 (Dao, 2023).

    Tile-based online-softmax with a single correction factor per KV tile.
    Compared to v1, it first finds the local maximum within each tile, then
    applies one correction ``exp(m_old - m_new)`` to the running state before
    accumulating tile contributions — fewer exponential evaluations and
    improved numerical stability.

    Parameters
    ----------
    q : torch.Tensor
        Queries of shape ``(batch, num_heads, seq_len_q, head_dim)``.
    k : torch.Tensor
        Keys of shape ``(batch, num_heads, seq_len_k, head_dim)``.
    v : torch.Tensor
        Values of shape ``(batch, num_heads, seq_len_k, head_dim)``.
    softmax_scale : float, optional
        Scale applied to QK^T before softmax (default: ``1 / sqrt(head_dim)``).
    out : torch.Tensor, optional
        Pre-allocated output of shape ``(batch, num_heads, seq_len_q, head_dim)``.

    Returns
    -------
    torch.Tensor
        Attention output, same shape as *q*.
    """
    return _attention_impl(
        "flash_attn_v2", q, k, v,
        softmax_scale=softmax_scale, out=out,
    )


__all__ = [
    "flash_attn_v1",
    "flash_attn_v2",
]
