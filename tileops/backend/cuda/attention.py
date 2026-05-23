"""CUDA / HIP Flash Attention v1 and v2.

Each kernel uses one thread block per query token.  Inputs are flat buffers:
*Q* and *O* are length ``M * D``; *K* and *V* are length ``N * D``, where
*M* = total query rows (batch × num_heads × seq_len_q), *N* = key/value
sequence length, and *D* = head dimension.

The facade :mod:`tileops.attention` dispatches here for ``cuda`` / ``hip`` /
``cutedsl`` targets.
"""

from typing import Any, Optional

import tilelang.language as T
from tilelang.language.tir import op as tir_op

from tileops.runtime import make_generic_cached_compiler


def flash_attn_v1_kernel(
    m: int, n: int, d: int, scale: float, dtype: Any,
) -> Any:
    """Flash Attention v1 (Dao et al. 2022): per-element online-softmax over K/V.

    Each Q row processes all K/V elements sequentially, maintaining a running
    max ``m`` and denominator ``l`` with online-softmax rescaling.  The
    correction factor ``exp(m_old - m_new)`` is computed per element whenever
    the max updates.
    """
    @T.prim_func
    def _kernel(
        arg0_Q: T.Tensor((m * d,), dtype),
        arg1_K: T.Tensor((n * d,), dtype),
        arg2_V: T.Tensor((n * d,), dtype),
        arg3_O: T.Tensor((m * d,), dtype),
    ):
        with T.Kernel(m, threads=1) as (row,):
            q_base = row * d
            o_base = row * d

            # reusable scratch buffers
            s_buf = T.alloc_local((1,), T.float32)
            m_buf = T.alloc_local((1,), T.float32)
            l_buf = T.alloc_local((1,), T.float32)

            # ---- initialise with j=0 ----
            s_buf[0] = 0.0
            for di in range(d):
                s_buf[0] = s_buf[0] + arg0_Q[q_base + di] * arg1_K[0 * d + di]
            s_buf[0] = s_buf[0] * scale
            m_buf[0] = s_buf[0]
            l_buf[0] = 1.0

            # O[0] = V[0]  (since exp(s_0 - m) = exp(0) = 1)
            for di in range(d):
                arg3_O[o_base + di] = arg2_V[0 * d + di]

            # ---- process j = 1 … n-1 (element-wise online softmax) ----
            for j in range(1, n):
                s_buf[0] = 0.0
                for di in range(d):
                    s_buf[0] = s_buf[0] + arg0_Q[q_base + di] * arg1_K[j * d + di]
                s_buf[0] = s_buf[0] * scale

                # m_new = max(m, s_j) — via identity (a+b+|a-b|)/2
                a = m_buf[0]
                b = s_buf[0]
                m_new = 0.5 * (a + b + tir_op.abs(a - b))

                # correction = exp(m - m_new),  p = exp(s_j - m_new)
                correction = tir_op.exp(a - m_new)
                p = tir_op.exp(b - m_new)

                # l = correction * l + p
                l_buf[0] = correction * l_buf[0] + p

                # O = correction * O + p * V[j]
                for di in range(d):
                    prev_o = arg3_O[o_base + di]
                    arg3_O[o_base + di] = (
                        correction * prev_o + p * arg2_V[j * d + di]
                    )

                m_buf[0] = m_new

            # ---- normalise: O = O / l ----
            for di in range(d):
                arg3_O[o_base + di] = arg3_O[o_base + di] / l_buf[0]

    return _kernel


def flash_attn_v2_kernel(
    m: int, n: int, d: int, scale: float, dtype: Any,
) -> Any:
    """Flash Attention v2 (Dao 2023): tile-based with a single correction per KV tile.

    Splits K/V into tiles of size *TILE_KV*.  For each tile the local maximum
    is found first, then a single correction factor ``exp(m_old - m_new)``
    rescales the running state before the tile contributions are accumulated.
    This reduces the number of exp evaluations and improves numerical
    stability over the per-element v1 update.
    """
    TILE_KV = 32

    @T.prim_func
    def _kernel(
        arg0_Q: T.Tensor((m * d,), dtype),
        arg1_K: T.Tensor((n * d,), dtype),
        arg2_V: T.Tensor((n * d,), dtype),
        arg3_O: T.Tensor((m * d,), dtype),
    ):
        with T.Kernel(m, threads=1) as (row,):
            q_base = row * d
            o_base = row * d

            # reusable scratch
            s_buf = T.alloc_local((1,), T.float32)
            tm_buf = T.alloc_local((1,), T.float32)   # tile max
            tl_buf = T.alloc_local((1,), T.float32)   # tile denominator
            m_buf = T.alloc_local((1,), T.float32)
            l_buf = T.alloc_local((1,), T.float32)
            correction = T.alloc_local((1,), T.float32)
            p_buf = T.alloc_local((1,), T.float32)

            # ---- running state ----
            m_buf[0] = -1.0e30
            l_buf[0] = 0.0
            for di in range(d):
                arg3_O[o_base + di] = 0.0

            # ---- tile loop over K/V ----
            for tile_start in range(0, n, TILE_KV):
                # upper bound (TILE_KV constant, infer at compile time)
                tile_end = tile_start + TILE_KV
                # clamp: max valid tile length; handled by loop bound below

                # --- step 1: find tile max ---
                s_buf[0] = 0.0
                for di in range(d):
                    s_buf[0] = s_buf[0] + arg0_Q[q_base + di] * arg1_K[tile_start * d + di]
                tm_buf[0] = s_buf[0] * scale

                for j in range(1, TILE_KV):
                    kj = tile_start + j
                    if kj < n:
                        s_buf[0] = 0.0
                        for di in range(d):
                            s_buf[0] = s_buf[0] + arg0_Q[q_base + di] * arg1_K[kj * d + di]
                        s_buf[0] = s_buf[0] * scale
                        a = tm_buf[0]
                        b = s_buf[0]
                        tm_buf[0] = 0.5 * (a + b + tir_op.abs(a - b))

                # --- step 2: global max & correction ---
                a = m_buf[0]
                b = tm_buf[0]
                m_new = 0.5 * (a + b + tir_op.abs(a - b))
                correction[0] = tir_op.exp(a - m_new)

                # --- step 3: rescale running state ---
                l_buf[0] = correction[0] * l_buf[0]
                for di in range(d):
                    arg3_O[o_base + di] = correction[0] * arg3_O[o_base + di]

                # --- step 4: accumulate tile ---
                tl_buf[0] = 0.0
                for j in range(TILE_KV):
                    kj = tile_start + j
                    if kj < n:
                        s_buf[0] = 0.0
                        for di in range(d):
                            s_buf[0] = s_buf[0] + arg0_Q[q_base + di] * arg1_K[kj * d + di]
                        s_buf[0] = s_buf[0] * scale
                        p_buf[0] = tir_op.exp(s_buf[0] - m_new)
                        tl_buf[0] = tl_buf[0] + p_buf[0]
                        for di in range(d):
                            arg3_O[o_base + di] = (
                                arg3_O[o_base + di]
                                + p_buf[0] * arg2_V[kj * d + di]
                            )

                l_buf[0] = l_buf[0] + tl_buf[0]
                m_buf[0] = m_new

            # ---- normalise: O = O / l ----
            for di in range(d):
                arg3_O[o_base + di] = arg3_O[o_base + di] / l_buf[0]

    return _kernel


_ATTENTION_PRIM: dict[str, Any] = {
    "flash_attn_v1": flash_attn_v1_kernel,
    "flash_attn_v2": flash_attn_v2_kernel,
}

_compile_cache = make_generic_cached_compiler(cache_prefix="cuda_attention_v1")


def _compile(
    op_name: str,
    m: int,
    n: int,
    d: int,
    scale: float,
    dtype: Any,
    *,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> Any:
    return _compile_cache(
        (op_name, m, n, d, scale, dtype),
        _ATTENTION_PRIM[op_name],
        m, n, d, scale, dtype,
        target=target,
        execution_backend=execution_backend,
    )


def flash_attn_v1(
    m: int,
    n: int,
    d: int,
    scale: float,
    *,
    dtype: Any,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> Any:
    """Flash Attention v1 compiled kernel.

    Parameters
    ----------
    m : int
        Total query rows (batch × num_heads × seq_len_q).
    n : int
        Key/value sequence length.
    d : int
        Head dimension.
    scale : float
        Softmax scale (typically ``1 / sqrt(d)``).
    """
    return _compile(
        "flash_attn_v1", m, n, d, scale, dtype,
        target=target, execution_backend=execution_backend,
    )


def flash_attn_v2(
    m: int,
    n: int,
    d: int,
    scale: float,
    *,
    dtype: Any,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> Any:
    """Flash Attention v2 compiled kernel (tile-based correction, fewer exp)."""
    return _compile(
        "flash_attn_v2", m, n, d, scale, dtype,
        target=target, execution_backend=execution_backend,
    )


__all__ = [
    "flash_attn_v1",
    "flash_attn_v2",
]
