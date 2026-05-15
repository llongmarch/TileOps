"""Host-side runtime helpers for TileLang kernel compilation and execution.

Centralises platform-specific logic so that kernel code, examples, and
benchmarks stay portable across CUDA and Metal / MPS backends.

Exported helpers fall into six categories:

1. **Metal workarounds** — :func:`setup_metal_workarounds`
2. **dtype helpers** — :func:`torch_to_tl_dtype`, :func:`normalize_elem_dtype`
3. **Target / device / backend detection** — :func:`default_tilelang_target`,
   :func:`default_torch_device`, :func:`default_execution_backend`, etc.
4. **Tile-size heuristic** — :func:`suggest_tile_config`
   (legacy alias: :func:`suggest_pointwise_config`)
5. **Compilation** — :func:`compile_prim`
6. **Kernel invocation & benchmarking** — :func:`invoke_kernel`,
   :func:`invoke_gemm_kernel`, :func:`invoke_gemv_kernel`,
   :func:`invoke_row_reduce_kernel`, :func:`invoke_row_index_reduce_kernel`,
   :func:`invoke_unary_kernel`, :func:`invoke_nary_kernel`,
   :func:`make_kernel_runner`,
   :func:`make_unary_kernel_runner`, :func:`sync_device`, :func:`bench_ms`

Environment variables
---------------------
``TILELANG_TARGET``
    When set, :func:`default_tilelang_target` returns it verbatim (may
    include arch flags, e.g. ``cuda -arch=sm_90``).
``TILELANG_DISABLE_CACHE``
    Set to ``"1"`` to skip TileLang's disk-cache (works around a Metal
    cache bug on arm64 macOS).
"""

from __future__ import annotations

import os
import platform
import sys
import time
from typing import Any, Callable, List, Optional, Union

import torch
import tilelang
import tilelang.language as T

__all__ = [
    "bench_ms",
    "compile_prim",
    "default_execution_backend",
    "default_tilelang_target",
    "default_torch_device",
    "heuristic_tilelang_target",
    "invoke_conv_kernel",
    "invoke_kernel",
    "invoke_gemm_kernel",
    "invoke_gemv_kernel",
    "invoke_row_reduce_kernel",
    "invoke_row_index_reduce_kernel",
    "invoke_row_topk_kernel",
    "invoke_nary_kernel",
    "invoke_quant_kernel",
    "invoke_unary_kernel",
    "make_kernel_runner",
    "make_unary_kernel_runner",
    "normalize_elem_dtype",
    "setup_metal_workarounds",
    "suggest_pointwise_config",
    "suggest_tile_config",
    "sync_device",
    "target_kind",
    "torch_to_tl_dtype",
]

# ── Metal / arm64 macOS workarounds ──────────────────────────────────────

def setup_metal_workarounds() -> None:
    """Disable TileLang disk-cache on arm64 macOS.

    TileLang's ``MetalKernelAdapter`` may call ``get_kernel_source()`` on a
    ``PrimFunc`` that lacks an ``.imports`` attribute, causing an
    ``AttributeError`` during cache saving.  Setting
    ``TILELANG_DISABLE_CACHE=1`` bypasses that code path.

    Safe to call on any platform — it is a no-op outside arm64 macOS.
    Must be called **before** ``import tilelang`` in scripts.
    """
    if sys.platform == "darwin" and platform.machine() == "arm64":
        os.environ.setdefault("TILELANG_DISABLE_CACHE", "1")


# ── dtype helpers ────────────────────────────────────────────────────────

_DTYPE_MAP: dict[str, Any] = {
    "float32": T.float32,
    "float": T.float32,
    "float16": T.float16,
    "half": T.float16,
    "bfloat16": T.bfloat16,
    "bf16": T.bfloat16,
}

_TORCH_TO_TL: dict[torch.dtype, Any] = {
    torch.float32: T.float32,
    torch.float16: T.float16,
    torch.bfloat16: T.bfloat16,
}


def normalize_elem_dtype(dtype: Any) -> Any:
    """Map common dtype string aliases to ``tilelang.language`` scalar types.

    Pass-through for values that are already TileLang dtype objects.

    >>> normalize_elem_dtype("float32")   # → T.float32
    >>> normalize_elem_dtype("half")      # → T.float16
    >>> normalize_elem_dtype(T.float32)   # → T.float32  (unchanged)
    """
    return _DTYPE_MAP.get(dtype, dtype)


def torch_to_tl_dtype(dtype: torch.dtype) -> Any:
    """Convert a :class:`torch.dtype` to the matching TileLang scalar type.

    Supported: ``float32``, ``float16``, ``bfloat16``.

    >>> torch_to_tl_dtype(torch.float16)  # → T.float16
    """
    tl = _TORCH_TO_TL.get(dtype)
    if tl is None:
        raise ValueError(
            f"Unsupported dtype {dtype}; expected one of "
            f"{', '.join(str(d) for d in _TORCH_TO_TL)}"
        )
    return tl



# ── Target / device / backend detection ─────────────────────────────────

def heuristic_tilelang_target() -> str:
    """Infer a TileLang target from the current hardware.

    Does **not** consult ``TILELANG_TARGET``.  Priority order:

    1. arm64 macOS → ``"metal"``
    2. x86 macOS   → ``"llvm"``
    3. ``torch.cuda.is_available()`` → ``"cuda"``
    4. Fallback    → ``"llvm"``
    """
    if sys.platform == "darwin":
        return "metal" if platform.machine() == "arm64" else "llvm"
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "llvm"


def default_tilelang_target() -> str:
    """Return the TileLang target string for this host.

    Reads ``TILELANG_TARGET`` first; falls back to
    :func:`heuristic_tilelang_target`.
    """
    env = os.environ.get("TILELANG_TARGET")
    return env if env else heuristic_tilelang_target()


def target_kind(tilelang_target: Optional[str]) -> str:
    """Extract the backend name (first token, lower-cased) from a target string.

    >>> target_kind("cuda -arch=sm_90")
    'cuda'
    >>> target_kind(None)
    'llvm'
    """
    if not tilelang_target:
        return "llvm"
    return tilelang_target.split()[0].lower()


def default_torch_device(target: Optional[str] = None) -> str:
    """Return the PyTorch device string that matches *target*.

    ============  =========
    target kind   device
    ============  =========
    cuda / hip    ``"cuda"``
    metal         ``"mps"`` (if available, else ``"cpu"``)
    llvm / other  ``"cpu"``
    ============  =========
    """
    tgt = target or default_tilelang_target()
    kind = target_kind(tgt)
    if kind == "auto":
        return default_torch_device(heuristic_tilelang_target())
    if kind in ("cuda", "cutedsl", "hip"):
        return "cuda"
    if kind == "metal":
        try:
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        raise RuntimeError(
            "TileLang target is 'metal' but PyTorch MPS backend is not "
            "available on this system.  Either set TILELANG_TARGET to a "
            "supported target (e.g. 'cuda', 'llvm') or run on a device "
            "with MPS support (Apple Silicon macOS)."
        )
    return "cpu"


def default_execution_backend(
    tilelang_target: str,
    torch_device: str,
) -> Optional[str]:
    """Suggest an ``execution_backend`` for :func:`compile_prim`.

    Metal + MPS requires ``"torch"``; everything else returns ``None``
    (TileLang auto-selects).
    """
    if target_kind(tilelang_target) == "metal" and torch_device == "mps":
        return "torch"
    return None


# ── Tile-size heuristic ─────────────────────────────────────────────────

def suggest_tile_config(
    num_elements: int,
    *,
    max_block_size: int = 1024,
    preferred_threads: int = 128,
) -> tuple[int, int]:
    """Choose ``(block_size, threads)`` for a 1-D element-wise kernel.

    The heuristic tries ``block_size = threads * k`` for
    ``k ∈ {8, 4, 2, 1}`` (largest first) and picks the first value
    that evenly divides *num_elements* — this eliminates tail-block
    boundary checks.  ``k ≥ 4`` also aligns with 128-bit vector loads
    for float32.

    Parameters
    ----------
    num_elements : int
        Total number of elements (``math.prod(shape)``).
    max_block_size : int
        Upper bound for *block_size*.
    preferred_threads : int
        Thread count per block (not auto-tuned; 128 is a safe default).

    Returns
    -------
    tuple[int, int]
        ``(block_size, threads)``
    """
    if num_elements <= 0:
        raise ValueError(
            f"num_elements must be positive, got {num_elements}"
        )

    threads = preferred_threads
    candidates = [
        threads * k
        for k in (8, 4, 2, 1)
        if threads * k <= max_block_size
    ]
    if not candidates:
        candidates = [max(1, min(max_block_size, num_elements))]

    for bs in candidates:
        if num_elements % bs == 0:
            return bs, threads

    return candidates[0], threads


suggest_pointwise_config = suggest_tile_config
"""Legacy alias for :func:`suggest_tile_config`."""


# ── Compilation ─────────────────────────────────────────────────────────

def compile_prim(
    prim: Any,
    *,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
    out_idx: Union[int, List[int], None] = None,
) -> Any:
    """Compile a TileLang ``PrimFunc`` into a callable kernel.

    Parameters
    ----------
    prim
        A ``@T.prim_func``-decorated function object.
    target
        TileLang target (e.g. ``"cuda"``, ``"metal"``).  Defaults to
        :func:`default_tilelang_target`.
    execution_backend
        Forwarded to ``tilelang.compile`` when not ``None``.
    out_idx
        Output-tensor position(s) in the function signature.  Defaults
        to ``[-1]`` (last parameter is the output).
    """
    if out_idx is None:
        out_idx = [-1]
    tgt = target if target is not None else default_tilelang_target()
    kwargs: dict[str, Any] = {"out_idx": out_idx, "target": tgt}
    if execution_backend is not None:
        kwargs["execution_backend"] = execution_backend
    return tilelang.compile(prim, **kwargs)


# ── Kernel invocation (unified) ─────────────────────────────────────────

def _invoke_impl(
    kernel: Any,
    *inputs: torch.Tensor,
    out: Optional[torch.Tensor],
    tilelang_target: Optional[str],
    execution_backend: Optional[str],
) -> torch.Tensor:
    """Shared logic for :func:`invoke_kernel` / :func:`invoke_unary_kernel`."""
    original_shape = inputs[0].shape
    flat_inputs = [inp.contiguous().view(-1) for inp in inputs]

    if out is None:
        out = torch.empty(original_shape, dtype=inputs[0].dtype,
                          device=inputs[0].device)
    out_contig = out.contiguous()
    out_flat = out_contig.view(-1)

    kind = target_kind(tilelang_target)
    if kind == "metal" and execution_backend == "torch":
        kernel(*flat_inputs, out_flat)
        if out_contig.data_ptr() != out.data_ptr():
            out.copy_(out_contig)
        return out

    ret = kernel(*flat_inputs)
    if ret is not None:
        out_flat.copy_(ret)
    else:
        kernel(*flat_inputs, out_flat)
    if out_contig.data_ptr() != out.data_ptr():
        out.copy_(out_contig)
    return out


def invoke_kernel(
    kernel: Any,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    tilelang_target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> torch.Tensor:
    """Call a compiled binary kernel portably.

    Input tensors of **any shape** are flattened to 1-D before being
    passed to the compiled kernel (which operates on flat buffers).
    The output is reshaped back to the original input shape.

    Parameters
    ----------
    kernel
        Compiled TileLang kernel (from :func:`compile_prim`).
    x, y
        Input tensors (same shape and dtype, must be contiguous).
    out
        Pre-allocated output (same shape as *x*).  Created automatically
        when ``None``.
    tilelang_target, execution_backend
        Used to select the calling convention (Metal needs 3-arg call).
    """
    return _invoke_impl(kernel, x, y, out=out,
                        tilelang_target=tilelang_target,
                        execution_backend=execution_backend)


def invoke_nary_kernel(
    kernel: Any,
    *inputs: Any,
    out: Optional[torch.Tensor] = None,
    tilelang_target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> torch.Tensor:
    """Call a compiled kernel with one or more input tensors (same convention
    as :func:`invoke_kernel`, but arity is not fixed at two).

    The output shape and dtype follow ``inputs[0]``.  All tensors are
    flattened to 1-D row-major buffers before the call.
    """
    if not inputs:
        raise ValueError("invoke_nary_kernel requires at least one input tensor")
    return _invoke_impl(
        kernel, *inputs, out=out,
        tilelang_target=tilelang_target,
        execution_backend=execution_backend,
    )


def invoke_quant_kernel(
    kernel: Any,
    x: torch.Tensor,
    scale: torch.Tensor,
    *,
    out: torch.Tensor,
    tilelang_target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> torch.Tensor:
    """Run a compiled quantize / dequantize kernel.

    *x* and *out* share shape; *scale* is ``float32`` (length 1 or channel
    count).  Buffers are flattened row-major before the call.
    """
    if x.shape != out.shape:
        raise ValueError(
            f"invoke_quant_kernel: x.shape={tuple(x.shape)} != "
            f"out.shape={tuple(out.shape)}"
        )
    if x.device != out.device or scale.device != x.device:
        raise TypeError("invoke_quant_kernel: all tensors must share device")
    if scale.dtype != torch.float32:
        raise TypeError("invoke_quant_kernel: scale must be float32")

    xf = x.contiguous().view(-1)
    sf = scale.contiguous().view(-1)
    of = out.contiguous().view(-1)

    tgt = tilelang_target if tilelang_target is not None else default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = (
        execution_backend
        if execution_backend is not None
        else default_execution_backend(tgt, dev)
    )
    kind = target_kind(tgt)
    if kind == "metal" and eb == "torch":
        kernel(xf, sf, of)
        if of.data_ptr() != out.data_ptr():
            out.copy_(of)
        return out

    ret = kernel(xf, sf)
    if ret is not None:
        of.copy_(ret)
    else:
        kernel(xf, sf, of)
    if of.data_ptr() != out.data_ptr():
        out.copy_(of)
    return out


def invoke_per_token_quant_int8_kernel(
    kernel: Any,
    x: torch.Tensor,
    *,
    out_work: torch.Tensor,
    out_scale: torch.Tensor,
    tilelang_target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run per-row dynamic INT8 quant: ``scale = absmax/127``, ``q = round(x/scale)``.

    *x* is 2-D ``(rows, cols)``; *out_work* is float32 clamped intermediates;
    *out_scale* is 1-D ``(rows,)`` dequant scales.
    """
    if x.ndim != 2:
        raise ValueError(
            f"invoke_per_token_quant_int8_kernel: x must be 2-D, got {x.ndim}D"
        )
    rows, cols = x.shape
    if out_work.shape != (rows, cols) or out_work.dtype != torch.float32:
        raise ValueError(
            f"invoke_per_token_quant_int8_kernel: out_work must be float32 "
            f"({rows}, {cols})"
        )
    if out_scale.shape != (rows,) or out_scale.dtype != torch.float32:
        raise ValueError(
            f"invoke_per_token_quant_int8_kernel: out_scale must be float32 "
            f"({rows},)"
        )
    if x.device != out_work.device or out_scale.device != x.device:
        raise TypeError(
            "invoke_per_token_quant_int8_kernel: all tensors must share device"
        )

    xf = x.contiguous().view(-1)
    yf = out_work.contiguous().view(-1)
    sf = out_scale.contiguous().view(-1)

    tgt = tilelang_target if tilelang_target is not None else default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = (
        execution_backend
        if execution_backend is not None
        else default_execution_backend(tgt, dev)
    )
    kind = target_kind(tgt)
    if kind == "metal" and eb == "torch":
        kernel(xf, yf, sf)
    else:
        ret = kernel(xf, yf)
        if ret is not None:
            yf.copy_(ret)
        else:
            kernel(xf, yf, sf)
    return out_work, out_scale


def invoke_gemm_kernel(
    kernel: Any,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    tilelang_target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> torch.Tensor:
    """Run a compiled ``gemm`` kernel: ``C = A @ B`` (row-major, 2-D only).

    *A* must be ``(M, K)``, *B* ``(K, N)``; output is ``(M, N)``.  Tensors must
    be contiguous on the device.  Metal + ``execution_backend="torch"`` uses
    ``kernel(A, B, C)``; CUDA follows the same calling convention as other
    TileLang torch kernels.
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("invoke_gemm_kernel expects 2-D A and B")
    m, k_a = a.shape
    k_b, n = b.shape
    if k_a != k_b:
        raise ValueError(
            f"gemm: inner dimensions mismatch {k_a} vs {k_b} "
            f"for shapes {tuple(a.shape)} @ {tuple(b.shape)}"
        )
    if a.dtype != b.dtype:
        raise TypeError("gemm: A and B must have the same dtype")
    tgt = tilelang_target if tilelang_target is not None else default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = (
        execution_backend
        if execution_backend is not None
        else default_execution_backend(tgt, dev)
    )
    aa = a.contiguous()
    bb = b.contiguous()
    if out is None:
        out = torch.empty(m, n, dtype=a.dtype, device=a.device)
    else:
        if out.shape != (m, n):
            raise ValueError(
                f"gemm: out.shape={tuple(out.shape)} != ({m}, {n})"
            )
        if out.dtype != a.dtype or out.device != a.device:
            raise TypeError("gemm: out dtype/device must match A")
    cc = out.contiguous()
    kind = target_kind(tgt)
    if kind == "metal" and eb == "torch":
        kernel(aa, bb, cc)
        if cc.data_ptr() != out.data_ptr():
            out.copy_(cc)
        return out
    ret = kernel(aa, bb)
    if ret is not None:
        cc.copy_(ret)
    else:
        kernel(aa, bb, cc)
    if cc.data_ptr() != out.data_ptr():
        out.copy_(cc)
    return out


def invoke_gemv_kernel(
    kernel: Any,
    a: torch.Tensor,
    x: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    tilelang_target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> torch.Tensor:
    """Run a compiled ``gemv`` kernel: ``y = A @ x`` (row-major).

    *A* is ``(M, K)``, *x* is ``(K,)``; output is ``(M,)``.
    """
    if a.ndim != 2 or x.ndim != 1:
        raise ValueError("invoke_gemv_kernel expects A 2-D and x 1-D")
    m, k_a = a.shape
    if x.shape[0] != k_a:
        raise ValueError(
            f"gemv: A.shape[1]={k_a} != len(x)={x.shape[0]}"
        )
    if a.dtype != x.dtype:
        raise TypeError("gemv: A and x must have the same dtype")
    tgt = tilelang_target if tilelang_target is not None else default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = (
        execution_backend
        if execution_backend is not None
        else default_execution_backend(tgt, dev)
    )
    aa = a.contiguous()
    xx = x.contiguous()
    if out is None:
        out = torch.empty(m, dtype=a.dtype, device=a.device)
    else:
        if out.shape != (m,):
            raise ValueError(f"gemv: out.shape={tuple(out.shape)} != ({m},)")
        if out.dtype != a.dtype or out.device != a.device:
            raise TypeError("gemv: out dtype/device must match A")
    yy = out.contiguous()
    kind = target_kind(tgt)
    if kind == "metal" and eb == "torch":
        kernel(aa, xx, yy)
        if yy.data_ptr() != out.data_ptr():
            out.copy_(yy)
        return out
    ret = kernel(aa, xx)
    if ret is not None:
        yy.copy_(ret)
    else:
        kernel(aa, xx, yy)
    if yy.data_ptr() != out.data_ptr():
        out.copy_(yy)
    return out


def invoke_row_reduce_kernel(
    kernel: Any,
    x_2d: torch.Tensor,
    out_1d: torch.Tensor,
    *,
    tilelang_target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> torch.Tensor:
    """Run a compiled row reduction: flat *x* ``(M * N,)`` layout of *x_2d*
    ``(M, N)`` row-major, *out_1d* length *M*.

    Used by ``tileops.reduction`` after merging the reduction axis to the
    last dimension.
    """
    if x_2d.ndim != 2:
        raise ValueError("invoke_row_reduce_kernel expects 2-D x_2d")
    m, n = x_2d.shape
    if out_1d.ndim != 1 or out_1d.numel() != m:
        raise ValueError(
            f"invoke_row_reduce_kernel: out_1d must be 1-D of length {m}, "
            f"got shape {tuple(out_1d.shape)}"
        )
    if x_2d.dtype != out_1d.dtype or x_2d.device != out_1d.device:
        raise TypeError("invoke_row_reduce_kernel: dtype/device mismatch")
    xf = x_2d.contiguous().reshape(-1)
    of = out_1d.contiguous().reshape(-1)
    tgt = tilelang_target if tilelang_target is not None else default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = (
        execution_backend
        if execution_backend is not None
        else default_execution_backend(tgt, dev)
    )
    kind = target_kind(tgt)
    if kind == "metal" and eb == "torch":
        kernel(xf, of)
        if of.data_ptr() != out_1d.data_ptr():
            out_1d.copy_(of)
        return out_1d
    ret = kernel(xf)
    if ret is not None:
        of.copy_(ret)
    else:
        kernel(xf, of)
    if of.data_ptr() != out_1d.data_ptr():
        out_1d.copy_(of)
    return out_1d


def invoke_row_index_reduce_kernel(
    kernel: Any,
    x_2d: torch.Tensor,
    out_index: torch.Tensor,
    *,
    tilelang_target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> torch.Tensor:
    """Same layout as :func:`invoke_row_reduce_kernel`, but *out_index* is
    ``torch.int64`` (one reduction index per row).  Input/output dtypes may
    differ; devices must match.
    """
    if x_2d.ndim != 2:
        raise ValueError("invoke_row_index_reduce_kernel expects 2-D x_2d")
    m = x_2d.shape[0]
    if out_index.ndim != 1 or out_index.numel() != m:
        raise ValueError(
            f"invoke_row_index_reduce_kernel: out_index must be 1-D "
            f"of length {m}, got shape {tuple(out_index.shape)}"
        )
    if out_index.dtype != torch.int64:
        raise TypeError(
            "invoke_row_index_reduce_kernel: out_index must be torch.int64"
        )
    if x_2d.device != out_index.device:
        raise TypeError(
            "invoke_row_index_reduce_kernel: x_2d and out_index "
            "must live on the same device"
        )
    xf = x_2d.contiguous().reshape(-1)
    oi = out_index.contiguous().reshape(-1)
    tgt = tilelang_target if tilelang_target is not None else default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = (
        execution_backend
        if execution_backend is not None
        else default_execution_backend(tgt, dev)
    )
    kind = target_kind(tgt)
    if kind == "metal" and eb == "torch":
        kernel(xf, oi)
        if oi.data_ptr() != out_index.data_ptr():
            out_index.copy_(oi)
        return out_index
    ret = kernel(xf)
    if ret is not None:
        oi.copy_(ret)
    else:
        kernel(xf, oi)
    if oi.data_ptr() != out_index.data_ptr():
        out_index.copy_(oi)
    return out_index


def invoke_row_topk_kernel(
    kernel: Any,
    x_2d: torch.Tensor,
    out_values: torch.Tensor,
    out_index: torch.Tensor,
    *,
    tilelang_target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row top-*k* along the last axis of a 2-D view.

    *out_values* matches *x_2d* dtype (flattened ``m * k``); *out_index* is
    ``torch.int64`` (flattened ``m * k``).
    """
    if x_2d.ndim != 2:
        raise ValueError("invoke_row_topk_kernel expects 2-D x_2d")
    m, _n = x_2d.shape
    k = out_index.numel() // m if m > 0 else 0
    if out_values.ndim != 1 or out_values.numel() != m * k:
        raise ValueError(
            f"invoke_row_topk_kernel: out_values must be 1-D of length {m * k}"
        )
    if out_index.ndim != 1 or out_index.numel() != m * k:
        raise ValueError(
            f"invoke_row_topk_kernel: out_index must be 1-D of length {m * k}"
        )
    if out_index.dtype != torch.int64:
        raise TypeError("invoke_row_topk_kernel: out_index must be torch.int64")
    if x_2d.device != out_values.device or out_index.device != x_2d.device:
        raise TypeError(
            "invoke_row_topk_kernel: all tensors must share device"
        )
    if out_values.dtype != x_2d.dtype:
        raise TypeError("invoke_row_topk_kernel: out_values dtype must match input")

    xf = x_2d.contiguous().reshape(-1)
    vf = out_values.contiguous().reshape(-1)
    idxf = out_index.contiguous().reshape(-1)
    tgt = tilelang_target if tilelang_target is not None else default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = (
        execution_backend
        if execution_backend is not None
        else default_execution_backend(tgt, dev)
    )
    kind = target_kind(tgt)
    if kind == "metal" and eb == "torch":
        kernel(xf, vf, idxf)
        if vf.data_ptr() != out_values.data_ptr():
            out_values.copy_(vf)
        if idxf.data_ptr() != out_index.data_ptr():
            out_index.copy_(idxf)
        return out_values, out_index
    ret = kernel(xf)
    if ret is not None:
        if isinstance(ret, (tuple, list)):
            vf.copy_(ret[0].reshape(-1))
            idxf.copy_(ret[1].reshape(-1))
        else:
            raise TypeError(
                "invoke_row_topk_kernel: expected (values, indices) from kernel"
            )
    else:
        kernel(xf, vf, idxf)
    if vf.data_ptr() != out_values.data_ptr():
        out_values.copy_(vf)
    if idxf.data_ptr() != out_index.data_ptr():
        out_index.copy_(idxf)
    return out_values, out_index


def invoke_unary_kernel(
    kernel: Any,
    x: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    tilelang_target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> torch.Tensor:
    """Call a compiled unary kernel portably.

    Flattens *x* to 1-D, runs the kernel, reshapes *out* to *x*.shape.
    Metal + ``execution_backend="torch"`` uses a two-argument call
    ``kernel(x_flat, out_flat)``; other backends use the same convention
    as :func:`invoke_kernel` but with a single input buffer.
    """
    return _invoke_impl(kernel, x, out=out,
                        tilelang_target=tilelang_target,
                        execution_backend=execution_backend)


def invoke_conv_kernel(
    kernel: Any,
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    tilelang_target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> torch.Tensor:
    """Run a compiled convolution kernel: ``out = conv(input, weight, bias)``.

    *input*, *weight* keep their logical shapes; *bias* is 1-D ``(C_out,)``.
    All tensors are contiguous on-device.  Metal + ``execution_backend="torch"``
    uses ``kernel(input, weight, bias, out)``; CUDA follows the same convention.
    """
    if input.device != weight.device or bias.device != input.device:
        raise TypeError("invoke_conv_kernel: all tensors must share device")
    if out is not None:
        if out.device != input.device:
            raise TypeError("invoke_conv_kernel: out device must match input")
        if out.dtype != input.dtype:
            raise TypeError("invoke_conv_kernel: out dtype must match input")

    inp_c = input.contiguous()
    w_c = weight.contiguous()
    b_c = bias.contiguous()
    if out is None:
        out_c = torch.empty(inp_c.shape[0], w_c.shape[0], *inp_c.shape[2:],
                            dtype=input.dtype, device=input.device)
        # For conv2d with (N, C_out, H_out, W_out); for conv1d (N, C_out, L_out)
        # Infer output spatial dims: input dims - kernel dims + 1 after stride.
        # Actually let's handle this by using the kernel's own output allocation.
        # Simpler: just use `input.new_empty(...)` with the expected output shape
        # from PyTorch convention.
        # For conv1d: output = (N, C_out, L_out)  where L_out = (L - K) // stride + 1
        # For conv2d: output = (N, C_out, H_out, W_out)
        # We don't know the exact output shape at this level; it's baked into
        # the compiled kernel. Let the caller always provide out.
        raise ValueError(
            "invoke_conv_kernel: out= is required (output shape depends on "
            "kernel hyper-parameters baked into the compiled kernel)"
        )
    else:
        out_c = out.contiguous()

    tgt = tilelang_target if tilelang_target is not None else default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = (
        execution_backend
        if execution_backend is not None
        else default_execution_backend(tgt, dev)
    )
    kind = target_kind(tgt)
    if kind == "metal" and eb == "torch":
        kernel(inp_c, w_c, b_c, out_c)
        if out_c.data_ptr() != out.data_ptr():
            out.copy_(out_c)
        return out

    ret = kernel(inp_c, w_c, b_c)
    if ret is not None:
        out_c.copy_(ret)
    else:
        kernel(inp_c, w_c, b_c, out_c)
    if out_c.data_ptr() != out.data_ptr():
        out.copy_(out_c)
    return out


# ── Benchmark runners (unified) ─────────────────────────────────────────

def _make_runner_impl(
    kernel: Any,
    *,
    n_inputs: int,
    tilelang_target: Optional[str],
    execution_backend: Optional[str],
) -> Callable[..., None]:
    """Shared logic for :func:`make_kernel_runner` / :func:`make_unary_kernel_runner`."""
    kind = target_kind(tilelang_target)

    def _run(*args: torch.Tensor) -> None:
        flat = [a.contiguous().view(-1) for a in args[:n_inputs]]
        out_flat = args[n_inputs].contiguous().view(-1)
        if kind == "metal" and execution_backend == "torch":
            kernel(*flat, out_flat)
            return
        ret = kernel(*flat)
        if ret is not None:
            out_flat.copy_(ret)
        else:
            kernel(*flat, out_flat)

    return _run


def make_kernel_runner(
    kernel: Any,
    *,
    tilelang_target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None]:
    """Return a ``run(x, y, out)`` callable suitable for benchmark loops.

    Tensors are flattened to 1-D internally.  Unlike :func:`invoke_kernel`,
    the runner always writes into a caller-provided *out* tensor.
    """
    return _make_runner_impl(kernel, n_inputs=2,
                             tilelang_target=tilelang_target,
                             execution_backend=execution_backend)


def make_unary_kernel_runner(
    kernel: Any,
    *,
    tilelang_target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> Callable[[torch.Tensor, torch.Tensor], None]:
    """Return ``run(x, out)`` for unary activation benchmark loops."""
    return _make_runner_impl(kernel, n_inputs=1,
                             tilelang_target=tilelang_target,
                             execution_backend=execution_backend)


# ── Benchmarking utilities ──────────────────────────────────────────────

def sync_device(device: str) -> None:
    """Block until all pending work on *device* has completed."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def bench_ms(
    fn: Callable[[], None],
    device: str,
    *,
    warmup: int = 10,
    repeat: int = 50,
) -> float:
    """Time *fn* and return the average latency in **milliseconds**.

    Runs *warmup* un-timed iterations, synchronises *device*, then times
    *repeat* iterations with a final synchronisation.
    """
    for _ in range(warmup):
        fn()
    sync_device(device)
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    sync_device(device)
    return (time.perf_counter() - t0) / repeat * 1000.0
