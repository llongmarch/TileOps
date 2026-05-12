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
   :func:`invoke_unary_kernel`, :func:`make_kernel_runner`,
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
    "invoke_kernel",
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
    out_flat = out.contiguous().view(-1)

    kind = target_kind(tilelang_target)
    if kind == "metal" and execution_backend == "torch":
        kernel(*flat_inputs, out_flat)
        return out

    ret = kernel(*flat_inputs)
    if ret is not None:
        out_flat.copy_(ret)
    else:
        kernel(*flat_inputs, out_flat)
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
