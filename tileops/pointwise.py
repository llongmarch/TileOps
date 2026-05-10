"""Binary pointwise TileLang kernels (``docs/ops.md`` — element-wise ops).

Kernels operate on **1-D flat buffers** internally — the public API
accepts an arbitrary N-D *shape* and computes ``num_elements = prod(shape)``.
Callers pass N-D PyTorch tensors; :func:`~tileops.runtime.invoke_kernel`
handles the flatten / reshape transparently.

Scheduling strategy — **GMEM-direct**: a 1-D grid where each thread-block
processes *block_size* contiguous elements.  With ``threads = 128`` and
``block_size = 1024`` each thread handles 8 elements, which gives the
TileLang compiler room for ``LDG.128`` / ``STG.128`` vectorised loads /
stores.  A tail-block guard (``if idx < num_elements``) ensures
correctness when *num_elements* is not divisible by *block_size*.

Backend-specific optimisations (e.g. shared-memory staging for CUDA) live
in ``tileops.cuda.pointwise`` / ``tileops.metal.pointwise`` and are
**dispatched automatically** — users always call ``pointwise_add(...)``
from the top-level module.

Public API is re-exported via ``tileops.__init__``; prefer::

    from tileops import pointwise_add
"""

import math
from typing import Any, Callable, Optional, Sequence

import tilelang.language as T
from tilelang.language.tir import op as tir_op

from tileops.runtime import (
    compile_prim,
    default_tilelang_target,
    require_float32_prim,
    target_kind,
)

# ── Compilation cache ────────────────────────────────────────────────────
#
# Keyed by (builder_id, num_elements, block_size, threads, target,
# execution_backend) so that identical compilation requests are served
# from memory.

_KERNEL_CACHE: dict[tuple, Any] = {}

# ── Backend dispatch helper ──────────────────────────────────────────────

_BACKEND_PACKAGES: dict[str, str] = {
    "cuda": "tileops.cuda",
    "hip":  "tileops.cuda",
    "metal": "tileops.metal",
}


def _dispatch_backend(
    op_name: str,
    target: Optional[str],
    shape: Sequence[int],
    block_size: int,
    threads: int,
    **kw: Any,
) -> Optional[Any]:
    """Try to load a backend-specific implementation for *op_name*.

    Returns a compiled kernel if a backend sub-package provides the
    function ``pointwise_<op_name>``, otherwise ``None`` so the caller
    can fall back to the generic GMEM-direct path.
    """
    tgt = target if target is not None else default_tilelang_target()
    kind = target_kind(tgt)
    pkg = _BACKEND_PACKAGES.get(kind)
    if pkg is None:
        return None

    try:
        import importlib
        mod = importlib.import_module(f"{pkg}.pointwise")
    except ImportError:
        return None

    fn = getattr(mod, f"pointwise_{op_name}", None)
    if fn is not None:
        return fn(shape, block_size, threads, target=tgt, **kw)
    return None


# ── Internal: PrimFunc builders ─────────────────────────────────────────
#
# Each builder returns a ``@T.prim_func`` that works on 1-D tensors of
# length *num_elements*, tiled into blocks of *block_size*.
# A tail-block guard ensures correctness for non-divisible sizes.
#
# Signature: ``(num_elements, block_size, threads) → PrimFunc``


def _pointwise_add_prim(
    num_elements: int, block_size: int, threads: int,
) -> Any:
    """GMEM-direct add: ``C[i] = A[i] + B[i]``."""

    @T.prim_func
    def main(
        A: T.Tensor((num_elements,), T.float32),
        B: T.Tensor((num_elements,), T.float32),
        C: T.Tensor((num_elements,), T.float32),
    ):
        with T.Kernel(T.ceildiv(num_elements, block_size), threads=threads) as (bx,):
            for i in T.Parallel(block_size):
                idx = bx * block_size + i
                if idx < num_elements:
                    C[idx] = A[idx] + B[idx]

    return main


def _pointwise_sub_prim(
    num_elements: int, block_size: int, threads: int,
) -> Any:
    """GMEM-direct sub: ``C[i] = A[i] - B[i]``."""

    @T.prim_func
    def main(
        A: T.Tensor((num_elements,), T.float32),
        B: T.Tensor((num_elements,), T.float32),
        C: T.Tensor((num_elements,), T.float32),
    ):
        with T.Kernel(T.ceildiv(num_elements, block_size), threads=threads) as (bx,):
            for i in T.Parallel(block_size):
                idx = bx * block_size + i
                if idx < num_elements:
                    C[idx] = A[idx] - B[idx]

    return main


def _pointwise_mul_prim(
    num_elements: int, block_size: int, threads: int,
) -> Any:
    """GMEM-direct mul: ``C[i] = A[i] * B[i]``."""

    @T.prim_func
    def main(
        A: T.Tensor((num_elements,), T.float32),
        B: T.Tensor((num_elements,), T.float32),
        C: T.Tensor((num_elements,), T.float32),
    ):
        with T.Kernel(T.ceildiv(num_elements, block_size), threads=threads) as (bx,):
            for i in T.Parallel(block_size):
                idx = bx * block_size + i
                if idx < num_elements:
                    C[idx] = A[idx] * B[idx]

    return main


def _pointwise_div_prim(
    num_elements: int, block_size: int, threads: int,
) -> Any:
    """GMEM-direct div: ``C[i] = A[i] / B[i]``."""

    @T.prim_func
    def main(
        A: T.Tensor((num_elements,), T.float32),
        B: T.Tensor((num_elements,), T.float32),
        C: T.Tensor((num_elements,), T.float32),
    ):
        with T.Kernel(T.ceildiv(num_elements, block_size), threads=threads) as (bx,):
            for i in T.Parallel(block_size):
                idx = bx * block_size + i
                if idx < num_elements:
                    C[idx] = A[idx] / B[idx]

    return main


def _pointwise_pow_prim(
    num_elements: int, block_size: int, threads: int,
) -> Any:
    """GMEM-direct pow: ``C[i] = A[i] ** B[i]``.

    Uses ``tir_op.pow`` because Python ``**`` is not supported on TIR
    ``BufferLoad`` objects.
    """

    @T.prim_func
    def main(
        A: T.Tensor((num_elements,), T.float32),
        B: T.Tensor((num_elements,), T.float32),
        C: T.Tensor((num_elements,), T.float32),
    ):
        with T.Kernel(T.ceildiv(num_elements, block_size), threads=threads) as (bx,):
            for i in T.Parallel(block_size):
                idx = bx * block_size + i
                if idx < num_elements:
                    C[idx] = tir_op.pow(A[idx], B[idx])

    return main


# Lookup table for GMEM-direct PrimFunc builders.
_POINTWISE_PRIM: dict[str, Callable[..., Any]] = {
    "add": _pointwise_add_prim,
    "sub": _pointwise_sub_prim,
    "mul": _pointwise_mul_prim,
    "div": _pointwise_div_prim,
    "pow": _pointwise_pow_prim,
}


# ── Internal: cached compile path ───────────────────────────────────────

def _compile_pointwise(
    op_name: str,
    num_elements: int,
    block_size: int,
    threads: int,
    *,
    target: Optional[str],
    execution_backend: Optional[str],
) -> Any:
    """Build a 1-D PrimFunc and compile it, with per-config caching."""
    tgt = target if target is not None else default_tilelang_target()
    eb = execution_backend
    cache_key = (op_name, num_elements, block_size, threads, tgt, eb)
    cached = _KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    prim_builder = _POINTWISE_PRIM[op_name]
    kernel = compile_prim(
        prim_builder(num_elements, block_size, threads),
        target=tgt,
        execution_backend=eb,
    )
    _KERNEL_CACHE[cache_key] = kernel
    return kernel


# ── Public API (factory-generated) ──────────────────────────────────────


def _make_pointwise_op(op_name: str, symbol: str) -> Callable[..., Any]:
    """Generate a ``pointwise_<op>`` public function for *op_name*."""

    def pointwise_op(
        shape: Sequence[int],
        block_size: int = 1024,
        threads: int = 128,
        *,
        in_dtype: Any = "float32",
        out_dtype: Any = "float32",
        target: Optional[str] = None,
        execution_backend: Optional[str] = None,
    ) -> Any:
        require_float32_prim(in_dtype, out_dtype, what=f"pointwise_{op_name}")
        opt = _dispatch_backend(
            op_name, target, shape, block_size, threads,
            execution_backend=execution_backend,
        )
        if opt is not None:
            return opt
        return _compile_pointwise(
            op_name, math.prod(shape), block_size, threads,
            target=target, execution_backend=execution_backend,
        )

    pointwise_op.__name__ = pointwise_op.__qualname__ = f"pointwise_{op_name}"
    pointwise_op.__doc__ = (
        f"Compile an element-wise **{op_name}** kernel: "
        f"``C[i] = A[i] {symbol} B[i]``.\n\n"
        "Parameters\n"
        "----------\n"
        "shape : sequence of int\n"
        "    Tensor shape (any number of dimensions).  Internally flattened to\n"
        "    ``num_elements = prod(shape)``.\n"
        "block_size : int\n"
        "    Elements per thread-block (default 1024).\n"
        "threads : int\n"
        "    Threads per block (default 128).  Each thread processes\n"
        "    ``block_size // threads`` consecutive elements.\n"
        "in_dtype, out_dtype\n"
        "    Must be ``\"float32\"`` (or ``T.float32``).  Reserved for future\n"
        "    multi-dtype support.\n"
        "target, execution_backend\n"
        "    Forwarded to :func:`~tileops.runtime.compile_prim`.\n"
    )
    return pointwise_op


pointwise_add = _make_pointwise_op("add", "+")
pointwise_sub = _make_pointwise_op("sub", "-")
pointwise_mul = _make_pointwise_op("mul", "*")
pointwise_div = _make_pointwise_op("div", "/")
pointwise_pow = _make_pointwise_op("pow", "**")
