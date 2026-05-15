"""Internal infrastructure shared by backend and facade modules.

**Backend helpers** (used by ``cuda/*.py``, ``metal/*.py``):

* :func:`make_cached_compiler` — wraps a prim-builder table with an
  in-memory kernel cache.
* :func:`make_backend_op` — generates a backend compile function from a
  cached compiler.

**Facade helpers** (used by ``binary.py``, ``activation.py``, ``fused.py``):

* :data:`BACKEND_PACKAGES` — target-kind → package mapping.
* :func:`dispatch_compile` — resolve backend, import module, compile op.
"""

from __future__ import annotations

import importlib
import math
from typing import Any, Callable, Optional, Sequence

from tileops.runtime import (
    compile_prim,
    default_tilelang_target,
    target_kind,
)

# ═══════════════════════════════════════════════════════════════════════════
# Backend helpers
# ═══════════════════════════════════════════════════════════════════════════


def make_cached_compiler(
    prim_table: dict[str, Callable[..., Any]],
    *,
    cache_prefix: str = "",
) -> Callable[..., Any]:
    """Return a compile function backed by an in-memory kernel cache.

    Parameters
    ----------
    prim_table
        ``{op_name: builder(num_elements, block_size, threads, dtype) -> PrimFunc}``.
    cache_prefix
        Disambiguates cache entries when multiple compilers share a process.
    """
    cache: dict[tuple, Any] = {}

    def compile_fn(
        op_name: str,
        num_elements: int,
        block_size: int,
        threads: int,
        dtype: Any,
        *,
        target: Optional[str] = None,
        execution_backend: Optional[str] = None,
    ) -> Any:
        tgt = target if target is not None else default_tilelang_target()
        key = (cache_prefix, op_name, num_elements, block_size, threads,
               dtype, tgt, execution_backend)
        hit = cache.get(key)
        if hit is not None:
            return hit
        kernel = compile_prim(
            prim_table[op_name](num_elements, block_size, threads, dtype),
            target=tgt,
            execution_backend=execution_backend,
        )
        cache[key] = kernel
        return kernel

    return compile_fn


def make_backend_op(
    op_name: str,
    *,
    compile_fn: Callable[..., Any],
    doc: str = "",
) -> Callable[..., Any]:
    """Build a backend compile function that delegates to *compile_fn*.

    The returned callable has signature::

        op(shape, block_size=1024, threads=128, *, dtype,
           target, execution_backend) -> compiled_kernel
    """

    def op(
        shape: Sequence[int],
        block_size: int = 1024,
        threads: int = 128,
        *,
        dtype: Any,
        target: Optional[str] = None,
        execution_backend: Optional[str] = None,
    ) -> Any:
        return compile_fn(
            op_name,
            math.prod(shape),
            block_size,
            threads,
            dtype,
            target=target,
            execution_backend=execution_backend,
        )

    op.__name__ = op.__qualname__ = op_name
    op.__doc__ = doc
    return op


# ═══════════════════════════════════════════════════════════════════════════
# Facade helpers
# ═══════════════════════════════════════════════════════════════════════════

BACKEND_PACKAGES: dict[str, str] = {
    "cuda": "tileops.cuda",
    "hip": "tileops.cuda",
    "cutedsl": "tileops.cuda",
    "metal": "tileops.metal",
}


def dispatch_compile(
    *,
    module_name: str,
    op_name: str,
    target: str,
    shape: Sequence[int],
    block_size: int,
    threads: int,
    dtype: Any,
    execution_backend: Optional[str] = None,
    backend_packages: dict[str, str] = BACKEND_PACKAGES,
) -> Any:
    """Resolve backend package and compile the kernel for *op_name*.

    Returns a compiled kernel object ready to be called via
    :func:`~tileops.runtime.invoke_kernel` /
    :func:`~tileops.runtime.invoke_unary_kernel`.
    """
    kind = target_kind(target)
    pkg = backend_packages.get(kind)
    if pkg is None:
        raise NotImplementedError(
            f"No {module_name} backend for target kind {kind!r} ({target!r}). "
            f"Supported: {', '.join(sorted(backend_packages))}."
        )
    try:
        mod = importlib.import_module(f"{pkg}.{module_name}")
    except ImportError as exc:
        raise ImportError(
            f"Failed to import {pkg}.{module_name} for target {target!r}"
        ) from exc

    fn = getattr(mod, op_name, None)
    if fn is None:
        raise NotImplementedError(
            f"{pkg}.{module_name} has no {op_name}() for {target!r}"
        )
    return fn(shape, block_size, threads, dtype=dtype, target=target,
              execution_backend=execution_backend)


def dispatch_compile_norm(
    *,
    op_name: str,
    target: str,
    rows: int,
    cols: int,
    threads: int,
    dtype: Any,
    execution_backend: Optional[str] = None,
    eps: Optional[float] = None,
    backend_packages: dict[str, str] = BACKEND_PACKAGES,
) -> Any:
    """Compile a row-wise norm / softmax kernel (``tileops.*.norm``).

    *rows* × *cols* is the logical 2-D view (leading dims merged, norm axis
    last).  *eps* is forwarded for normalization cache keys (
    ``layer_norm``, ``rms_norm``, ``skip_rms_norm``, ``skip_layer_norm``).
    """
    kind = target_kind(target)
    pkg = backend_packages.get(kind)
    if pkg is None:
        raise NotImplementedError(
            f"No norm backend for target kind {kind!r} ({target!r}). "
            f"Supported: {', '.join(sorted(backend_packages))}."
        )
    try:
        mod = importlib.import_module(f"{pkg}.norm")
    except ImportError as exc:
        raise ImportError(
            f"Failed to import {pkg}.norm for target {target!r}"
        ) from exc

    fn = getattr(mod, op_name, None)
    if fn is None:
        raise NotImplementedError(
            f"{pkg}.norm has no {op_name}() for {target!r}"
        )
    return fn(
        rows, cols, threads,
        dtype=dtype,
        target=target,
        execution_backend=execution_backend,
        eps=eps,
    )


def dispatch_compile_blas(
    *,
    op_name: str,
    target: str,
    m: int,
    k: int,
    dtype: Any,
    execution_backend: Optional[str] = None,
    n: Optional[int] = None,
    backend_packages: dict[str, str] = BACKEND_PACKAGES,
) -> Any:
    """Compile a BLAS-style kernel from ``tileops.*.blas``.

    *op_name* is ``"gemm"`` or ``"gemv"``.  Facade wrappers such as ``mm``,
    ``outer``, ``bmm``, ``addmm`` reuse ``gemm`` with the shapes described in
    :mod:`tileops.blas`.  For ``gemm``, pass the inner dimension *k* and the
    trailing matrix width *n* (``C`` is *m* × *n*).
    For ``gemv``, *n* must be ``None`` — only *m* and *k* are used.
    """
    kind = target_kind(target)
    pkg = backend_packages.get(kind)
    if pkg is None:
        raise NotImplementedError(
            f"No blas backend for target kind {kind!r} ({target!r}). "
            f"Supported: {', '.join(sorted(backend_packages))}."
        )
    try:
        mod = importlib.import_module(f"{pkg}.blas")
    except ImportError as exc:
        raise ImportError(
            f"Failed to import {pkg}.blas for target {target!r}"
        ) from exc

    fn = getattr(mod, op_name, None)
    if fn is None:
        raise NotImplementedError(
            f"{pkg}.blas has no {op_name}() for {target!r}"
        )
    if op_name == "gemm":
        if n is None:
            raise ValueError("dispatch_compile_blas: gemm requires n=")
        return fn(
            m, n, k,
            dtype=dtype,
            target=target,
            execution_backend=execution_backend,
        )
    if op_name == "gemv":
        return fn(
            m, k,
            dtype=dtype,
            target=target,
            execution_backend=execution_backend,
        )
    raise ValueError(f"dispatch_compile_blas: unknown op_name {op_name!r}")


_REDUCE_OP_TO_BACKEND_FN: dict[str, str] = {
    "sum": "row_sum",
    "mean": "row_mean",
    "prod": "row_prod",
    "amax": "row_amax",
    "amin": "row_amin",
    "argmax": "row_argmax",
    "argmin": "row_argmin",
    "all": "row_all",
    "any": "row_any",
    "cumsum": "row_cumsum",
}


def dispatch_compile_reduce(
    *,
    op_name: str,
    target: str,
    rows: int,
    cols: int,
    threads: int,
    dtype: Any,
    execution_backend: Optional[str] = None,
    backend_packages: dict[str, str] = BACKEND_PACKAGES,
) -> Any:
    """Compile a row-wise reduction kernel (``tileops.*.reduce``).

    *rows* × *cols* is the logical 2-D view after merging the reduction *dim*
    to the last axis (``cols`` is the extent along that axis).
    """
    kind = target_kind(target)
    pkg = backend_packages.get(kind)
    if pkg is None:
        raise NotImplementedError(
            f"No reduce backend for target kind {kind!r} ({target!r}). "
            f"Supported: {', '.join(sorted(backend_packages))}."
        )
    try:
        mod = importlib.import_module(f"{pkg}.reduce")
    except ImportError as exc:
        raise ImportError(
            f"Failed to import {pkg}.reduce for target {target!r}"
        ) from exc

    fn_name = _REDUCE_OP_TO_BACKEND_FN.get(op_name)
    if fn_name is None:
        raise ValueError(
            f"dispatch_compile_reduce: unknown op_name {op_name!r}; "
            f"expected one of {sorted(_REDUCE_OP_TO_BACKEND_FN)}"
        )
    fn = getattr(mod, fn_name, None)
    if fn is None:
        raise NotImplementedError(
            f"{pkg}.reduce has no {fn_name}() for {target!r}"
        )
    return fn(
        rows, cols, threads,
        dtype=dtype,
        target=target,
        execution_backend=execution_backend,
    )
