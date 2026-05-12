"""Internal infrastructure shared by backend and facade modules.

**Backend helpers** (used by ``cuda/*.py``, ``metal/*.py``):

* :func:`make_cached_compiler` — wraps a prim-builder table with an
  in-memory kernel cache.
* :func:`make_backend_op` — generates a backend compile function from a
  cached compiler.

**Facade helpers** (used by ``binary.py``, ``activation.py``):

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
