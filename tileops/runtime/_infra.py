"""Internal infrastructure shared by backend and facade modules.

**Backend helpers** (used by ``backend/cuda/*.py``, ``backend/metal/*.py``):

* :func:`make_cached_compiler` — wraps a prim-builder table with an
  in-memory kernel cache.
* :func:`build_op` — generates a backend compile function from a
  cached compiler.

**Facade helpers** (used by ``binary.py``, ``activation.py``, ``fused.py``,
``quant``):

* :data:`BACKEND_PACKAGES` — target-kind → package mapping.
* :func:`dispatch_compile` — resolve backend, import module, compile op.
"""

from __future__ import annotations

import importlib
import math
from typing import Any, Callable, Optional, Sequence

from . import compile_prim, default_tilelang_target, target_kind

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


def build_op(
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


def make_generic_cached_compiler(
    *,
    cache_prefix: str = "",
) -> Callable[..., Any]:
    """Return a compile function with in-memory kernel caching.

    Unlike :func:`make_cached_compiler`, this does **not** prescribe a
    ``(num_elements, block_size, threads, dtype)`` calling convention.
    The caller controls the cache key and prim-builder arguments.

    The returned ``compile_fn`` has signature::

        compile_fn(key_parts, prim_builder, *args,
                   target=None, execution_backend=None, **kwargs) -> kernel

    where *key_parts* is an iterable of hashable values joined with
    *cache_prefix* to form the cache key, *prim_builder* is a callable
    that returns a ``@T.prim_func``, and *args* / *kwargs* are forwarded
    to the builder.
    """
    cache: dict[tuple, Any] = {}

    def compile_fn(
        key_parts: Sequence[Any],
        prim_builder: Callable[..., Any],
        *args: Any,
        target: Optional[str] = None,
        execution_backend: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        tgt = target if target is not None else default_tilelang_target()
        key = (cache_prefix,) + tuple(key_parts) + (tgt, execution_backend)
        hit = cache.get(key)
        if hit is not None:
            return hit
        kernel = compile_prim(
            prim_builder(*args, **kwargs),
            target=tgt,
            execution_backend=execution_backend,
        )
        cache[key] = kernel
        return kernel

    return compile_fn


# ═══════════════════════════════════════════════════════════════════════════
# Facade helpers
# ═══════════════════════════════════════════════════════════════════════════

BACKEND_PACKAGES: dict[str, str] = {
    "cuda": "tileops.backend.cuda",
    "hip": "tileops.backend.cuda",
    "cutedsl": "tileops.backend.cuda",
    "metal": "tileops.backend.metal",
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
    **kwargs: Any,
) -> Any:
    """Resolve backend package and compile the kernel for *op_name*.

    Returns a compiled kernel object ready to be called via
    :func:`~tileops.runtime.invoke_kernel` /
    :func:`~tileops.runtime.invoke_unary_kernel`.

    Extra keyword arguments are forwarded to the backend op function
    (e.g. ``min_val`` / ``max_val`` for ``clamp``).
    """
    kind = target_kind(target)
    pkg = BACKEND_PACKAGES.get(kind)
    if pkg is None:
        raise NotImplementedError(
            f"No {module_name} backend for target kind {kind!r} ({target!r}). "
            f"Supported: {', '.join(sorted(BACKEND_PACKAGES))}."
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
              execution_backend=execution_backend, **kwargs)


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
) -> Any:
    """Compile a row-wise norm / softmax kernel (``tileops.*.norm``).

    *rows* × *cols* is the logical 2-D view (leading dims merged, norm axis
    last).  *eps* is forwarded for normalization cache keys (
    ``layer_norm``, ``rms_norm``, ``skip_rms_norm``, ``skip_layer_norm``).
    """
    kind = target_kind(target)
    pkg = BACKEND_PACKAGES.get(kind)
    if pkg is None:
        raise NotImplementedError(
            f"No norm backend for target kind {kind!r} ({target!r}). "
            f"Supported: {', '.join(sorted(BACKEND_PACKAGES))}."
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
) -> Any:
    """Compile a BLAS-style kernel from ``tileops.*.blas``.

    *op_name* is ``"gemm"`` or ``"gemv"``.  Facade wrappers such as ``mm``,
    ``outer``, ``bmm``, ``addmm`` reuse ``gemm`` with the shapes described in
    :mod:`tileops.blas`.  For ``gemm``, pass the inner dimension *k* and the
    trailing matrix width *n* (``C`` is *m* × *n*).
    For ``gemv``, *n* must be ``None`` — only *m* and *k* are used.
    """
    kind = target_kind(target)
    pkg = BACKEND_PACKAGES.get(kind)
    if pkg is None:
        raise NotImplementedError(
            f"No blas backend for target kind {kind!r} ({target!r}). "
            f"Supported: {', '.join(sorted(BACKEND_PACKAGES))}."
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
    "topk": "row_topk",
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
    k: Optional[int] = None,
    largest: bool = True,
) -> Any:
    """Compile a row-wise reduction kernel (``tileops.*.reduce``).

    *rows* × *cols* is the logical 2-D view after merging the reduction *dim*
    to the last axis (``cols`` is the extent along that axis).
    For ``op_name="topk"``, pass *k* and *largest*.
    """
    kind = target_kind(target)
    pkg = BACKEND_PACKAGES.get(kind)
    if pkg is None:
        raise NotImplementedError(
            f"No reduce backend for target kind {kind!r} ({target!r}). "
            f"Supported: {', '.join(sorted(BACKEND_PACKAGES))}."
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
    if op_name == "topk":
        if k is None:
            raise ValueError("dispatch_compile_reduce: topk requires k=")
        return fn(
            rows, cols, k, threads,
            dtype=dtype,
            largest=largest,
            target=target,
            execution_backend=execution_backend,
        )
    return fn(
        rows, cols, threads,
        dtype=dtype,
        target=target,
        execution_backend=execution_backend,
    )


_QUANT_POINTWISE_OPS = frozenset({
    "quantize_per_tensor",
    "dequantize_per_tensor",
})
_QUANT_CHANNEL_OPS = frozenset({
    "quantize_per_channel",
    "dequantize_per_channel",
})
_QUANT_ROW_OPS = frozenset({
    "per_token_quant_int8",
})


def dispatch_compile_quant(
    *,
    op_name: str,
    target: str,
    shape: Sequence[int],
    block_size: int,
    threads: int,
    dtype: Any,
    execution_backend: Optional[str] = None,
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    eps: float = 1e-10,
) -> Any:
    """Compile a kernel from backend ``tileops.<cuda|metal>.quant``.

    User-facing API is :mod:`tileops.quant` (and :mod:`tileops.quant_int8` for
    dynamic per-token ops).  For per-tensor ops, pass *shape* only.  For
    per-channel ops, pass *rows* and *cols*.  For per-row dynamic quant, use
    ``op_name="per_token_quant_int8"`` with *rows* and *cols*.
    """
    kind = target_kind(target)
    pkg = BACKEND_PACKAGES.get(kind)
    if pkg is None:
        raise NotImplementedError(
            f"No quant backend for target kind {kind!r} ({target!r}). "
            f"Supported: {', '.join(sorted(BACKEND_PACKAGES))}."
        )
    try:
        mod = importlib.import_module(f"{pkg}.quant")
    except ImportError as exc:
        raise ImportError(
            f"Failed to import {pkg}.quant for target {target!r}"
        ) from exc

    fn = getattr(mod, op_name, None)
    if fn is None:
        raise NotImplementedError(
            f"{pkg}.quant has no {op_name}() for {target!r}"
        )

    if op_name in _QUANT_POINTWISE_OPS:
        return fn(
            tuple(shape),
            block_size,
            threads,
            dtype=dtype,
            target=target,
            execution_backend=execution_backend,
        )
    if op_name in _QUANT_CHANNEL_OPS:
        if rows is None or cols is None:
            raise ValueError(
                f"dispatch_compile_quant: {op_name} requires rows= and cols="
            )
        return fn(
            rows,
            cols,
            block_size,
            threads,
            dtype=dtype,
            target=target,
            execution_backend=execution_backend,
        )
    if op_name in _QUANT_ROW_OPS:
        if rows is None or cols is None:
            raise ValueError(
                f"dispatch_compile_quant: {op_name} requires rows= and cols="
            )
        return fn(
            rows,
            cols,
            threads,
            dtype=dtype,
            eps=eps,
            target=target,
            execution_backend=execution_backend,
        )
    raise ValueError(
        f"dispatch_compile_quant: unknown op_name {op_name!r}; "
        f"expected one of {sorted(_QUANT_POINTWISE_OPS | _QUANT_CHANNEL_OPS | _QUANT_ROW_OPS)}"
    )


def dispatch_compile_conv(
    *,
    op_name: str,
    target: str,
    N: int,
    C_out: int,
    C_in: int,
    L: int,
    kernel_size: int,
    stride: int,
    groups: int,
    dtype: Any,
    execution_backend: Optional[str] = None,
    H: Optional[int] = None,
    W: Optional[int] = None,
    kernel_h: Optional[int] = None,
    kernel_w: Optional[int] = None,
    stride_h: Optional[int] = None,
    stride_w: Optional[int] = None,
) -> Any:
    """Compile a convolution kernel from ``tileops.*.conv``.

    For ``conv1d``, pass *L*, *kernel_size*, *stride* (H/W/*_h/*_w are None).
    For ``conv2d``, pass *H*, *W*, *kernel_h*, *kernel_w*, *stride_h*, *stride_w*.
    """
    kind = target_kind(target)
    pkg = BACKEND_PACKAGES.get(kind)
    if pkg is None:
        raise NotImplementedError(
            f"No conv backend for target kind {kind!r} ({target!r}). "
            f"Supported: {', '.join(sorted(BACKEND_PACKAGES))}."
        )
    try:
        mod = importlib.import_module(f"{pkg}.conv")
    except ImportError as exc:
        raise ImportError(
            f"Failed to import {pkg}.conv for target {target!r}"
        ) from exc

    fn = getattr(mod, op_name, None)
    if fn is None:
        raise NotImplementedError(
            f"{pkg}.conv has no {op_name}() for {target!r}"
        )

    if op_name == "conv1d":
        return fn(
            N, C_out, C_in, L,
            kernel_size, stride, groups,
            dtype=dtype,
            target=target,
            execution_backend=execution_backend,
        )
    if op_name == "conv2d":
        if any(v is None for v in (H, W, kernel_h, kernel_w, stride_h, stride_w)):
            raise ValueError(
                "dispatch_compile_conv: conv2d requires H=, W=, kernel_h=, "
                "kernel_w=, stride_h=, stride_w="
            )
        return fn(
            N, C_out, C_in, H, W,
            kernel_h, kernel_w, stride_h, stride_w, groups,
            dtype=dtype,
            target=target,
            execution_backend=execution_backend,
        )
    raise ValueError(
        f"dispatch_compile_conv: unknown op_name {op_name!r}; "
        f"expected 'conv1d' or 'conv2d'"
    )
