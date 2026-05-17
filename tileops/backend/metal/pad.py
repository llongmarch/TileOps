"""Metal constant-value padding kernels.

Mirrors :mod:`tileops.backend.cuda.pad` with a separate in-process cache prefix.

Two kernels are provided:

* ``pad1d`` — pad the last dimension only.
* ``pad2d`` — pad the last two dimensions.

The output buffer is pre-filled with the constant value by the host; the
kernel only copies original elements to their correct output positions.
"""

from typing import Any, Optional

import tilelang.language as T

from tileops.runtime import make_generic_cached_compiler


def pad1d_kernel(
    m: int, n: int, pad_left: int, pad_right: int, dtype: Any, _value: float,
) -> Any:
    n_out = n + pad_left + pad_right

    @T.prim_func
    def _kernel(
        arg0_X: T.Tensor((m * n,), dtype),
        arg1_Y: T.Tensor((m * n_out,), dtype),
    ):
        with T.Kernel(m, threads=1) as (row,):
            src_base = row * n
            dst_base = row * n_out + pad_left
            for j in range(n):
                arg1_Y[dst_base + j] = arg0_X[src_base + j]

    return _kernel


def pad2d_kernel(
    m: int,
    cols_in: int,
    pad_top: int,
    pad_bottom: int,
    pad_left: int,
    pad_right: int,
    dtype: Any,
    _value: float,
    *,
    h_in: int,
    w_in: int,
) -> Any:
    h_out = h_in + pad_top + pad_bottom
    w_out = w_in + pad_left + pad_right
    n_out = h_out * w_out

    @T.prim_func
    def _kernel(
        arg0_X: T.Tensor((m * h_in * w_in,), dtype),
        arg1_Y: T.Tensor((m * n_out,), dtype),
    ):
        with T.Kernel(m, threads=1) as (slice,):
            src_slice = slice * h_in * w_in
            dst_slice = slice * n_out
            for hi in range(h_in):
                ho = hi + pad_top
                src_row = src_slice + hi * w_in
                dst_row = dst_slice + ho * w_out + pad_left
                for wi in range(w_in):
                    arg1_Y[dst_row + wi] = arg0_X[src_row + wi]

    return _kernel


_PAD_PRIM: dict[str, Any] = {
    "pad1d": pad1d_kernel,
    "pad2d": pad2d_kernel,
}

_compile_cache = make_generic_cached_compiler(cache_prefix="metal_pad_v1")


def _compile(
    op_name: str,
    threads: int,
    dtype: Any,
    *,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    return _compile_cache(
        (op_name, threads, dtype) + tuple(sorted(kwargs.items())),
        _PAD_PRIM[op_name],
        dtype=dtype,
        **kwargs,
        target=target,
        execution_backend=execution_backend,
    )


def pad1d(
    rows: int,
    cols: int,
    threads: int = 1,
    *,
    dtype: Any,
    pad_left: int,
    pad_right: int,
    value: float = 0.0,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> Any:
    return _compile(
        "pad1d", threads, dtype,
        m=rows, n=cols,
        pad_left=pad_left, pad_right=pad_right,
        _value=value,
        target=target, execution_backend=execution_backend,
    )


def pad2d(
    rows: int,
    cols: int,
    threads: int = 1,
    *,
    dtype: Any,
    pad_top: int,
    pad_bottom: int,
    pad_left: int,
    pad_right: int,
    h_in: int,
    w_in: int,
    value: float = 0.0,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> Any:
    return _compile(
        "pad2d", threads, dtype,
        m=rows,
        cols_in=h_in * w_in,
        pad_top=pad_top, pad_bottom=pad_bottom,
        pad_left=pad_left, pad_right=pad_right,
        h_in=h_in, w_in=w_in,
        _value=value,
        target=target, execution_backend=execution_backend,
    )


__all__ = [
    "pad1d",
    "pad2d",
]
