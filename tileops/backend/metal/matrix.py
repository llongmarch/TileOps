"""Metal tril / triu — mirrors :mod:`tileops.backend.cuda.matrix`."""

from typing import Any, Optional

import tilelang.language as T

from tileops.runtime import make_generic_cached_compiler

_compile = make_generic_cached_compiler(cache_prefix="metal_matrix")


def tril_kernel(m: int, n: int, diagonal: int, dtype: Any) -> Any:
    d = int(diagonal)

    @T.prim_func
    def _kernel(
        arg0_in: T.Tensor((m, n), dtype),
        arg1_out: T.Tensor((m, n), dtype),
    ):
        with T.Kernel(m, n, threads=1) as (i, j):
            if j <= i + d:
                arg1_out[i, j] = arg0_in[i, j]
            else:
                arg1_out[i, j] = 0.0
    return _kernel


def tril(
    m: int, n: int, diagonal: int, *,
    dtype: Any,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> Any:
    return _compile(
        ("tril", m, n, diagonal, dtype),
        tril_kernel,
        m, n, diagonal, dtype,
        target=target,
        execution_backend=execution_backend,
    )


def triu_kernel(m: int, n: int, diagonal: int, dtype: Any) -> Any:
    d = int(diagonal)

    @T.prim_func
    def _kernel(
        arg0_in: T.Tensor((m, n), dtype),
        arg1_out: T.Tensor((m, n), dtype),
    ):
        with T.Kernel(m, n, threads=1) as (i, j):
            if j >= i + d:
                arg1_out[i, j] = arg0_in[i, j]
            else:
                arg1_out[i, j] = 0.0
    return _kernel


def triu(
    m: int, n: int, diagonal: int, *,
    dtype: Any,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> Any:
    return _compile(
        ("triu", m, n, diagonal, dtype),
        triu_kernel,
        m, n, diagonal, dtype,
        target=target,
        execution_backend=execution_backend,
    )


__all__ = ["tril", "triu"]
