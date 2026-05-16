"""Metal gather kernel — mirrors :mod:`tileops.backend.cuda.indexing`."""

from typing import Any, Optional

import tilelang.language as T

from tileops.runtime import make_generic_cached_compiler

_compile = make_generic_cached_compiler(cache_prefix="metal_indexing")


def gather_kernel(m: int, n_data: int, n_index: int, dtype: Any) -> Any:
    @T.prim_func
    def _kernel(
        arg0_data: T.Tensor((m * n_data,), dtype),
        arg1_index: T.Tensor((m * n_index,), T.int64),
        arg2_output: T.Tensor((m * n_index,), dtype),
    ):
        with T.Kernel(m, threads=1) as (row,):
            base = row * n_data
            obase = row * n_index
            for c in range(n_index):
                idx = arg1_index[obase + c]
                arg2_output[obase + c] = arg0_data[base + idx]

    return _kernel


def gather(
    m: int,
    n_data: int,
    n_index: int,
    *,
    dtype: Any,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> Any:
    return _compile(
        ("gather", m, n_data, n_index, dtype),
        gather_kernel,
        m, n_data, n_index, dtype,
        target=target,
        execution_backend=execution_backend,
    )


__all__ = ["gather"]
