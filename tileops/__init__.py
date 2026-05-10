"""TileOps — high-performance deep-learning operators built on TileLang.

Prefer importing from the package root::

    from tileops import pointwise_add, default_tilelang_target

All public symbols are listed in :data:`__all__`.  Kernel implementations
live in submodules (e.g. ``tileops.pointwise``); runtime helpers live in
``tileops.runtime``.
"""

# ── Re-exports: runtime ─────────────────────────────────────────────────
from tileops.runtime import (
    bench_ms,
    compile_prim,
    default_execution_backend,
    default_tilelang_target,
    default_torch_device,
    heuristic_tilelang_target,
    invoke_kernel,
    make_kernel_runner,
    normalize_elem_dtype,
    require_float32_prim,
    setup_metal_workarounds,
    suggest_pointwise_config,
    sync_device,
    target_kind,
)

# ── Re-exports: pointwise kernels ───────────────────────────────────────
from tileops.pointwise import (
    pointwise_add,
    pointwise_div,
    pointwise_mul,
    pointwise_pow,
    pointwise_sub,
)

__all__ = [
    # runtime / host helpers
    "bench_ms",
    "compile_prim",
    "default_execution_backend",
    "default_tilelang_target",
    "default_torch_device",
    "heuristic_tilelang_target",
    "invoke_kernel",
    "make_kernel_runner",
    "normalize_elem_dtype",
    "require_float32_prim",
    "setup_metal_workarounds",
    "suggest_pointwise_config",
    "sync_device",
    "target_kind",
    # pointwise kernels
    "pointwise_add",
    "pointwise_div",
    "pointwise_mul",
    "pointwise_pow",
    "pointwise_sub",
]
