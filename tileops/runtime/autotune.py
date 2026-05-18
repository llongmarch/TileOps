"""Autotune helpers for searching optimal kernel launch configurations.

Integrates with TileLang's profiler to benchmark candidate ``(block_size, threads)``
pairs and return the fastest one.  Results are cached in-memory so repeated calls
for the same operator shape are instant.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Sequence

from tileops.runtime import (
    bench_ms,
    compile_prim,
    default_execution_backend,
    default_tilelang_target,
    default_torch_device,
    make_unary_kernel_runner,
    target_kind,
    sync_device,
)

# ── Default search space ──────────────────────────────────────────────────

_DEFAULT_THREAD_CANDIDATES = (64, 128, 256, 512, 1024)
_DEFAULT_BLOCK_K = (1, 2, 4, 8)  # block_size = threads * k


def default_search_space(
    num_elements: int,
    *,
    max_block_size: int = 1024,
    thread_candidates: tuple[int, ...] = _DEFAULT_THREAD_CANDIDATES,
    k_candidates: tuple[int, ...] = _DEFAULT_BLOCK_K,
) -> list[dict[str, int]]:
    """Build the default ``(block_size, threads)`` search space for a given element count.

    Returns a list of ``{"block_size": bs, "threads": t}`` dicts.
    """
    space: list[dict[str, int]] = []
    for t in thread_candidates:
        for k in k_candidates:
            bs = t * k
            if bs <= max_block_size:
                space.append({"block_size": bs, "threads": t})
    if not space:
        space.append({
            "block_size": min(256, num_elements),
            "threads": min(128, num_elements),
        })
    return space


# ── In-memory cache ───────────────────────────────────────────────────────

_autotune_cache: dict[tuple, dict[str, int]] = {}


def _cache_key(
    op_name: str,
    num_elements: int,
    dtype: Any,
    target: str,
    execution_backend: Optional[str],
    config_space: Sequence[dict[str, int]],
    **extra: Any,
) -> tuple:
    cfg_sig = tuple(
        (k, v) for cfg in sorted(config_space, key=lambda d: (d.get("block_size", 0), d.get("threads", 0)))
        for k, v in sorted(cfg.items())
    )
    extra_sig = tuple(sorted(extra.items()))
    return (op_name, num_elements, str(dtype), target, execution_backend, cfg_sig, extra_sig)


# ── Main API ──────────────────────────────────────────────────────────────

def search_best_config(
    *,
    op_name: str,
    prim_builder: Callable[..., Any],
    builder_args_fn: Callable[[dict[str, int]], tuple],
    config_space: list[dict[str, int]],
    num_elements: int,
    dtype: Any,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
    warmup: int = 25,
    rep: int = 100,
    device: Optional[str] = None,
    extra_cache_keys: Optional[dict[str, Any]] = None,
) -> dict[str, int]:
    """Grid-search ``(block_size, threads)`` and return the fastest config dict.

    Parameters
    ----------
    op_name
        Operator name (for logging / cache disambiguation).
    prim_builder
        A callable that returns a ``@T.prim_func`` when called with positional
        args returned by *builder_args_fn*.
    builder_args_fn
        ``fn(config) -> tuple`` — maps a config dict (e.g. ``{"block_size": 256,
        "threads": 128}``) to positional arguments for *prim_builder*.
    config_space
        List of config dicts to try.
    num_elements
        Total number of elements the kernel will process.
    dtype
        TileLang dtype.
    target
        TileLang target string.  Defaults to ``default_tilelang_target()``.
    execution_backend
        Forwarded to ``tilelang.compile``.  Defaults determined by target/device.
    warmup, rep
        Profiling iterations.
    device
        PyTorch device string.  Defaults determined by *target*.
    extra_cache_keys
        Additional key-value pairs that affect the kernel code (e.g. min_val,
        max_val for clamp).  Included in the cache key.

    Returns
    -------
    dict
        The best config found, e.g. ``{"block_size": 512, "threads": 128}``.
    """
    tgt = target or default_tilelang_target()
    dev = device or default_torch_device(tgt)
    eb = execution_backend if execution_backend is not None else default_execution_backend(tgt, dev)

    extra = dict(extra_cache_keys or {})
    key = _cache_key(op_name, num_elements, dtype, tgt, eb, config_space, **extra)

    cached = _autotune_cache.get(key)
    if cached is not None:
        return cached

    best_latency = float("inf")
    best_config: dict[str, int] = config_space[0] if config_space else {}

    for config in config_space:
        args = builder_args_fn(config)
        prim = prim_builder(*args)
        kernel = compile_prim(prim, target=tgt, execution_backend=eb)
        runner = make_unary_kernel_runner(kernel, tilelang_target=tgt, execution_backend=eb)

        # Allocate dummy tensors sized to num_elements.
        # dtype mapping from TileLang → torch
        import torch
        import tilelang.language as T_lang
        _TL_TO_TORCH = {
            T_lang.float32: torch.float32,
            T_lang.float16: torch.float16,
            T_lang.bfloat16: torch.bfloat16,
        }
        torch_dtype = _TL_TO_TORCH.get(dtype, torch.float32)
        x = torch.randn(num_elements, device=dev, dtype=torch_dtype)
        out = torch.empty(num_elements, device=dev, dtype=torch_dtype)

        def _bench_fn(_r=runner, _x=x, _o=out):
            _r(_x, _o)

        ms = bench_ms(_bench_fn, dev, warmup=warmup, repeat=rep)

        if ms < best_latency:
            best_latency = ms
            best_config = config

    _autotune_cache[key] = best_config
    return best_config
