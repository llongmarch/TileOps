"""Shared benchmark infrastructure for TileOps operator evaluation.

Provides reusable utilities for benchmarking activation, binary, and unary
operators, including CLI argument parsing, result collection, table formatting,
and matplotlib-based visualization.

Usage::

    from benchmark.common import (
        BenchConfig, BenchResult,
        create_arg_parser, resolve_config, print_setup_info,
        format_results_table, plot_results,
        benchmark_op,
        ALL_UNARY_OPS, ALL_BINARY_OPS, ALL_ACTIVATION_OPS,
    )
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tileops.runtime import setup_metal_workarounds

setup_metal_workarounds()

from tileops import (
    bench_ms,
    default_execution_backend,
    default_tilelang_target,
    default_torch_device,
    suggest_tile_config,
)

# ═══════════════════════════════════════════════════════════════════════════
# Operator lists
# ═══════════════════════════════════════════════════════════════════════════

ALL_UNARY_OPS = [
    # exponential / logarithmic
    "exp", "log", "exp2", "exp10", "log2", "log10", "log1p",
    # trigonometry
    "sin", "cos", "tan",
    # inverse trigonometry
    "asin", "acos", "atan",
    # hyperbolic
    "sinh", "cosh",
    # inverse hyperbolic
    "asinh", "acosh", "atanh",
    # power / root
    "sqrt", "rsqrt", "square",
    # error function
    "erf",
    # sign / absolute
    "abs", "sign", "neg",
    # special-value detection
    "isnan", "isinf", "isfinite",
    # rounding
    "round", "floor", "ceil", "trunc",
    # reciprocal
    "reciprocal",
    # bitwise
    "bitwise_not",
]

ALL_BINARY_OPS = [
    # arithmetic
    "add", "sub", "mul", "div", "pow",
    "fmod", "remainder", "floor_div",
    # extrema
    "maximum", "minimum",
    # comparison
    "eq", "ne", "gt", "ge", "lt", "le",
    # math
    "atan2", "copysign", "hypot", "xlogy", "logaddexp",
    # logical
    "logical_and", "logical_or", "logical_xor",
    # bitwise
    "bitwise_and", "bitwise_or", "bitwise_xor",
    "shift_left", "shift_right",
]

ALL_ACTIVATION_OPS = [
    # classic
    "relu", "sigmoid", "tanh",
    # gelu
    "gelu", "gelu_exact",
    # swish family
    "silu", "hardswish", "hardsigmoid",
    # relu variants
    "leaky_relu", "relu6", "elu", "selu", "celu", "hardtanh",
    # smooth
    "softplus", "mish", "softsign",
    # log
    "log_sigmoid",
]


# ═══════════════════════════════════════════════════════════════════════════
# Autotune defaults
# ═══════════════════════════════════════════════════════════════════════════

_DEFAULT_TUNE_WARMUP = 3
_DEFAULT_TUNE_REPEAT = 10
_DEFAULT_AUTOTUNE_THREADS = (64, 128, 256)
_DEFAULT_AUTOTUNE_K = (1, 2, 4, 8)  # block_size = threads × k
_AUTOTUNE_CACHE_DIR = Path.home() / ".cache" / "tileops"
_AUTOTUNE_CACHE_PATH = _AUTOTUNE_CACHE_DIR / "autotune.json"
_AUTOTUNE_MEM_CACHE: dict[str, dict] = {}


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass
class BenchConfig:
    """Shared benchmark configuration."""
    shape: tuple[int, ...] = (4096, 4096)
    block_size: int = 0
    threads: int = 0
    warmup: int = 10
    repeat: int = 50
    target: str = ""
    execution_backend: str = ""
    device_str: str = ""
    ops: list[str] = dataclasses.field(default_factory=list)
    autotune: bool = False
    autotune_warmup: int = _DEFAULT_TUNE_WARMUP
    autotune_repeat: int = _DEFAULT_TUNE_REPEAT
    autotune_no_cache: bool = False

    @property
    def num_elements(self) -> int:
        return math.prod(self.shape)


@dataclasses.dataclass
class BenchResult:
    """Benchmark result for a single operator."""
    op: str
    pytorch_ms: float
    tilelang_ms: float

    @property
    def speedup(self) -> float:
        if self.tilelang_ms > 0:
            return self.pytorch_ms / self.tilelang_ms
        return float("inf")

    @property
    def winner(self) -> str:
        if self.speedup > 1.005:
            return "TileLang"
        if self.speedup < 0.995:
            return "PyTorch"
        return "tie"


# ═══════════════════════════════════════════════════════════════════════════
# CLI helpers
# ═══════════════════════════════════════════════════════════════════════════


def create_arg_parser(description: str) -> argparse.ArgumentParser:
    """Return an ArgumentParser with common benchmark options."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--shape", type=int, nargs="+", default=[4096, 4096])
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--execution-backend", type=str, default=None)
    parser.add_argument("--ops", type=str, nargs="*", default=None,
                        help="Subset of ops to benchmark (default: all)")
    parser.add_argument("--plot", action="store_true",
                        help="Show matplotlib bar chart after benchmark")
    parser.add_argument("--plot-save", type=str, default=None,
                        help="Save plot to file (e.g. results.png)")
    parser.add_argument("--no-print-table", action="store_true",
                        help="Suppress terminal table output")
    parser.add_argument("--autotune", action="store_true",
                        help="Grid search (block_size, threads) per op")
    parser.add_argument("--autotune-warmup", type=int,
                        default=_DEFAULT_TUNE_WARMUP,
                        help="Warmup iterations per autotune trial")
    parser.add_argument("--autotune-repeat", type=int,
                        default=_DEFAULT_TUNE_REPEAT,
                        help="Bench iterations per autotune trial")
    parser.add_argument("--autotune-no-cache", action="store_true",
                        help="Disable persistent autotune cache")
    parser.add_argument("--autotune-search-space", type=str, default=None,
                        help="Colon-separated threads:k list, "
                             "e.g. '64,128,256:1,2,4,8'")
    return parser


def resolve_config(args: argparse.Namespace, all_ops: Sequence[str]) -> BenchConfig:
    """Resolve default values and return a populated :class:`BenchConfig`."""
    shape = tuple(args.shape)
    tgt = args.target or default_tilelang_target()
    device_str = default_torch_device(tgt)
    eb = args.execution_backend or default_execution_backend(tgt, device_str) or ""

    block_size, threads = suggest_tile_config(math.prod(shape))
    if args.block_size is not None:
        block_size = args.block_size
    if args.threads is not None:
        threads = args.threads

    return BenchConfig(
        shape=shape,
        block_size=block_size,
        threads=threads,
        warmup=args.warmup,
        repeat=args.repeat,
        target=tgt,
        execution_backend=eb,
        device_str=device_str,
        ops=list(args.ops) if args.ops else list(all_ops),
        autotune=args.autotune,
        autotune_warmup=args.autotune_warmup,
        autotune_repeat=args.autotune_repeat,
        autotune_no_cache=args.autotune_no_cache,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Terminal output
# ═══════════════════════════════════════════════════════════════════════════


def print_setup_info(cfg: BenchConfig, category: str) -> None:
    """Print benchmark setup header to stdout."""
    print(f"shape={cfg.shape}  num_elements={cfg.num_elements}  "
          f"dtype=float32  device={cfg.device_str}")
    print(f"TileLang target={cfg.target}  "
          f"execution_backend={cfg.execution_backend or 'auto'}")
    print(f"block_size={cfg.block_size}  threads={cfg.threads}")
    print(f"warmup={cfg.warmup}  repeat={cfg.repeat}  "
          f"category={category}  ops={len(cfg.ops)}")
    if cfg.autotune:
        cache_status = "off" if cfg.autotune_no_cache else "on"
        print(f"autotune=enabled  tune_warmup={cfg.autotune_warmup}  "
              f"tune_repeat={cfg.autotune_repeat}  cache={cache_status}")
    print()


def format_results_table(results: list[BenchResult]) -> str:
    """Format benchmark results as an aligned terminal table.

    Returns a multi-line string suitable for ``print()``.
    """
    lines: list[str] = []
    header = f"{'op':<16} {'pytorch_ms':>12} {'tilelang_ms':>12} {'speedup':>8}  {'winner':>8}"
    sep = "-" * len(header)
    lines.append(header)
    lines.append(sep)

    for r in results:
        lines.append(
            f"{r.op:<16} {r.pytorch_ms:12.4f} {r.tilelang_ms:12.4f} "
            f"{r.speedup:8.3f}  {r.winner:>8}"
        )
    lines.append(sep)

    # Summary row
    total_pt = sum(r.pytorch_ms for r in results)
    total_tl = sum(r.tilelang_ms for r in results)
    geo_mean = math.exp(
        sum(math.log(max(r.speedup, 1e-9)) for r in results) / len(results)
    ) if results else 0.0
    lines.append(
        f"{'TOTAL':<16} {total_pt:12.4f} {total_tl:12.4f} "
        f"{total_pt/total_tl if total_tl > 0 else float('inf'):8.3f}"
        f"  (geo-mean speedup: {geo_mean:.3f})"
    )
    lines.append("")
    lines.append("speedup = pytorch_ms / tilelang_ms  (>1 means TileLang is faster)")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════


def plot_results(
    results: list[BenchResult],
    *,
    title: str = "TileOps Benchmark",
    save_path: Optional[str] = None,
) -> None:
    """Create a dual-panel benchmark visualization.

    Left panel: grouped bar chart (PyTorch vs TileLang time per operator).
    Right panel: speedup ratio horizontal bar chart.

    Automatically scales to fit any number of operators.  If *save_path*
    is provided, the figure is saved to disk before display.
    """
    import matplotlib.pyplot as plt

    n = len(results)
    if n == 0:
        return

    ops = [r.op for r in results]
    pt_times = [r.pytorch_ms for r in results]
    tl_times = [r.tilelang_ms for r in results]
    speedups = [r.speedup for r in results]

    # Dynamic sizing — each bar row gets enough vertical space
    bar_row_height = 0.32 if n <= 12 else 0.22 if n <= 25 else 0.18
    fig_height = max(6, n * bar_row_height + 1.5)
    fig_width = max(16, n * 0.18) if n > 15 else 14
    label_fontsize = 9 if n <= 15 else 7.5 if n <= 25 else 6.5
    annotate_fontsize = 6.5 if n <= 15 else 5.5
    annotate = n <= 30  # skip per-bar annotations when too crowded

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # ── Left: grouped bar chart ──────────────────────────────────────────
    y = range(n)
    bar_h = 0.32 if n <= 15 else 0.24
    bars_pt = ax1.barh([i + bar_h / 2 for i in y], pt_times,
                        bar_h, label="PyTorch", color="#ed6a5a", alpha=0.85)
    bars_tl = ax1.barh([i - bar_h / 2 for i in y], tl_times,
                        bar_h, label="TileLang", color="#5ca4a9", alpha=0.85)
    ax1.set_yticks(y)
    ax1.set_yticklabels(ops, fontsize=label_fontsize)
    ax1.set_xlabel("Time (ms)")
    ax1.set_title("Runtime Comparison")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.invert_yaxis()
    ax1.grid(axis="x", alpha=0.3)

    if annotate:
        max_time = max(max(pt_times), max(tl_times)) if pt_times and tl_times else 1.0
        for bar, val in zip(bars_pt, pt_times):
            ax1.text(val + max_time * 0.015, bar.get_y() + bar.get_height() / 2,
                     f"{val:.2f}", va="center", fontsize=annotate_fontsize)
        for bar, val in zip(bars_tl, tl_times):
            ax1.text(val + max_time * 0.015, bar.get_y() + bar.get_height() / 2,
                     f"{val:.2f}", va="center", fontsize=annotate_fontsize)

    # ── Right: speedup horizontal bar chart ──────────────────────────────
    colors = ["#5ca4a9" if s >= 1.0 else "#ed6a5a" for s in speedups]
    bars_sp = ax2.barh(y, speedups, color=colors, alpha=0.85)
    ax2.axvline(x=1.0, color="gray", linewidth=0.8, linestyle="--")
    ax2.set_yticks(y)
    ax2.set_yticklabels(ops, fontsize=label_fontsize)
    ax2.set_xlabel("Speedup (PyTorch / TileLang)")
    ax2.set_title("Speedup Ratio")
    ax2.invert_yaxis()
    ax2.grid(axis="x", alpha=0.3)

    if annotate:
        max_sp = max(speedups) if speedups else 1.0
        for bar, val in zip(bars_sp, speedups):
            label_x = val + max_sp * 0.02 if val >= 1.0 else val - max_sp * 0.12
            ha = "left" if val >= 1.0 else "right"
            ax2.text(label_x, bar.get_y() + bar.get_height() / 2,
                     f"{val:.2f}x", va="center", ha=ha, fontsize=annotate_fontsize)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# Autotune
# ═══════════════════════════════════════════════════════════════════════════


def _autotune_cache_key(
    module_name: str, op_name: str, cfg: BenchConfig,
) -> str:
    shape_str = "x".join(str(s) for s in cfg.shape)
    raw = f"{cfg.target}|{module_name}|{op_name}|{shape_str}|float32"
    return hashlib.sha256(raw.encode()).hexdigest()


def _load_autotune_cache() -> dict[str, dict]:
    if not _AUTOTUNE_MEM_CACHE and _AUTOTUNE_CACHE_PATH.exists():
        try:
            with open(_AUTOTUNE_CACHE_PATH) as f:
                _AUTOTUNE_MEM_CACHE.update(json.load(f))
        except Exception:
            pass
    return _AUTOTUNE_MEM_CACHE


def _save_autotune_cache_entry(key: str, value: dict) -> None:
    _AUTOTUNE_MEM_CACHE[key] = value
    try:
        _AUTOTUNE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _AUTOTUNE_CACHE_PATH.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(_AUTOTUNE_MEM_CACHE, f, indent=2)
        os.replace(tmp, _AUTOTUNE_CACHE_PATH)
    except OSError:
        pass


def _default_search_space(
    num_elements: int,
    max_block_size: int = 1024,
) -> list[tuple[int, int]]:
    space: list[tuple[int, int]] = []
    for t in _DEFAULT_AUTOTUNE_THREADS:
        for k in _DEFAULT_AUTOTUNE_K:
            bs = t * k
            if bs <= max_block_size and bs <= num_elements * 4:
                space.append((bs, t))
    if not space:
        space.append((min(256, num_elements), min(128, num_elements)))
    return space


def _parse_search_space(
    spec: str, num_elements: int, max_block_size: int = 1024,
) -> list[tuple[int, int]]:
    parts = spec.split(":")
    if len(parts) == 2:
        threads_list = [int(x.strip()) for x in parts[0].split(",") if x.strip()]
        k_list = [int(x.strip()) for x in parts[1].split(",") if x.strip()]
    else:
        threads_list = list(_DEFAULT_AUTOTUNE_THREADS)
        k_list = list(_DEFAULT_AUTOTUNE_K)

    space: list[tuple[int, int]] = []
    for t in threads_list:
        for k in k_list:
            bs = t * k
            if bs <= max_block_size:
                space.append((bs, t))
    if not space:
        space.append((min(256, num_elements), min(128, num_elements)))
    return space


def autotune_config(
    *,
    module_name: str,
    op_name: str,
    cfg: BenchConfig,
    runner_factory: Callable[..., Callable[..., Any]],
    runner_args: tuple = (),
    compile_kwargs: Optional[dict[str, Any]] = None,
    tune_warmup: Optional[int] = None,
    tune_repeat: Optional[int] = None,
    search_space: Optional[list[tuple[int, int]]] = None,
    force: bool = False,
) -> tuple[int, int]:
    """Grid search ``(block_size, threads)`` for the fastest kernel config.

    Returns the best ``(block_size, threads)`` found.
    """
    from tileops.runtime import dispatch_compile, bench_ms
    import tilelang.language as T

    num_elements = math.prod(cfg.shape)
    tw = tune_warmup if tune_warmup is not None else cfg.autotune_warmup
    tr = tune_repeat if tune_repeat is not None else cfg.autotune_repeat
    if search_space is None:
        search_space = _default_search_space(num_elements)

    # Check persistent cache
    if not force and not cfg.autotune_no_cache:
        key = _autotune_cache_key(module_name, op_name, cfg)
        cache = _load_autotune_cache()
        entry = cache.get(key)
        if entry is not None:
            cached_pair = (entry["block_size"], entry["threads"])
            if cached_pair in search_space:
                if cfg.autotune:
                    print(f"  autotune: {op_name}: cached "
                          f"bs={entry['block_size']} t={entry['threads']} "
                          f"({entry['time_ms']:.3f}ms)")
                return cached_pair

    if cfg.autotune:
        print(f"  autotune: {op_name}: searching {len(search_space)} configs "
              f"(warmup={tw} repeat={tr})")

    best_time = float("inf")
    best_config = (cfg.block_size, cfg.threads)

    base_ckw: dict[str, Any] = {
        "module_name": module_name,
        "op_name": op_name,
        "target": cfg.target,
        "shape": cfg.shape,
        "dtype": T.float32,
        "execution_backend": cfg.execution_backend or None,
    }
    if compile_kwargs:
        base_ckw.update(compile_kwargs)

    for bs, t in search_space:
        ckw = dict(base_ckw)
        ckw["block_size"] = bs
        ckw["threads"] = t

        kernel = dispatch_compile(**ckw)
        runner = runner_factory(
            kernel,
            tilelang_target=cfg.target,
            execution_backend=cfg.execution_backend or None,
        )

        def _bench_fn(_run=runner, _args=runner_args):
            _run(*_args)

        ms = bench_ms(_bench_fn, cfg.device_str,
                       warmup=tw, repeat=tr)

        if ms < best_time:
            best_time = ms
            best_config = (bs, t)

    if cfg.autotune:
        print(f"  autotune: {op_name}: best bs={best_config[0]} "
              f"t={best_config[1]} ({best_time:.3f}ms)")

    if not cfg.autotune_no_cache:
        key = _autotune_cache_key(module_name, op_name, cfg)
        _save_autotune_cache_entry(key, {
            "block_size": best_config[0],
            "threads": best_config[1],
            "time_ms": best_time,
        })

    return best_config


# ═══════════════════════════════════════════════════════════════════════════
# Benchmark runner
# ═══════════════════════════════════════════════════════════════════════════


def benchmark_op(
    *,
    module_name: str,
    op_name: str,
    cfg: BenchConfig,
    runner_factory: Callable[..., Callable[..., Any]],
    compile_kwargs: Optional[dict[str, Any]] = None,
) -> BenchResult:
    """Compile, run, and measure a single operator.

    Parameters
    ----------
    module_name: "unary", "binary", or "activation".
    op_name: operator name (e.g. "add", "sin", "relu").
    cfg: shared benchmark configuration.
    runner_factory: callable that returns a kernel runner function.
        Signature: ``runner_factory(kernel, tilelang_target=..., execution_backend=...) -> Callable``.
    compile_kwargs: extra kwargs forwarded to ``dispatch_compile``
        (e.g. ``min_val``, ``max_val`` for clamp).
    """
    from tileops.runtime import dispatch_compile
    import tilelang.language as T

    ckw: dict[str, Any] = {
        "module_name": module_name,
        "op_name": op_name,
        "target": cfg.target,
        "shape": cfg.shape,
        "block_size": cfg.block_size,
        "threads": cfg.threads,
        "dtype": T.float32,
        "execution_backend": cfg.execution_backend or None,
    }
    if compile_kwargs:
        ckw.update(compile_kwargs)

    kernel = dispatch_compile(**ckw)
    runner = runner_factory(
        kernel,
        tilelang_target=cfg.target,
        execution_backend=cfg.execution_backend or None,
    )
    bkw = {"warmup": cfg.warmup, "repeat": cfg.repeat}
    ms_tl = bench_ms(runner, cfg.device_str, **bkw)
    return runner, ms_tl


def run_benchmarks(
    *,
    cfg: BenchConfig,
    module_name: str,
    pt_ref_factory: Callable[[str], Callable[[], None]],
    tl_runner_factory: Callable[..., Callable[..., Any]],
    compile_kwargs_fn: Optional[Callable[[str], dict[str, Any]]] = None,
) -> list[BenchResult]:
    """Run benchmarks for a list of operators and return results.

    Parameters
    ----------
    cfg: benchmark configuration.
    module_name: "unary", "binary", or "activation".
    pt_ref_factory: ``(op_name) -> callable`` that returns a PyTorch reference function.
    tl_runner_factory: ``(kernel, **kw) -> callable`` that returns a TileLang runner.
    compile_kwargs_fn: optional ``(op_name) -> dict`` providing extra compile kwargs.
    """
    results: list[BenchResult] = []
    bkw = {"warmup": cfg.warmup, "repeat": cfg.repeat}

    for name in cfg.ops:
        ckw = compile_kwargs_fn(name) if compile_kwargs_fn else {}

        result_item = _bench_single(
            module_name=module_name,
            op_name=name,
            cfg=cfg,
            pt_ref_fn=pt_ref_factory(name),
            tl_runner_factory=tl_runner_factory,
            compile_kwargs=ckw,
            bkw=bkw,
        )
        results.append(result_item)

    return results


def _bench_single(
    *,
    module_name: str,
    op_name: str,
    cfg: BenchConfig,
    pt_ref_fn: Callable[[], None],
    tl_runner_factory: Callable[..., Callable[..., Any]],
    compile_kwargs: dict[str, Any],
    bkw: dict[str, int],
) -> BenchResult:
    """Internal: benchmark a single op and return its result."""
    import tilelang.language as T
    from tileops.runtime import dispatch_compile

    ckw_all: dict[str, Any] = {
        "module_name": module_name,
        "op_name": op_name,
        "target": cfg.target,
        "shape": cfg.shape,
        "block_size": cfg.block_size,
        "threads": cfg.threads,
        "dtype": T.float32,
        "execution_backend": cfg.execution_backend or None,
    }
    ckw_all.update(compile_kwargs)

    kernel = dispatch_compile(**ckw_all)
    runner = tl_runner_factory(
        kernel,
        tilelang_target=cfg.target,
        execution_backend=cfg.execution_backend or None,
    )

    ms_pt = bench_ms(pt_ref_fn, cfg.device_str, **bkw)
    ms_tl = bench_ms(runner, cfg.device_str, **bkw)

    return BenchResult(op=op_name, pytorch_ms=ms_pt, tilelang_ms=ms_tl)
