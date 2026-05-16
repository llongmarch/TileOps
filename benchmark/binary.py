"""Benchmark TileLang binary kernels vs PyTorch element-wise ops.

Usage::

    python benchmark/binary.py
    python benchmark/binary.py --shape 4096 4096 --repeat 50
    python benchmark/binary.py --compare-shared-add   # CUDA / HIP only
    python benchmark/binary.py --ops add mul maximum   # run selected ops only
    python benchmark/binary.py --plot
    python benchmark/binary.py --plot-save results_binary.png
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Callable, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tileops.runtime import setup_metal_workarounds

setup_metal_workarounds()

import tilelang.language as T
import torch

from tileops import (
    bench_ms,
    default_execution_backend,
    default_tilelang_target,
    default_torch_device,
    make_kernel_runner,
    suggest_tile_config,
    target_kind,
)
from tileops.runtime import dispatch_compile

from benchmark.common import (
    ALL_BINARY_OPS,
    BenchResult,
    autotune_config,
    create_arg_parser,
    format_results_table,
    plot_results,
    print_setup_info,
    resolve_config,
)


def _make_torch_ref(
    name: str,
    a: torch.Tensor,
    b: torch.Tensor,
    b_div: torch.Tensor,
    a_pow: torch.Tensor,
    b_pow: torch.Tensor,
    out: torch.Tensor,
    a_int: Optional[torch.Tensor] = None,
    b_int: Optional[torch.Tensor] = None,
) -> Callable[[], None]:
    """Build a PyTorch reference callable for *name*."""
    _map: dict[str, Callable[[], None]] = {
        # arithmetic
        "add":         lambda: torch.add(a, b, out=out),
        "sub":         lambda: torch.sub(a, b, out=out),
        "mul":         lambda: torch.mul(a, b, out=out),
        "div":         lambda: torch.div(a, b_div, out=out),
        "pow":         lambda: torch.pow(a_pow, b_pow, out=out),
        "fmod":        lambda: torch.fmod(a, b_div, out=out),
        "remainder":   lambda: torch.remainder(a, b_div, out=out),
        "floor_div":   lambda: torch.div(a, b_div, rounding_mode="floor", out=out),
        # extrema
        "maximum":     lambda: torch.maximum(a, b, out=out),
        "minimum":     lambda: torch.minimum(a, b, out=out),
        # comparison
        "eq":          lambda: torch.eq(a, b),
        "ne":          lambda: torch.ne(a, b),
        "gt":          lambda: torch.gt(a, b),
        "ge":          lambda: torch.ge(a, b),
        "lt":          lambda: torch.lt(a, b),
        "le":          lambda: torch.le(a, b),
        # math
        "atan2":       lambda: torch.atan2(a, b, out=out),
        "copysign":    lambda: torch.copysign(a, b, out=out),
        "hypot":       lambda: torch.hypot(a, b, out=out),
        "xlogy":       lambda: torch.xlogy(a, b, out=out),
        "logaddexp":   lambda: torch.logaddexp(a, b, out=out),
        # logical
        "logical_and": lambda: torch.logical_and(a, b),
        "logical_or":  lambda: torch.logical_or(a, b),
        "logical_xor": lambda: torch.logical_xor(a, b),
    }

    if name in _map:
        return _map[name]

    # Bitwise ops — require integer inputs
    if a_int is None or b_int is None:
        raise ValueError(f"bitwise op {name} requires integer inputs")
    _bitwise_map: dict[str, Callable[[], None]] = {
        "bitwise_and":  lambda: torch.bitwise_and(a_int, b_int),
        "bitwise_or":   lambda: torch.bitwise_or(a_int, b_int),
        "bitwise_xor":  lambda: torch.bitwise_xor(a_int, b_int),
        "shift_left":   lambda: torch.bitwise_left_shift(a_int, b_int),
        "shift_right":  lambda: torch.bitwise_right_shift(a_int, b_int),
    }
    if name in _bitwise_map:
        return _bitwise_map[name]

    raise ValueError(f"Unknown binary op: {name!r}")


def _select_inputs(
    name: str,
    a: torch.Tensor,
    b: torch.Tensor,
    b_div: torch.Tensor,
    a_pow: torch.Tensor,
    b_pow: torch.Tensor,
    a_int: torch.Tensor,
    b_int: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if name in ("div", "fmod", "remainder", "floor_div"):
        return a, b_div
    if name == "pow":
        return a_pow, b_pow
    if name.startswith("bitwise_") or name.startswith("shift_"):
        return a_int, b_int
    return a, b


def main() -> None:
    parser = create_arg_parser("Binary ops: TileLang vs PyTorch benchmark")
    parser.add_argument("--compare-shared-add", action="store_true",
                        help="Compare GMEM add vs shared-tile add (CUDA / HIP only)")
    args = parser.parse_args()
    cfg = resolve_config(args, ALL_BINARY_OPS)

    print_setup_info(cfg, "binary")
    device = torch.device(cfg.device_str)

    # ── Data ────────────────────────────────────────────────────────────
    a = torch.randn(*cfg.shape, dtype=torch.float32, device=device)
    b = torch.randn(*cfg.shape, dtype=torch.float32, device=device)
    b_div = torch.where(b.abs() < 1e-3, torch.full_like(b, 1e-2), b)
    a_pow = torch.rand(*cfg.shape, dtype=torch.float32, device=device) + 0.05
    b_pow = torch.clamp(b, min=0.25, max=4.0)
    a_int = torch.randint(0, 256, cfg.shape, dtype=torch.int64, device=device)
    b_int = torch.randint(0, 32, cfg.shape, dtype=torch.int64, device=device)
    out_pt = torch.empty(*cfg.shape, dtype=torch.float32, device=device)
    out_tl = torch.empty(*cfg.shape, dtype=torch.float32, device=device)

    # ── Compile & run ───────────────────────────────────────────────────
    results: list[BenchResult] = []
    bkw = {"warmup": cfg.warmup, "repeat": cfg.repeat}

    for name in cfg.ops:
        dtype = T.int64 if (name.startswith("bitwise_") or name.startswith("shift_")) else T.float32
        x, y = _select_inputs(name, a, b, b_div, a_pow, b_pow, a_int, b_int)

        best_bs, best_t = cfg.block_size, cfg.threads
        if cfg.autotune:
            best_bs, best_t = autotune_config(
                module_name="binary",
                op_name=name,
                cfg=cfg,
                compile_kwargs={"dtype": dtype},
                runner_factory=make_kernel_runner,
                runner_args=(x, y, out_tl),
            )

        kernel = dispatch_compile(
            module_name="binary", op_name=name, target=cfg.target,
            shape=cfg.shape, block_size=best_bs, threads=best_t,
            dtype=dtype, execution_backend=cfg.execution_backend or None,
        )
        runner = make_kernel_runner(
            kernel,
            tilelang_target=cfg.target,
            execution_backend=cfg.execution_backend or None,
        )

        fn_pt = _make_torch_ref(name, a, b, b_div, a_pow, b_pow, out_pt,
                                a_int, b_int)
        fn_tl = lambda x=x, y=y: runner(x, y, out_tl)

        ms_pt = bench_ms(fn_pt, cfg.device_str, **bkw)
        ms_tl = bench_ms(fn_tl, cfg.device_str, **bkw)
        results.append(BenchResult(op=name, pytorch_ms=ms_pt, tilelang_ms=ms_tl))

    if not args.no_print_table:
        print(format_results_table(results))

    if args.plot or args.plot_save:
        plot_results(
            results,
            title=f"Binary Ops  —  shape={cfg.shape}  target={cfg.target}",
            save_path=args.plot_save,
        )

    # ── Optional: GMEM add vs shared-tile add ───────────────────────────
    if args.compare_shared_add:
        print()
        if target_kind(cfg.target) not in ("cuda", "hip"):
            print(f"--compare-shared-add skipped: requires cuda/hip "
                  f"(current: {cfg.target!r})")
        else:
            from tileops.backend.cuda.binary import add_shared

            ckw = {"target": cfg.target}
            if cfg.execution_backend:
                ckw["execution_backend"] = cfg.execution_backend

            k_direct = dispatch_compile(
                module_name="binary", op_name="add", target=cfg.target,
                shape=cfg.shape, block_size=cfg.block_size,
                threads=cfg.threads, dtype=T.float32,
                execution_backend=cfg.execution_backend or None,
            )
            k_shared = add_shared(cfg.shape, cfg.block_size, cfg.threads,
                                  dtype=T.float32, **ckw)
            run_d = make_kernel_runner(
                k_direct,
                tilelang_target=cfg.target,
                execution_backend=cfg.execution_backend or None,
            )
            run_s = make_kernel_runner(
                k_shared,
                tilelang_target=cfg.target,
                execution_backend=cfg.execution_backend or None,
            )
            ms_d = bench_ms(lambda: run_d(a, b, out_tl), cfg.device_str, **bkw)
            ms_s = bench_ms(lambda: run_s(a, b, out_tl), cfg.device_str, **bkw)
            ratio = ms_d / ms_s if ms_s > 0 else float("inf")
            print("add: GMEM-direct vs shared-tile")
            print(f"  direct={ms_d:.4f} ms  shared={ms_s:.4f} ms  "
                  f"ratio={ratio:.3f}")
            print("  (ratio > 1 → shared is faster)")


if __name__ == "__main__":
    main()
