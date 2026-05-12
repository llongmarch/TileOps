"""Benchmark TileLang binary kernels vs PyTorch element-wise ops.

Usage::

    python benchmark/binary.py
    python benchmark/binary.py --shape 4096 4096 --repeat 50
    python benchmark/binary.py --compare-shared-add   # CUDA / HIP only
    python benchmark/binary.py --ops add mul maximum   # run selected ops only
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Callable

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
from tileops._infra import dispatch_compile

_ALL_OPS = [
    "add", "sub", "mul", "div", "pow",
    "fmod", "remainder", "floor_div",
    "maximum", "minimum",
    "eq", "ne", "gt", "ge", "lt", "le",
    "atan2", "copysign", "hypot", "xlogy",
    "logical_and", "logical_or", "logical_xor",
]


def _make_torch_ref(name: str, a: torch.Tensor, b: torch.Tensor,
                    b_div: torch.Tensor, a_pow: torch.Tensor,
                    b_pow: torch.Tensor, out: torch.Tensor) -> Callable[[], None]:
    """Build a PyTorch reference callable for *name*."""
    _map: dict[str, Callable[[], None]] = {
        "add":         lambda: torch.add(a, b, out=out),
        "sub":         lambda: torch.sub(a, b, out=out),
        "mul":         lambda: torch.mul(a, b, out=out),
        "div":         lambda: torch.div(a, b_div, out=out),
        "pow":         lambda: torch.pow(a_pow, b_pow, out=out),
        "fmod":        lambda: torch.fmod(a, b_div, out=out),
        "remainder":   lambda: torch.remainder(a, b_div, out=out),
        "floor_div":   lambda: torch.div(a, b_div, rounding_mode="floor", out=out),
        "maximum":     lambda: torch.maximum(a, b, out=out),
        "minimum":     lambda: torch.minimum(a, b, out=out),
        "eq":          lambda: torch.eq(a, b),
        "ne":          lambda: torch.ne(a, b),
        "gt":          lambda: torch.gt(a, b),
        "ge":          lambda: torch.ge(a, b),
        "lt":          lambda: torch.lt(a, b),
        "le":          lambda: torch.le(a, b),
        "atan2":       lambda: torch.atan2(a, b, out=out),
        "copysign":    lambda: torch.copysign(a, b, out=out),
        "hypot":       lambda: torch.hypot(a, b, out=out),
        "xlogy":       lambda: torch.xlogy(a, b, out=out),
        "logical_and": lambda: torch.logical_and(a, b),
        "logical_or":  lambda: torch.logical_or(a, b),
        "logical_xor": lambda: torch.logical_xor(a, b),
    }
    return _map[name]


def _select_inputs(name: str, a: torch.Tensor, b: torch.Tensor,
                   b_div: torch.Tensor, a_pow: torch.Tensor,
                   b_pow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if name in ("div", "fmod", "remainder", "floor_div"):
        return a, b_div
    if name == "pow":
        return a_pow, b_pow
    return a, b


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Binary ops: TileLang vs PyTorch benchmark",
    )
    parser.add_argument("--shape", type=int, nargs="+", default=[4096, 4096])
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--execution-backend", type=str, default=None)
    parser.add_argument("--ops", type=str, nargs="*", default=None,
                        help="Subset of ops to benchmark (default: all)")
    parser.add_argument("--compare-shared-add", action="store_true",
                        help="Compare GMEM add vs shared-tile add (CUDA / HIP only)")
    args = parser.parse_args()

    shape = tuple(args.shape)
    num_elements = math.prod(shape)
    tgt = args.target or default_tilelang_target()
    device_str = default_torch_device(tgt)
    device = torch.device(device_str)
    eb = args.execution_backend or default_execution_backend(tgt, device_str)

    block_size, threads = suggest_tile_config(num_elements)
    if args.block_size is not None:
        block_size = args.block_size
    if args.threads is not None:
        threads = args.threads

    ckw = {"target": tgt}
    if eb is not None:
        ckw["execution_backend"] = eb

    ops = args.ops or _ALL_OPS

    print(f"shape={shape}  num_elements={num_elements}  dtype=float32  device={device_str}")
    print(f"TileLang target={tgt}  execution_backend={eb or 'auto'}")
    print(f"block_size={block_size}  threads={threads}")
    print(f"warmup={args.warmup}  repeat={args.repeat}  ops={len(ops)}")
    print()

    # ── Data ────────────────────────────────────────────────────────────
    a = torch.randn(*shape, dtype=torch.float32, device=device)
    b = torch.randn(*shape, dtype=torch.float32, device=device)
    b_div = torch.where(b.abs() < 1e-3, torch.full_like(b, 1e-2), b)
    a_pow = torch.rand(*shape, dtype=torch.float32, device=device) + 0.05
    b_pow = torch.clamp(b, min=0.25, max=4.0)
    out_pt = torch.empty(*shape, dtype=torch.float32, device=device)
    out_tl = torch.empty(*shape, dtype=torch.float32, device=device)

    # ── Compile & run ───────────────────────────────────────────────────
    header = f"{'op':<14} {'pytorch_ms':>12} {'tilelang_ms':>12} {'speedup':>8}"
    print(header)
    print("-" * len(header))

    bkw = {"warmup": args.warmup, "repeat": args.repeat}

    for name in ops:
        kernel = dispatch_compile(
            module_name="binary", op_name=name, target=tgt,
            shape=shape, block_size=block_size, threads=threads,
            dtype=T.float32, execution_backend=eb,
        )
        runner = make_kernel_runner(kernel, tilelang_target=tgt, execution_backend=eb)
        x, y = _select_inputs(name, a, b, b_div, a_pow, b_pow)

        fn_pt = _make_torch_ref(name, a, b, b_div, a_pow, b_pow, out_pt)
        fn_tl = lambda x=x, y=y: runner(x, y, out_tl)

        ms_pt = bench_ms(fn_pt, device_str, **bkw)
        ms_tl = bench_ms(fn_tl, device_str, **bkw)
        ratio = ms_pt / ms_tl if ms_tl > 0 else float("inf")
        print(f"{name:<14} {ms_pt:12.4f} {ms_tl:12.4f} {ratio:8.3f}")

    print()
    print("speedup = pytorch_ms / tilelang_ms  (>1 means TileLang is faster)")

    # ── Optional: GMEM add vs shared-tile add ───────────────────────────
    if args.compare_shared_add:
        print()
        if target_kind(tgt) not in ("cuda", "hip"):
            print(f"--compare-shared-add skipped: requires cuda/hip (current: {tgt!r})")
        else:
            from tileops.cuda.binary import add_shared

            k_direct = dispatch_compile(
                module_name="binary", op_name="add", target=tgt,
                shape=shape, block_size=block_size, threads=threads,
                dtype=T.float32, execution_backend=eb,
            )
            k_shared = add_shared(shape, block_size, threads, dtype=T.float32, **ckw)
            run_d = make_kernel_runner(k_direct, tilelang_target=tgt, execution_backend=eb)
            run_s = make_kernel_runner(k_shared, tilelang_target=tgt, execution_backend=eb)
            ms_d = bench_ms(lambda: run_d(a, b, out_tl), device_str, **bkw)
            ms_s = bench_ms(lambda: run_s(a, b, out_tl), device_str, **bkw)
            ratio = ms_d / ms_s if ms_s > 0 else float("inf")
            print("add: GMEM-direct vs shared-tile")
            print(f"  direct={ms_d:.4f} ms  shared={ms_s:.4f} ms  ratio={ratio:.3f}")
            print("  (ratio > 1 → shared is faster)")


if __name__ == "__main__":
    main()
