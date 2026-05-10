"""Benchmark TileLang pointwise kernels vs PyTorch element-wise ops.

Usage::

    pip install -e .                     # one-time
    python benchmark/pointwise.py
    python benchmark/pointwise.py --shape 4096 4096 --repeat 50
    python benchmark/pointwise.py --auto-config
    python benchmark/pointwise.py --compare-shared-add   # CUDA / HIP only
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Callable

# Fallback for running without ``pip install -e .``.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tileops.runtime import setup_metal_workarounds

setup_metal_workarounds()

import torch

from tileops import (
    bench_ms,
    default_execution_backend,
    default_tilelang_target,
    default_torch_device,
    make_kernel_runner,
    pointwise_add,
    pointwise_div,
    pointwise_mul,
    pointwise_pow,
    pointwise_sub,
    suggest_pointwise_config,
    target_kind,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pointwise: TileLang vs PyTorch benchmark",
    )
    parser.add_argument("--shape", type=int, nargs="+", default=[4096, 4096],
                        help="Tensor shape, e.g. --shape 4096 4096")
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--execution-backend", type=str, default=None)
    parser.add_argument(
        "--auto-config", action="store_true",
        help="Use suggest_pointwise_config() for block_size / threads",
    )
    parser.add_argument(
        "--compare-shared-add", action="store_true",
        help="Compare GMEM add vs shared-tile add (CUDA / HIP only)",
    )
    args = parser.parse_args()

    # ── Platform setup ──────────────────────────────────────────────────
    shape = tuple(args.shape)
    num_elements = math.prod(shape)
    tgt = args.target or default_tilelang_target()
    device_str = default_torch_device(tgt)
    device = torch.device(device_str)
    eb = args.execution_backend or default_execution_backend(tgt, device_str)

    block_size, threads = args.block_size, args.threads
    if args.auto_config:
        block_size, threads = suggest_pointwise_config(num_elements)

    kw: dict[str, str] = {"target": tgt}
    if eb is not None:
        kw["execution_backend"] = eb

    print(f"shape={shape}  num_elements={num_elements}  dtype=float32  device={device_str}")
    print(f"TileLang target={tgt}  execution_backend={eb or 'auto'}")
    print(f"block_size={block_size}  threads={threads}")
    print(f"warmup={args.warmup}  repeat={args.repeat}")
    print()

    # ── Data ────────────────────────────────────────────────────────────
    a = torch.randn(*shape, dtype=torch.float32, device=device)
    b = torch.randn(*shape, dtype=torch.float32, device=device)
    b_div = torch.where(b.abs() < 1e-3, torch.full_like(b, 1e-2), b)
    a_pow = torch.rand(*shape, dtype=torch.float32, device=device) + 0.05
    b_pow = torch.clamp(b, min=0.25, max=4.0)

    out_pt = torch.empty(*shape, dtype=torch.float32, device=device)
    out_tl = torch.empty(*shape, dtype=torch.float32, device=device)

    # ── Compile kernels ─────────────────────────────────────────────────
    kernels = {
        "add": pointwise_add(shape, block_size, threads, **kw),
        "sub": pointwise_sub(shape, block_size, threads, **kw),
        "mul": pointwise_mul(shape, block_size, threads, **kw),
        "div": pointwise_div(shape, block_size, threads, **kw),
        "pow": pointwise_pow(shape, block_size, threads, **kw),
    }

    runners = {
        name: make_kernel_runner(
            k, tilelang_target=tgt, execution_backend=eb,
        )
        for name, k in kernels.items()
    }

    # ── Benchmark cases ─────────────────────────────────────────────────
    cases: list[tuple[str, Callable[[], None], Callable[[], None]]] = [
        ("add", lambda: torch.add(a, b, out=out_pt),
               lambda: runners["add"](a, b, out_tl)),
        ("sub", lambda: torch.sub(a, b, out=out_pt),
               lambda: runners["sub"](a, b, out_tl)),
        ("mul", lambda: torch.mul(a, b, out=out_pt),
               lambda: runners["mul"](a, b, out_tl)),
        ("div", lambda: torch.div(a, b_div, out=out_pt),
               lambda: runners["div"](a, b_div, out_tl)),
        ("pow", lambda: torch.pow(a_pow, b_pow, out=out_pt),
               lambda: runners["pow"](a_pow, b_pow, out_tl)),
    ]

    header = f"{'op':<6} {'pytorch_ms':>12} {'tilelang_ms':>12} {'speedup':>8}"
    print(header)
    print("-" * len(header))

    bkw = {"warmup": args.warmup, "repeat": args.repeat}
    for name, fn_pt, fn_tl in cases:
        ms_pt = bench_ms(fn_pt, device_str, **bkw)
        ms_tl = bench_ms(fn_tl, device_str, **bkw)
        ratio = ms_pt / ms_tl if ms_tl > 0 else float("inf")
        print(f"{name:<6} {ms_pt:12.4f} {ms_tl:12.4f} {ratio:8.3f}")

    print()
    print("speedup = pytorch_ms / tilelang_ms  (>1 means TileLang is faster)")

    # ── Optional: GMEM add vs shared-tile add ───────────────────────────
    if args.compare_shared_add:
        print()
        if target_kind(tgt) not in ("cuda", "hip"):
            print(
                "--compare-shared-add skipped: shared-tile add requires "
                f"cuda or hip target (current: {tgt!r})"
            )
        else:
            from tileops.cuda.pointwise import pointwise_add_shared

            k_direct = pointwise_add(shape, block_size, threads, **kw)
            k_shared = pointwise_add_shared(shape, block_size, threads, **kw)
            run_d = make_kernel_runner(
                k_direct, tilelang_target=tgt, execution_backend=eb,
            )
            run_s = make_kernel_runner(
                k_shared, tilelang_target=tgt, execution_backend=eb,
            )
            ms_d = bench_ms(lambda: run_d(a, b, out_tl), device_str, **bkw)
            ms_s = bench_ms(lambda: run_s(a, b, out_tl), device_str, **bkw)
            ratio = ms_d / ms_s if ms_s > 0 else float("inf")
            print("add: GMEM-direct vs shared-tile")
            print(f"  direct={ms_d:.4f} ms  shared={ms_s:.4f} ms  "
                  f"ratio={ratio:.3f}")
            print("  (ratio > 1 → shared is faster)")


if __name__ == "__main__":
    main()
