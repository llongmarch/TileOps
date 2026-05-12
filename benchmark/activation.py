"""Benchmark TileLang activation kernels vs PyTorch.

Usage::

    python benchmark/activation.py
    python benchmark/activation.py --shape 4096 4096 --repeat 50
    python benchmark/activation.py --ops relu gelu silu mish
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
import torch.nn.functional as F

from tileops import (
    bench_ms,
    default_execution_backend,
    default_tilelang_target,
    default_torch_device,
    make_unary_kernel_runner,
    suggest_tile_config,
)
from tileops._infra import dispatch_compile

_ALL_OPS = [
    "relu", "sigmoid", "tanh",
    "gelu", "gelu_exact",
    "silu", "hardswish", "hardsigmoid",
    "leaky_relu", "relu6", "elu", "selu", "celu", "hardtanh",
    "softplus", "mish", "softsign",
    "log_sigmoid",
]


def _make_torch_ref(name: str, x: torch.Tensor) -> Callable[[], None]:
    """Build a PyTorch reference callable for *name*."""
    _map: dict[str, Callable[[], None]] = {
        "relu":        lambda: torch.relu(x),
        "sigmoid":     lambda: torch.sigmoid(x),
        "tanh":        lambda: torch.tanh(x),
        "gelu":        lambda: F.gelu(x, approximate="tanh"),
        "gelu_exact":  lambda: F.gelu(x, approximate="none"),
        "silu":        lambda: F.silu(x),
        "hardswish":   lambda: F.hardswish(x),
        "hardsigmoid": lambda: F.hardsigmoid(x),
        "leaky_relu":  lambda: F.leaky_relu(x, 0.01),
        "relu6":       lambda: F.relu6(x),
        "elu":         lambda: F.elu(x, alpha=1.0),
        "selu":        lambda: F.selu(x),
        "celu":        lambda: F.celu(x, alpha=1.0),
        "hardtanh":    lambda: F.hardtanh(x),
        "softplus":    lambda: F.softplus(x),
        "mish":        lambda: F.mish(x),
        "softsign":    lambda: F.softsign(x),
        "log_sigmoid": lambda: F.logsigmoid(x),
    }
    return _map[name]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Activation ops: TileLang vs PyTorch benchmark",
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

    ops = args.ops or _ALL_OPS

    print(f"shape={shape}  num_elements={num_elements}  dtype=float32  device={device_str}")
    print(f"TileLang target={tgt}  execution_backend={eb or 'auto'}")
    print(f"block_size={block_size}  threads={threads}")
    print(f"warmup={args.warmup}  repeat={args.repeat}  ops={len(ops)}")
    print()

    # ── Data ────────────────────────────────────────────────────────────
    x = torch.randn(*shape, dtype=torch.float32, device=device)
    out_tl = torch.empty(*shape, dtype=torch.float32, device=device)

    # ── Compile & run ───────────────────────────────────────────────────
    header = f"{'op':<14} {'pytorch_ms':>12} {'tilelang_ms':>12} {'speedup':>8}"
    print(header)
    print("-" * len(header))

    bkw = {"warmup": args.warmup, "repeat": args.repeat}

    for name in ops:
        kernel = dispatch_compile(
            module_name="activation", op_name=name, target=tgt,
            shape=shape, block_size=block_size, threads=threads,
            dtype=T.float32, execution_backend=eb,
        )
        runner = make_unary_kernel_runner(
            kernel, tilelang_target=tgt, execution_backend=eb,
        )
        fn_pt = _make_torch_ref(name, x)
        fn_tl = lambda: runner(x, out_tl)

        ms_pt = bench_ms(fn_pt, device_str, **bkw)
        ms_tl = bench_ms(fn_tl, device_str, **bkw)
        ratio = ms_pt / ms_tl if ms_tl > 0 else float("inf")
        print(f"{name:<14} {ms_pt:12.4f} {ms_tl:12.4f} {ratio:8.3f}")

    print()
    print("speedup = pytorch_ms / tilelang_ms  (>1 means TileLang is faster)")


if __name__ == "__main__":
    main()
