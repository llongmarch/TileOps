"""Benchmark TileLang activation kernels vs PyTorch.

Usage::

    python benchmark/activation.py
    python benchmark/activation.py --shape 4096 4096 --repeat 50
    python benchmark/activation.py --ops relu gelu silu mish
    python benchmark/activation.py --plot
    python benchmark/activation.py --plot-save results_activation.png
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

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
    make_unary_kernel_runner,
)
from tileops.runtime import dispatch_compile

from benchmark.common import (
    ALL_ACTIVATION_OPS,
    BenchResult,
    autotune_config,
    create_arg_parser,
    format_results_table,
    plot_results,
    print_setup_info,
    resolve_config,
)


def _make_torch_ref(name: str, x: torch.Tensor) -> Callable[[], None]:
    """Build a PyTorch reference callable for *name*."""
    _map: dict[str, Callable[[], None]] = {
        # classic
        "relu":        lambda: torch.relu(x),
        "sigmoid":     lambda: torch.sigmoid(x),
        "tanh":        lambda: torch.tanh(x),
        # gelu
        "gelu":        lambda: F.gelu(x, approximate="tanh"),
        "gelu_exact":  lambda: F.gelu(x, approximate="none"),
        # swish family
        "silu":        lambda: F.silu(x),
        "hardswish":   lambda: F.hardswish(x),
        "hardsigmoid": lambda: F.hardsigmoid(x),
        # relu variants
        "leaky_relu":  lambda: F.leaky_relu(x, 0.01),
        "relu6":       lambda: F.relu6(x),
        "elu":         lambda: F.elu(x, alpha=1.0),
        "selu":        lambda: F.selu(x),
        "celu":        lambda: F.celu(x, alpha=1.0),
        "hardtanh":    lambda: F.hardtanh(x),
        # smooth
        "softplus":    lambda: F.softplus(x),
        "mish":        lambda: F.mish(x),
        "softsign":    lambda: F.softsign(x),
        # log
        "log_sigmoid": lambda: F.logsigmoid(x),
    }
    if name in _map:
        return _map[name]
    raise ValueError(f"Unknown activation op: {name!r}")


def main() -> None:
    parser = create_arg_parser("Activation ops: TileLang vs PyTorch benchmark")
    args = parser.parse_args()
    cfg = resolve_config(args, ALL_ACTIVATION_OPS)

    print_setup_info(cfg, "activation")
    device = torch.device(cfg.device_str)

    # ── Data ────────────────────────────────────────────────────────────
    x = torch.randn(*cfg.shape, dtype=torch.float32, device=device)
    out_tl = torch.empty(*cfg.shape, dtype=torch.float32, device=device)

    # ── Compile & run ───────────────────────────────────────────────────
    results: list[BenchResult] = []
    bkw = {"warmup": cfg.warmup, "repeat": cfg.repeat}

    for name in cfg.ops:
        best_bs, best_t = cfg.block_size, cfg.threads
        if cfg.autotune:
            best_bs, best_t = autotune_config(
                module_name="activation",
                op_name=name,
                cfg=cfg,
                runner_factory=make_unary_kernel_runner,
                runner_args=(x, out_tl),
            )

        kernel = dispatch_compile(
            module_name="activation", op_name=name, target=cfg.target,
            shape=cfg.shape, block_size=best_bs, threads=best_t,
            dtype=T.float32, execution_backend=cfg.execution_backend or None,
        )
        runner = make_unary_kernel_runner(
            kernel,
            tilelang_target=cfg.target,
            execution_backend=cfg.execution_backend or None,
        )
        fn_pt = _make_torch_ref(name, x)
        fn_tl = lambda: runner(x, out_tl)

        ms_pt = bench_ms(fn_pt, cfg.device_str, **bkw)
        ms_tl = bench_ms(fn_tl, cfg.device_str, **bkw)
        results.append(BenchResult(op=name, pytorch_ms=ms_pt, tilelang_ms=ms_tl))

    if not args.no_print_table:
        print(format_results_table(results))

    if args.plot or args.plot_save:
        plot_results(
            results,
            title=f"Activation Ops  —  shape={cfg.shape}  target={cfg.target}",
            save_path=args.plot_save,
        )


if __name__ == "__main__":
    main()
