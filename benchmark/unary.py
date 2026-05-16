"""Benchmark TileLang unary kernels vs PyTorch.

Usage::

    python benchmark/unary.py
    python benchmark/unary.py --shape 4096 4096 --repeat 50
    python benchmark/unary.py --ops sin cos erf log2
    python benchmark/unary.py --plot
    python benchmark/unary.py --plot-save results_unary.png
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

import torch

from tileops import bench_ms, make_unary_kernel_runner
from tileops.runtime import dispatch_compile

from benchmark.common import (
    ALL_UNARY_OPS,
    BenchConfig,
    BenchResult,
    autotune_config,
    create_arg_parser,
    format_results_table,
    plot_results,
    print_setup_info,
    resolve_config,
)

def _make_unary_compile_kwargs(op_name: str) -> dict[str, Any]:
    """Extra kwargs for dispatch_compile per op name."""
    if op_name == "clamp":
        return {"min_val": -1.0, "max_val": 1.0}
    return {}


def _make_unary_pt_ref(op_name: str, x: torch.Tensor) -> Callable[[], None]:
    """Build a PyTorch reference callable for a unary op."""
    _map: dict[str, Callable[[], None]] = {
        # exponential / logarithmic
        "exp":        lambda: torch.exp(x),
        "log":        lambda: torch.log(x),
        "exp2":       lambda: torch.pow(2.0, x),
        "exp10":      lambda: torch.pow(10.0, x),
        "log2":       lambda: torch.log2(x),
        "log10":      lambda: torch.log10(x),
        "log1p":      lambda: torch.log1p(x),
        # trigonometry
        "sin":        lambda: torch.sin(x),
        "cos":        lambda: torch.cos(x),
        "tan":        lambda: torch.tan(x),
        # inverse trigonometry
        "asin":       lambda: torch.asin(x),
        "acos":       lambda: torch.acos(x),
        "atan":       lambda: torch.atan(x),
        # hyperbolic
        "sinh":       lambda: torch.sinh(x),
        "cosh":       lambda: torch.cosh(x),
        # inverse hyperbolic
        "asinh":      lambda: torch.asinh(x),
        "acosh":      lambda: torch.acosh(x),
        "atanh":      lambda: torch.atanh(x),
        # power / root
        "sqrt":       lambda: torch.sqrt(x),
        "rsqrt":      lambda: torch.rsqrt(x),
        "square":     lambda: torch.square(x),
        # error function
        "erf":        lambda: torch.erf(x),
        # sign / absolute
        "abs":        lambda: torch.abs(x),
        "sign":       lambda: torch.sign(x),
        "neg":        lambda: torch.neg(x),
        # rounding
        "round":      lambda: torch.round(x),
        "floor":      lambda: torch.floor(x),
        "ceil":       lambda: torch.ceil(x),
        "trunc":      lambda: torch.trunc(x),
        # reciprocal
        "reciprocal": lambda: torch.reciprocal(x),
    }
    if op_name in _map:
        return _map[op_name]

    # special-value detection
    if op_name == "isnan":
        return lambda: torch.isnan(x)
    if op_name == "isinf":
        return lambda: torch.isinf(x)
    if op_name == "isfinite":
        return lambda: torch.isfinite(x)

    # Clamp — needs compile-time min/max
    if op_name == "clamp":
        return lambda: torch.clamp(x, -1.0, 1.0)

    raise ValueError(f"Unknown unary op: {op_name!r}")


def _setup_unary_data(cfg: BenchConfig) -> dict[str, Any]:
    """Create input tensors for unary benchmarks."""
    device = torch.device(cfg.device_str)

    # Generic input: normal distribution, avoid extremes for trig/hyperbolic
    x = torch.randn(*cfg.shape, dtype=torch.float32, device=device)

    # For reciprocal, avoid near-zero values
    x_recip = torch.where(x.abs() < 1e-3, torch.full_like(x, 1e-2), x)

    # For asin/acos, clamp to (-1, 1)
    x_trig = torch.clamp(x * 0.9, -0.99, 0.99)

    # For log/log1p/log2/log10, ensure positive input
    x_log = torch.rand(*cfg.shape, dtype=torch.float32, device=device) + 0.1

    # For sqrt/rsqrt, positive input
    x_sqrt = torch.rand(*cfg.shape, dtype=torch.float32, device=device) + 1e-6

    # For isnan/isinf/isfinite — use mixed normal + some NaN/Inf
    x_special = torch.randn(*cfg.shape, dtype=torch.float32, device=device)
    # Inject some NaN and ±inf
    nan_mask = torch.rand(*cfg.shape, device=device) < 0.02
    inf_mask = torch.rand(*cfg.shape, device=device) < 0.02
    x_special[nan_mask] = float("nan")
    x_special[inf_mask] = float("inf")

    out_tl = torch.empty(*cfg.shape, dtype=torch.float32, device=device)

    return {
        "x": x, "x_recip": x_recip, "x_trig": x_trig,
        "x_log": x_log, "x_sqrt": x_sqrt, "x_special": x_special,
        "out": out_tl,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = create_arg_parser("Unary ops: TileLang vs PyTorch benchmark")
    args = parser.parse_args()
    cfg = resolve_config(args, ALL_UNARY_OPS)

    print_setup_info(cfg, "unary")
    bkw = {"warmup": cfg.warmup, "repeat": cfg.repeat}
    data = _setup_unary_data(cfg)

    results: list[BenchResult] = []

    for name in cfg.ops:
        # ── Select appropriate input tensor ───────────────────────────────
        if name == "clamp":
            x = data["x"]  # clamp works with any input
        elif name in ("asin", "acos", "atanh"):
            x = data["x_trig"]
        elif name in ("log", "log1p", "log2", "log10", "exp2", "exp10"):
            x = data["x_log"]
        elif name in ("sqrt", "rsqrt"):
            x = data["x_sqrt"]
        elif name == "reciprocal":
            x = data["x_recip"]
        elif name in ("isnan", "isinf", "isfinite"):
            x = data["x_special"]
        elif name == "bitwise_not":
            # bitwise_not needs integer tensor; skip if benchmark is float-only
            continue
        else:
            x = data["x"]

        out = data["out"]
        ckw = _make_unary_compile_kwargs(name)

        import tilelang.language as T

        best_bs, best_t = cfg.block_size, cfg.threads
        if cfg.autotune:
            best_bs, best_t = autotune_config(
                module_name="unary",
                op_name=name,
                cfg=cfg,
                compile_kwargs=ckw,
                runner_factory=make_unary_kernel_runner,
                runner_args=(x, out),
            )

        ckw_all: dict[str, Any] = {
            "module_name": "unary",
            "op_name": name,
            "target": cfg.target,
            "shape": cfg.shape,
            "block_size": best_bs,
            "threads": best_t,
            "dtype": T.float32,
            "execution_backend": cfg.execution_backend or None,
        }
        ckw_all.update(ckw)

        kernel = dispatch_compile(**ckw_all)
        runner = make_unary_kernel_runner(
            kernel,
            tilelang_target=cfg.target,
            execution_backend=cfg.execution_backend or None,
        )

        fn_pt = _make_unary_pt_ref(name, x)
        fn_tl = lambda: runner(x, out)

        ms_pt = bench_ms(fn_pt, cfg.device_str, **bkw)
        ms_tl = bench_ms(fn_tl, cfg.device_str, **bkw)

        results.append(BenchResult(op=name, pytorch_ms=ms_pt, tilelang_ms=ms_tl))

    if not args.no_print_table:
        print(format_results_table(results))

    if args.plot or args.plot_save:
        plot_results(
            results,
            title=f"Unary Ops  —  shape={cfg.shape}  target={cfg.target}",
            save_path=args.plot_save,
        )


if __name__ == "__main__":
    main()
