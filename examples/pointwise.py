"""Example: run and verify TileLang pointwise kernels (add / sub / mul / div / pow).

Usage::

    pip install -e .                     # one-time
    python examples/pointwise.py
    python examples/pointwise.py --m 512 --n 512
    python examples/pointwise.py --shape 8 3 224 224   # N-D tensors

The script auto-detects the TileLang target and matching PyTorch device
so that it works transparently on Apple Silicon (metal + mps),
CPU (llvm), and CUDA.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Fallback for running without ``pip install -e .``.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tileops.runtime import setup_metal_workarounds

setup_metal_workarounds()

import torch

from tileops import (
    default_execution_backend,
    default_tilelang_target,
    default_torch_device,
    invoke_kernel,
    pointwise_add,
    pointwise_div,
    pointwise_mul,
    pointwise_pow,
    pointwise_sub,
)


def _build_kernels(
    shape: tuple[int, ...],
    block_size: int,
    threads: int,
    target: str,
    execution_backend: str | None,
) -> dict[str, object]:
    """Compile all five pointwise kernels for *shape*."""
    kw: dict[str, str] = {"target": target}
    if execution_backend is not None:
        kw["execution_backend"] = execution_backend
    return {
        "add": pointwise_add(shape, block_size, threads, **kw),
        "sub": pointwise_sub(shape, block_size, threads, **kw),
        "mul": pointwise_mul(shape, block_size, threads, **kw),
        "div": pointwise_div(shape, block_size, threads, **kw),
        "pow": pointwise_pow(shape, block_size, threads, **kw),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="TileLang pointwise correctness check")
    parser.add_argument("--shape", type=int, nargs="+", default=[1024, 1024],
                        help="Tensor shape, e.g. --shape 1024 1024 or --shape 8 3 224 224")
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--target", type=str, default=None,
                        help="TileLang target (default: auto-detect)")
    parser.add_argument("--execution-backend", type=str, default=None,
                        help="TileLang execution backend (default: auto)")
    args = parser.parse_args()

    shape = tuple(args.shape)
    tgt = args.target or default_tilelang_target()
    device = default_torch_device(tgt)
    eb = args.execution_backend or default_execution_backend(tgt, device)

    print(f"Tensor shape    : {shape}")
    print(f"TileLang target : {tgt}")
    print(f"PyTorch device  : {device}")
    if eb:
        print(f"Execution backend: {eb}")
    print()

    a = torch.randn(*shape, dtype=torch.float32, device=device)
    b = torch.randn(*shape, dtype=torch.float32, device=device)
    b_div = torch.where(b.abs() < 1e-3, torch.full_like(b, 1e-2), b)
    a_pow = torch.rand(*shape, dtype=torch.float32, device=device) + 0.05
    b_pow = torch.clamp(b, min=0.25, max=4.0)

    kernels = _build_kernels(shape, args.block_size, args.threads, tgt, eb)

    checks: list[tuple[str, object, torch.Tensor, torch.Tensor, torch.Tensor]] = [
        ("add", kernels["add"], a,     b,     a + b),
        ("sub", kernels["sub"], a,     b,     a - b),
        ("mul", kernels["mul"], a,     b,     a * b),
        ("div", kernels["div"], a,     b_div, a / b_div),
        ("pow", kernels["pow"], a_pow, b_pow, torch.pow(a_pow, b_pow)),
    ]

    for name, kernel, x, y, ref in checks:
        out = invoke_kernel(
            kernel, x, y,
            tilelang_target=tgt, execution_backend=eb,
        )
        assert out.shape == ref.shape, f"{name}: shape mismatch {out.shape} vs {ref.shape}"
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
        print(f"  ok: {name}")

    print("\nAll pointwise checks passed.")


if __name__ == "__main__":
    main()
