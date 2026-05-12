"""Example: run and verify TileLang pointwise kernels.

Usage::

    pip install -e .
    python examples/pointwise.py
    python examples/pointwise.py --shape 512 512
    python examples/pointwise.py --shape 8 3 224 224   # N-D tensors

Covers basic binary ops, comparison ops, activation functions, and
demonstrates the ``out=`` pre-allocated output feature.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tileops.runtime import setup_metal_workarounds

setup_metal_workarounds()

import torch

from tileops import (
    add, sub, mul, div, pow,
    maximum, minimum, gt,
    relu, gelu, silu, sigmoid, mish,
    default_tilelang_target, default_torch_device,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="TileLang pointwise correctness check")
    parser.add_argument("--shape", type=int, nargs="+", default=[1024, 1024],
                        help="Tensor shape, e.g. --shape 1024 1024 or --shape 8 3 224 224")
    args = parser.parse_args()

    shape = tuple(args.shape)
    tgt = default_tilelang_target()
    device = default_torch_device(tgt)

    print(f"Tensor shape    : {shape}")
    print(f"TileLang target : {tgt}")
    print(f"PyTorch device  : {device}")
    print()

    a = torch.randn(*shape, dtype=torch.float32, device=device)
    b = torch.randn(*shape, dtype=torch.float32, device=device)
    b_div = torch.where(b.abs() < 1e-3, torch.full_like(b, 1e-2), b)
    a_pow = torch.rand(*shape, dtype=torch.float32, device=device) + 0.05
    b_pow = torch.clamp(b, min=0.25, max=4.0)

    rtol, atol = 1e-2, 1e-2

    # ── Binary ops ──────────────────────────────────────────────────────
    print("Binary ops:")
    binary_checks = [
        ("add",     add(a, b),             a + b),
        ("sub",     sub(a, b),             a - b),
        ("mul",     mul(a, b),             a * b),
        ("div",     div(a, b_div),         a / b_div),
        ("pow",     pow(a_pow, b_pow),     torch.pow(a_pow, b_pow)),
        ("maximum", maximum(a, b),         torch.maximum(a, b)),
        ("minimum", minimum(a, b),         torch.minimum(a, b)),
        ("gt",      gt(a, b),              torch.gt(a, b).float()),
    ]
    for name, out, ref in binary_checks:
        assert out.shape == ref.shape, f"{name}: shape mismatch"
        torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
        print(f"  ok: {name}")

    # ── out= pre-allocated output ───────────────────────────────────────
    print("\nout= pre-allocated output:")
    out_buf = torch.empty_like(a)
    result = add(a, b, out=out_buf)
    assert result.data_ptr() == out_buf.data_ptr()
    torch.testing.assert_close(result, a + b, rtol=rtol, atol=atol)
    print("  ok: add(a, b, out=buf)")

    # ── Activation ops ──────────────────────────────────────────────────
    print("\nActivation ops:")
    import torch.nn.functional as F
    activation_checks = [
        ("relu",    relu(a),       torch.relu(a)),
        ("sigmoid", sigmoid(a),    torch.sigmoid(a)),
        ("gelu",    gelu(a),       F.gelu(a, approximate="tanh")),
        ("silu",    silu(a),       F.silu(a)),
        ("mish",    mish(a),       F.mish(a)),
    ]
    for name, out, ref in activation_checks:
        assert out.shape == ref.shape, f"{name}: shape mismatch"
        torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
        print(f"  ok: {name}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
