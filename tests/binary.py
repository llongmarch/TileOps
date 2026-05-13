"""Correctness tests for binary element-wise kernels (``tileops.binary``).

Run with::

    pip install -e ".[dev]"
    pytest tests/ -v
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tileops import (
    add, sub, mul, div, pow,
    fmod, remainder, floor_div,
    maximum, minimum,
    eq, ne, gt, ge, lt, le,
    atan2, copysign, hypot, xlogy,
    logical_and, logical_or, logical_xor,
    suggest_tile_config, target_kind,
)

# ── Constants ────────────────────────────────────────────────────────────

SHAPE_2D = (256, 256)
SHAPE_4D = (2, 4, 32, 32)
RTOL, ATOL = 1e-5, 1e-5             # tight: exact / near-exact ops
RTOL_MATH, ATOL_MATH = 1e-4, 1e-4   # relaxed: transcendental / multi-step ops


# ── Tensor fixtures ─────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tensors_2d(platform):
    _, dev = platform
    a = torch.randn(*SHAPE_2D, dtype=torch.float32, device=dev)
    b = torch.randn(*SHAPE_2D, dtype=torch.float32, device=dev)
    b_div = torch.where(b.abs() < 1e-3, torch.full_like(b, 1e-2), b)
    a_pow = torch.rand(*SHAPE_2D, dtype=torch.float32, device=dev) + 0.05
    b_pow = torch.clamp(b, min=0.25, max=4.0)
    return a, b, b_div, a_pow, b_pow


@pytest.fixture(scope="module")
def tensors_4d(platform):
    _, dev = platform
    a = torch.randn(*SHAPE_4D, dtype=torch.float32, device=dev)
    b = torch.randn(*SHAPE_4D, dtype=torch.float32, device=dev)
    b_div = torch.where(b.abs() < 1e-3, torch.full_like(b, 1e-2), b)
    a_pow = torch.rand(*SHAPE_4D, dtype=torch.float32, device=dev) + 0.05
    b_pow = torch.clamp(b, min=0.25, max=4.0)
    return a, b, b_div, a_pow, b_pow


# ═══════════════════════════════════════════════════════════════════════════
# Basic arithmetic (2-D)
# ═══════════════════════════════════════════════════════════════════════════

def test_add_2d(platform, tensors_2d):
    a, b, *_ = tensors_2d
    out = add(a, b)
    assert out.shape == a.shape
    torch.testing.assert_close(out, a + b, rtol=RTOL, atol=ATOL)


def test_sub_2d(platform, tensors_2d):
    a, b, *_ = tensors_2d
    out = sub(a, b)
    assert out.shape == a.shape
    torch.testing.assert_close(out, a - b, rtol=RTOL, atol=ATOL)


def test_mul_2d(platform, tensors_2d):
    a, b, *_ = tensors_2d
    out = mul(a, b)
    assert out.shape == a.shape
    torch.testing.assert_close(out, a * b, rtol=RTOL, atol=ATOL)


def test_div_2d(platform, tensors_2d):
    a, _, b_div, *_ = tensors_2d
    out = div(a, b_div)
    assert out.shape == a.shape
    torch.testing.assert_close(out, a / b_div, rtol=RTOL, atol=ATOL)


def test_pow_2d(platform, tensors_2d):
    *_, a_pow, b_pow = tensors_2d
    out = pow(a_pow, b_pow)
    assert out.shape == a_pow.shape
    torch.testing.assert_close(out, torch.pow(a_pow, b_pow), rtol=RTOL_MATH, atol=ATOL_MATH)


# ═══════════════════════════════════════════════════════════════════════════
# Basic arithmetic (4-D)
# ═══════════════════════════════════════════════════════════════════════════

def test_add_4d(platform, tensors_4d):
    a, b, *_ = tensors_4d
    out = add(a, b)
    assert out.shape == SHAPE_4D
    torch.testing.assert_close(out, a + b, rtol=RTOL, atol=ATOL)


def test_mul_4d(platform, tensors_4d):
    a, b, *_ = tensors_4d
    out = mul(a, b)
    assert out.shape == SHAPE_4D
    torch.testing.assert_close(out, a * b, rtol=RTOL, atol=ATOL)


def test_pow_4d(platform, tensors_4d):
    *_, a_pow, b_pow = tensors_4d
    out = pow(a_pow, b_pow)
    assert out.shape == SHAPE_4D
    torch.testing.assert_close(out, torch.pow(a_pow, b_pow), rtol=RTOL_MATH, atol=ATOL_MATH)


# ═══════════════════════════════════════════════════════════════════════════
# Extended arithmetic — fmod, remainder, floor_div
# ═══════════════════════════════════════════════════════════════════════════

def test_fmod(platform, tensors_2d):
    a, _, b_div, *_ = tensors_2d
    out = fmod(a, b_div)
    torch.testing.assert_close(out, torch.fmod(a, b_div), rtol=RTOL_MATH, atol=ATOL_MATH)


def test_remainder(platform, tensors_2d):
    a, _, b_div, *_ = tensors_2d
    out = remainder(a, b_div)
    torch.testing.assert_close(out, torch.remainder(a, b_div), rtol=RTOL_MATH, atol=ATOL_MATH)


def test_floor_div(platform, tensors_2d):
    a, _, b_div, *_ = tensors_2d
    out = floor_div(a, b_div)
    torch.testing.assert_close(
        out, torch.div(a, b_div, rounding_mode="floor"), rtol=RTOL_MATH, atol=ATOL_MATH,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Extrema — maximum, minimum
# ═══════════════════════════════════════════════════════════════════════════

def test_maximum(platform, tensors_2d):
    a, b, *_ = tensors_2d
    out = maximum(a, b)
    torch.testing.assert_close(out, torch.maximum(a, b), rtol=RTOL, atol=ATOL)


def test_minimum(platform, tensors_2d):
    a, b, *_ = tensors_2d
    out = minimum(a, b)
    torch.testing.assert_close(out, torch.minimum(a, b), rtol=RTOL, atol=ATOL)


# ═══════════════════════════════════════════════════════════════════════════
# Comparison — output 0.0 / 1.0 (same dtype as input)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def cmp_tensors(platform):
    _, dev = platform
    a = torch.tensor([1.0, 2.0, 3.0, 4.0], device=dev)
    b = torch.tensor([4.0, 2.0, 1.0, 4.0], device=dev)
    return a, b


_CMP_CASES = [
    ("eq", eq, torch.eq),
    ("ne", ne, torch.ne),
    ("gt", gt, torch.gt),
    ("ge", ge, torch.ge),
    ("lt", lt, torch.lt),
    ("le", le, torch.le),
]


@pytest.mark.parametrize("name,tl_fn,pt_fn", _CMP_CASES, ids=[c[0] for c in _CMP_CASES])
def test_comparison(platform, cmp_tensors, name, tl_fn, pt_fn):
    a, b = cmp_tensors
    out = tl_fn(a, b)
    ref = pt_fn(a, b).float()
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


# ═══════════════════════════════════════════════════════════════════════════
# Math — atan2, copysign, hypot, xlogy
# ═══════════════════════════════════════════════════════════════════════════

def test_atan2(platform):
    _, dev = platform
    a = torch.randn(128, dtype=torch.float32, device=dev)
    b = torch.randn(128, dtype=torch.float32, device=dev)
    b = torch.where(b.abs() < 0.01, torch.full_like(b, 0.1), b)
    out = atan2(a, b)
    torch.testing.assert_close(out, torch.atan2(a, b), rtol=RTOL_MATH, atol=ATOL_MATH)


def test_copysign(platform):
    _, dev = platform
    a = torch.randn(128, dtype=torch.float32, device=dev)
    b = torch.randn(128, dtype=torch.float32, device=dev)
    out = copysign(a, b)
    torch.testing.assert_close(out, torch.copysign(a, b), rtol=RTOL, atol=ATOL)


def test_hypot(platform):
    _, dev = platform
    a = torch.randn(128, dtype=torch.float32, device=dev)
    b = torch.randn(128, dtype=torch.float32, device=dev)
    out = hypot(a, b)
    torch.testing.assert_close(out, torch.hypot(a, b), rtol=RTOL_MATH, atol=ATOL_MATH)


def test_xlogy(platform):
    _, dev = platform
    a = torch.tensor([0.0, 1.0, 2.0, 0.0, 3.0], dtype=torch.float32, device=dev)
    b = torch.tensor([0.5, 1.0, 2.0, 0.1, 0.5], dtype=torch.float32, device=dev)
    out = xlogy(a, b)
    torch.testing.assert_close(out, torch.xlogy(a, b), rtol=RTOL_MATH, atol=ATOL_MATH)


# ═══════════════════════════════════════════════════════════════════════════
# Logical — logical_and, logical_or, logical_xor
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def logic_tensors(platform):
    _, dev = platform
    a = torch.tensor([0.0, 0.0, 1.0, 2.5], dtype=torch.float32, device=dev)
    b = torch.tensor([0.0, 3.0, 0.0, -1.0], dtype=torch.float32, device=dev)
    return a, b


def test_logical_and(platform, logic_tensors):
    a, b = logic_tensors
    out = logical_and(a, b)
    ref = torch.logical_and(a, b).float()
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_logical_or(platform, logic_tensors):
    a, b = logic_tensors
    out = logical_or(a, b)
    ref = torch.logical_or(a, b).float()
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_logical_xor(platform, logic_tensors):
    a, b = logic_tensors
    out = logical_xor(a, b)
    ref = torch.logical_xor(a, b).float()
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


# ═══════════════════════════════════════════════════════════════════════════
# Shared-memory add (CUDA / HIP only)
# ═══════════════════════════════════════════════════════════════════════════

def test_add_shared_cuda(platform, tensors_2d):
    a, b, *_ = tensors_2d
    tgt, _ = platform
    if target_kind(tgt) not in ("cuda", "hip"):
        pytest.skip(f"shared-tile add requires cuda/hip (current: {tgt!r})")
    import tilelang.language as T

    from tileops.cuda.binary import add_shared
    from tileops.runtime import default_execution_backend, invoke_kernel

    eb = default_execution_backend(tgt, _)
    kernel = add_shared(SHAPE_2D, 1024, 128, dtype=T.float32, target=tgt, execution_backend=eb)
    out = invoke_kernel(kernel, a, b, tilelang_target=tgt, execution_backend=eb)
    torch.testing.assert_close(out, a + b, rtol=RTOL, atol=ATOL)


# ═══════════════════════════════════════════════════════════════════════════
# out= parameter
# ═══════════════════════════════════════════════════════════════════════════

def test_add_with_out(platform, tensors_2d):
    a, b, *_ = tensors_2d
    out_buf = torch.empty_like(a)
    result = add(a, b, out=out_buf)
    torch.testing.assert_close(result, a + b, rtol=RTOL, atol=ATOL)
    assert result.data_ptr() == out_buf.data_ptr()


# ═══════════════════════════════════════════════════════════════════════════
# Facade input validation
# ═══════════════════════════════════════════════════════════════════════════

def test_shape_mismatch_raises():
    a = torch.randn(4, dtype=torch.float32)
    b = torch.randn(8, dtype=torch.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        add(a, b)


def test_dtype_mismatch_raises():
    a = torch.randn(4, dtype=torch.float32)
    b = torch.randn(4, dtype=torch.float16)
    with pytest.raises(TypeError, match="dtype mismatch"):
        add(a, b)


def test_noncontiguous_out_is_written_back(platform):
    _, dev = platform
    a = torch.randn(4, 3, dtype=torch.float32, device=dev)
    b = torch.randn(4, 3, dtype=torch.float32, device=dev)
    out = torch.empty(3, 4, dtype=torch.float32, device=dev).t()
    assert not out.is_contiguous()
    ret = add(a, b, out=out)
    assert ret is out
    torch.testing.assert_close(out, a + b, rtol=RTOL, atol=ATOL)


# ═══════════════════════════════════════════════════════════════════════════
# Runtime helpers
# ═══════════════════════════════════════════════════════════════════════════

class TestSuggestTileConfig:
    """Unit tests for :func:`suggest_tile_config`."""

    def test_power_of_two(self):
        bs, t = suggest_tile_config(1024 * 1024)
        assert (1024 * 1024) % bs == 0

    def test_non_power_of_two(self):
        bs, t = suggest_tile_config(1000 * 1000)
        assert bs >= 128

    def test_small_elements(self):
        bs, t = suggest_tile_config(64)
        assert 1 <= bs <= 1024

    def test_invalid(self):
        with pytest.raises(ValueError):
            suggest_tile_config(0)
        with pytest.raises(ValueError):
            suggest_tile_config(-1)
