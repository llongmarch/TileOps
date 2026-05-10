"""Correctness tests for ``tileops.pointwise`` kernels.

Run with::

    pip install -e ".[dev]"
    pytest tests/ -v
"""

from __future__ import annotations

import math

import pytest
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
    suggest_pointwise_config,
    target_kind,
)

# ── Fixtures ────────────────────────────────────────────────────────────

SHAPE_2D = (256, 256)
SHAPE_4D = (2, 4, 32, 32)
BLOCK_SIZE = 1024
THREADS = 128
RTOL, ATOL = 1e-2, 1e-2


@pytest.fixture(scope="module")
def platform():
    """Resolve target / device / execution-backend once per test module."""
    tgt = default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = default_execution_backend(tgt, dev)
    return tgt, dev, eb


@pytest.fixture(scope="module")
def tensors_2d(platform):
    """Create shared 2-D input tensors."""
    _, dev, _ = platform
    a = torch.randn(*SHAPE_2D, dtype=torch.float32, device=dev)
    b = torch.randn(*SHAPE_2D, dtype=torch.float32, device=dev)
    b_div = torch.where(b.abs() < 1e-3, torch.full_like(b, 1e-2), b)
    a_pow = torch.rand(*SHAPE_2D, dtype=torch.float32, device=dev) + 0.05
    b_pow = torch.clamp(b, min=0.25, max=4.0)
    return a, b, b_div, a_pow, b_pow


@pytest.fixture(scope="module")
def tensors_4d(platform):
    """Create shared 4-D input tensors."""
    _, dev, _ = platform
    a = torch.randn(*SHAPE_4D, dtype=torch.float32, device=dev)
    b = torch.randn(*SHAPE_4D, dtype=torch.float32, device=dev)
    b_div = torch.where(b.abs() < 1e-3, torch.full_like(b, 1e-2), b)
    a_pow = torch.rand(*SHAPE_4D, dtype=torch.float32, device=dev) + 0.05
    b_pow = torch.clamp(b, min=0.25, max=4.0)
    return a, b, b_div, a_pow, b_pow


def _compile_kw(platform) -> dict:
    tgt, _, eb = platform
    kw: dict[str, str] = {"target": tgt}
    if eb is not None:
        kw["execution_backend"] = eb
    return kw


# ── 2-D kernel correctness ──────────────────────────────────────────────

def test_pointwise_add_2d(platform, tensors_2d):
    a, b, *_ = tensors_2d
    tgt, _, eb = platform
    kernel = pointwise_add(SHAPE_2D, BLOCK_SIZE, THREADS, **_compile_kw(platform))
    out = invoke_kernel(kernel, a, b, tilelang_target=tgt, execution_backend=eb)
    assert out.shape == a.shape
    torch.testing.assert_close(out, a + b, rtol=RTOL, atol=ATOL)


def test_pointwise_sub_2d(platform, tensors_2d):
    a, b, *_ = tensors_2d
    tgt, _, eb = platform
    kernel = pointwise_sub(SHAPE_2D, BLOCK_SIZE, THREADS, **_compile_kw(platform))
    out = invoke_kernel(kernel, a, b, tilelang_target=tgt, execution_backend=eb)
    assert out.shape == a.shape
    torch.testing.assert_close(out, a - b, rtol=RTOL, atol=ATOL)


def test_pointwise_mul_2d(platform, tensors_2d):
    a, b, *_ = tensors_2d
    tgt, _, eb = platform
    kernel = pointwise_mul(SHAPE_2D, BLOCK_SIZE, THREADS, **_compile_kw(platform))
    out = invoke_kernel(kernel, a, b, tilelang_target=tgt, execution_backend=eb)
    assert out.shape == a.shape
    torch.testing.assert_close(out, a * b, rtol=RTOL, atol=ATOL)


def test_pointwise_div_2d(platform, tensors_2d):
    a, _, b_div, *_ = tensors_2d
    tgt, _, eb = platform
    kernel = pointwise_div(SHAPE_2D, BLOCK_SIZE, THREADS, **_compile_kw(platform))
    out = invoke_kernel(kernel, a, b_div, tilelang_target=tgt, execution_backend=eb)
    assert out.shape == a.shape
    torch.testing.assert_close(out, a / b_div, rtol=RTOL, atol=ATOL)


def test_pointwise_pow_2d(platform, tensors_2d):
    *_, a_pow, b_pow = tensors_2d
    tgt, _, eb = platform
    kernel = pointwise_pow(SHAPE_2D, BLOCK_SIZE, THREADS, **_compile_kw(platform))
    out = invoke_kernel(kernel, a_pow, b_pow, tilelang_target=tgt, execution_backend=eb)
    assert out.shape == a_pow.shape
    torch.testing.assert_close(out, torch.pow(a_pow, b_pow), rtol=RTOL, atol=ATOL)


# ── 4-D (N-D) kernel correctness ────────────────────────────────────────

def test_pointwise_add_4d(platform, tensors_4d):
    a, b, *_ = tensors_4d
    tgt, _, eb = platform
    kernel = pointwise_add(SHAPE_4D, BLOCK_SIZE, THREADS, **_compile_kw(platform))
    out = invoke_kernel(kernel, a, b, tilelang_target=tgt, execution_backend=eb)
    assert out.shape == SHAPE_4D
    torch.testing.assert_close(out, a + b, rtol=RTOL, atol=ATOL)


def test_pointwise_mul_4d(platform, tensors_4d):
    a, b, *_ = tensors_4d
    tgt, _, eb = platform
    kernel = pointwise_mul(SHAPE_4D, BLOCK_SIZE, THREADS, **_compile_kw(platform))
    out = invoke_kernel(kernel, a, b, tilelang_target=tgt, execution_backend=eb)
    assert out.shape == SHAPE_4D
    torch.testing.assert_close(out, a * b, rtol=RTOL, atol=ATOL)


def test_pointwise_pow_4d(platform, tensors_4d):
    *_, a_pow, b_pow = tensors_4d
    tgt, _, eb = platform
    kernel = pointwise_pow(SHAPE_4D, BLOCK_SIZE, THREADS, **_compile_kw(platform))
    out = invoke_kernel(kernel, a_pow, b_pow, tilelang_target=tgt, execution_backend=eb)
    assert out.shape == SHAPE_4D
    torch.testing.assert_close(out, torch.pow(a_pow, b_pow), rtol=RTOL, atol=ATOL)


# ── Shared-memory add (CUDA / HIP only) ─────────────────────────────────

def test_pointwise_add_shared_cuda(platform, tensors_2d):
    a, b, *_ = tensors_2d
    tgt, _, eb = platform
    if target_kind(tgt) not in ("cuda", "hip"):
        pytest.skip(f"shared-tile add requires cuda/hip (current: {tgt!r})")
    from tileops.cuda.pointwise import pointwise_add_shared

    kernel = pointwise_add_shared(SHAPE_2D, BLOCK_SIZE, THREADS, **_compile_kw(platform))
    out = invoke_kernel(kernel, a, b, tilelang_target=tgt, execution_backend=eb)
    torch.testing.assert_close(out, a + b, rtol=RTOL, atol=ATOL)


# ── Runtime helpers ─────────────────────────────────────────────────────

class TestSuggestPointwiseConfig:
    """Unit tests for :func:`suggest_pointwise_config`."""

    def test_power_of_two(self):
        bs, t = suggest_pointwise_config(1024 * 1024)
        assert (1024 * 1024) % bs == 0

    def test_non_power_of_two(self):
        bs, t = suggest_pointwise_config(1000 * 1000)
        assert bs >= 128

    def test_small_elements(self):
        bs, t = suggest_pointwise_config(64)
        assert 1 <= bs <= 1024

    def test_invalid(self):
        with pytest.raises(ValueError):
            suggest_pointwise_config(0)
        with pytest.raises(ValueError):
            suggest_pointwise_config(-1)
