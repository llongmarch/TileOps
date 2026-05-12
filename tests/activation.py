"""Correctness tests for ``tileops.activation`` unary kernels.

Run with::

    pip install -e ".[dev]"
    pytest tests/ -v
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tileops import (
    relu, sigmoid, tanh,
    gelu, gelu_exact,
    silu, hardswish, hardsigmoid,
    leaky_relu, relu6, elu, selu, celu, hardtanh,
    softplus, mish, softsign,
    log_sigmoid,
)

SHAPE = (128, 128)
RTOL, ATOL = 1e-5, 1e-5             # tight: branchless / simple ops
RTOL_MATH, ATOL_MATH = 1e-4, 1e-4   # relaxed: transcendental / multi-step ops


@pytest.fixture(scope="module")
def x_tensor(platform):
    _, dev = platform
    return torch.randn(*SHAPE, dtype=torch.float32, device=dev)


# ═══════════════════════════════════════════════════════════════════════════
# Classic
# ═══════════════════════════════════════════════════════════════════════════

def test_relu(platform, x_tensor):
    out = relu(x_tensor)
    torch.testing.assert_close(out, torch.relu(x_tensor), rtol=RTOL, atol=ATOL)


def test_sigmoid(platform, x_tensor):
    out = sigmoid(x_tensor)
    torch.testing.assert_close(out, torch.sigmoid(x_tensor), rtol=RTOL_MATH, atol=ATOL_MATH)


def test_tanh(platform, x_tensor):
    out = tanh(x_tensor)
    torch.testing.assert_close(out, torch.tanh(x_tensor), rtol=RTOL_MATH, atol=ATOL_MATH)


# ═══════════════════════════════════════════════════════════════════════════
# GELU
# ═══════════════════════════════════════════════════════════════════════════

def test_gelu(platform, x_tensor):
    out = gelu(x_tensor)
    ref = F.gelu(x_tensor, approximate="tanh")
    torch.testing.assert_close(out, ref, rtol=RTOL_MATH, atol=ATOL_MATH)


def test_gelu_exact(platform, x_tensor):
    out = gelu_exact(x_tensor)
    ref = F.gelu(x_tensor, approximate="none")
    torch.testing.assert_close(out, ref, rtol=RTOL_MATH, atol=ATOL_MATH)


# ═══════════════════════════════════════════════════════════════════════════
# Swish family
# ═══════════════════════════════════════════════════════════════════════════

def test_silu(platform, x_tensor):
    out = silu(x_tensor)
    torch.testing.assert_close(out, F.silu(x_tensor), rtol=RTOL_MATH, atol=ATOL_MATH)


def test_hardswish(platform, x_tensor):
    out = hardswish(x_tensor)
    torch.testing.assert_close(out, F.hardswish(x_tensor), rtol=RTOL, atol=ATOL)


def test_hardsigmoid(platform, x_tensor):
    out = hardsigmoid(x_tensor)
    torch.testing.assert_close(out, F.hardsigmoid(x_tensor), rtol=RTOL, atol=ATOL)


# ═══════════════════════════════════════════════════════════════════════════
# ReLU variants
# ═══════════════════════════════════════════════════════════════════════════

def test_leaky_relu(platform, x_tensor):
    out = leaky_relu(x_tensor)
    torch.testing.assert_close(out, F.leaky_relu(x_tensor, 0.01), rtol=RTOL, atol=ATOL)


def test_relu6(platform, x_tensor):
    out = relu6(x_tensor)
    torch.testing.assert_close(out, F.relu6(x_tensor), rtol=RTOL, atol=ATOL)


def test_elu(platform, x_tensor):
    out = elu(x_tensor)
    torch.testing.assert_close(out, F.elu(x_tensor, alpha=1.0), rtol=RTOL_MATH, atol=ATOL_MATH)


def test_selu(platform, x_tensor):
    out = selu(x_tensor)
    torch.testing.assert_close(out, F.selu(x_tensor), rtol=RTOL_MATH, atol=ATOL_MATH)


def test_celu(platform, x_tensor):
    out = celu(x_tensor)
    torch.testing.assert_close(out, F.celu(x_tensor, alpha=1.0), rtol=RTOL_MATH, atol=ATOL_MATH)


def test_hardtanh(platform, x_tensor):
    out = hardtanh(x_tensor)
    torch.testing.assert_close(out, F.hardtanh(x_tensor), rtol=RTOL, atol=ATOL)


# ═══════════════════════════════════════════════════════════════════════════
# Smooth
# ═══════════════════════════════════════════════════════════════════════════

def test_softplus(platform, x_tensor):
    out = softplus(x_tensor)
    torch.testing.assert_close(out, F.softplus(x_tensor), rtol=RTOL_MATH, atol=ATOL_MATH)


def test_mish(platform, x_tensor):
    out = mish(x_tensor)
    torch.testing.assert_close(out, F.mish(x_tensor), rtol=RTOL_MATH, atol=ATOL_MATH)


def test_softsign(platform, x_tensor):
    out = softsign(x_tensor)
    torch.testing.assert_close(out, F.softsign(x_tensor), rtol=RTOL, atol=ATOL)


# ═══════════════════════════════════════════════════════════════════════════
# Log
# ═══════════════════════════════════════════════════════════════════════════

def test_log_sigmoid(platform, x_tensor):
    out = log_sigmoid(x_tensor)
    torch.testing.assert_close(out, F.logsigmoid(x_tensor), rtol=RTOL_MATH, atol=ATOL_MATH)


# ═══════════════════════════════════════════════════════════════════════════
# out= parameter
# ═══════════════════════════════════════════════════════════════════════════

def test_relu_with_out(platform, x_tensor):
    out_buf = torch.empty_like(x_tensor)
    result = relu(x_tensor, out=out_buf)
    torch.testing.assert_close(result, torch.relu(x_tensor), rtol=RTOL, atol=ATOL)
    assert result.data_ptr() == out_buf.data_ptr()


# ═══════════════════════════════════════════════════════════════════════════
# Facade input validation
# ═══════════════════════════════════════════════════════════════════════════

def test_empty_tensor_raises():
    x = torch.empty(0, dtype=torch.float32)
    with pytest.raises(ValueError, match="empty"):
        relu(x)
