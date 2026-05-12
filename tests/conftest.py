"""Shared pytest configuration for the TileOps test suite."""

import pytest

from tileops.runtime import (
    default_tilelang_target,
    default_torch_device,
    setup_metal_workarounds,
    target_kind,
)

# Must run before any TileLang kernel compilation.
setup_metal_workarounds()

_SUPPORTED_KINDS = ("cuda", "hip", "cutedsl", "metal")


@pytest.fixture(scope="module")
def platform():
    """Resolve target / device once per test module; skip on unsupported hosts."""
    tgt = default_tilelang_target()
    kind = target_kind(tgt)
    if kind not in _SUPPORTED_KINDS:
        pytest.skip(f"requires cuda/hip/metal target; got {tgt!r} ({kind})")
    dev = default_torch_device(tgt)
    return tgt, dev
