"""Metal-specific kernel implementations for TileOps.

This sub-package will house optimised kernels that exploit Apple GPU
features (simdgroup intrinsics, threadgroup memory layout, etc.).

Currently a placeholder — all Metal-compatible operations are served
by the generic GMEM-direct kernels in :mod:`tileops.pointwise`.
"""
