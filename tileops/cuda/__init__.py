"""CUDA / HIP-specific kernel implementations for TileOps.

Binary GMEM element-wise kernels and the shared-memory ``add_shared``
live in :mod:`tileops.cuda.binary`.  Unary activations are in
:mod:`tileops.cuda.activation`.  Top-level facades dispatch here for
``cuda``, ``hip``, and ``cutedsl`` targets.
"""
