"""CUDA / HIP-specific kernel implementations for TileOps.

Binary GMEM element-wise kernels and the shared-memory ``add_shared``
live in :mod:`tileops.backend.cuda.binary`.  Unary activations are in
:mod:`tileops.backend.cuda.activation`.  Row-wise softmax / norm kernels are in
:mod:`tileops.backend.cuda.norm`; BLAS-style ``gemm`` / ``gemv`` are in
:mod:`tileops.backend.cuda.blas`; row-wise reductions are in :mod:`tileops.backend.cuda.reduce`;
fused activation–multiply kernels are in :mod:`tileops.backend.cuda.fused`.
Top-level facades dispatch here for ``cuda``, ``hip``, and ``cutedsl`` targets.
"""
