"""Metal-specific kernel implementations for TileOps.

Binary GMEM kernels live in :mod:`tileops.backend.metal.binary`.
Unary activations are in :mod:`tileops.backend.metal.activation`.
Row-wise softmax / norm kernels and BLAS-style ``gemm`` / ``gemv`` live in
:mod:`tileops.backend.metal.norm` and :mod:`tileops.backend.metal.blas`; row-wise reductions
are in :mod:`tileops.backend.metal.reduce`; fused activation–multiply kernels are in
:mod:`tileops.backend.metal.fused`.
Top-level facades dispatch here for ``metal`` targets.
"""
