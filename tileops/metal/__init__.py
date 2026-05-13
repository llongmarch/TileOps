"""Metal-specific kernel implementations for TileOps.

Binary GMEM kernels live in :mod:`tileops.metal.binary`.
Unary activations are in :mod:`tileops.metal.activation`.
Row-wise softmax / norm kernels and BLAS-style ``gemm`` / ``gemv`` live in
:mod:`tileops.metal.norm` and :mod:`tileops.metal.blas`; row-wise reductions
are in :mod:`tileops.metal.reduce`; fused activation–multiply kernels are in
:mod:`tileops.metal.fused`.
Top-level facades dispatch here for ``metal`` targets.
"""
