"""CUDA-specific kernel implementations for TileOps.

This sub-package contains optimised kernels that exploit CUDA / HIP
hardware features (shared memory, warp-level primitives, etc.).
They are **not** imported directly by users — the top-level
``tileops.pointwise`` dispatch layer falls back to these
automatically when running on a CUDA target.

You can also import them explicitly::

    from tileops.cuda.pointwise import pointwise_add_shared
"""
