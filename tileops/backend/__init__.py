"""Hardware backend implementations for TileOps.

CUDA / HIP / cutedsl kernels live in :mod:`tileops.backend.cuda`.
Metal / MPS kernels live in :mod:`tileops.backend.metal`.

End-users should not import from this package directly; use the top-level
facades (e.g., :func:`tileops.add`, :func:`tileops.relu`) instead.
"""
