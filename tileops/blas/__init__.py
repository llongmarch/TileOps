"""BLAS-style ops: matrix–matrix / matrix–vector multiply and PyTorch aliases.

``gemm`` / ``gemv`` are the TileLang-backed primitives.  ``mm``, ``mv``,
``outer``, ``bmm`` wrap them with :class:`torch.Tensor` shapes matching
``torch.mm``, ``torch.mv``, etc.  ``addmm`` / ``baddbmm`` compute the
matrix product with the same kernels, then apply ``beta`` / ``alpha`` with
PyTorch element-wise ops (on-device, broadcast rules match ``torch.addmm``).

Dtypes: ``float32``, ``float16``, ``bfloat16``.
"""

from tileops.blas.gemm import (
    addmm,
    baddbmm,
    bmm,
    gemm,
    mm,
    outer,
)
from tileops.blas.gemv import (
    gemv,
    mv,
)

__all__ = [
    "addmm",
    "baddbmm",
    "bmm",
    "gemm",
    "gemv",
    "mm",
    "mv",
    "outer",
]
