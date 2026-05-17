"""Tensor interpolation / resizing (upsample / downsample).

Delegates to :func:`torch.nn.functional.interpolate` for all modes; a future
TileLang kernel may accelerate nearest / bilinear on common shapes.

**API**

* :func:`interpolate` — resize a tensor.

::

    from tileops import interpolate

    y = interpolate(x, size=(224, 224), mode="bilinear")
    y = interpolate(x, scale_factor=2.0, mode="nearest")
    y = interpolate(x, size=(64, 64), mode="bicubic", align_corners=False)
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn.functional as F


def interpolate(
    input: torch.Tensor,
    size: Optional[Union[int, tuple[int, ...]]] = None,
    scale_factor: Optional[Union[float, tuple[float, ...]]] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None,
    antialias: bool = False,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Resize *input* to *size* or by *scale_factor*.

    Exactly one of *size* or *scale_factor* must be provided.

    Supported *mode* values match :func:`torch.nn.functional.interpolate`:

    - ``"nearest"`` — nearest-neighbour (default)
    - ``"nearest-exact"`` — exact nearest-neighbour (PyTorch >= 1.11)
    - ``"linear"`` — linear interpolation (3-D input only)
    - ``"bilinear"`` — bilinear (4-D input)
    - ``"bicubic"`` — bicubic (4-D input)
    - ``"trilinear"`` — trilinear (5-D input)
    - ``"area"`` — area / average pooling
    """
    if size is not None and scale_factor is not None:
        raise ValueError(
            "interpolate: only one of size or scale_factor may be given"
        )
    if size is None and scale_factor is None:
        raise ValueError(
            "interpolate: either size or scale_factor must be provided"
        )

    result = F.interpolate(
        input,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
        antialias=antialias,
    )
    if out is not None:
        out.copy_(result)
        return out
    return result
