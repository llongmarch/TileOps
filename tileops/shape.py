"""Shape manipulation operators: ``cat``, ``stack``, ``split``, ``chunk``,
``permute``, ``transpose``, ``flip``, ``repeat``, ``expand``.

All delegate to PyTorch — these are view / copy operations that do not
benefit from custom TileLang kernels.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import torch


def cat(
    tensors: Sequence[torch.Tensor],
    dim: int = 0,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Concatenate tensors along *dim* (``torch.cat``)."""
    if out is not None:
        return torch.cat(tensors, dim=dim, out=out)
    return torch.cat(tensors, dim=dim)


def stack(
    tensors: Sequence[torch.Tensor],
    dim: int = 0,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Stack tensors along a new *dim* (``torch.stack``)."""
    if out is not None:
        return torch.stack(tensors, dim=dim, out=out)
    return torch.stack(tensors, dim=dim)


def split(
    tensor: torch.Tensor,
    split_size_or_sections: Union[int, List[int]],
    dim: int = 0,
) -> List[torch.Tensor]:
    """Split *tensor* into chunks (``torch.split``)."""
    return torch.split(tensor, split_size_or_sections, dim=dim)


def chunk(
    input: torch.Tensor,
    chunks: int,
    dim: int = 0,
) -> List[torch.Tensor]:
    """Split *input* into *chunks* (``torch.chunk``)."""
    return torch.chunk(input, chunks, dim=dim)


def permute(
    input: torch.Tensor,
    *dims: int,
) -> torch.Tensor:
    """Permute dimensions (``torch.permute``)."""
    return input.permute(*dims)


def transpose(
    input: torch.Tensor,
    dim0: int,
    dim1: int,
) -> torch.Tensor:
    """Transpose *dim0* and *dim1* (``torch.transpose``)."""
    return input.transpose(dim0, dim1)


def flip(
    input: torch.Tensor,
    dims: Sequence[int],
) -> torch.Tensor:
    """Reverse the order of elements along *dims* (``torch.flip``)."""
    return torch.flip(input, dims)


def repeat(
    input: torch.Tensor,
    *repeats: int,
) -> torch.Tensor:
    """Repeat *input* along each dimension (``torch.repeat``)."""
    return input.repeat(*repeats)


def expand(
    input: torch.Tensor,
    *sizes: int,
) -> torch.Tensor:
    """Broadcast *input* to *sizes* without allocating new memory (``torch.expand``)."""
    return input.expand(*sizes)


__all__ = [
    "cat", "stack", "split", "chunk",
    "permute", "transpose", "flip",
    "repeat", "expand",
]
