"""vLLM ``awq_triton.py``-compatible AWQ dequantization and GEMM helpers.

Semantics follow vLLM v0.21.0
``vllm.model_executor.layers.quantization.awq_triton``:

* ``qweight`` / ``qzeros`` pack eight 4-bit values per ``int32`` with AWQ nibble
  order ``[0, 4, 1, 5, 2, 6, 3, 7]``.
* Dequant: ``(unpacked_weight - unpacked_zero) * scale`` per group along *K*.
* ``awq_gemm_triton`` reference path: dequantize weights then ``input @ W``.

Triton kernels can replace the PyTorch reference later.
"""

from __future__ import annotations

import torch

# vLLM: supported group sizes; ``-1`` is a sentinel, ``K`` is inferred per tensor.
AWQ_TRITON_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]

# Nibble unpack order used by AWQ Triton kernels.
REVERSE_AWQ_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)


def _infer_group_size(k: int, num_groups: int) -> int:
    if num_groups <= 0 or k % num_groups != 0:
        raise ValueError(f"invalid AWQ group layout: K={k}, num_groups={num_groups}")
    return k // num_groups


def _validate_group_size(group_size: int, k: int) -> None:
    if group_size > k:
        raise ValueError(f"group_size {group_size} must be <= K={k}")
    if group_size not in AWQ_TRITON_SUPPORTED_GROUP_SIZES and group_size != k:
        raise ValueError(
            f"group_size {group_size} not in {AWQ_TRITON_SUPPORTED_GROUP_SIZES} "
            f"and not full K={k}"
        )


def unpack_awq_int4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack AWQ ``int32`` tiles to int64 nibbles in ``[0, 15]``.

    Last dimension expands by 8 (each ``int32`` holds eight 4-bit values).
    """
    if packed.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"unpack_awq_int4: expected int32 packed tensor, got {packed.dtype}")
    word = packed.to(torch.int64)
    pieces = [(word >> (shift * 4)) & 0xF for shift in REVERSE_AWQ_ORDER]
    # Stack nibbles for each packed word, then flatten (AWQ column order).
    stacked = torch.stack(pieces, dim=-1)
    return stacked.reshape(*packed.shape[:-1], packed.shape[-1] * 8)


def pack_awq_int4(unpacked: torch.Tensor) -> torch.Tensor:
    """Pack int64/int32 nibbles ``[..., cols]`` (``cols % 8 == 0``) to AWQ ``int32``."""
    if unpacked.shape[-1] % 8 != 0:
        raise ValueError("pack_awq_int4: last dimension must be divisible by 8")
    cols8 = unpacked.shape[-1] // 8
    lead = unpacked.shape[:-1]
    tiles = unpacked.reshape(*lead, cols8, 8).to(torch.int64)
    packed = torch.zeros(*lead, cols8, dtype=torch.int64, device=unpacked.device)
    for i, shift in enumerate(REVERSE_AWQ_ORDER):
        packed |= (tiles[..., i] & 0xF) << (shift * 4)
    return packed.to(torch.int32)


def awq_dequantize_triton(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    block_size_x: int = 32,
    block_size_y: int = 32,
) -> torch.Tensor:
    """Dequantize AWQ ``qweight`` → floating matrix ``[K, M]``.

    * ``qweight``: ``[K, M // 8]`` ``int32``
    * ``scales``: ``[K // G, M]``
    * ``zeros``: ``[K // G, M // 8]`` ``int32``
    * ``block_size_*``: accepted for API compatibility (unused in reference path).
    """
    del block_size_x, block_size_y  # reserved for a future TileLang/Triton kernel

    if qweight.ndim != 2 or scales.ndim != 2 or zeros.ndim != 2:
        raise ValueError("awq_dequantize_triton: qweight, scales, zeros must be 2-D")
    k, packed_m = qweight.shape
    m = packed_m * 8
    num_groups = scales.shape[0]
    group_size = _infer_group_size(k, num_groups)

    if k <= 0 or m <= 0:
        raise ValueError(f"awq_dequantize_triton: invalid shape K={k}, M={m}")
    if scales.shape[1] != m:
        raise ValueError(
            f"awq_dequantize_triton: scales.shape[1]={scales.shape[1]} != M={m}"
        )
    if zeros.shape != (num_groups, packed_m):
        raise ValueError(
            f"awq_dequantize_triton: zeros shape {tuple(zeros.shape)} expected "
            f"({num_groups}, {packed_m})"
        )
    _validate_group_size(group_size, k)

    weights = unpack_awq_int4(qweight).to(scales.dtype)
    z_unpacked = unpack_awq_int4(zeros).to(scales.dtype)
    group_idx = torch.arange(k, device=qweight.device) // group_size
    z_rows = z_unpacked[group_idx]
    s_rows = scales[group_idx]
    return (weights - z_rows) * s_rows


def awq_gemm_triton(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    split_k_iters: int,
    block_size_m: int = 32,
    block_size_n: int = 32,
    block_size_k: int = 32,
) -> torch.Tensor:
    """AWQ GEMM: ``input @ dequant(qweight)`` with shape ``[M, K] @ [K, N]``."""
    del block_size_m, block_size_n, block_size_k  # reserved for fused kernel

    if input.ndim != 2:
        raise ValueError("awq_gemm_triton: input must be 2-D [M, K]")
    m, k = input.shape
    if qweight.shape[0] != k:
        raise ValueError(
            f"awq_gemm_triton: qweight rows {qweight.shape[0]} != K={k}"
        )
    n = qweight.shape[1] * 8
    num_groups = qzeros.shape[0]
    group_size = _infer_group_size(k, num_groups)

    if n <= 0 or m <= 0:
        raise ValueError(f"awq_gemm_triton: invalid M={m}, N={n}, K={k}")
    if qweight.shape[1] != n // 8:
        raise ValueError("awq_gemm_triton: qweight column count mismatch")
    if qzeros.shape[1] != n // 8:
        raise ValueError("awq_gemm_triton: qzeros column count mismatch")
    if scales.shape != (num_groups, n):
        raise ValueError(
            f"awq_gemm_triton: scales shape {tuple(scales.shape)} expected "
            f"({num_groups}, {n})"
        )
    if split_k_iters <= 0 or (split_k_iters & (split_k_iters - 1)) != 0:
        raise ValueError("awq_gemm_triton: split_k_iters must be a positive power of 2")
    if split_k_iters > 32:
        raise ValueError("awq_gemm_triton: split_k_iters must be <= 32")
    _validate_group_size(group_size, k)

    weight = awq_dequantize_triton(qweight, scales, qzeros)
    out_dtype = scales.dtype
    return (input.to(out_dtype) @ weight.to(out_dtype)).to(out_dtype)


# vLLM names kept as primary entry points.
awq_dequantize = awq_dequantize_triton
awq_gemm = awq_gemm_triton


__all__ = [
    "AWQ_TRITON_SUPPORTED_GROUP_SIZES",
    "REVERSE_AWQ_ORDER",
    "awq_dequantize",
    "awq_dequantize_triton",
    "awq_gemm",
    "awq_gemm_triton",
    "pack_awq_int4",
    "unpack_awq_int4",
]
