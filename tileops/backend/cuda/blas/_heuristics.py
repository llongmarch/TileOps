"""Tile-size heuristics shared across CUDA BLAS kernel variants."""

from __future__ import annotations


def gemm_config(m: int, n: int, k: int) -> dict:
    """Standard GEMM (f16/bf16/f32) tile sizes."""
    total = m * n
    if total <= 256 * 256 or k <= 256:
        return {"block_M": 32, "block_N": 32, "block_K": 16, "num_stages": 2, "threads": 128}
    if total <= 1024 * 1024:
        return {"block_M": 64, "block_N": 64, "block_K": 32, "num_stages": 2, "threads": 128}
    if total <= 4096 * 4096:
        return {"block_M": 128, "block_N": 128, "block_K": 32, "num_stages": 3, "threads": 128}
    return {"block_M": 128, "block_N": 256, "block_K": 32, "num_stages": 3, "threads": 256}


def gemv_config(m: int, k: int) -> dict:
    """Standard GEMV tile sizes."""
    if m <= 1024 and k <= 1024:
        return {"block_M": 64, "block_K": 64, "num_stages": 2, "threads": 128}
    if m <= 4096 and k <= 4096:
        return {"block_M": 128, "block_K": 128, "num_stages": 2, "threads": 256}
    return {"block_M": 128, "block_K": 128, "num_stages": 3, "threads": 256}


def gemm_fp8_config(m: int, n: int, k: int) -> dict:
    """FP8 GEMM tile sizes (tuned for SM89+ Tensor Core)."""
    total = m * n
    if total <= 512 * 512:
        return {"block_M": 64, "block_N": 64, "block_K": 64, "num_stages": 2, "threads": 128, "k_pack": 1}
    if total <= 2048 * 2048:
        return {"block_M": 128, "block_N": 128, "block_K": 64, "num_stages": 2, "threads": 256, "k_pack": 2}
    return {"block_M": 128, "block_N": 128, "block_K": 128, "num_stages": 3, "threads": 256, "k_pack": 2}


def gemm_int_config(m: int, n: int, k: int) -> dict:
    """INT4/INT8 GEMM tile sizes."""
    total = m * n
    if total <= 512 * 512:
        return {"block_M": 64, "block_N": 64, "block_K": 64, "num_stages": 2, "threads": 128}
    return {"block_M": 128, "block_N": 128, "block_K": 64, "num_stages": 3, "threads": 128}


def gemm_sm100_config(m: int, n: int, k: int) -> dict:
    """SM100 (Blackwell) GEMM tile sizes with TMA / tcgen05."""
    return {"block_M": 64, "block_N": 256, "block_K": 32, "num_stages": 2, "threads": 256}
