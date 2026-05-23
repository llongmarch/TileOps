"""CUDA BLAS dispatch — routes to the right kernel builder by dtype and SM arch.

===================================== ==========================================
Path                                  Trigger
===================================== ==========================================
:mod:`.gemm`                          f16 / bf16 / f32 GEMM
:mod:`.gemv`                          f16 / bf16 / f32 GEMV
:mod:`.gemm_fp8`                      float8_e4m3fn / float8_e5m2 GEMM
:mod:`.gemm_int`                      int4 / int8 GEMM
:mod:`.gemm_sm100`                    SM ≥ 100 (Blackwell) — tcgen05 + TMA
===================================== ==========================================

Dispatch is dtype-aware and falls back to the standard :mod:`.gemm` path for
any unrecognised dtype.
"""

from __future__ import annotations

from typing import Any, Optional

import tilelang.language as T

from tileops.runtime import make_generic_cached_compiler

# ── Dtype categories ─────────────────────────────────────────────────────

try:
    _FP8_DTYPES: frozenset = frozenset({T.float8_e4m3fn, T.float8_e5m2})
except AttributeError:
    _FP8_DTYPES = frozenset()

try:
    _INT_DTYPES: frozenset = frozenset({T.int4, T.int8})
except AttributeError:
    _INT_DTYPES = frozenset()


def _is_sm100(target: Optional[str]) -> bool:
    """Detect Blackwell (SM 10.0+) from target string or device query."""
    if target:
        t = target.lower()
        if "sm_10" in t or "sm100" in t or "blackwell" in t:
            return True
        if " -arch=sm_" in t:
            try:
                arch_str = t.split("-arch=sm_")[1].split()[0]
                if int(arch_str) >= 100:
                    return True
            except (ValueError, IndexError):
                pass
    try:
        import torch
        if torch.cuda.is_available():
            major, _minor = torch.cuda.get_device_capability()
            return major >= 10
    except Exception:
        pass
    return False


# ── Compilation cache ────────────────────────────────────────────────────

_compile = make_generic_cached_compiler(cache_prefix="cuda_blas_v3")


# ── GEMM dispatch ────────────────────────────────────────────────────────

def gemm(
    m: int,
    n: int,
    k: int,
    *,
    dtype: Any,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> Any:
    """Compile and return a GEMM kernel, routing by dtype and SM architecture."""

    # ── SM100 (Blackwell) ──────────────────────────────────────────────
    if _is_sm100(target):
        from tileops.backend.cuda.blas.gemm_sm100 import gemm_sm100_kernel
        from tileops.backend.cuda.blas._heuristics import gemm_sm100_config

        cfg = gemm_sm100_config(m, n, k)
        accum_dtype = T.float32 if dtype in _FP8_DTYPES else T.float16
        return _compile(
            ("gemm_sm100", m, n, k, dtype, cfg["block_M"], cfg["block_N"], cfg["block_K"], cfg["num_stages"], cfg["threads"]),
            gemm_sm100_kernel,
            m, n, k,
            cfg["block_M"], cfg["block_N"], cfg["block_K"], cfg["num_stages"], cfg["threads"],
            dtype, dtype, accum_dtype,
            target=target,
            execution_backend=execution_backend,
        )

    # ── FP8 ────────────────────────────────────────────────────────────
    if dtype in _FP8_DTYPES:
        from tileops.backend.cuda.blas.gemm_fp8 import gemm_fp8_kernel
        from tileops.backend.cuda.blas._heuristics import gemm_fp8_config

        cfg = gemm_fp8_config(m, n, k)
        return _compile(
            ("gemm_fp8", m, n, k, dtype, cfg["block_M"], cfg["block_N"], cfg["block_K"], cfg["num_stages"], cfg["threads"], cfg["k_pack"]),
            gemm_fp8_kernel,
            m, n, k,
            cfg["block_M"], cfg["block_N"], cfg["block_K"], cfg["num_stages"], cfg["threads"], cfg["k_pack"], dtype,
            target=target,
            execution_backend=execution_backend,
        )

    # ── INT4 / INT8 ────────────────────────────────────────────────────
    if dtype in _INT_DTYPES:
        from tileops.backend.cuda.blas.gemm_int import gemm_int_kernel
        from tileops.backend.cuda.blas._heuristics import gemm_int_config

        cfg = gemm_int_config(m, n, k)
        accum_dtype = T.int32
        return _compile(
            ("gemm_int", m, n, k, dtype, cfg["block_M"], cfg["block_N"], cfg["block_K"], cfg["num_stages"], cfg["threads"]),
            gemm_int_kernel,
            m, n, k,
            cfg["block_M"], cfg["block_N"], cfg["block_K"], cfg["num_stages"], cfg["threads"], dtype, accum_dtype,
            target=target,
            execution_backend=execution_backend,
        )

    # ── Standard (f16 / bf16 / f32) ────────────────────────────────────
    from tileops.backend.cuda.blas.gemm import gemm_kernel
    from tileops.backend.cuda.blas._heuristics import gemm_config

    cfg = gemm_config(m, n, k)
    return _compile(
        ("gemm", m, n, k, dtype, cfg["block_M"], cfg["block_N"], cfg["block_K"], cfg["num_stages"], cfg["threads"]),
        gemm_kernel,
        m, n, k,
        cfg["block_M"], cfg["block_N"], cfg["block_K"], cfg["num_stages"], cfg["threads"], dtype,
        target=target,
        execution_backend=execution_backend,
    )


# ── GEMV dispatch ────────────────────────────────────────────────────────

def gemv(
    m: int,
    k: int,
    *,
    dtype: Any,
    target: Optional[str] = None,
    execution_backend: Optional[str] = None,
) -> Any:
    """Compile and return a GEMV kernel (block-reduction path)."""

    from tileops.backend.cuda.blas.gemv import gemv_kernel
    from tileops.backend.cuda.blas._heuristics import gemv_config

    cfg = gemv_config(m, k)
    return _compile(
        ("gemv", m, k, dtype, cfg["block_M"], cfg["block_K"], cfg["num_stages"], cfg["threads"]),
        gemv_kernel,
        m, k,
        cfg["block_M"], cfg["block_K"], cfg["num_stages"], cfg["threads"], dtype,
        target=target,
        execution_backend=execution_backend,
    )


__all__ = ["gemm", "gemv"]
