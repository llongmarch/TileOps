# TileOps

**TileOps** reimplements deep-learning operators using
[TileLang](https://github.com/tile-ai/tilelang) — a Python DSL that
lowers to high-performance GPU / CPU kernels via TVM-style compilation.
The goal is to mirror PyTorch's operator surface while fully owning the
kernel schedule and compilation path.

## Goals

- **Correct, comparable** TileLang kernels for common operator families
  (pointwise, reductions, BLAS-like, fusions, …).
- A thin **runtime layer** (target / device / compilation helpers) so the
  same entry-points run on **CUDA**, **Apple Metal (MPS)**, **CPU (LLVM)**,
  and other TileLang-supported backends.
- **Examples** and **benchmarks** side-by-side with the code for regression
  checks and performance baselines against PyTorch.

Operator scope follows the living taxonomy in [`docs/ops.md`](docs/ops.md).

## Repository layout

| Path | Purpose |
|------|---------|
| [`tileops/`](tileops/) | Python package — kernels + host utilities |
| [`tileops/runtime.py`](tileops/runtime.py) | Target selection, device mapping, `compile_prim`, `invoke_kernel` (N-D ↔ 1-D reshape), dtype helpers, tile-size heuristic, benchmarking utilities |
| [`tileops/pointwise.py`](tileops/pointwise.py) | Generic GMEM-direct pointwise kernels for **N-D tensors**: `add`, `sub`, `mul`, `div`, `pow` — with automatic backend dispatch |
| [`tileops/cuda/`](tileops/cuda/) | CUDA / HIP-specific optimised kernels (e.g. shared-memory tiled `add`) |
| [`tileops/metal/`](tileops/metal/) | Metal-specific optimised kernels (placeholder for future simdgroup ops) |
| [`tileops/__init__.py`](tileops/__init__.py) | Public re-exports (`from tileops import …`) |
| [`examples/`](examples/) | Runnable correctness demos |
| [`benchmark/`](benchmark/) | Performance comparisons (TileLang vs PyTorch) |
| [`tests/`](tests/) | pytest-based regression tests |
| [`scripts/benchmark.sh`](scripts/benchmark.sh) | One-shot runner for every `benchmark/*.py` |

## Requirements

- **Python 3.9+**
- **TileLang** and **PyTorch** in the same environment
- A supported backend: **NVIDIA GPU + CUDA**, **Apple Silicon + Metal / MPS**, or **CPU**

## Quick start

```bash
# Install in development mode
pip install -e ".[dev]"

# Correctness smoke test (auto-detects target & device)
python examples/pointwise.py --shape 512 512

# N-D tensor support
python examples/pointwise.py --shape 8 3 224 224

# Run all benchmarks
./scripts/benchmark.sh
./scripts/benchmark.sh --shape 4096 4096 --repeat 50

# Run tests
pytest tests/ -v
```

Import the public API from the package root:

```python
from tileops import (
    pointwise_add,
    default_tilelang_target,
    suggest_pointwise_config,
    invoke_kernel,
)

# Compile a kernel for any N-D shape
kernel = pointwise_add(shape=(8, 3, 224, 224))
# invoke_kernel handles flatten/reshape transparently
out = invoke_kernel(kernel, x, y, tilelang_target=...)
```

## License

See [`LICENSE`](LICENSE).
