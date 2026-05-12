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
  same entry-points run on **CUDA** and **Apple Metal (MPS)** for the
  operators implemented here (`tileops.cuda` / `tileops.metal` only;
  TileLang `llvm` / CPU kernels are not shipped).
- **Examples** and **benchmarks** side-by-side with the code for regression
  checks and performance baselines against PyTorch.

Operator scope follows the living taxonomy in [`docs/ops.md`](docs/ops.md).

## Repository layout

| Path | Purpose |
|------|---------|
| [`tileops/`](tileops/) | Python package — kernels + host utilities |
| [`tileops/runtime.py`](tileops/runtime.py) | Target selection, `compile_prim`, `invoke_kernel` / `invoke_unary_kernel`, `suggest_tile_config`, `bench_ms` |
| [`tileops/_infra.py`](tileops/_infra.py) | Shared infrastructure: cached compiler, backend-op factory, facade dispatch |
| [`tileops/binary.py`](tileops/binary.py) | Binary ops facade: `add`, `sub`, `mul`, `div`, `pow` |
| [`tileops/activation.py`](tileops/activation.py) | Activation facade: `relu`, `gelu`, `silu`, `sigmoid`, `tanh` |
| [`tileops/cuda/binary.py`](tileops/cuda/binary.py) | CUDA / HIP binary GMEM kernels + `add_shared` |
| [`tileops/cuda/activation.py`](tileops/cuda/activation.py) | CUDA / HIP unary activations |
| [`tileops/metal/binary.py`](tileops/metal/binary.py) | Metal binary GMEM kernels (separate copy) |
| [`tileops/metal/activation.py`](tileops/metal/activation.py) | Metal unary activations (separate copy) |
| [`tileops/__init__.py`](tileops/__init__.py) | Public re-exports (`from tileops import …`) |
| [`examples/`](examples/) | Runnable correctness demos |
| [`benchmark/`](benchmark/) | Performance comparisons (TileLang vs PyTorch) |
| [`tests/`](tests/) | pytest-based regression tests |
| [`scripts/benchmark.sh`](scripts/benchmark.sh) | One-shot runner for every `benchmark/*.py` |

## Requirements

- **Python 3.9+**
- **TileLang** and **PyTorch** in the same environment
- A supported **GPU** backend: **NVIDIA + CUDA** or **Apple Silicon + Metal / MPS**.

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

Import and use — just like PyTorch:

```python
from tileops import add, relu

out = add(x, y)    # element-wise addition
act = relu(x)      # ReLU activation
```

## License

See [`LICENSE`](LICENSE).
