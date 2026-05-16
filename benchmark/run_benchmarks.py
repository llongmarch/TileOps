"""Run all three operator-category benchmarks and produce a combined report.

Usage::

    python benchmark/run_benchmarks.py
    python benchmark/run_benchmarks.py --shape 2048 2048 --repeat 30
    python benchmark/run_benchmarks.py --plot-savedir results/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tileops.runtime import setup_metal_workarounds

setup_metal_workarounds()

from benchmark.common import (
    ALL_ACTIVATION_OPS,
    ALL_BINARY_OPS,
    ALL_UNARY_OPS,
)

_SECTIONS: list[tuple[str, str, list[str]]] = [
    ("activation", "benchmark/activation.py", ALL_ACTIVATION_OPS),
    ("binary", "benchmark/binary.py", ALL_BINARY_OPS),
    ("unary", "benchmark/unary.py", ALL_UNARY_OPS),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all TileOps benchmarks (activation, binary, unary)",
    )
    parser.add_argument("--shape", type=int, nargs="+", default=[4096, 4096])
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--execution-backend", type=str, default=None)
    parser.add_argument("--plot", action="store_true",
                        help="Show individual category visualizations")
    parser.add_argument("--plot-savedir", type=str, default=None,
                        help="Save all category plots into the given directory")
    parser.add_argument("--autotune", action="store_true",
                        help="Grid search (block_size, threads) per op")
    parser.add_argument("--autotune-warmup", type=int, default=3)
    parser.add_argument("--autotune-repeat", type=int, default=10)
    parser.add_argument("--autotune-no-cache", action="store_true")
    parser.add_argument("--autotune-search-space", type=str, default=None)
    args = parser.parse_args()

    plot_save_dir: Path | None = None
    if args.plot_savedir:
        plot_save_dir = Path(args.plot_savedir)
        plot_save_dir.mkdir(parents=True, exist_ok=True)

    for category, script_path, ops in _SECTIONS:
        print(f"\n{'='*60}")
        print(f"  {category.upper()}  ({len(ops)} ops)")
        print(f"{'='*60}")

        shape_args = [str(s) for s in args.shape]
        cmd = [
            sys.executable, script_path,
            "--shape", *shape_args,
            "--warmup", str(args.warmup),
            "--repeat", str(args.repeat),
        ]
        if args.plot:
            cmd.append("--plot")
        if plot_save_dir:
            cmd.extend(["--plot-save", str(plot_save_dir / f"{category}.png")])
        if args.target:
            cmd.extend(["--target", args.target])
        if args.execution_backend:
            cmd.extend(["--execution-backend", args.execution_backend])
        if args.block_size is not None:
            cmd.extend(["--block-size", str(args.block_size)])
        if args.threads is not None:
            cmd.extend(["--threads", str(args.threads)])
        if args.autotune:
            cmd.append("--autotune")
        if args.autotune_warmup != 3:
            cmd.extend(["--autotune-warmup", str(args.autotune_warmup)])
        if args.autotune_repeat != 10:
            cmd.extend(["--autotune-repeat", str(args.autotune_repeat)])
        if args.autotune_no_cache:
            cmd.append("--autotune-no-cache")
        if args.autotune_search_space:
            cmd.extend(["--autotune-search-space", args.autotune_search_space])

        subprocess.run(cmd)


if __name__ == "__main__":
    main()
