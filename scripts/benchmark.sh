#!/usr/bin/env bash
# Run every Python benchmark under benchmark/.
#
# Usage:
#   ./scripts/benchmark.sh
#   ./scripts/benchmark.sh --m 2048 --n 2048 --repeat 30
#   ./scripts/benchmark.sh --auto-blocks
#
# Extra arguments are forwarded to each benchmark script.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Ensure the tileops package is importable even without `pip install -e .`.
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

shopt -s nullglob
scripts=(benchmark/*.py)
shopt -u nullglob

if [[ ${#scripts[@]} -eq 0 ]]; then
  echo "No benchmark/*.py found under ${ROOT}/benchmark" >&2
  exit 1
fi

for py in "${scripts[@]}"; do
  echo "========================================"
  echo "==> python3 ${py#./} $*"
  echo "========================================"
  python3 "$py" "$@"
  echo
done

echo "All benchmark scripts finished."
