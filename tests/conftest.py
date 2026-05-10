"""Shared pytest configuration for the TileOps test suite."""

from tileops.runtime import setup_metal_workarounds

# Must run before any TileLang kernel compilation.
setup_metal_workarounds()
