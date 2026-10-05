"""Runtime configuration shared by CLI and batch experiment entry points."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile


def configure_headless_matplotlib() -> str:
    """Select a safe Matplotlib backend and return its resolved name.

    An explicit ``MPLBACKEND`` remains authoritative.  On workers without a
    display, the non-interactive Agg backend is selected before pyplot is
    imported.
    """

    cache_root = Path(tempfile.gettempdir()) / "hldbea-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    matplotlib_cache = cache_root / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

    import matplotlib

    requested = os.environ.get("MPLBACKEND")
    if requested:
        matplotlib.use(requested, force=True)
    elif not os.environ.get("DISPLAY"):
        matplotlib.use("Agg", force=True)
    return str(matplotlib.get_backend())
