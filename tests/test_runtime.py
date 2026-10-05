import importlib
import sys


def _clear_matplotlib_modules() -> None:
    for name in list(sys.modules):
        if name == "matplotlib" or name.startswith("matplotlib."):
            sys.modules.pop(name, None)


def test_configure_headless_matplotlib_uses_agg_without_display(monkeypatch, tmp_path):
    """Catches selecting an interactive backend on a headless batch worker."""
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("MPLBACKEND", raising=False)
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    _clear_matplotlib_modules()

    from hldbea.runtime import configure_headless_matplotlib

    assert "agg" in configure_headless_matplotlib().lower()


def test_legacy_stats_import_is_headless(monkeypatch, tmp_path):
    """Catches the legacy statistics module forcing an unavailable Qt backend."""
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("MPLBACKEND", raising=False)
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    _clear_matplotlib_modules()
    sys.modules.pop("ibea_stats", None)

    stats = importlib.import_module("ibea_stats")

    assert "agg" in stats.matplotlib.get_backend().lower()
