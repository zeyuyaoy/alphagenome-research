"""Shared fixtures for isolated offline tests."""

import pytest


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch, tmp_path_factory):
    """Redirects the cache and clears cache and results overrides for each test.

    Args:
        monkeypatch: Fixture that restores environment variables after the test.
        tmp_path_factory: Factory for temporary cache directories.
    """
    monkeypatch.delenv("AG2KIT_RESULTS_ROOT", raising=False)
    monkeypatch.delenv("AG2KIT_AG2_CACHE", raising=False)
    monkeypatch.setenv("AG2KIT_CACHE", str(tmp_path_factory.mktemp("cache")))
