"""Wiring. Each function builds one object; FastAPI injects them into routes.

Everything is created once and reused (lru_cache). Tests replace these
functions with fakes through app.dependency_overrides.
"""

from functools import lru_cache

from agrivibe.config import Settings


@lru_cache
def get_settings() -> Settings:
    """Returns the application settings, loaded once."""
    return Settings()
