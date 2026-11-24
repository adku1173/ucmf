"""Shared test setup and fixtures."""

# Acoular must be loaded before NumPy (imported via ucmf/torch).
import acoular  # noqa: F401

from helpers import Recorder

import pytest


@pytest.fixture
def recorder():
    """Callback recording begin/end events and the end-callback keywords."""
    return Recorder()
