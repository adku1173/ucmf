"""Shared test setup and fixtures."""

# Acoular must be loaded before NumPy (imported via ucmf/torch).
import acoular  # noqa: F401

import pytest
from helpers import Recorder


@pytest.fixture
def recorder():
    """Callback recording begin/end events and the end-callback keywords."""
    return Recorder()
