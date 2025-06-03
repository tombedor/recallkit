import pytest

@pytest.fixture
def completion_model():
    """
    Fixture that provides the default completion model to use for tests.

    Returns:
        str: The model identifier to use for completions.
    """
    return "openai/gpt-4o-mini"
