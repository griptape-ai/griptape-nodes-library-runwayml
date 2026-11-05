from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _api_key_is_configured():
    """Nodes check the key before anything else. Tests that cover a missing key patch it themselves."""
    with patch("runway_node.get_api_key", return_value="test-key"):
        yield
