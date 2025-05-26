"""
Unit tests for ReAgent components.

This package contains unit tests that test individual components
in isolation with mocked dependencies.
"""

# Unit test configuration
UNIT_TEST_TIMEOUT = 5  # seconds

# Common unit test utilities
def assert_valid_json_response(response: str) -> dict:
    """Assert that a response is valid JSON and return parsed content."""
    import json
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        raise AssertionError(f"Invalid JSON response: {e}\nResponse: {response}")

def assert_message_format(message: dict, required_fields: list):
    """Assert that a message contains required fields."""
    missing_fields = [field for field in required_fields if field not in message]
    if missing_fields:
        raise AssertionError(f"Missing required fields: {missing_fields}")
