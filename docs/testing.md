# Testing

## Organization

- **Unit** (`tests/unit/`): fast, mocked, isolated. Run in ms.
- **Integration** (`tests/integration/`): real APIs/services. Don't mock unless user asks.

## Running

```bash
uv run pytest                          # all
uv run pytest tests/unit               # unit only
uv run pytest tests/integration        # integration only
uv run pytest tests/unit/test_x.py     # single file
uv run pytest -v                       # verbose
uv run pytest --cov=src/llm_evolution  # coverage
```

## Patterns

### Fixtures

```python
import pytest
from unittest.mock import Mock

@pytest.fixture
def mock_api_client():
    client = Mock()
    client.fetch_data.return_value = {"status": "success"}
    return client

def test_function_with_mock(mock_api_client):
    result = my_function(mock_api_client)
    assert result == expected_value
    mock_api_client.fetch_data.assert_called_once()
```

### Parametrize

```python
@pytest.mark.parametrize(
    "input_value, expected_output",
    [(1, 2), (5, 10), (0, 0)],
)
def test_function_with_params(input_value, expected_output):
    assert my_function(input_value) == expected_output
```

### Exceptions

```python
def test_invalid_input_raises():
    with pytest.raises(ValueError):
        my_function(invalid_input)
```

### Integration

```python
def test_api_integration():
    client = create_api_client()
    result = client.fetch_data()
    assert isinstance(result, dict)
    assert "status" in result
```

## Philosophy

- TDD when possible
- Mock externals in unit tests
- Test edge cases and errors
- Unit tests run in ms
- Integration verifies real-world
