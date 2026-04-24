# Code Style

## Typing

- Built-in generics: `list[str]`, `dict[str, int]` (not `List`, `Dict`)
- `| None` instead of `Optional`
- `typing.Protocol` for structural typing
- Type hints for all new functions

```python
from typing import Literal

def process_data(
    data: list[dict],
    mode: Literal["fast", "accurate"] = "fast"
) -> dict[str, int]:
    return {"count": len(data)}
```

## SOLID (CRITICAL)

- **SRP** — one reason to change per class/function
- **OCP** — extend via protocols/ABCs, don't modify
- **LSP** — subtypes substitutable for base
- **ISP** — many small protocols > one fat
- **DIP** — depend on protocols, not concrete classes

```python
# Good DIP
class Service:
    def __init__(self, repository: DataRepository):
        self.repository = repository

# Bad
class Service:
    def __init__(self):
        self.repository = PostgreSQLRepository()
```

## File Length (MANDATORY)

Keep Python files **< 150 lines**. If exceeded:

1. Split into focused modules
2. Extract classes/functions to separate files
3. Group related functionality into subpackages

```
# Before: large_module.py (200+ lines)
# After:
large_module/
├── core.py
├── validators.py
├── processors.py
└── __init__.py  # exports public API
```

## Patterns

### Protocol-based DI

```python
from typing import Protocol

class DataProvider(Protocol):
    def fetch_data(self, query: str) -> dict: ...

class MyService:
    def __init__(self, provider: DataProvider):
        self.provider = provider
```

### Dataclasses

```python
from dataclasses import dataclass

@dataclass
class Result:
    success: bool
    data: dict
    error_message: str | None = None

@dataclass(frozen=True)  # immutable config
class Config:
    api_key: str
    timeout: int
```

### Guard Clauses

```python
def process_order(order: Order) -> None:
    if not order.is_valid():
        raise ValueError("Invalid order")
    if order.is_cancelled():
        return
    if order.is_completed():
        return
    process_payment(order)
```

### Context Managers

```python
from contextlib import contextmanager

@contextmanager
def database_connection():
    conn = create_connection()
    try:
        yield conn
    finally:
        conn.close()
```

## Best Practices

- **DRY** — extract repeated code
- **YAGNI** — don't add until needed
- **Composition over inheritance**
- **Immutability when possible** (`@dataclass(frozen=True)`)
- **Explicit over implicit** — no hidden side effects
- **Fail fast** — raise early on invalid input
- **No magic numbers** — named constants (`MAX_RETRIES = 3`)

## Error Handling

```python
try:
    result = risky_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
    raise
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise RuntimeError("Operation failed") from e
```

## Docstrings

Always on modules, classes, functions. For non-trivial functions, add inline comments for:
- Invariants/assumptions
- Non-obvious algorithm steps
- Units / coordinate systems / domain constraints

```python
def calculate_score(correct: int, total: int, weight: float = 1.0) -> float:
    """
    Calculate weighted score.

    Args:
        correct: Number of correct answers.
        total: Total number of questions.
        weight: Score weight multiplier.

    Returns:
        Weighted score between 0 and 1.

    Raises:
        ValueError: If total is zero or negative.
    """
    if total <= 0:
        raise ValueError("Total must be positive")
    return (correct / total) * weight
```
