# llm-evolution

A Python library combining LLMs and evolutionary algorithms to optimize programs and systems across multiple domains and target languages (e.g. CUDA, RISC-V, algorithmic trading strategies).

## 🚀 Features

- **Library-first layout**: `src/`-based packaging for reliable imports
- **Modern Python**: Python 3.12+
- **Testing ready**: pytest with unit + integration structure
- **Code quality**: Ruff (formatting + linting)
- **Type checking**: Optional basedpyright configuration

## 📋 Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

## 🛠️ Installation

### Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Clone and Setup

```bash
# Clone the repository
git clone https://github.com/alexfdez1010/llm-evolution
cd llm-evolution

# Sync dependencies (creates .venv and installs packages)
uv sync

# Install dev tools (pytest/ruff)
uv sync --extra dev
```

### Install the library locally

```bash
# Install in editable mode (recommended for development)
uv pip install -e .

# Or install with dev extras
uv pip install -e ".[dev]"
```

## 🎯 Usage

### Importing the library

```python
import llm_evolution

print(llm_evolution.__version__)
```

## 🧪 Testing

The repository separates:

- **Unit Tests** (`tests/unit/`): Fast, isolated tests using mocks
- **Integration Tests** (`tests/integration/`): Tests with real APIs/services

### Running Tests

```bash
# Ensure dev extras are installed
uv sync --extra dev

# All tests
uv run pytest

# Unit tests only
uv run pytest tests/unit

# Integration tests only
uv run pytest tests/integration

# With verbose output
uv run pytest -v

# With coverage
uv run pytest --cov=src/llm_evolution
```

## 🎨 Code Quality

```bash
# Ensure dev extras are installed
uv sync --extra dev

# Format
uv run ruff format

# Lint
uv run ruff check
```

## 📦 Dependency Management

### Adding Dependencies

```bash
# Add runtime dependency
uv add <package-name>

# Add development dependency
uv add --dev <package-name>

# Example: Add requests library
uv add requests

# Example: Add pytest plugin
uv add --dev pytest-cov
```

### Updating Dependencies

```bash
# Update a specific package
uv lock --upgrade-package <package-name>

# Update all packages
uv lock --upgrade

# Sync after updating
uv sync
```

### Removing Dependencies

```bash
uv remove <package-name>
```

## 🎨 Code Quality

### Formatting

```bash
# Auto-format all code
uv run ruff format

# Check formatting without changes
uv run ruff format --check
```

### Linting

```bash
# Run linter
uv run ruff check

# Auto-fix issues where possible
uv run ruff check --fix
```

## 📁 Project Structure

```
.
├── src/
│   └── llm_evolution/            # Main package source code
│       ├── __init__.py
│       └── version.py
├── tests/
│   ├── unit/                     # Unit tests with mocks
│   └── integration/              # Integration tests (real APIs/services)
├── .python-version               # Python version (3.12)
├── pyproject.toml                # Project metadata & dependencies
├── uv.lock                       # Locked dependencies (DO NOT edit manually)
├── .gitignore                    # Git ignore patterns
├── AGENTS.md                     # AI coding assistant guidelines
└── README.md                     # This file
```

## 🔧 Configuration

### pyproject.toml

The `pyproject.toml` file contains:
- Project metadata (name, version, description)
- Python version requirement (>=3.12)
- Dependencies list
- Build system configuration (Hatchling)
- Tool configurations (pytest, basedpyright)

### Environment Variables

For sensitive configuration, create a `.env` file (already in `.gitignore`):

```bash
# .env
API_KEY=your-secret-key
DATABASE_URL=postgresql://localhost/db
```

Load with `python-dotenv` (already included):

```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("API_KEY")
```

## 🚀 Development Workflow

1. **Make changes** to code in `src/llm_evolution/`
2. **Write tests** in `tests/unit/` or `tests/integration/`
3. **Run tests**: `uv run pytest`
4. **Format code**: `uv run ruff format`
5. **Lint code**: `uv run ruff check`
6. **Commit** your changes

## 👥 Authors

Alejandro Fernández Camello & Claude Sonnet 4.5

## 🙏 Acknowledgments

- Built with [uv](https://docs.astral.sh/uv/) for fast Python package management
- Code quality powered by [Ruff](https://github.com/astral-sh/ruff)
- Testing with [pytest](https://pytest.org/)
