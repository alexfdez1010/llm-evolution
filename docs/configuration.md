# Configuration

## pyproject.toml

```toml
[project]
name = "package-name"
version = "0.1.0"
description = "Package description"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "package1>=1.0.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/llm_evolution"]

[tool.pytest.ini_options]
addopts = "-v --capture=no"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W"]
ignore = []
```

## Environment Variables

Use `.env` (in `.gitignore`):

```bash
API_KEY=your-secret-key
DATABASE_URL=postgresql://localhost/db
```

Load via `python-dotenv`:

```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("API_KEY")
```

## .gitignore

```gitignore
# venv
.venv
venv/
ENV/

# cache
__pycache__/
*.py[cod]
*$py.class
*.so

# testing
.pytest_cache/
.coverage
htmlcov/

# IDE
.vscode/
.idea/
*.swp

# env
.env
.env.local

# build
dist/
build/
*.egg-info/

# uv / ruff
.ruff_cache/
```

## Module Layout

```
src/package_name/
├── __init__.py   # package init + public API
├── main.py       # CLI entry
├── core.py       # core logic
├── models.py     # dataclasses, protocols
├── api.py        # external integrations
├── utils.py      # utilities
└── config.py     # config management
```
