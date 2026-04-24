# uv Workflow

## Initial Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync                  # install runtime deps
uv sync --extra dev      # install dev tooling (pytest, ruff)
```

## Common Commands

| Command | Purpose |
|---------|---------|
| `uv sync` | Install/sync deps from lockfile to `.venv` |
| `uv add <package>` | Add runtime dependency |
| `uv add --dev <package>` | Add dev dependency |
| `uv remove <package>` | Remove dependency |
| `uv run <command>` | Run command in project env (auto-syncs) |
| `uv lock` | Update lockfile without installing |
| `uv lock --upgrade-package <pkg>` | Upgrade one package |
| `uv lock --upgrade` | Upgrade all (careful) |
| `uv python install 3.12` | Install Python version |

## Running Code

```bash
uv run python -c "import llm_evolution; print(llm_evolution.__version__)"
```

`uv run` auto-syncs env. No manual venv activation.

## Adding Dependencies

```bash
uv add requests
uv add "pandas>=2.0.0"
uv add --dev pytest
uv add -r requirements.txt
```

## Updating

```bash
uv lock --upgrade-package requests
uv sync
```

## Debugging

```bash
uv run python -m pdb -c continue -m llm_evolution
uv run pytest -s -v
uv run pytest tests/unit/test_file.py::test_function -s
```
