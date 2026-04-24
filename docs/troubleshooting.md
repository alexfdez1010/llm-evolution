# Troubleshooting

## `uv sync` fails with dependency conflicts

```bash
uv lock --upgrade-package problematic-package
uv sync
```

Check `pyproject.toml` for conflicting version pins.

## Tests fail with import errors

```bash
uv sync
# or force editable install
uv pip install -e .
```

## Ruff formatting conflicts

```bash
# inspect [tool.ruff] in pyproject.toml
uv run ruff format
```

## Resources

- uv: https://docs.astral.sh/uv/
- pytest: https://docs.pytest.org/
- Ruff: https://docs.astral.sh/ruff/
- Python Packaging: https://packaging.python.org/
