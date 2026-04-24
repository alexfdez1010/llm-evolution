# AGENT INSTRUCTIONS

Guidance for AI coding assistants on this Python **library** project. Concise essentials live here; details link into `docs/`.

## Stack

- **Language**: Python 3.12+
- **Package manager**: [uv](https://docs.astral.sh/uv/)
- **Build**: Hatchling
- **Test**: pytest
- **Lint/format**: Ruff
- **Types**: basedpyright (optional)

## Repo Layout

```
.
├── src/llm_evolution/           # package source
│   ├── ai/
│   │   ├── interfaces/          # LLM / embedding protocols
│   │   └── implementations/     # concrete AI impls
│   ├── algorithm/               # evolutionary algorithm
│   └── interfaces/              # mutation, crossover, etc.
├── tests/
│   ├── unit/                    # mocked, fast
│   └── integration/             # real services
├── docs/                        # extended guides (see below)
├── pyproject.toml
├── uv.lock                      # DO NOT edit manually
└── README.md
```

## Quick Commands

```bash
uv sync --extra dev              # install deps + dev tooling
uv run pytest                    # all tests
uv run pytest tests/unit         # unit only
uv run pytest tests/integration  # integration only
uv run ruff format               # format
uv run ruff check                # lint
```

Always prefix with `uv run`. Never activate venv manually.

## Core Rules

- **Modern typing**: built-in generics (`list[str]`, `dict[str, int]`), `| None` not `Optional`, `typing.Protocol` for interfaces.
- **SOLID**: follow strictly. Details → `docs/code-style.md`.
- **File length**: keep Python files **< 150 lines**. Refactor if exceeded.
- **Protocols** for dependency injection; **dataclasses** for data.
- **Docstrings** on every module/class/function. Public APIs stay stable.
- **Mock** externals in unit tests. **Never** mock in integration tests unless explicitly asked.
- **No magic numbers**, guard clauses, fail fast, composition over inheritance.

## Extended Docs (progressive disclosure)

Read when relevant to the task:

- **[docs/uv-workflow.md](docs/uv-workflow.md)** — uv setup, commands, dep management, debugging
- **[docs/testing.md](docs/testing.md)** — unit/integration split, fixtures, parametrize, patterns
- **[docs/code-style.md](docs/code-style.md)** — SOLID deep-dive, typing, patterns, error handling, docstrings
- **[docs/configuration.md](docs/configuration.md)** — pyproject.toml, env vars, .gitignore, module layout
- **[docs/troubleshooting.md](docs/troubleshooting.md)** — common issues, external resources

## Workflow: Adding a Feature

1. `uv add <package>` if new dep needed
2. Implement in `src/llm_evolution/`
3. Unit tests in `tests/unit/test_<feature>.py`
4. Integration tests in `tests/integration/` if applicable
5. `uv run pytest` — **must be 100% pass**
6. `uv run ruff format && uv run ruff check`

## Completion Checklist

Before finishing any task:

- [ ] **All** tests pass: `uv run pytest`
- [ ] Unit tests pass: `uv run pytest tests/unit`
- [ ] Integration tests pass (if applicable): `uv run pytest tests/integration`
- [ ] Formatted: `uv run ruff format`
- [ ] Lint clean: `uv run ruff check`
- [ ] Type hints on new functions
- [ ] Docstrings on public APIs
- [ ] Edge cases + error handling tested

## Hard Requirements

1. `uv run` for all Python commands
2. **100% test pass rate** before finishing
3. Format + lint before commit
4. Update tests when behavior changes
5. Follow existing patterns — read neighboring files first
6. Protocols for interfaces, dataclasses for data
