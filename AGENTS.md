AGENTS

This file documents developer-facing commands and code-style rules for agentic coding agents working in this repository.
Keep changes conservative: prefer small, well-tested edits and follow the repository pre-commit hooks.

1) Quick environment
- Python: >= 3.10 (project targets 3.10+). See `setup.py` and `pyproject.toml` for details.
- Source packages: `slime`, `slime_plugins`.

2) Build / install commands
- Install editable for local development:

```bash
python -m pip install -e .
```

- Build a wheel (custom wheel tag handling lives in `setup.py`):

```bash
python setup.py bdist_wheel
```

- Create conda environment (helper script):

```bash
./build_conda.sh
```

3) Lint / format / pre-commit (local & CI)
- Run pre-commit hooks (recommended before committing):

```bash
pre-commit install
pre-commit run --all-files --show-diff-on-failure --color=always
```

- Individual tools (pre-commit configured):
  - Black: `black .` (project-configured line-length 119)
  - isort: `isort .` (profile=black, known_first_party includes `slime`, `slime_plugins`)
  - Ruff: `ruff check . --fix` (configured via pyproject)
  - Autoflake: `autoflake --remove-all-unused-imports --in-place <files>`

4) Tests
- Run full test suite (repo default pytest options are set in `pyproject.toml`):

```bash
python -m pytest
```

- Run a single test file:

```bash
python -m pytest tests/path/to/test_file.py
```

- Run a single test function in a file:

```bash
python -m pytest tests/path/to/test_file.py::test_function_name
```

- Run tests by keyword (substring match):

```bash
python -m pytest -k "substring"   # runs tests whose names or node ids match
```

- Run tests by marker (markers are defined in `pyproject.toml`):

```bash
python -m pytest -m unit        # run tests marked as unit
python -m pytest -m "not integration"   # deselect a marker
```

- Notes for plugin-contract tests: some plugin contract tests use a small helper that manipulates env vars and may call pytest on a single file. See `tests/plugin_contracts/_shared.py` for usage hints — set the expected environment variables and call pytest on the file if replicating CI behaviour.

5) Running quickly in CI or debug
- Use `-q` for quieter output, `-k` for filtering, `-x` to stop on first failure, `-q -k name -x` is useful for iterating on a failing single test.

Example: iterate on one failing test function

```bash
python -m pytest -q tests/path/to/test_file.py::test_function_name -x
```

6) Code style and conventions (enforced or expected)

- Formatting
  - Use Black with line length 119. Do not change Black rules locally.
  - isort aligns with Black (`profile = "black"`) and places imports in the order: FUTURE, STDLIB, THIRDPARTY, FIRSTPARTY, LOCALFOLDER.
  - Ruff is configured to catch pyflakes/pycodestyle/bugbear issues and can auto-fix many problems via pre-commit.

- Imports
  - Group imports in the order enforced by isort (see above).
  - Known first-party modules: `slime`, `slime_plugins` (configured in `pyproject.toml`).
  - Keep imports at top of module unless there is a documented, tested reason for local import (e.g., optional dependency, avoid circular import, lazy import for heavy deps).
  - Avoid wildcard imports (`from foo import *`).

- Typing and annotations
  - Target Python 3.10+: prefer modern typing (PEP 585 types like `list[int]` when appropriate) but keep compatibility with codebase patterns.
  - Use type hints for public functions/methods and complex internal functions. Small local helpers may omit them but prefer adding types when they clarify behavior.
  - Prefer `typing` constructs when needed (e.g., `Protocol`, `TypedDict`, `Generic`) for public APIs.

- Naming
  - Functions and variables: snake_case
  - Classes: PascalCase
  - Constants: UPPER_SNAKE_CASE
  - Private/internal symbols: single underscore prefix `_internal` for module-private names; double-underscore only for name-mangling when needed.

- Docstrings & comments
  - Public modules, classes and functions should have concise docstrings describing purpose, important args and return values.
  - For short internal helpers, a one-line docstring or a brief inline comment is fine.
  - Avoid redundant comments that restate obvious code; prefer comments that explain non-obvious rationale or constraints.

- Error handling and logging
  - Do not use bare `except:` or `except Exception:` without re-raising or explicitly handling the error. Prefer specific exception classes.
  - Use `raise` (re-raise) when augmenting an exception to keep original traceback, or `raise from` to attach context.
  - For expected runtime errors, raise appropriate built-in exceptions (ValueError, TypeError, RuntimeError) or define clear, documented custom exceptions in a `exceptions.py` module when appropriate.
  - Prefer structured logging (`logging` module) over `print()` for library code. Use `logging.getLogger(__name__)` in each module.

- Resource management
  - Use context managers (`with`) for files, locks, streams, and other resources that need deterministic closing.

- Concurrency and distributed code
  - The project contains distributed training/rollout code — prefer explicit synchronization and avoid assumptions about shared global state.
  - Add clear comments whenever code relies on a particular process ordering, environment variables, or non-obvious side-effects.

- Tests
  - Tests use pytest and rely on markers declared in `pyproject.toml` (unit, integration, system, acceptance, docs, etc.).
  - Tests should be deterministic when possible. If a test must be non-deterministic, document the source of nondeterminism and mark appropriately.
  - Use fixtures for common setup/teardown. Keep tests small and focused.

- Performance-sensitive code
  - When optimizing for throughput or memory, add clear profiling notes and a short benchmark or unit test that verifies the optimization does not regress correctness.

7) Repo-specific tips & config references
- pyproject.toml: black, isort, ruff and pytest config live here. Respect configured line-length and isort profile.
- .pre-commit-config.yaml: autotools for ruff, isort, black, autoflake. Run pre-commit locally before committing.
- setup.py: packaging configuration and custom wheel tag behaviour.
- tests/plugin_contracts/_shared.py: helper for plugin contract tests — useful if you're adapting or running plugin tests manually.

8) Cursor / Copilot rules
- This repository contains no Cursor rules (no `.cursor` or `.cursorrules` found) and no explicit Copilot instruction file in `.github/`. If you add repository-level agent rules, ensure they live in either `.cursor/rules/` or `.github/copilot-instructions.md` and document them here.

9) When you are unsure — conservative defaults for agents
- Make minimal, well-tested changes; run local tests for files you modify.
- Run `pre-commit run --all-files` before suggesting a commit.
- Prefer creating small, well-scoped PRs with a short explanation of why the change is required and which tests were run locally.

10) Example quick workflow (one failing test iteration)

```bash
# run single failing test and iterate
python -m pytest -q tests/path/to/test_file.py::test_function_name -x
# fix code
pre-commit run --all-files --show-diff-on-failure
python -m pytest tests/path/to/test_file.py::test_function_name -q
```

Appendix: contact maintainers or open an issue if a change touches major infrastructure (packaging, CI matrix, major refactor) or requires secret/credentials.
