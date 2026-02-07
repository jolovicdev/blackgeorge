# Development

This project uses pytest for tests, ruff for linting, and mypy for type checking.

## Development workflow

Typical local workflow:

1. Install dev dependencies.
2. Make code and docs changes.
3. Run ruff to catch lint and style issues early.
4. Run mypy to ensure type correctness under the strict config.
5. Run pytest to validate behavior.
6. Run MkDocs serve/build to verify documentation changes.

This order keeps fast feedback loops first (ruff and mypy), then tests, then docs.

MkDocs uses `docs/index.md` as the site home. The `docs/README.md` file is excluded to avoid a name conflict with the home page.

## Install dev dependencies

```
uv pip install -e .[dev]
```

## Run tests

```
uv run pytest
```

## Lint

```
uv run ruff check .
```

```
uv run ruff format .
```

## Docs site

```
uv run mkdocs serve
```

```
uv run mkdocs build
```

## Type check

```
uv run mypy src
```
