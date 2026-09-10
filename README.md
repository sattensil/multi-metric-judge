# More Signal Per Call

## How Many Metrics Can One LLM Judge Score Before It Lies?

This project tests when scoring more metrics in one LLM call starts to make the
judge less reliable.

## Start here

- [`presentation/index.html`](presentation/index.html): the results
- [`METHODS.md`](METHODS.md): what was tested and how
- `src/packing/`: experiment and analysis code
- `tests/`: offline checks
- `replication/`: pinned results and release checksums
- `scripts/`: dataset and release utilities

## Verify the results

```bash
uv sync --frozen
uv run pytest -q
uv run packing reproduce --strict
```

These commands do not make model API calls.

Generated datasets, live runs, internal plans, technical notes, and presenter
support files are intentionally excluded from the public repository.

## Explore the CLI

```bash
uv run packing --help
```

Live commands require explicit spend confirmation. Use dry-run mode first.
