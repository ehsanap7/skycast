# CLAUDE.md

Guidance for anyone (human or AI agent) working in this repository. These rules are mandatory.

## Project overview

SkyCast is an end-to-end MLOps weather forecasting platform following the AWS Well-Architected
Machine Learning Lens lifecycle, built with local / open-source tools:

1. **Process Data**: collect (historical dataset + hourly API) → preprocess → engineer features
   → feature store (offline = full history, online = latest values).
2. **Develop Model**: train, tune, evaluate against a baseline → model registry.
3. **Deploy**: batch inference (scheduled) + real-time inference (FastAPI) → Streamlit app.
4. **Monitor**: data drift, performance vs. observed weather, system health → alerts → scheduler.

Feedback loops (monitor → retrain, evaluation → data processing) and full lineage
(data + code + params → model → prediction) are first-class requirements.

**Multi-city rule:** the first city is Toronto, but cities are configuration (name, lat/lon,
timezone). Never put a city name in package, module, class, or function names, or in logic.

## Repository structure

```
src/skycast/          # application package (src layout)
tests/                # pytest suite, mirrors src/skycast/
.github/workflows/    # CI
.claude/              # shared tool permissions and hooks
pyproject.toml        # metadata, dependencies, ruff + pytest config
uv.lock               # locked dependencies (always committed)
```

## Commands

```bash
uv sync                             # install project + dev tools into .venv
uv run pre-commit install           # once per clone
uv run skycast                      # run the CLI
uv run pytest                       # tests
uv run ruff check . && uv run ruff format .   # lint + format
uv run pre-commit run --all-files   # everything CI runs
```

## Workflow (mandatory)

`Issue → Branch → Commits → Pull Request → CI → Squash-merge → Delete branch`

- **Issue first.** Every task starts as a GitHub issue with a clear explanation, label,
  task checklist, and acceptance criteria.
- **Separate branch per issue:** `<type>/<issue#>-<short-kebab-name>`, created with
  `gh issue develop <#> --name <branch> --checkout`.
- **One task at a time.** Do not combine several tasks into one step, unless they are listed
  under [Approved task groups](#approved-task-groups).
- **Commits:** Conventional Commits (`feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`,
  `ci:`, `build:`), small and logical, each leaving the repo in a working state.
  Subject line (+ optional body) only: **no trailers.**
- **PRs:** conventional-style title; description with Summary, Changes, How to test, and
  `Closes #<n>`. CI (`Lint & test`) must pass.
- **Merge:** squash-merge only, delete the branch, switch to `main`, pull, `git fetch --prune`.

Tool permissions (allowed, confirmed, and blocked actions) are defined in `.claude/settings.json`.

### Approved task groups

Task bundles that may run together without separate approvals. Anything not listed here needs
its own approval.

- _None yet._

## Code standards

- **Small, single-purpose functions.** If a function does two things, split it.
- **Type hints everywhere** (parameters and return values).
- **One-line docstring at the top of every function/class**, plain and human: what it does.
  No comments inside functions unless the logic is genuinely non-obvious.
- **Design patterns when they solve a real problem** (e.g. Strategy for interchangeable data
  sources, Factory for building them from config, Repository for storage). Never for show.
- **Logging, not `print`**, in library code (`logging.getLogger(__name__)`). `print` is only
  for CLI output.
- **Configuration over hard-coding:** cities, paths, URLs, and parameters come from config
  or environment variables.
- Ruff (lint + format) is the source of truth for style; line length 100.

## Error handling

- **Handle expected failures at boundaries**: network/API calls, file I/O, parsing, config
  loading. Catch specific exceptions there and retry, wrap, or log with context.
- **Raise domain exceptions** from a `SkyCastError` base class (e.g. `DataSourceError`,
  `ConfigError`), chaining the cause with `raise ... from err`.
- **Let unexpected errors propagate** to the entry point (CLI/API/scheduler), which logs them
  and exits or responds cleanly.
- **Never swallow errors** (no bare `except:`, no `except Exception: pass`).

## Testing & ML rules

- Every piece of new logic gets tests; external services (APIs) are mocked.
- **Time-based splits only** (no random splits) to avoid leakage.
- Every model must beat a baseline (persistence / seasonal average) to be registered.

## Dependencies & CI

- The project targets **Python 3.14**. Before adding a dependency, verify it supports 3.14;
  if not, report it and propose options. Never change the Python version silently.
- Add dependencies with `uv add` (runtime) or `uv add --dev` (tooling); commit `uv.lock`.
- Keep the ruff version in `.pre-commit-config.yaml` in sync with `uv.lock`.
- **GitHub Actions:** pin third-party actions to a full commit SHA with a version comment,
  and verify the tag/ref actually exists before using it.

## Security

- Secrets (API keys) live only in `.env` (git-ignored); keep `.env.example` up to date.
- Never commit secrets, data dumps, or model binaries (data/models will be tracked with DVC).

## Decision log

| Decision | Choice | Reason |
|---|---|---|
| Packaging / env | uv + `pyproject.toml` | Fast, single tool, lockfile |
| Build backend | `uv_build` | Zero-config for `src/` layout, same toolchain |
| Layout | `src/` | Tests run against the installed package |
| Version source | `importlib.metadata` | Single source of truth in `pyproject.toml` |
| CI | GitHub Actions, required check `Lint & test` | Nothing broken reaches `main` |
| Branch protection | Ruleset: PR required, squash-only, 0 approvals, no bypass | Solo repo that still follows its own rules |

**Open decisions** (ask the maintainer, do not assume): prediction target(s), forecast horizon,
historical data source, recent-observations API and key setup, preprocessing design patterns
(to be adopted from the maintainer's existing project).
