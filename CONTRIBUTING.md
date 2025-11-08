# Contributing Guide

Thank you for your interest in contributing! This guide helps you set up your environment, follow coding conventions, and submit high-quality pull requests.

## Environment Setup
1. Use Python 3.10+.
2. Create a virtual environment and install dependencies:
```
python -m venv .venv
. .venv/Scripts/Activate.ps1   # PowerShell (Windows)
pip install -e .[dev]
```

## Development Workflow
- Lint: `ruff check .`
- Type check: `mypy src`
- Test: `pytest`
- CLI help: `aiflow --help`

## Commit and PR Conventions
- Use conventional commits where possible (e.g., `feat:`, `fix:`, `docs:`, `test:`, `refactor:`).
- Include tests for new features and bug fixes.

## Project Architecture
- Prefer adding new functionality via the plugin system (`aiflow.plugins`) when feasible.
- Keep core abstractions stable: parsers, IR, optimizer passes, kernel backends, profilers, autotuner strategies.
- Separate graph-level and kernel-level concerns.

## Code Style
- Follow Ruff (PEP8 + selected rules). See `ruff.toml`.
- Keep functions and classes small and focused.
- Add concise, high-signal comments only for non-obvious logic or invariants.

## Opening Issues
- For bugs: use the bug report template and include a minimal reproducer when possible.
- For features: outline motivation, proposed design, and alternatives.

## Security
- See SECURITY.md for vulnerability reporting.

## Getting Help
- Open a discussion or issue with your question and relevant context.




