# Contributing to paperly

Thank you for your interest in contributing!

## Getting Started

1. Fork the repository
2. Clone your fork and create a feature branch:
   ```bash
   git checkout -b feat/my-feature
   ```
3. Set up the local dev environment:
   ```bash
   pip install -e .
   cp .env.example .env
   # fill in .env
   uvicorn paperly.app:app --reload --port 8002
   ```
4. Make your changes
5. Open a pull request against `main`

## Guidelines

- **Python style**: type hints on all functions, async everywhere, Pydantic for models
- **Templates**: Jinja2 + HTMX — keep JS minimal; run `npm ci && npm run build:css` when changing utility classes
- **Tests**: install `pip install -e '.[test]'`, run `python -m playwright install --with-deps chromium`, then `python -m unittest discover -s tests -v`; use synthetic data, never real documents or credentials
- **Commits**: use [Conventional Commits](https://www.conventionalcommits.org/):
  `feat:`, `fix:`, `docs:`, `chore:`

## Reporting Bugs

Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.yml).

## Suggesting Features

Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.yml).

## Questions

Open a Discussion on GitHub.
