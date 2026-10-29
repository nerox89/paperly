# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed
- Run bulk suggestion application as a single background job with live progress, refreshed review results, and visible failures.
- Serialize document updates and stop on upstream timeouts without retrying uncertain writes or discarding unapplied suggestions.
- Use Paperless's native inbox definition, including renamed or multiple inbox tags.
- Show retryable classification failures instead of crashing or caching failed results.
- Validate AI output and dates before they can become actionable suggestions.
- Share suggestion application across review and audit, including new tags, correspondents, dates, and accurate history.
- Respect explicit metadata clearing, preserve filtered audit navigation, and remove duplicate keyboard handlers.
- Reserve batch jobs before fetching documents, count fetch failures, and clean up cancelled workers.
- Keep conflicting suggestions out of audit batch confirmation and show page-level audit counts accurately.
- Validate replacement settings before discarding the working connection; preserve review cache when changing models.
- Paginate taxonomy and cleanup document queries before deleting merged entities.
- Serve bundled styles and keep the interface visible when external animation services are unavailable.

### Added
- Isolated API, batch, metadata, learning, and browser regression checks in CI.

## [0.1.0] — 2026-04-05

### Added
- FastAPI web UI for reviewing AI-generated document metadata suggestions
- Side-by-side PDF preview and editable suggestion form
- Batch classification mode with configurable concurrency
- Auto-apply above a confidence threshold (`PAPERLY_AUTO_APPLY_CONFIDENCE`)
- Anthropic Claude backend (`claude-haiku-4-5` default)
- Ollama backend for self-hosted, API-key-free operation
- AI provider and model switching via Settings UI (no restart required)
- Taxonomy cleanup: merge duplicate correspondents/document types, remove orphans
- Classification history with provider and confidence tracking
- Keyboard shortcuts: `n` next, `s` skip, `Enter` apply
- SQLite database for caching and history (`paperly.db`)
- Docker image published to `ghcr.io/nerox89/paperly`
