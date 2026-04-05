# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
