# Paperly — Copilot Instructions

## Project purpose
Paperly is a Python/FastAPI web app that uses Claude AI to improve document metadata quality in a self-hosted Paperless-NGX instance. It processes inbox documents and suggests better titles, correspondents, tags, and document types.

## Tech stack
- **Backend**: Python 3.12, FastAPI, Jinja2 templates, httpx (async HTTP)
- **Frontend**: HTMX + Tailwind CSS (served via CDN, no build step)
- **AI**: Anthropic Claude API (`claude-haiku-4-5` by default)
- **Deployment**: Docker container on Unraid server

## Key context
- Paperless-NGX API at `http://paperless-ngx:8000` (Docker network) or `http://localhost:18000` (local dev via SSH tunnel)
- ~1,367 inbox documents to process (out of 1,693 total)
- Documents are primarily German (deu+eng OCR)
- Existing taxonomy: 40 tags, 94 correspondents, 44 document types (many duplicates)
- Server: faunetserver (Unraid), accessible via `faunetserverremote` SSH alias

## Code conventions
- Async everywhere (httpx.AsyncClient, async FastAPI routes)
- Pydantic models for API responses
- Type hints on all functions
- Keep templates simple — HTMX handles dynamic updates, no heavy JS framework
- Error states shown inline (no full-page error pages)

## File structure
```
src/paperly/
├── app.py          # FastAPI app, routes, startup
├── paperless.py    # Paperless-NGX API client
├── classifier.py   # Claude classifier
└── templates/      # Jinja2 HTML templates
```

## Paperless API patterns
- Auth: `Authorization: Token <token>` header
- Inbox filter: `GET /api/documents/?tags__id__all=<inbox_tag_id>&page_size=25`
- Update: `PATCH /api/documents/<id>/` with JSON body
- Tags/correspondents/doc types: `/api/tags/`, `/api/correspondents/`, `/api/document_types/`

## Claude classifier contract
Input: OCR text (truncated to ~3000 chars) + existing taxonomy lists
Output JSON:
```json
{
  "title": "suggested title",
  "correspondent_id": 42,
  "correspondent_name": "Sparkasse Lübeck",
  "tag_ids": [5, 12],
  "document_type_id": 3,
  "confidence": 0.85,
  "reasoning": "brief explanation"
}
```
