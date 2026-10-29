# paperly

> AI-powered document quality improvement for [Paperless-NGX](https://docs.paperless-ngx.com/)

Paperly processes your Paperless-NGX inbox using an AI model and suggests better titles, correspondents, tags, and document types. Review suggestions individually, or explicitly apply high-confidence suggestions from the review page. Batch classification alone never changes documents.

---

## Features

- 🤖 **AI classification** — reads OCR text and suggests metadata (title, correspondent, tags, document type)
- 🔍 **Review UI** — side-by-side PDF preview with editable suggestions
- ⚡ **Batch mode** — classify all inbox documents, then review or apply above a confidence threshold
- 🏷️ **Taxonomy cleanup** — merge duplicate correspondents/document types, remove orphans
- ⚙️ **Settings UI** — switch between Claude and Ollama at runtime, no restart needed
- 📜 **History** — track which documents were processed and what was changed
- ⌨️ **Keyboard shortcuts** — `s` mark reviewed, `Enter` apply, `Esc` return

## Supported AI backends

| Provider | Models | Notes |
|---|---|---|
| [Anthropic Claude](https://www.anthropic.com/) | `claude-haiku-4-5` (default), any Claude model | Requires API key |
| [Ollama](https://ollama.com/) | `gemma4:e4b` (default), any Ollama model | Self-hosted, no API key |
| OpenAI-compatible API | `gpt-4o-mini` (default), models available at your endpoint | Configure the base URL and API key in Settings |

---

## Quick Start (Docker)

### Option A — `docker run`

```bash
docker run -d \
  --name paperly \
  -p 8002:8002 \
  -e PAPERLESS_URL=http://paperless-ngx:8000 \
  -e PAPERLESS_TOKEN=your_token_here \
  -e ANTHROPIC_API_KEY=your_key_here \
  --network your_docker_network \
  ghcr.io/nerox89/paperly:latest
```

Then open **http://localhost:8002**.

### Option B — Docker Compose

```bash
curl -O https://raw.githubusercontent.com/nerox89/paperly/main/docker-compose.yml
curl -O https://raw.githubusercontent.com/nerox89/paperly/main/.env.example
cp .env.example .env
# Edit .env and set PAPERLESS_URL, PAPERLESS_TOKEN, ANTHROPIC_API_KEY
docker compose up -d
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| [Paperless-NGX](https://github.com/paperless-ngx/paperless-ngx) | Running instance with API access |
| Paperless API token | Settings → API Tokens in Paperless-NGX |
| Anthropic API key **or** Ollama | At least one AI backend |
| Docker (recommended) | Or Python 3.12+ for local dev |

### Get a Paperless-NGX API token

In Paperless-NGX, go to **Settings → API Tokens** and create a token for your user.

---

## Configuration

All configuration is done via environment variables (or `.env` file).

### Required

| Variable | Description |
|---|---|
| `PAPERLESS_URL` | URL to your Paperless-NGX instance, e.g. `http://paperless-ngx:8000` |
| `PAPERLESS_TOKEN` | Paperless-NGX API token |

### AI backend (at least one required)

| Variable | Default | Description |
|---|---|---|
| `PAPERLY_AI_PROVIDER` | `claude` | `claude`, `ollama`, or `openai` |
| `ANTHROPIC_API_KEY` | — | Required when using Claude |
| `PAPERLY_CLAUDE_MODEL` | `claude-haiku-4-5` | Any Claude model |
| `PAPERLY_OLLAMA_URL` | `http://localhost:11434` | Ollama base URL |
| `PAPERLY_OLLAMA_MODEL` | `gemma4:e4b` | Any model pulled in Ollama |
| `PAPERLY_OPENAI_BASE_URL` | `http://copilot-gateway:8080/v1` | OpenAI-compatible API base URL |
| `OPENAI_API_KEY` | — | API key for the OpenAI-compatible endpoint |
| `PAPERLY_OPENAI_MODEL` | `gpt-4o-mini` | Use a currently supported model from the Settings model picker |

### Optional

| Variable | Default | Description |
|---|---|---|
| `PAPERLY_AUTO_APPLY_CONFIDENCE` | `0.85` | Batch mode: auto-apply suggestions above this threshold (0–1) |
| `PAPERLY_BATCH_CONCURRENCY` | `1` | Number of parallel classification workers in batch mode (minimum 1) |
| `PAPERLY_DATA_DIR` | `.` | Directory for the SQLite database (`paperly.db`) |
| `PAPERLY_PORT` | `8002` | Port to listen on |
| `PAPERLY_HOST` | `0.0.0.0` | Host/address to bind |
| `PAPERLY_PAPERLESS_PUBLIC_URL` | *(same as `PAPERLESS_URL`)* | Public URL for PDF preview links (useful when Paperless is on an internal network) |

The AI provider and model can also be changed at runtime via the **Settings** page — no restart needed.

### Inbox and review behavior

Paperly uses Paperless's native inbox filter, including every tag marked as an
inbox tag, regardless of its name. A Paperless saved view named "INBOX" still
needs an actual inbox filter; its name alone does not filter documents.

Applying a suggestion removes inbox tags. **Skip** marks the document reviewed
by removing inbox tags without changing its other metadata. Audit's **confirm
current metadata** action records a training example without applying the AI
suggestion. **Confirm matching** skips conflicting suggestions, including inbox
documents; apply those individually after reviewing the differences.

Failed classifications are never cached or applied. Bulk and individual AI
application both resolve new correspondents and tags and apply the suggested
date. Settings cannot be changed during an inbox batch.

---

## Local Development

```bash
# 1. Clone the repo
git clone https://github.com/nerox89/paperly.git
cd paperly

# 2. Set up environment
cp .env.example .env
# Edit .env: set PAPERLESS_URL, PAPERLESS_TOKEN, ANTHROPIC_API_KEY

# 3. Install and run
pip install -e .
uvicorn paperly.app:app --reload --port 8002
```

If Paperless runs on another machine, create an SSH tunnel first:

```bash
ssh -f -N -L 18000:paperless-ngx-host:8000 your-server
# Then set PAPERLESS_URL=http://localhost:18000 in .env
```

### With Ollama (no API key needed)

```bash
# Start Ollama and pull a model
ollama pull gemma4:e4b

# Set in .env:
# PAPERLY_AI_PROVIDER=ollama
# PAPERLY_OLLAMA_URL=http://localhost:11434
# PAPERLY_OLLAMA_MODEL=gemma4:e4b
```

### Regression checks and bundled styles

```bash
pip install -e '.[test]'
python -m playwright install --with-deps chromium
python -m unittest discover -s tests -v

# After changing template utility classes or stylesheet sources:
npm ci
npm run build:css
```

Tests use synthetic documents and an isolated Paperless API fixture. They do
not require real credentials or change your Paperless instance. Node.js is
needed for the keyboard-dispatch check. The bundled stylesheet works without
external CDNs; optional fonts and animations are not required to use the UI.

---

## Taxonomy Cleanup CLI

Remove duplicate correspondents, document types, and orphaned tags:

```bash
# Preview changes without applying
paperly-cleanup --dry-run

# Interactive cleanup
paperly-cleanup
```

---

## Deployment Examples

### Unraid (community template)

Add via **Community Applications** or manually:

```bash
docker run -d \
  --name paperly \
  --network paperless_network \
  -p 8002:8002 \
  --env-file /path/to/paperly.env \
  -v /mnt/user/appdata/paperly:/data \
  -e PAPERLY_DATA_DIR=/data \
  ghcr.io/nerox89/paperly:latest
```

### With a reverse proxy (Caddy example)

```
paperly.example.com {
    reverse_proxy paperly:8002
}
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE)
