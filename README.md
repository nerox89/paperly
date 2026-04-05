# paperly

> AI-powered document quality improvement for [Paperless-NGX](https://docs.paperless-ngx.com/)

Paperly processes your Paperless-NGX inbox using an AI model (Claude or Ollama) and suggests better titles, correspondents, tags, and document types. You review each suggestion in a side-by-side UI and apply them with a single click — or let batch mode handle it automatically above a confidence threshold.

---

## Features

- 🤖 **AI classification** — reads OCR text and suggests metadata (title, correspondent, tags, document type)
- 🔍 **Review UI** — side-by-side PDF preview with editable suggestions
- ⚡ **Batch mode** — process all inbox documents automatically, auto-apply above a confidence threshold
- 🏷️ **Taxonomy cleanup** — merge duplicate correspondents/document types, remove orphans
- ⚙️ **Settings UI** — switch between Claude and Ollama at runtime, no restart needed
- 📜 **History** — track which documents were processed and what was changed
- ⌨️ **Keyboard shortcuts** — `n` next, `s` skip, `Enter` apply

## Supported AI backends

| Provider | Models | Notes |
|---|---|---|
| [Anthropic Claude](https://www.anthropic.com/) | `claude-haiku-4-5` (default), any Claude model | Requires API key |
| [Ollama](https://ollama.com/) | `gemma4:e4b` (default), any Ollama model | Self-hosted, no API key |

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
| `PAPERLY_AI_PROVIDER` | `claude` | `claude` or `ollama` |
| `ANTHROPIC_API_KEY` | — | Required when using Claude |
| `PAPERLY_CLAUDE_MODEL` | `claude-haiku-4-5` | Any Claude model |
| `PAPERLY_OLLAMA_URL` | `http://localhost:11434` | Ollama base URL |
| `PAPERLY_OLLAMA_MODEL` | `gemma4:e4b` | Any model pulled in Ollama |

### Optional

| Variable | Default | Description |
|---|---|---|
| `PAPERLY_AUTO_APPLY_CONFIDENCE` | `0.85` | Batch mode: auto-apply suggestions above this threshold (0–1) |
| `PAPERLY_BATCH_CONCURRENCY` | `3` | Number of parallel classification workers in batch mode |
| `PAPERLY_DATA_DIR` | `.` | Directory for the SQLite database (`paperly.db`) |
| `PAPERLY_PORT` | `8002` | Port to listen on |
| `PAPERLY_HOST` | `0.0.0.0` | Host/address to bind |
| `PAPERLY_PAPERLESS_PUBLIC_URL` | *(same as `PAPERLESS_URL`)* | Public URL for PDF preview links (useful when Paperless is on an internal network) |

The AI provider and model can also be changed at runtime via the **Settings** page — no restart needed.

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
