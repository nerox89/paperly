# paperly

AI-powered document quality improvement for [Paperless-NGX](https://docs.paperless-ngx.com/).

Processes inbox documents using Claude AI to suggest better titles, correspondents, tags, and document types. Review and apply suggestions via a web UI.

## Features

- **AI classification**: Claude reads OCR text and suggests metadata
- **Review UI**: side-by-side PDF preview + editable suggestions
- **Taxonomy cleanup**: merge duplicate correspondents/document types, remove orphans
- **Keyboard shortcuts**: `n` next, `s` skip, `Enter` apply

## Running on Unraid (production)

The app runs as a Docker container on the same network as Paperless-NGX.
Secrets are managed via Infisical (`/paperly` folder).

```bash
docker pull ghcr.io/nerox89/paperly:latest
docker run -d --name paperly --env-file /boot/config/secrets/paperly.env \
  --network paperless-network -p 8002:8002 ghcr.io/nerox89/paperly:latest
```

## Local development

```bash
# Start SSH tunnel to paperless
ssh -f -N -L 18000:172.22.0.11:8000 faunetserverremote

# Set up env
cp .env.example .env
# Edit .env: set PAPERLESS_URL=http://localhost:18000 and tokens

# Install and run
pip install -e .
uvicorn paperly.app:app --reload --port 8002
```

## Taxonomy cleanup

```bash
paperly cleanup --dry-run   # preview changes
paperly cleanup             # interactive cleanup
```
