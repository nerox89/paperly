"""FastAPI application — paperly."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

from paperly.classifier import Classifier, ClassificationResult
from paperly.paperless import Document, PaperlessClient, Taxonomy

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ---------------------------------------------------------------------------
# App state (loaded at startup, refreshed on demand)
# ---------------------------------------------------------------------------

class AppState:
    paperless: PaperlessClient
    classifier: Classifier
    taxonomy: Taxonomy
    # Simple in-memory cache: doc_id -> ClassificationResult
    suggestion_cache: dict[int, ClassificationResult]

    def __init__(self) -> None:
        self.suggestion_cache = {}


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    paperless_url = os.environ["PAPERLESS_URL"]
    paperless_token = os.environ["PAPERLESS_TOKEN"]
    anthropic_key = os.environ["ANTHROPIC_API_KEY"]

    state.paperless = PaperlessClient(paperless_url, paperless_token)
    state.classifier = Classifier(anthropic_key)
    state.taxonomy = await state.paperless.get_taxonomy()

    logger.info(
        "Loaded taxonomy: %d tags, %d correspondents, %d document types, %d storage paths, inbox_tag=%s",
        len(state.taxonomy.tags),
        len(state.taxonomy.correspondents),
        len(state.taxonomy.document_types),
        len(state.taxonomy.storage_paths),
        state.taxonomy.inbox_tag_id,
    )
    yield


app = FastAPI(title="paperly", lifespan=lifespan)

_static = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_static), name="static")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request, page: int = 1):
    if not state.taxonomy.inbox_tag_id:
        raise HTTPException(500, "INBOX tag not found in Paperless")

    docs, total = await state.paperless.get_inbox_documents(
        state.taxonomy.inbox_tag_id, page=page, page_size=25
    )
    stats = await state.paperless.get_statistics()
    total_docs = stats.get("documents_total", 0)
    processed = total_docs - total  # rough estimate

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "docs": docs,
            "total": total,
            "total_docs": total_docs,
            "processed": processed,
            "page": page,
            "page_size": 25,
            "taxonomy": state.taxonomy,
        },
    )


@app.get("/document/{doc_id}", response_class=HTMLResponse)
async def document_view(request: Request, doc_id: int):
    doc = await state.paperless.get_document(doc_id)
    suggestion = state.suggestion_cache.get(doc_id)
    return templates.TemplateResponse(
        "document.html",
        {
            "request": request,
            "doc": doc,
            "suggestion": suggestion,
            "taxonomy": state.taxonomy,
            "paperless_url": os.environ["PAPERLESS_URL"],
        },
    )


@app.get("/document/{doc_id}/classify", response_class=HTMLResponse)
async def classify_document(request: Request, doc_id: int):
    """Trigger AI classification and return the suggestion partial (HTMX target)."""
    if doc_id not in state.suggestion_cache:
        doc = await state.paperless.get_document(doc_id)
        suggestion = await state.classifier.classify(
            doc.content, state.taxonomy, original_title=doc.title
        )
        state.suggestion_cache[doc_id] = suggestion

    suggestion = state.suggestion_cache[doc_id]
    doc = await state.paperless.get_document(doc_id)
    return templates.TemplateResponse(
        "partials/suggestion.html",
        {
            "request": request,
            "doc": doc,
            "suggestion": suggestion,
            "taxonomy": state.taxonomy,
        },
    )


@app.post("/document/{doc_id}/apply")
async def apply_document(
    request: Request,
    doc_id: int,
    title: Annotated[str, Form()],
    created: Annotated[str, Form()] = "",
    correspondent_id: Annotated[str, Form()] = "",
    correspondent_new: Annotated[str, Form()] = "",
    document_type_id: Annotated[str, Form()] = "",
    document_type_new: Annotated[str, Form()] = "",
    storage_path_id: Annotated[str, Form()] = "",
    tag_ids: Annotated[list[str], Form()] = [],
    tag_new: Annotated[str, Form()] = "",
):
    """Apply chosen metadata and remove INBOX tag."""
    # Resolve correspondent
    corr_id: int | None = None
    if correspondent_id:
        corr_id = int(correspondent_id)
    elif correspondent_new.strip():
        new_corr = await state.paperless.create_correspondent(correspondent_new.strip())
        corr_id = new_corr.id
        state.taxonomy = await state.paperless.get_taxonomy()

    # Resolve document type
    dt_id: int | None = None
    if document_type_id:
        dt_id = int(document_type_id)
    elif document_type_new.strip():
        new_dt = await state.paperless.create_document_type(document_type_new.strip())
        dt_id = new_dt.id
        state.taxonomy = await state.paperless.get_taxonomy()

    # Resolve storage path
    sp_id: int | None = None
    if storage_path_id:
        sp_id = int(storage_path_id)

    # Resolve tags — create new tag if provided
    chosen_tags = [int(t) for t in tag_ids if t]
    if tag_new.strip():
        new_tag = await state.paperless.create_tag(tag_new.strip())
        chosen_tags.append(new_tag.id)
        state.taxonomy = await state.paperless.get_taxonomy()

    # Remove INBOX tag
    if state.taxonomy.inbox_tag_id in chosen_tags:
        chosen_tags.remove(state.taxonomy.inbox_tag_id)

    await state.paperless.update_document(
        doc_id,
        title=title,
        created=created or None,
        correspondent=corr_id,
        document_type=dt_id,
        storage_path=sp_id,
        tags=chosen_tags,
    )

    state.suggestion_cache.pop(doc_id, None)
    return RedirectResponse("/", status_code=303)


@app.post("/document/{doc_id}/skip")
async def skip_document(doc_id: int):
    """Remove INBOX tag without changing metadata (mark as reviewed)."""
    if state.taxonomy.inbox_tag_id:
        doc = await state.paperless.get_document(doc_id)
        new_tags = [t for t in doc.tags if t != state.taxonomy.inbox_tag_id]
        await state.paperless.update_document(doc_id, tags=new_tags)
    state.suggestion_cache.pop(doc_id, None)
    return RedirectResponse("/", status_code=303)


@app.post("/document/{doc_id}/delete")
async def delete_document(doc_id: int):
    """Permanently delete a document from Paperless."""
    await state.paperless.delete_document(doc_id)
    state.suggestion_cache.pop(doc_id, None)
    return RedirectResponse("/", status_code=303)


@app.get("/proxy/thumb/{doc_id}")
async def proxy_thumb(doc_id: int):
    """Proxy document thumbnail through paperly (avoids CORS issues)."""
    import httpx
    paperless_url = os.environ["PAPERLESS_URL"]
    token = os.environ["PAPERLESS_TOKEN"]
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{paperless_url}/api/documents/{doc_id}/thumb/",
            headers={"Authorization": f"Token {token}"},
            timeout=10.0,
        )
    return StreamingResponse(
        iter([r.content]),
        media_type=r.headers.get("content-type", "image/jpeg"),
    )


@app.get("/proxy/preview/{doc_id}")
async def proxy_preview(doc_id: int):
    """Proxy document preview PDF through paperly."""
    import httpx
    paperless_url = os.environ["PAPERLESS_URL"]
    token = os.environ["PAPERLESS_TOKEN"]
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(
            f"{paperless_url}/api/documents/{doc_id}/preview/",
            headers={"Authorization": f"Token {token}"},
        )
    return StreamingResponse(
        iter([r.content]),
        media_type="application/pdf",
    )


@app.get("/stats")
async def stats():
    return await state.paperless.get_statistics()


@app.post("/refresh-taxonomy")
async def refresh_taxonomy():
    state.taxonomy = await state.paperless.get_taxonomy()
    return {"status": "ok", "correspondents": len(state.taxonomy.correspondents)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    port = int(os.environ.get("PAPERLY_PORT", 8002))
    host = os.environ.get("PAPERLY_HOST", "0.0.0.0")
    uvicorn.run("paperly.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
