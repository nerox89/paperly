"""FastAPI application — paperly."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from html import escape
from pathlib import Path
from typing import Annotated
from urllib.parse import urlencode

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from paperly.classifier import (
    AnthropicProvider,
    BaseProvider,
    Classifier,
    ClassificationResult,
    OllamaProvider,
    OpenAIProvider,
)
from paperly.database import Database
from paperly.paperless import Document, PaperlessClient, Taxonomy

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

BATCH_CONCURRENCY = max(1, int(os.environ.get("PAPERLY_BATCH_CONCURRENCY", "1")))
AUTO_APPLY_MIN_CONFIDENCE = float(os.environ.get("PAPERLY_AUTO_APPLY_CONFIDENCE", "0.85"))

# ---------------------------------------------------------------------------
# App state (loaded at startup, refreshed on demand)
# ---------------------------------------------------------------------------

class AppState:
    paperless: PaperlessClient
    classifier: Classifier
    taxonomy: Taxonomy
    db: Database
    # Batch classification state
    batch_running: bool
    batch_total: int
    batch_done: int
    batch_errors: int
    batch_cancel: bool
    batch_start_time: float
    batch_current_doc: str
    batch_log: list[dict]  # last N completed items

    taxonomy_refreshed_at: float

    def __init__(self) -> None:
        self.batch_running = False
        self.batch_total = 0
        self.batch_done = 0
        self.batch_errors = 0
        self.batch_cancel = False
        self.batch_start_time = 0.0
        self.batch_current_doc = ""
        self.batch_log = []
        self.taxonomy_refreshed_at = 0.0
        self.batch_task: asyncio.Task | None = None
        self.taxonomy_lock = asyncio.Lock()
        self.metadata_lock = asyncio.Lock()
        self.suggestion_generation = 0
        self.review_apply_task: asyncio.Task | None = None
        self.review_apply = {
            "started": False, "running": False, "total": 0, "done": 0,
            "applied": 0, "skipped": 0, "errors": 0, "failures": [], "stopped": False,
        }


state = AppState()


def _build_provider(db: Database, overrides: dict[str, str] | None = None) -> BaseProvider:
    """Build the active LLM provider from DB settings + env vars."""
    def setting(key: str, default: str) -> str:
        return (overrides or {}).get(key, db.get_setting(key, default))

    provider_name = setting("ai_provider", os.environ.get("PAPERLY_AI_PROVIDER", "claude"))

    if provider_name == "ollama":
        url = setting("ollama_url", os.environ.get("PAPERLY_OLLAMA_URL", "http://localhost:11434"))
        model = setting("ollama_model", os.environ.get("PAPERLY_OLLAMA_MODEL", "gemma4:e4b"))
        logger.info("Using Ollama provider: %s model=%s", url, model)
        return OllamaProvider(base_url=url, model=model)
    elif provider_name == "openai":
        api_key = setting("openai_api_key", os.environ.get("OPENAI_API_KEY", "dummy")) or "dummy"
        base_url = setting("openai_base_url", os.environ.get("PAPERLY_OPENAI_BASE_URL", "http://copilot-gateway:8080/v1"))
        model = setting("openai_model", os.environ.get("PAPERLY_OPENAI_MODEL", "gpt-4o-mini"))
        logger.info("Using OpenAI-compatible provider: %s model=%s", base_url, model)
        return OpenAIProvider(api_key=api_key, base_url=base_url, model=model)
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        model = setting("claude_model", os.environ.get("PAPERLY_CLAUDE_MODEL", "claude-haiku-4-5"))
        logger.info("Using Anthropic provider: model=%s", model)
        return AnthropicProvider(api_key=api_key, model=model)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # SQLite database for persistent cache, history, and settings
    data_dir = os.environ.get("PAPERLY_DATA_DIR", ".")
    state.db = Database(Path(data_dir) / "paperly.db")
    state.db.open()

    # Paperless connection: DB settings override env vars
    paperless_url = state.db.get_setting("paperless_url", os.environ.get("PAPERLESS_URL", ""))
    paperless_token = state.db.get_setting("paperless_token", os.environ.get("PAPERLESS_TOKEN", ""))
    if not paperless_url or not paperless_token:
        raise RuntimeError("PAPERLESS_URL and PAPERLESS_TOKEN must be set (env or settings)")

    state.paperless = PaperlessClient(paperless_url, paperless_token)
    await state.paperless.open()

    # Build classifier with provider from settings
    provider = _build_provider(state.db)
    custom_prompt = state.db.get_setting("custom_prompt", "")
    state.classifier = Classifier(provider, custom_prompt=custom_prompt)

    state.taxonomy = await state.paperless.get_taxonomy()
    state.taxonomy_refreshed_at = time.time()

    logger.info(
        "Loaded taxonomy: %d tags, %d correspondents, %d document types, %d storage paths, inbox_tag=%s",
        len(state.taxonomy.tags),
        len(state.taxonomy.correspondents),
        len(state.taxonomy.document_types),
        len(state.taxonomy.storage_paths),
        state.taxonomy.inbox_tag_id,
    )
    logger.info("Cached suggestions: %d", state.db.suggestion_count())
    logger.info("AI provider: %s / %s", state.classifier.provider.name, state.classifier.provider.model)
    try:
        yield
    finally:
        if state.batch_task and not state.batch_task.done():
            state.batch_cancel = True
            state.batch_task.cancel()
            await asyncio.gather(state.batch_task, return_exceptions=True)
        if state.review_apply_task and not state.review_apply_task.done():
            state.review_apply_task.cancel()
            await asyncio.gather(state.review_apply_task, return_exceptions=True)
        await state.paperless.close()
        state.db.close()


app = FastAPI(title="paperly", lifespan=lifespan)

_static = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_static), name="static")

@app.exception_handler(httpx.HTTPError)
async def paperless_error(request: Request, exc: httpx.HTTPError):
    status = 404 if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 404 else 502
    logger.warning("Paperless request failed: %s", type(exc).__name__)
    return templates.TemplateResponse(
        request, "partials/error.html",
        {"error": "Dokument in Paperless nicht gefunden." if status == 404 else
                  "Paperless ist nicht erreichbar oder hat die Anfrage abgelehnt.",
         "doc_id": request.path_params.get("doc_id") if request.url.path.endswith("/classify") else None,
         "ref": request.query_params.get("ref", "")},
        status_code=status,
        headers={"X-Paperly-Error": "1"},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

TAXONOMY_MAX_AGE = 300  # 5 minutes

async def _ensure_fresh_taxonomy() -> None:
    """Refresh taxonomy from Paperless if older than TAXONOMY_MAX_AGE seconds."""
    if time.time() - state.taxonomy_refreshed_at > TAXONOMY_MAX_AGE:
        async with state.taxonomy_lock:
            if time.time() - state.taxonomy_refreshed_at > TAXONOMY_MAX_AGE:
                await _refresh_taxonomy()


async def _refresh_taxonomy() -> None:
    """Force-refresh taxonomy from Paperless."""
    taxonomy = await state.paperless.get_taxonomy()
    previous = getattr(state, "taxonomy", None)
    if previous and any(
        {item.id for item in getattr(previous, field)} - {item.id for item in getattr(taxonomy, field)}
        for field in ("tags", "correspondents", "document_types", "storage_paths")
    ):
        _invalidate_suggestions()
    state.taxonomy = taxonomy
    state.taxonomy_refreshed_at = time.time()


def _invalidate_suggestions() -> None:
    state.suggestion_generation += 1
    state.db.clear_all_suggestions()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request, page: int = 1):
    await _ensure_fresh_taxonomy()
    docs, total = await state.paperless.get_inbox_documents(
        page=page, page_size=25
    )
    stats = await state.paperless.get_statistics()
    total_docs = stats.get("documents_total", 0)
    processed = total_docs - total  # rough estimate

    return templates.TemplateResponse(
            request,
            "index.html",
            {
            "docs": docs,
            "total": total,
            "total_docs": total_docs,
            "processed": processed,
            "page": page,
            "page_size": 25,
            "taxonomy": state.taxonomy,
            "running": state.batch_running,
            "batch_total": state.batch_total,
            "batch_done": state.batch_done,
            "batch_errors": state.batch_errors,
            "cached_count": state.db.suggestion_count(),
        },
    )


@app.get("/document/{doc_id}", response_class=HTMLResponse)
async def document_view(request: Request, doc_id: int, ref: str = ""):
    await _ensure_fresh_taxonomy()
    doc = await state.paperless.get_document(doc_id)
    suggestion = state.db.get_suggestion(doc_id)
    if suggestion and not 0 < suggestion.confidence <= 1:
        state.db.clear_suggestion(doc_id)
        suggestion = None
    return templates.TemplateResponse(
            request,
            "document.html",
            {
            "doc": doc,
            "suggestion": suggestion,
            "taxonomy": state.taxonomy,
            "paperless_url": (state.db.get_setting("paperless_public_url")
                              or os.environ.get("PAPERLY_PAPERLESS_PUBLIC_URL")
                              or state.db.get_setting("paperless_url")
                              or os.environ.get("PAPERLESS_URL", "")),
            "ref": ref,
        },
    )


@app.get("/document/{doc_id}/classify", response_class=HTMLResponse)
async def classify_document(request: Request, doc_id: int):
    """Trigger AI classification and return the suggestion partial (HTMX target)."""
    await _ensure_fresh_taxonomy()
    generation = state.suggestion_generation
    doc = await state.paperless.get_document(doc_id)
    try:
        suggestion = await _get_or_classify(doc, generation=generation)
    except Exception as e:
        logger.error("Classification failed for doc %d: %s", doc_id, e)
        return templates.TemplateResponse(
            request, "partials/error.html",
            {"error": str(e), "doc_id": doc_id, "ref": request.query_params.get("ref", "")},
        )
    return templates.TemplateResponse(
            request,
            "partials/suggestion.html",
            {
            "doc": doc,
            "suggestion": suggestion,
            "taxonomy": state.taxonomy,
            "ref": request.query_params.get("ref", ""),
        },
    )


@app.get("/document/{doc_id}/apply")
async def apply_document_get(doc_id: int):
    """Redirect stale GET requests back to the document page."""
    return RedirectResponse(f"/document/{doc_id}", status_code=303)


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
    new_tag_names: Annotated[list[str], Form()] = [],
):
    """Apply chosen metadata and remove INBOX tag."""
    await _ensure_fresh_taxonomy()
    doc = await state.paperless.get_document(doc_id)

    # Resolve correspondent
    corr_id: int | None = None
    if correspondent_id:
        corr_id = _optional_id(correspondent_id)
    elif correspondent_new.strip():
        new_corr = await state.paperless.create_correspondent(correspondent_new.strip())
        corr_id = new_corr.id
        await _refresh_taxonomy()

    # Resolve document type
    dt_id: int | None = None
    if document_type_id:
        dt_id = _optional_id(document_type_id)
    elif document_type_new.strip():
        new_dt = await state.paperless.create_document_type(document_type_new.strip())
        dt_id = new_dt.id
        await _refresh_taxonomy()

    # Resolve storage path
    sp_id: int | None = None
    if storage_path_id:
        sp_id = _optional_id(storage_path_id)

    # Resolve tags — create new tags if provided
    chosen_tags = [_optional_id(t) for t in tag_ids if t]
    if tag_new.strip():
        new_tag = await state.paperless.create_tag(tag_new.strip())
        chosen_tags.append(new_tag.id)
        await _refresh_taxonomy()
    # Create AI-suggested new tags
    for new_name in new_tag_names:
        if new_name.strip():
            new_tag = await state.paperless.create_tag(new_name.strip())
            chosen_tags.append(new_tag.id)
    if new_tag_names:
        await _refresh_taxonomy()

    # Remove INBOX tag
    chosen_tags = list(dict.fromkeys(t for t in chosen_tags if t not in state.taxonomy.inbox_tag_ids))

    updated = await state.paperless.update_document(
        doc_id,
        title=title,
        created=created or None,
        correspondent=corr_id,
        document_type=dt_id,
        storage_path=sp_id,
        tags=chosen_tags,
    )

    suggestion = state.db.get_suggestion(doc_id)
    _record_applied(doc, updated, suggestion)
    return _redirect_to_next(request)


def _optional_id(value: str) -> int:
    try:
        result = int(value)
        if result > 0:
            return result
    except ValueError:
        pass
    raise HTTPException(422, "Invalid metadata ID")


def _metadata(doc: Document) -> dict:
    return {key: getattr(doc, key) for key in
            ("title", "created", "correspondent", "document_type", "storage_path", "tags")}


def _record_applied(
    doc: Document, updated: Document, suggestion: ClassificationResult | None, action: str = "apply",
) -> None:
    state.db.record_feedback(
        doc.id, action="apply", suggestion=suggestion,
        final_title=updated.title, final_correspondent_id=updated.correspondent,
        final_document_type_id=updated.document_type, final_storage_path_id=updated.storage_path,
        final_tag_ids=updated.tags, content_preview=doc.content[:500],
    )
    state.db.log_action(doc.id, action, old_values=_metadata(doc), new_values=_metadata(updated))
    state.db.clear_suggestion(doc.id)


async def _apply_suggestion(doc: Document, suggestion: ClassificationResult, action: str = "apply") -> None:
    if not 0 < suggestion.confidence <= 1:
        raise HTTPException(409, "No successful classification available")
    await _ensure_fresh_taxonomy()
    corr_id = suggestion.correspondent_id or doc.correspondent
    tags = list(suggestion.tag_ids)
    async with state.metadata_lock:
        changed = False
        if suggestion.correspondent_id is None and suggestion.correspondent_name:
            corr = await state.paperless.create_correspondent(suggestion.correspondent_name.strip())
            corr_id = corr.id
            changed = True
        for name in suggestion.new_tags:
            tag = await state.paperless.create_tag(name.strip())
            tags.append(tag.id)
            changed = True
        if changed:
            await _refresh_taxonomy()
    tags = list(dict.fromkeys(t for t in tags if t not in state.taxonomy.inbox_tag_ids))
    resolved = replace(
        suggestion, correspondent_id=corr_id, tag_ids=tags,
        document_type_id=suggestion.document_type_id or doc.document_type,
        storage_path_id=suggestion.storage_path_id or doc.storage_path,
    )
    updated = await state.paperless.update_document(
        doc.id, title=resolved.title, created=resolved.created,
        correspondent=resolved.correspondent_id, document_type=resolved.document_type_id,
        storage_path=resolved.storage_path_id, tags=tags,
    )
    _record_applied(doc, updated, resolved, action)


def _redirect_to_next(request: Request) -> RedirectResponse | JSONResponse:
    """Redirect to next suggestion, audit, or inbox depending on context."""
    ref = request.query_params.get("ref", "")

    if ref == "audit":
        target = "/audit"
    elif ref.startswith("audit:"):
        # ref=audit:/audit?filter_query — return to exact filtered audit URL
        target = ref[len("audit:"):]
        if not target.startswith("/audit"):
            target = "/audit"
    else:
        next_id = state.db.next_suggestion_doc_id()
        target = f"/document/{next_id}" if next_id else "/"

    if request.headers.get("HX-Request"):
        return JSONResponse({}, headers={"HX-Redirect": target})
    return RedirectResponse(target, status_code=303)


@app.post("/document/{doc_id}/skip")
async def skip_document(request: Request, doc_id: int):
    """Remove INBOX tag without changing metadata (mark as reviewed)."""
    await _ensure_fresh_taxonomy()
    suggestion = state.db.get_suggestion(doc_id)
    doc = await state.paperless.get_document(doc_id)
    new_tags = [t for t in doc.tags if t not in state.taxonomy.inbox_tag_ids]
    await state.paperless.update_document(doc_id, tags=new_tags)
    if suggestion:
        state.db.record_feedback(doc_id, action="skip", suggestion=suggestion)
    state.db.log_action(doc_id, "skip")
    state.db.clear_suggestion(doc_id)
    return _redirect_to_next(request)


@app.post("/document/{doc_id}/delete")
async def delete_document(request: Request, doc_id: int):
    """Permanently delete a document from Paperless."""
    # Record feedback before clearing
    suggestion = state.db.get_suggestion(doc_id)
    await state.paperless.delete_document(doc_id)
    if suggestion:
        state.db.record_feedback(doc_id, action="delete", suggestion=suggestion)
    state.db.log_action(doc_id, "delete")
    state.db.clear_suggestion(doc_id)
    return _redirect_to_next(request)


@app.get("/proxy/thumb/{doc_id}")
async def proxy_thumb(doc_id: int):
    """Proxy document thumbnail through paperly (avoids CORS issues)."""
    r = await state.paperless._c.get(f"/api/documents/{doc_id}/thumb/", timeout=10.0)
    r.raise_for_status()
    return StreamingResponse(
        iter([r.content]),
        media_type=r.headers.get("content-type", "image/jpeg"),
    )


@app.get("/proxy/preview/{doc_id}")
async def proxy_preview(doc_id: int):
    """Proxy document preview PDF through paperly."""
    r = await state.paperless._c.get(f"/api/documents/{doc_id}/preview/", timeout=30.0)
    r.raise_for_status()
    return StreamingResponse(
        iter([r.content]),
        media_type=r.headers.get("content-type", "application/pdf"),
    )


@app.get("/stats")
async def stats():
    return await state.paperless.get_statistics()


@app.post("/refresh-taxonomy")
async def refresh_taxonomy():
    await _refresh_taxonomy()
    return {
        "status": "ok",
        "tags": len(state.taxonomy.tags),
        "correspondents": len(state.taxonomy.correspondents),
        "document_types": len(state.taxonomy.document_types),
        "storage_paths": len(state.taxonomy.storage_paths),
    }


# ---------------------------------------------------------------------------
# Batch classification
# ---------------------------------------------------------------------------

async def _get_or_classify(
    doc: Document, *, force: bool = False, generation: int | None = None,
) -> ClassificationResult:
    if generation is None:
        generation = state.suggestion_generation
    if generation != state.suggestion_generation:
        raise ValueError("Settings or taxonomy changed. Please retry classification.")
    cached = state.db.get_suggestion(doc.id)
    if cached and 0 < cached.confidence <= 1 and not force:
        return cached
    state.db.clear_suggestion(doc.id)
    result = await state.classifier.classify(
        doc.content, state.taxonomy, original_title=doc.title,
        filename=doc.original_file_name or "", db=state.db,
    )
    if not 0 < result.confidence <= 1:
        raise ValueError(result.reasoning or "Classification failed")
    if generation != state.suggestion_generation:
        raise ValueError("Settings or taxonomy changed. Please retry classification.")
    state.db.set_suggestion(doc.id, result)
    return result


async def _run_batch(doc_ids: list[int]) -> None:
    """Background task: classify documents with concurrency limit."""
    state.batch_running = True
    state.batch_total = len(doc_ids)
    state.batch_done = 0
    state.batch_errors = 0
    state.batch_start_time = time.time()
    state.batch_current_doc = ""
    state.batch_log = []

    queue = iter(doc_ids)

    async def _worker() -> None:
        for doc_id in queue:
            if state.batch_cancel:
                return
            title = f"Doc #{doc_id}"
            result = None
            try:
                generation = state.suggestion_generation
                doc = await state.paperless.get_document(doc_id)
                title = doc.title or title
                state.batch_current_doc = title
                result = await _get_or_classify(doc, generation=generation)
            except Exception as e:
                logger.error("Batch classify failed for doc %d: %s", doc_id, e)
                state.batch_errors += 1
            state.batch_done += 1
            state.batch_log.append({
                "doc_id": doc_id, "title": title,
                "status": "ok" if result else "error",
                "confidence": round(result.confidence, 2) if result else None,
                "provider": f"{result.provider_name}/{result.provider_model}" if result else "",
            })
            state.batch_log = state.batch_log[-30:]

    workers = [asyncio.create_task(_worker()) for _ in range(min(BATCH_CONCURRENCY, len(doc_ids)))]
    try:
        await asyncio.gather(*workers)
    finally:
        for task in workers:
            if not task.done():
                task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        state.batch_running = False
        state.batch_current_doc = ""
        logger.info("Batch finished: %d/%d done, %d errors",
                    state.batch_done, state.batch_total, state.batch_errors)


@app.post("/batch/start")
async def batch_start(request: Request):
    """Start batch classification for all inbox documents (or remaining unclassified ones)."""
    if state.batch_running:
        return JSONResponse({"status": "already_running", "done": state.batch_done, "total": state.batch_total})

    state.batch_running = True
    state.batch_cancel = False
    try:
        await _ensure_fresh_taxonomy()
        docs = await _all_inbox_documents()
        uncached = []
        for doc in docs:
            cached = state.db.get_suggestion(doc.id)
            if not cached or not 0 < cached.confidence <= 1:
                uncached.append(doc.id)
        if not uncached or state.batch_cancel:
            state.batch_running = False
            return JSONResponse({"status": "cancelled" if state.batch_cancel else "nothing_to_do",
                                 "cached": len(docs) - len(uncached)})
        state.batch_task = asyncio.create_task(_run_batch(uncached))
        return JSONResponse({"status": "started", "total": len(uncached), "already_cached": len(docs) - len(uncached)})
    except BaseException:
        state.batch_running = False
        raise


@app.get("/batch/status")
async def batch_status():
    """Return current batch classification progress with detail."""
    elapsed = time.time() - state.batch_start_time if state.batch_start_time else 0
    avg_per_doc = elapsed / state.batch_done if state.batch_done > 0 else 0
    remaining = state.batch_total - state.batch_done
    eta = avg_per_doc * remaining if avg_per_doc > 0 else 0

    return JSONResponse({
        "running": state.batch_running,
        "total": state.batch_total,
        "done": state.batch_done,
        "errors": state.batch_errors,
        "cancelled": state.batch_cancel,
        "current_doc": state.batch_current_doc,
        "elapsed": round(elapsed, 1),
        "eta": round(eta, 1),
        "provider": f"{state.classifier.provider.name}/{state.classifier.provider.model}",
        "log": state.batch_log[-10:],
    })


@app.post("/batch/cancel")
async def batch_cancel():
    """Cancel a running batch classification."""
    if state.batch_running:
        state.batch_cancel = True
        if state.batch_task and not state.batch_task.done():
            state.batch_task.cancel()
            await asyncio.gather(state.batch_task, return_exceptions=True)
            state.batch_running = False
        return JSONResponse({"status": "cancelling"})
    return JSONResponse({"status": "not_running"})


@app.get("/batch/progress", response_class=HTMLResponse)
async def batch_progress(request: Request):
    """HTMX partial: batch progress bar."""
    elapsed = time.time() - state.batch_start_time if state.batch_start_time else 0
    avg_per_doc = elapsed / state.batch_done if state.batch_done > 0 else 0
    remaining = state.batch_total - state.batch_done
    eta = avg_per_doc * remaining if avg_per_doc > 0 else 0

    return templates.TemplateResponse(
            request,
            "partials/batch_progress.html",
            {
            "running": state.batch_running,
            "total": state.batch_total,
            "done": state.batch_done,
            "errors": state.batch_errors,
            "cancelled": state.batch_cancel,
            "current_doc": state.batch_current_doc,
            "elapsed": round(elapsed, 1),
            "eta": round(eta, 1),
            "provider": f"{state.classifier.provider.name}/{state.classifier.provider.model}",
            "log": state.batch_log[-10:],
        },
    )


# ---------------------------------------------------------------------------
# Batch review
# ---------------------------------------------------------------------------

async def _all_inbox_documents() -> list[Document]:
    all_docs: list[Document] = []
    page = 1
    while True:
        docs, total = await state.paperless.get_inbox_documents(page=page, page_size=100)
        all_docs.extend(docs)
        if not docs or len(all_docs) >= total:
            return all_docs
        page += 1


@app.get("/review", response_class=HTMLResponse)
async def review_list(request: Request, min_conf: float = 0.0, max_conf: float = 1.0, sort: str = "confidence"):
    """Show all cached suggestions for review."""
    await _ensure_fresh_taxonomy()
    all_docs = await _all_inbox_documents()

    # Match with cached suggestions
    items: list[dict] = []
    for doc in all_docs:
        suggestion = state.db.get_suggestion(doc.id)
        if suggestion is None or not 0 < suggestion.confidence <= 1:
            continue
        if suggestion.confidence < min_conf or suggestion.confidence > max_conf:
            continue
        items.append({
            "doc": doc,
            "suggestion": suggestion,
            "correspondent_name": (
                state.taxonomy.correspondent_by_id(suggestion.correspondent_id).name
                if suggestion.correspondent_id and state.taxonomy.correspondent_by_id(suggestion.correspondent_id)
                else suggestion.correspondent_name or "–"
            ),
            "document_type_name": (
                state.taxonomy.document_type_by_id(suggestion.document_type_id).name
                if suggestion.document_type_id and state.taxonomy.document_type_by_id(suggestion.document_type_id)
                else "–"
            ),
            "tag_names": [
                state.taxonomy.tag_by_id(tid).name
                for tid in suggestion.tag_ids
                if state.taxonomy.tag_by_id(tid)
            ],
        })

    # Sort
    if sort == "confidence":
        items.sort(key=lambda x: x["suggestion"].confidence)
    elif sort == "confidence_desc":
        items.sort(key=lambda x: x["suggestion"].confidence, reverse=True)
    elif sort == "title":
        items.sort(key=lambda x: x["suggestion"].title.lower())

    return templates.TemplateResponse(
            request,
            "review.html",
            {
            "items": items,
            "total_inbox": len(all_docs),
            "total_cached": len(items),
            "min_conf": min_conf,
            "max_conf": max_conf,
            "sort": sort,
            "taxonomy": state.taxonomy,
            "auto_apply_threshold": AUTO_APPLY_MIN_CONFIDENCE,
            "review_apply": state.review_apply,
        },
    )


@app.post("/review/apply-all")
async def review_apply_all(request: Request, min_confidence: Annotated[float, Form(ge=0, le=1)] = 0.85):
    """Reserve one background application job before doing any upstream I/O."""
    if not state.review_apply["running"]:
        state.review_apply = {
            "started": True, "running": True, "total": 0, "done": 0,
            "applied": 0, "skipped": 0, "errors": 0, "failures": [], "stopped": False,
        }
        state.review_apply_task = asyncio.create_task(_run_review_apply(min_confidence))
    return await review_apply_progress(request)


@app.get("/review/apply-progress", response_class=HTMLResponse)
async def review_apply_progress(request: Request):
    job = state.review_apply
    headers = {}
    if request.method == "GET" and request.headers.get("HX-Request") and job["started"] and not job["running"]:
        headers["HX-Refresh"] = "true"
    return templates.TemplateResponse(
        request, "partials/apply_all_results.html", {"review_apply": job}, headers=headers,
    )


async def _run_review_apply(min_confidence: float):
    job = state.review_apply
    try:
        await _ensure_fresh_taxonomy()
        generation = state.suggestion_generation
        docs = await _all_inbox_documents()
        eligible = [
            doc.id for doc in docs
            if (suggestion := state.db.get_suggestion(doc.id))
            and 0 < suggestion.confidence <= 1 and suggestion.confidence >= min_confidence
        ]
        job["total"] = len(eligible)
        for doc_id in eligible:
            if generation != state.suggestion_generation:
                job["stopped"] = True
                break
            try:
                suggestion = state.db.get_suggestion(doc_id)
                doc = await state.paperless.get_document(doc_id)
                if (not suggestion or not min_confidence <= suggestion.confidence <= 1
                        or not 0 < suggestion.confidence
                        or not state.taxonomy.inbox_tag_ids.intersection(doc.tags)):
                    job["skipped"] += 1
                    continue
                await _apply_suggestion(doc, suggestion, action="auto_apply")
                job["applied"] += 1
            except Exception as exc:
                job["errors"] += 1
                job["failures"].append({"doc_id": doc_id, "error": type(exc).__name__})
                logger.exception("Auto-apply failed for doc %d (%s)", doc_id, type(exc).__name__)
                if isinstance(exc, httpx.RequestError) or (
                    isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code >= 500
                ):
                    # A timed-out write may still be running in Paperless. Never replay it.
                    job["stopped"] = True
                    break
            finally:
                job["done"] += 1
    except asyncio.CancelledError:
        job["stopped"] = True
        raise
    except Exception as exc:
        job["errors"] += 1
        job["failures"].append({"doc_id": None, "error": type(exc).__name__})
        job["stopped"] = True
        logger.exception("Could not prepare bulk application (%s)", type(exc).__name__)
    finally:
        job["running"] = False


# ---------------------------------------------------------------------------
# Cleanup routes
# ---------------------------------------------------------------------------

@app.get("/cleanup", response_class=HTMLResponse)
async def cleanup_view(request: Request):
    """Show taxonomy cleanup analysis page."""
    from paperly.cleanup import _analyse

    await _refresh_taxonomy()
    taxonomy = state.taxonomy
    merge_actions, delete_actions = _analyse(taxonomy)

    return templates.TemplateResponse(
            request,
            "cleanup.html",
            {
            "merge_actions": merge_actions,
            "delete_actions": delete_actions,
            "taxonomy": taxonomy,
        },
    )


@app.post("/cleanup/execute", response_class=HTMLResponse)
async def cleanup_execute(request: Request):
    """Execute cleanup actions (merges and deletes)."""
    from paperly.cleanup import _analyse, _reassign_and_delete

    await _refresh_taxonomy()
    taxonomy = state.taxonomy
    merge_actions, delete_actions = _analyse(taxonomy)

    results: list[dict] = []
    if merge_actions or delete_actions:
        _invalidate_suggestions()

    for action in merge_actions:
        try:
            await _reassign_and_delete(state.paperless, taxonomy, action)
            results.append({"action": f"Zusammengeführt: {action.remove_name} → {action.keep_name}", "ok": True})
        except Exception as e:
            results.append({"action": f"Fehler bei {action.remove_name}: {e}", "ok": False})

    for action in delete_actions:
        try:
            if action.kind == "document_type":
                await state.paperless.delete_document_type(action.item_id)
            else:
                await state.paperless.delete_correspondent(action.item_id)
            results.append({"action": f"Gelöscht: {action.name}", "ok": True})
        except Exception as e:
            results.append({"action": f"Fehler beim Löschen von {action.name}: {e}", "ok": False})

    await _refresh_taxonomy()

    return templates.TemplateResponse(            request,
            "partials/cleanup_results.html",
            {"results": results},
    )


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

@app.post("/cache/clear")
async def clear_cache(request: Request):
    """Clear all cached suggestions."""
    _invalidate_suggestions()
    referer = request.headers.get("referer", "/")
    return RedirectResponse(referer, status_code=303)


@app.post("/cache/clear/{doc_id}")
async def clear_suggestion_cache(request: Request, doc_id: int):
    """Clear cached suggestion for a single document."""
    state.db.clear_suggestion(doc_id)
    if request.headers.get("HX-Request"):
        return HTMLResponse("")
    referer = request.headers.get("referer", "/review")
    return RedirectResponse(referer, status_code=303)


@app.get("/history")
async def history_view(request: Request):
    """Show processing history."""
    entries = state.db.get_history(limit=100)
    return templates.TemplateResponse(
            request,
            "history.html",
            {"entries": entries, "taxonomy": state.taxonomy},
    )


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def _settings_values() -> dict[str, str]:
    return {
        "ai_provider": state.db.get_setting("ai_provider", os.environ.get("PAPERLY_AI_PROVIDER", "claude")),
        "claude_model": state.db.get_setting("claude_model", os.environ.get("PAPERLY_CLAUDE_MODEL", "claude-haiku-4-5")),
        "ollama_url": state.db.get_setting("ollama_url", os.environ.get("PAPERLY_OLLAMA_URL", "http://localhost:11434")),
        "ollama_model": state.db.get_setting("ollama_model", os.environ.get("PAPERLY_OLLAMA_MODEL", "gemma4:e4b")),
        "openai_api_key": state.db.get_setting("openai_api_key", os.environ.get("OPENAI_API_KEY", "")),
        "openai_base_url": state.db.get_setting("openai_base_url", os.environ.get("PAPERLY_OPENAI_BASE_URL", "http://copilot-gateway:8080/v1")),
        "openai_model": state.db.get_setting("openai_model", os.environ.get("PAPERLY_OPENAI_MODEL", "gpt-4o-mini")),
        "custom_prompt": state.db.get_setting("custom_prompt", ""),
        "paperless_url": state.db.get_setting("paperless_url", os.environ.get("PAPERLESS_URL", "")),
        "paperless_token": state.db.get_setting("paperless_token", os.environ.get("PAPERLESS_TOKEN", "")),
        "paperless_public_url": state.db.get_setting("paperless_public_url", os.environ.get("PAPERLY_PAPERLESS_PUBLIC_URL", "")),
    }


@app.get("/settings", response_class=HTMLResponse)
async def settings_view(request: Request):
    """Settings page — AI provider configuration."""
    return templates.TemplateResponse(
        request,
        "settings.html",
        {
            "settings": _settings_values(),
            "active_provider": state.classifier.provider.name,
            "active_model": state.classifier.provider.model,
        },
    )


@app.post("/settings", response_class=HTMLResponse)
async def settings_save(
    request: Request,
    ai_provider: Annotated[str, Form()],
    claude_model: Annotated[str, Form()] = "claude-haiku-4-5",
    ollama_url: Annotated[str, Form()] = "http://localhost:11434",
    ollama_model: Annotated[str, Form()] = "gemma4:e4b",
    openai_api_key: Annotated[str, Form()] = "",
    openai_base_url: Annotated[str, Form()] = "http://copilot-gateway:8080/v1",
    openai_model: Annotated[str, Form()] = "gpt-4o-mini",
    custom_prompt: Annotated[str, Form()] = "",
    paperless_url: Annotated[str, Form()] = "",
    paperless_token: Annotated[str, Form()] = "",
    paperless_public_url: Annotated[str, Form()] = "",
):
    """Save settings and switch provider."""
    if state.batch_running or state.review_apply["running"]:
        raise HTTPException(409, "Wait for the running batch to finish before changing settings")
    if ai_provider not in {"claude", "ollama", "openai"}:
        raise HTTPException(422, "Unknown AI provider")
    updates = {
        "ai_provider": ai_provider, "claude_model": claude_model.strip(),
        "ollama_url": ollama_url.strip(), "ollama_model": ollama_model.strip(),
        "openai_base_url": openai_base_url.strip(), "openai_model": openai_model.strip(),
        "custom_prompt": custom_prompt.strip(), "paperless_public_url": paperless_public_url.strip(),
    }
    if openai_api_key.strip():
        updates["openai_api_key"] = openai_api_key.strip()
    if paperless_url.strip():
        updates["paperless_url"] = paperless_url.strip()
    if paperless_token.strip():
        updates["paperless_token"] = paperless_token.strip()
    previous = _settings_values()
    current = {**previous, **updates}
    candidate = None
    try:
        if not current["paperless_url"] or not current["paperless_token"]:
            raise ValueError("Paperless URL and token are required")
        if any(current[key] != previous[key] for key in ("paperless_url", "paperless_token")):
            candidate = PaperlessClient(current["paperless_url"], current["paperless_token"])
            await candidate.open()
            taxonomy = await candidate.get_taxonomy()
        provider = _build_provider(state.db, current)
    except Exception:
        if candidate:
            await candidate.close()
        return templates.TemplateResponse(
            request, "settings.html",
            {"settings": current, "active_provider": state.classifier.provider.name,
             "active_model": state.classifier.provider.model,
             "error": "Einstellungen nicht gespeichert. Bitte Verbindung und Zugangsdaten prüfen."},
            status_code=400,
        )
    if state.batch_running or state.review_apply["running"]:
        if candidate:
            await candidate.close()
        raise HTTPException(409, "A batch started while settings were being validated")
    state.db.set_settings(updates)
    state.suggestion_generation += 1
    previous_client = state.paperless
    if candidate:
        state.paperless = candidate
        state.taxonomy = taxonomy
        state.taxonomy_refreshed_at = time.time()
    state.classifier = Classifier(provider, custom_prompt=current["custom_prompt"])
    if any(current[key] != previous[key] for key in ("paperless_url", "paperless_token", "custom_prompt")):
        state.db.clear_all_suggestions()
    if candidate:
        await previous_client.close()
    return templates.TemplateResponse(
        request,
        "settings.html",
        {
            "settings": current,
            "active_provider": state.classifier.provider.name,
            "active_model": state.classifier.provider.model,
            "saved": True,
        },
    )


@app.post("/settings/test-ollama")
async def test_ollama(request: Request):
    """Test Ollama connectivity and return status."""
    # Accept URL from form body or fall back to stored setting
    try:
        body = await request.json()
        url = body.get("url", "").strip()
    except Exception:
        url = ""
    if not url:
        url = state.db.get_setting("ollama_url", os.environ.get("PAPERLY_OLLAMA_URL", "http://localhost:11434"))
    try:
        provider = OllamaProvider(base_url=url)
        info = await provider.test_connection()
        return JSONResponse(info)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=502)


@app.post("/settings/list-openai-models")
async def list_openai_models(request: Request):
    """Fetch model list from an OpenAI-compatible endpoint."""
    try:
        body = await request.json()
        base_url = body.get("base_url", "").strip()
        api_key = body.get("api_key", "").strip()
    except Exception:
        base_url = ""
        api_key = ""
    if not base_url:
        base_url = state.db.get_setting("openai_base_url", os.environ.get("PAPERLY_OPENAI_BASE_URL", "http://copilot-gateway:8080/v1"))
    if not api_key:
        api_key = state.db.get_setting("openai_api_key", os.environ.get("OPENAI_API_KEY", "dummy")) or "dummy"
    try:
        from openai import OpenAI
        import asyncio
        client = OpenAI(api_key=api_key, base_url=base_url)
        loop = asyncio.get_event_loop()
        try:
            models = await loop.run_in_executor(None, lambda: client.models.list())
        finally:
            client.close()
        ids = sorted(set(m.id for m in models.data))
        return JSONResponse({"ok": True, "models": ids})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=502)




def _compute_doc_diffs(doc: Document, suggestion: ClassificationResult) -> list[tuple[str, str, str]]:
    """Compute field-level diffs between current doc metadata and AI suggestion."""
    diffs = []
    if suggestion.title and suggestion.title != doc.title:
        diffs.append(("Titel", doc.title, suggestion.title))
    if suggestion.created and suggestion.created != doc.created[:10]:
        diffs.append(("Datum", doc.created[:10], suggestion.created))
    if suggestion.correspondent_id and doc.correspondent and suggestion.correspondent_id != doc.correspondent:
        curr = state.taxonomy.correspondent_by_id(doc.correspondent)
        sugg = state.taxonomy.correspondent_by_id(suggestion.correspondent_id)
        diffs.append(("Absender", curr.name if curr else "–", sugg.name if sugg else suggestion.correspondent_name or f"ID {suggestion.correspondent_id}"))
    if suggestion.document_type_id and doc.document_type and suggestion.document_type_id != doc.document_type:
        curr = state.taxonomy.document_type_by_id(doc.document_type)
        sugg = state.taxonomy.document_type_by_id(suggestion.document_type_id)
        diffs.append(("Dokumenttyp", curr.name if curr else "–", sugg.name if sugg else f"ID {suggestion.document_type_id}"))
    if suggestion.storage_path_id and doc.storage_path and suggestion.storage_path_id != doc.storage_path:
        curr = state.taxonomy.storage_path_by_id(doc.storage_path)
        sugg = state.taxonomy.storage_path_by_id(suggestion.storage_path_id)
        diffs.append(("Speicherpfad", curr.name if curr else "–", sugg.name if sugg else f"ID {suggestion.storage_path_id}"))
    if suggestion.correspondent_id is None and suggestion.correspondent_name and doc.correspondent:
        curr = state.taxonomy.correspondent_by_id(doc.correspondent)
        if not curr or curr.name.casefold() != suggestion.correspondent_name.strip().casefold():
            diffs.append(("Absender", curr.name if curr else "–", suggestion.correspondent_name))
    current_tags = set(doc.tags) - state.taxonomy.inbox_tag_ids
    suggested_tags = set(suggestion.tag_ids) - state.taxonomy.inbox_tag_ids
    removed_tags = current_tags - suggested_tags  # tags on doc that AI wants to remove
    if removed_tags:  # only flag if AI removes existing tags, not when it just adds new ones
        curr_names = ", ".join(state.taxonomy.tag_by_id(t).name for t in sorted(current_tags) if state.taxonomy.tag_by_id(t)) or "–"
        sugg_names = ", ".join(state.taxonomy.tag_by_id(t).name for t in sorted(suggested_tags) if state.taxonomy.tag_by_id(t)) or "–"
        diffs.append(("Tags", curr_names, sugg_names))
    return diffs


@app.get("/audit", response_class=HTMLResponse)
async def audit_view(
    request: Request,
    page: int = 1,
    correspondent: str = "",
    document_type: str = "",
    storage_path: str = "",
    tag: str = "",
    search: str = "",
    scope: str = "processed",
    show: str = "all",
):
    """Browse all documents with filters for audit/review."""
    await _ensure_fresh_taxonomy()

    # Parse optional int filters (HTML sends "" for unselected)
    correspondent_id = _optional_id(correspondent) if correspondent else None
    document_type_id = _optional_id(document_type) if document_type else None
    storage_path_id = _optional_id(storage_path) if storage_path else None
    tag_id = _optional_id(tag) if tag else None

    if scope not in {"processed", "inbox", "all"} or show not in {"all", "changes", "unchecked"}:
        raise HTTPException(422, "Invalid audit filter")

    docs, total = await state.paperless.get_documents(
        page=page,
        page_size=25,
        correspondent_id=correspondent_id,
        document_type_id=document_type_id,
        storage_path_id=storage_path_id,
        tag_id=tag_id,
        in_inbox=None if scope == "all" else scope == "inbox",
        search=search or None,
    )

    # Pre-compute status for each doc
    doc_ids = [d.id for d in docs]
    confirmed_ids = state.db.get_confirmed_doc_ids(doc_ids)

    # status: "confirmed" | "agree" | "differs" | "unchecked"
    # diffs: list of (field, current, suggested) tuples
    doc_status: dict[int, dict] = {}
    for doc in docs:
        suggestion = state.db.get_suggestion(doc.id)
        if suggestion and not 0 < suggestion.confidence <= 1:
            suggestion = None
        if doc.id in confirmed_ids and not suggestion:
            doc_status[doc.id] = {"status": "confirmed", "diffs": [], "suggestion": None}
        elif suggestion:
            diffs = _compute_doc_diffs(doc, suggestion)
            if diffs:
                doc_status[doc.id] = {"status": "differs", "diffs": diffs, "suggestion": suggestion}
            else:
                doc_status[doc.id] = {"status": "agree", "diffs": [], "suggestion": suggestion}
        else:
            doc_status[doc.id] = {"status": "unchecked", "diffs": [], "suggestion": None}

    # Filter docs if show=changes
    if show == "changes":
        docs = [d for d in docs if doc_status[d.id]["status"] == "differs"]
    elif show == "unchecked":
        docs = [d for d in docs if doc_status[d.id]["status"] == "unchecked"]

    # Count statuses for summary
    status_counts = {"confirmed": 0, "agree": 0, "differs": 0, "unchecked": 0}
    for ds in doc_status.values():
        status_counts[ds["status"]] += 1

    filters = {
        "correspondent": correspondent_id,
        "document_type": document_type_id,
        "storage_path": storage_path_id,
        "tag": tag_id,
        "search": search or "",
        "scope": scope,
        "show": show,
    }

    return templates.TemplateResponse(
        request,
        "audit.html",
        {
            "docs": docs,
            "total": total,
            "page": page,
            "page_size": 25,
            "taxonomy": state.taxonomy,
            "doc_status": doc_status,
            "status_counts": status_counts,
            "filters": filters,
            "filter_query": urlencode({k: v for k, v in filters.items()
                                       if k != "show" and v and (k != "scope" or v != "processed")}),
            "audit_ref": f"audit:{request.url.path}?{request.url.query}",
        },
    )


@app.post("/audit/{doc_id}/confirm")
async def audit_confirm(request: Request, doc_id: int):
    """Confirm current document metadata is correct — feeds learning as accepted."""
    doc = await state.paperless.get_document(doc_id)

    # Build a pseudo-suggestion from the current metadata (as if AI got it right)
    from paperly.classifier import ClassificationResult
    pseudo_suggestion = ClassificationResult(
        title=doc.title,
        created=doc.created,
        correspondent_id=doc.correspondent,
        correspondent_name="",
        tag_ids=doc.tags,
        document_type_id=doc.document_type,
        storage_path_id=doc.storage_path,
        confidence=1.0,
        reasoning="Confirmed by user during audit",
        provider_name="audit",
        provider_model="manual",
    )

    state.db.record_feedback(
        doc_id,
        action="accept",
        suggestion=pseudo_suggestion,
        final_title=doc.title,
        final_correspondent_id=doc.correspondent,
        final_document_type_id=doc.document_type,
        final_storage_path_id=doc.storage_path,
        final_tag_ids=doc.tags,
        content_preview=doc.content[:500] if doc.content else "",
    )
    state.db.clear_suggestion(doc_id)

    return HTMLResponse(
        '<span class="chip chip-green text-xs">✅ Bestätigt — als Trainingsbeispiel gespeichert</span>'
    )


@app.post("/audit/{doc_id}/classify")
async def audit_classify(request: Request, doc_id: int):
    """Classify a document with AI and return diff view."""
    await _ensure_fresh_taxonomy()
    generation = state.suggestion_generation
    doc = await state.paperless.get_document(doc_id)

    try:
        result = await _get_or_classify(doc, force=True, generation=generation)
    except Exception as e:
        return HTMLResponse(
            f'<span class="chip chip-rose text-xs">❌ Fehler: {escape(str(e))}</span>'
        )

    diffs = _compute_doc_diffs(doc, result)

    return templates.TemplateResponse(
        request,
        "partials/audit_diff.html",
        {
            "doc": doc,
            "result": result,
            "diffs": diffs,
            "taxonomy": state.taxonomy,
        },
    )


@app.post("/audit/{doc_id}/apply-suggestion")
async def audit_apply_suggestion(request: Request, doc_id: int):
    """Apply AI suggestion to a document and record feedback."""
    doc = await state.paperless.get_document(doc_id)
    suggestion = state.db.get_suggestion(doc_id)
    if not suggestion:
        return HTMLResponse('<span class="chip chip-rose text-xs">❌ Kein Vorschlag vorhanden</span>')

    await _apply_suggestion(doc, suggestion)

    return HTMLResponse(
        '<span class="chip chip-green text-xs">✅ KI-Vorschlag angewendet</span>'
    )


@app.post("/audit/{doc_id}/create-template")
async def audit_create_template(request: Request, doc_id: int):
    """Create a correction rule from a document's current metadata as a reusable template."""
    doc = await state.paperless.get_document(doc_id)

    # Build human-readable description of the doc's metadata
    corr = state.taxonomy.correspondent_by_id(doc.correspondent) if doc.correspondent else None
    dt = state.taxonomy.document_type_by_id(doc.document_type) if doc.document_type else None
    sp = state.taxonomy.storage_path_by_id(doc.storage_path) if doc.storage_path else None
    tags = [t for tid in doc.tags
            if tid not in state.taxonomy.inbox_tag_ids and (t := state.taxonomy.tag_by_id(tid))]

    # Build prompt text
    parts = []
    scope = corr.name if corr else doc.title[:40]
    if corr:
        parts.append(f'Absender: "{corr.name}" (ID {corr.id})')
    if dt:
        parts.append(f'Dokumenttyp: "{dt.name}" (ID {dt.id})')
    if sp:
        parts.append(f'Speicherpfad: "{sp.name}" (ID {sp.id})')
    if tags:
        tag_parts = ", ".join(f'"{t.name}" (ID {t.id})' for t in tags)
        parts.append(f"Tags: {tag_parts}")

    prompt_text = f'Für Dokumente von "{scope}" verwende immer:\n' + "\n".join(f"- {p}" for p in parts)
    description = f"Vorlage: {scope} → {dt.name if dt else '?'}"

    return templates.TemplateResponse(
        request,
        "partials/audit_template_form.html",
        {
            "doc": doc,
            "description": description,
            "prompt_text": prompt_text,
        },
    )


@app.post("/audit/batch-classify")
async def audit_batch_classify(request: Request):
    """Classify a batch of document IDs and return summary."""
    form = await request.form()
    doc_ids = list(dict.fromkeys(_optional_id(x) for x in form.get("doc_ids", "").split(",") if x.strip()))
    if not doc_ids:
        return JSONResponse({"classified": 0, "diffs": 0})

    classified = 0
    diffs = 0
    errors = 0
    await _ensure_fresh_taxonomy()
    generation = state.suggestion_generation
    for doc_id in doc_ids:
        try:
            doc = await state.paperless.get_document(doc_id)
            result = await _get_or_classify(doc, generation=generation)

            classified += 1
            if _compute_doc_diffs(doc, result):
                diffs += 1
        except Exception as e:
            errors += 1
            logger.error("Audit batch classify failed for doc %d: %s", doc_id, e)

    return JSONResponse({"classified": classified, "diffs": diffs, "errors": errors})


@app.post("/audit/batch-confirm")
async def audit_batch_confirm(request: Request):
    """Confirm all docs on the page where AI agrees (no diffs). Feeds learning."""
    form = await request.form()
    doc_ids = list(dict.fromkeys(_optional_id(x) for x in form.get("doc_ids", "").split(",") if x.strip()))

    confirmed = 0
    errors = 0
    await _ensure_fresh_taxonomy()
    for doc_id in doc_ids:
        try:
            doc = await state.paperless.get_document(doc_id)
            suggestion = state.db.get_suggestion(doc_id)
            if not suggestion or not 0 < suggestion.confidence <= 1:
                continue
            diffs = _compute_doc_diffs(doc, suggestion)
            if diffs:
                continue  # skip docs with differences
            await _apply_suggestion(doc, suggestion)
            confirmed += 1
        except Exception as e:
            errors += 1
            logger.error("Audit batch confirm failed for doc %d: %s", doc_id, e)

    return JSONResponse({"confirmed": confirmed, "errors": errors})


# ---------------------------------------------------------------------------
# Learning dashboard & rules
# ---------------------------------------------------------------------------

@app.get("/learning", response_class=HTMLResponse)
async def learning_dashboard(request: Request):
    """Show learning progress and feedback statistics."""
    stats = state.db.get_feedback_stats()
    corrections = state.db.get_top_corrections(limit=10)

    # Resolve names for corrections
    for c in corrections:
        if c["type"] == "correspondent":
            from_obj = state.taxonomy.correspondent_by_id(c["from_id"])
            to_obj = state.taxonomy.correspondent_by_id(c["to_id"])
            c["from_name"] = from_obj.name if from_obj else f"ID {c['from_id']}"
            c["to_name"] = to_obj.name if to_obj else f"ID {c['to_id']}"
        elif c["type"] == "document_type":
            from_obj = state.taxonomy.document_type_by_id(c["from_id"])
            to_obj = state.taxonomy.document_type_by_id(c["to_id"])
            c["from_name"] = from_obj.name if from_obj else f"ID {c['from_id']}"
            c["to_name"] = to_obj.name if to_obj else f"ID {c['to_id']}"

    rules = state.db.get_all_rules()
    new_patterns = state.db.detect_correction_patterns(min_occurrences=3)

    # Resolve names and generate context-aware prompt text for patterns
    for p in new_patterns:
        if p["rule_type"] == "correspondent":
            from_obj = state.taxonomy.correspondent_by_id(p["from_id"])
            to_obj = state.taxonomy.correspondent_by_id(p["to_id"])
            p["from_name"] = from_obj.name if from_obj else f"ID {p['from_id']}"
            p["to_name"] = to_obj.name if to_obj else f"ID {p['to_id']}"
            label = "Absender"
        else:
            from_obj = state.taxonomy.document_type_by_id(p["from_id"])
            to_obj = state.taxonomy.document_type_by_id(p["to_id"])
            p["from_name"] = from_obj.name if from_obj else f"ID {p['from_id']}"
            p["to_name"] = to_obj.name if to_obj else f"ID {p['to_id']}"
            label = "Dokumenttyp"

        titles = p.get("example_titles", [])
        context_hint = ""
        if titles:
            examples_str = ", ".join(f'"{t}"' for t in titles[:3])
            context_hint = f" Betrifft Dokumente wie: {examples_str}."

        p["suggested_description"] = (
            f"{label}: {p['from_name']} → {p['to_name']} ({p['count']}× korrigiert)"
        )
        p["suggested_prompt"] = (
            f"Wenn du {label} \"{p['from_name']}\" (ID {p['from_id']}) vorschlagen würdest, "
            f"verwende stattdessen \"{p['to_name']}\" (ID {p['to_id']}).{context_hint}"
        )

    return templates.TemplateResponse(
        request,
        "learning.html",
        {
            "stats": stats,
            "corrections": corrections,
            "rules": rules,
            "new_patterns": new_patterns,
            "feedback_count": state.db.feedback_count(),
            "recent_feedback": state.db.get_recent_feedback(limit=20),
            "taxonomy": state.taxonomy,
        },
    )


@app.post("/learning/rules/add")
async def add_rule(
    request: Request,
    rule_type: Annotated[str, Form()] = "general",
    description: Annotated[str, Form()] = "",
    prompt_text: Annotated[str, Form()] = "",
):
    """Add a manual correction rule."""
    if description.strip() and prompt_text.strip():
        state.db.add_rule(rule_type, description.strip(), prompt_text.strip(), auto_generated=False)
        _invalidate_suggestions()
    return RedirectResponse("/learning", status_code=303)


@app.post("/learning/rules/{rule_id}/toggle")
async def toggle_rule(rule_id: int):
    """Toggle a correction rule on/off."""
    state.db.toggle_rule(rule_id)
    _invalidate_suggestions()
    return RedirectResponse("/learning", status_code=303)


@app.post("/learning/rules/{rule_id}/delete")
async def delete_rule(rule_id: int):
    """Delete a correction rule."""
    state.db.delete_rule(rule_id)
    _invalidate_suggestions()
    return RedirectResponse("/learning", status_code=303)


@app.post("/learning/rules/{rule_id}/edit")
async def edit_rule(
    rule_id: int,
    description: Annotated[str, Form()] = "",
    prompt_text: Annotated[str, Form()] = "",
):
    """Edit description and prompt text of a correction rule."""
    if description.strip() and prompt_text.strip():
        state.db.update_rule(rule_id, description.strip(), prompt_text.strip())
        _invalidate_suggestions()
    return RedirectResponse("/learning", status_code=303)


@app.post("/learning/patterns/accept")
async def accept_pattern(
    request: Request,
    source_pattern: Annotated[str, Form()] = "",
    description: Annotated[str, Form()] = "",
    prompt_text: Annotated[str, Form()] = "",
):
    """Accept a detected pattern as a correction rule."""
    if source_pattern and prompt_text:
        state.db.add_rule("auto", description, prompt_text, source_pattern=source_pattern, auto_generated=True)
        _invalidate_suggestions()
    return RedirectResponse("/learning", status_code=303)


@app.post("/learning/feedback/{feedback_id}/delete")
async def delete_feedback(feedback_id: int):
    """Delete a single feedback entry (removes it from learning data)."""
    state.db.delete_feedback(feedback_id)
    return RedirectResponse("/learning", status_code=303)


@app.post("/learning/feedback/clear")
async def clear_all_feedback():
    """Delete all feedback data and auto-generated rules. Manual rules are kept."""
    state.db.clear_all_feedback()
    return RedirectResponse("/learning", status_code=303)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    port = int(os.environ.get("PAPERLY_PORT", 8002))
    host = os.environ.get("PAPERLY_HOST", "0.0.0.0")
    uvicorn.run("paperly.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
