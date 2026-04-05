"""FastAPI application — paperly."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

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
)
from paperly.database import Database
from paperly.paperless import Document, PaperlessClient, Taxonomy

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

BATCH_CONCURRENCY = int(os.environ.get("PAPERLY_BATCH_CONCURRENCY", "1"))
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

    def __init__(self) -> None:
        self.batch_running = False
        self.batch_total = 0
        self.batch_done = 0
        self.batch_errors = 0
        self.batch_cancel = False
        self.batch_start_time = 0.0
        self.batch_current_doc = ""
        self.batch_log = []


state = AppState()


def _build_provider(db: Database) -> BaseProvider:
    """Build the active LLM provider from DB settings + env vars."""
    provider_name = db.get_setting("ai_provider", os.environ.get("PAPERLY_AI_PROVIDER", "claude"))

    if provider_name == "ollama":
        url = db.get_setting("ollama_url", os.environ.get("PAPERLY_OLLAMA_URL", "http://localhost:11434"))
        model = db.get_setting("ollama_model", os.environ.get("PAPERLY_OLLAMA_MODEL", "gemma4:e4b"))
        logger.info("Using Ollama provider: %s model=%s", url, model)
        return OllamaProvider(base_url=url, model=model)
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        model = db.get_setting("claude_model", os.environ.get("PAPERLY_CLAUDE_MODEL", "claude-haiku-4-5"))
        logger.info("Using Anthropic provider: model=%s", model)
        return AnthropicProvider(api_key=api_key, model=model)


@asynccontextmanager
async def lifespan(app: FastAPI):
    paperless_url = os.environ["PAPERLESS_URL"]
    paperless_token = os.environ["PAPERLESS_TOKEN"]

    state.paperless = PaperlessClient(paperless_url, paperless_token)
    await state.paperless.open()

    # SQLite database for persistent cache, history, and settings
    data_dir = os.environ.get("PAPERLY_DATA_DIR", ".")
    state.db = Database(Path(data_dir) / "paperly.db")
    state.db.open()

    # Build classifier with provider from settings
    provider = _build_provider(state.db)
    custom_prompt = state.db.get_setting("custom_prompt", "")
    state.classifier = Classifier(provider, custom_prompt=custom_prompt)

    state.taxonomy = await state.paperless.get_taxonomy()

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
    yield
    state.db.close()
    await state.paperless.close()


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
async def document_view(request: Request, doc_id: int):
    doc = await state.paperless.get_document(doc_id)
    suggestion = state.db.get_suggestion(doc_id)
    return templates.TemplateResponse(
            request,
            "document.html",
            {
            "doc": doc,
            "suggestion": suggestion,
            "taxonomy": state.taxonomy,
            "paperless_url": os.environ.get("PAPERLY_PAPERLESS_PUBLIC_URL", os.environ["PAPERLESS_URL"]),
        },
    )


@app.get("/document/{doc_id}/classify", response_class=HTMLResponse)
async def classify_document(request: Request, doc_id: int):
    """Trigger AI classification and return the suggestion partial (HTMX target)."""
    suggestion = state.db.get_suggestion(doc_id)

    # Don't serve cached error results — reclassify instead
    if suggestion is not None and suggestion.confidence <= 0.0:
        state.db.clear_suggestion(doc_id)
        suggestion = None

    if suggestion is None:
        doc = await state.paperless.get_document(doc_id)
        try:
            suggestion = await state.classifier.classify(
                doc.content, state.taxonomy,
                original_title=doc.title,
                filename=doc.original_file_name or "",
            )
        except Exception as e:
            logger.error("Classification failed for doc %d: %s", doc_id, e)
            return templates.TemplateResponse(
            request,
            "partials/error.html",
            {"error": f"Klassifizierung fehlgeschlagen: {e}", "doc_id": doc_id},
            )
        # Only cache successful results (confidence > 0)
        if suggestion.confidence > 0:
            state.db.set_suggestion(doc_id, suggestion)

    suggestion = state.db.get_suggestion(doc_id)
    doc = await state.paperless.get_document(doc_id)
    return templates.TemplateResponse(
            request,
            "partials/suggestion.html",
            {
            "doc": doc,
            "suggestion": suggestion,
            "taxonomy": state.taxonomy,
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
    # Capture old values for audit log
    doc = await state.paperless.get_document(doc_id)
    old_values = {
        "title": doc.title,
        "created": doc.created,
        "correspondent": doc.correspondent,
        "document_type": doc.document_type,
        "storage_path": doc.storage_path,
        "tags": doc.tags,
    }

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

    # Resolve tags — create new tags if provided
    chosen_tags = [int(t) for t in tag_ids if t]
    if tag_new.strip():
        new_tag = await state.paperless.create_tag(tag_new.strip())
        chosen_tags.append(new_tag.id)
        state.taxonomy = await state.paperless.get_taxonomy()
    # Create AI-suggested new tags
    for new_name in new_tag_names:
        if new_name.strip():
            new_tag = await state.paperless.create_tag(new_name.strip())
            chosen_tags.append(new_tag.id)
    if new_tag_names:
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

    # Audit log
    new_values = {
        "title": title,
        "created": created or None,
        "correspondent": corr_id,
        "document_type": dt_id,
        "storage_path": sp_id,
        "tags": chosen_tags,
    }
    state.db.log_action(doc_id, "apply", old_values=old_values, new_values=new_values)
    state.db.clear_suggestion(doc_id)

    # Redirect to next suggested document, or back to inbox
    next_id = state.db.next_suggestion_doc_id()
    if next_id:
        return RedirectResponse(f"/document/{next_id}", status_code=303)
    return RedirectResponse("/", status_code=303)


def _redirect_to_next(request: Request) -> RedirectResponse | JSONResponse:
    """Redirect to next suggestion or inbox. Supports both regular and HTMX requests."""
    next_id = state.db.next_suggestion_doc_id()
    target = f"/document/{next_id}" if next_id else "/"
    if request.headers.get("HX-Request"):
        return JSONResponse({}, headers={"HX-Redirect": target})
    return RedirectResponse(target, status_code=303)


@app.post("/document/{doc_id}/skip")
async def skip_document(request: Request, doc_id: int):
    """Remove INBOX tag without changing metadata (mark as reviewed)."""
    if state.taxonomy.inbox_tag_id:
        doc = await state.paperless.get_document(doc_id)
        new_tags = [t for t in doc.tags if t != state.taxonomy.inbox_tag_id]
        await state.paperless.update_document(doc_id, tags=new_tags)
    state.db.log_action(doc_id, "skip")
    state.db.clear_suggestion(doc_id)
    return _redirect_to_next(request)


@app.post("/document/{doc_id}/delete")
async def delete_document(request: Request, doc_id: int):
    """Permanently delete a document from Paperless."""
    await state.paperless.delete_document(doc_id)
    state.db.log_action(doc_id, "delete")
    state.db.clear_suggestion(doc_id)
    return _redirect_to_next(request)


@app.get("/proxy/thumb/{doc_id}")
async def proxy_thumb(doc_id: int):
    """Proxy document thumbnail through paperly (avoids CORS issues)."""
    r = await state.paperless._c.get(f"/api/documents/{doc_id}/thumb/", timeout=10.0)
    return StreamingResponse(
        iter([r.content]),
        media_type=r.headers.get("content-type", "image/jpeg"),
    )


@app.get("/proxy/preview/{doc_id}")
async def proxy_preview(doc_id: int):
    """Proxy document preview PDF through paperly."""
    r = await state.paperless._c.get(f"/api/documents/{doc_id}/preview/", timeout=30.0)
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
# Batch classification
# ---------------------------------------------------------------------------

async def _classify_one(doc: Document) -> tuple[int, ClassificationResult | None]:
    """Classify a single document, returning (doc_id, result_or_none)."""
    if state.db.get_suggestion(doc.id) is not None:
        return doc.id, state.db.get_suggestion(doc.id)
    try:
        result = await state.classifier.classify(
            doc.content, state.taxonomy,
            original_title=doc.title,
            filename=doc.original_file_name or "",
        )
        state.db.set_suggestion(doc.id, result)
        return doc.id, result
    except Exception as e:
        logger.error("Batch classify failed for doc %d: %s", doc.id, e)
        return doc.id, None


async def _run_batch(doc_ids: list[int]) -> None:
    """Background task: classify documents with concurrency limit."""
    state.batch_running = True
    state.batch_total = len(doc_ids)
    state.batch_done = 0
    state.batch_errors = 0
    state.batch_cancel = False
    state.batch_start_time = time.time()
    state.batch_current_doc = ""
    state.batch_log = []

    sem = asyncio.Semaphore(BATCH_CONCURRENCY)
    pending_tasks: set[asyncio.Task] = set()

    async def _worker(doc_id: int) -> None:
        async with sem:
            if state.batch_cancel:
                return
            doc = await state.paperless.get_document(doc_id)
            state.batch_current_doc = doc.title or f"Doc #{doc_id}"
            if state.batch_cancel:
                return
            _, result = await _classify_one(doc)

            entry = {
                "doc_id": doc_id,
                "title": doc.title or f"Doc #{doc_id}",
                "status": "ok" if result else "error",
                "confidence": round(result.confidence, 2) if result else None,
                "provider": f"{result.provider_name}/{result.provider_model}" if result else "",
            }
            state.batch_log.append(entry)
            if len(state.batch_log) > 30:
                state.batch_log = state.batch_log[-30:]

            if result is None:
                state.batch_errors += 1
            state.batch_done += 1

    for did in doc_ids:
        if state.batch_cancel:
            break
        task = asyncio.create_task(_worker(did))
        pending_tasks.add(task)
        task.add_done_callback(pending_tasks.discard)

        # Limit how many tasks are created ahead (concurrency + small buffer)
        while len(pending_tasks) >= BATCH_CONCURRENCY + 1:
            if state.batch_cancel:
                break
            await asyncio.sleep(0.2)

    # Cancel any in-flight tasks on abort
    if state.batch_cancel and pending_tasks:
        for t in pending_tasks:
            t.cancel()

    # Wait for remaining in-flight tasks
    if pending_tasks:
        await asyncio.gather(*pending_tasks, return_exceptions=True)

    state.batch_running = False
    state.batch_current_doc = ""
    cancelled = state.batch_cancel
    logger.info(
        "Batch %s: %d/%d done, %d errors",
        "cancelled" if cancelled else "complete",
        state.batch_done, state.batch_total, state.batch_errors,
    )


@app.post("/batch/start")
async def batch_start(request: Request):
    """Start batch classification for all inbox documents (or remaining unclassified ones)."""
    if state.batch_running:
        return JSONResponse({"status": "already_running", "done": state.batch_done, "total": state.batch_total})

    if not state.taxonomy.inbox_tag_id:
        raise HTTPException(500, "INBOX tag not found")

    # Fetch all inbox doc IDs (paginate through all pages)
    all_doc_ids: list[int] = []
    page = 1
    while True:
        docs, total = await state.paperless.get_inbox_documents(
            state.taxonomy.inbox_tag_id, page=page, page_size=100
        )
        all_doc_ids.extend(d.id for d in docs)
        if page * 100 >= total:
            break
        page += 1

    # Filter out already-cached
    uncached = [did for did in all_doc_ids if state.db.get_suggestion(did) is None]

    if not uncached:
        return JSONResponse({"status": "nothing_to_do", "cached": len(all_doc_ids)})

    asyncio.create_task(_run_batch(uncached))
    return JSONResponse({"status": "started", "total": len(uncached), "already_cached": len(all_doc_ids) - len(uncached)})


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

@app.get("/review", response_class=HTMLResponse)
async def review_list(request: Request, min_conf: float = 0.0, max_conf: float = 1.0, sort: str = "confidence"):
    """Show all cached suggestions for review."""
    if not state.taxonomy.inbox_tag_id:
        raise HTTPException(500, "INBOX tag not found")

    # Get all inbox docs (paginate)
    all_docs: list[Document] = []
    page = 1
    while True:
        docs, total = await state.paperless.get_inbox_documents(
            state.taxonomy.inbox_tag_id, page=page, page_size=100
        )
        all_docs.extend(docs)
        if page * 100 >= total:
            break
        page += 1

    # Match with cached suggestions
    items: list[dict] = []
    for doc in all_docs:
        suggestion = state.db.get_suggestion(doc.id)
        if suggestion is None:
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
        },
    )


@app.post("/review/apply-all")
async def review_apply_all(request: Request, min_confidence: Annotated[float, Form()] = 0.85):
    """Apply all suggestions above the confidence threshold."""
    if not state.taxonomy.inbox_tag_id:
        raise HTTPException(500, "INBOX tag not found")

    # Get all inbox docs
    all_docs: list[Document] = []
    page = 1
    while True:
        docs, total = await state.paperless.get_inbox_documents(
            state.taxonomy.inbox_tag_id, page=page, page_size=100
        )
        all_docs.extend(docs)
        if page * 100 >= total:
            break
        page += 1

    applied = 0
    skipped = 0
    errors = 0

    for doc in all_docs:
        suggestion = state.db.get_suggestion(doc.id)
        if suggestion is None or suggestion.confidence < min_confidence:
            skipped += 1
            continue

        try:
            # Build tag list without INBOX
            tags = [tid for tid in suggestion.tag_ids if tid != state.taxonomy.inbox_tag_id]

            await state.paperless.update_document(
                doc.id,
                title=suggestion.title,
                created=suggestion.created,
                correspondent=suggestion.correspondent_id,
                document_type=suggestion.document_type_id,
                tags=tags,
            )
            state.db.log_action(
                doc.id, "auto_apply",
                old_values={"title": doc.title, "tags": doc.tags},
                new_values={
                    "title": suggestion.title,
                    "correspondent": suggestion.correspondent_id,
                    "document_type": suggestion.document_type_id,
                    "tags": tags,
                    "confidence": suggestion.confidence,
                },
            )
            state.db.clear_suggestion(doc.id)
            applied += 1
        except Exception as e:
            logger.error("Auto-apply failed for doc %d: %s", doc.id, e)
            errors += 1

    return templates.TemplateResponse(
            request,
            "partials/apply_all_results.html",
            {"applied": applied, "skipped": skipped, "errors": errors},
    )


# ---------------------------------------------------------------------------
# Cleanup routes
# ---------------------------------------------------------------------------

@app.get("/cleanup", response_class=HTMLResponse)
async def cleanup_view(request: Request):
    """Show taxonomy cleanup analysis page."""
    from paperly.cleanup import _analyse

    taxonomy = await state.paperless.get_taxonomy()
    state.taxonomy = taxonomy
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

    taxonomy = await state.paperless.get_taxonomy()
    merge_actions, delete_actions = _analyse(taxonomy)

    results: list[dict] = []

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

    state.taxonomy = await state.paperless.get_taxonomy()

    return templates.TemplateResponse(
            request,
            "partials/cleanup_results.html",
            {"results": results},
    )


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

@app.post("/cache/clear")
async def clear_cache(request: Request):
    """Clear all cached suggestions."""
    state.db.clear_all_suggestions()
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

@app.get("/settings", response_class=HTMLResponse)
async def settings_view(request: Request):
    """Settings page — AI provider configuration."""
    current = {
        "ai_provider": state.db.get_setting("ai_provider", os.environ.get("PAPERLY_AI_PROVIDER", "claude")),
        "claude_model": state.db.get_setting("claude_model", os.environ.get("PAPERLY_CLAUDE_MODEL", "claude-haiku-4-5")),
        "ollama_url": state.db.get_setting("ollama_url", os.environ.get("PAPERLY_OLLAMA_URL", "http://localhost:11434")),
        "ollama_model": state.db.get_setting("ollama_model", os.environ.get("PAPERLY_OLLAMA_MODEL", "gemma4:e4b")),
        "custom_prompt": state.db.get_setting("custom_prompt", ""),
    }
    return templates.TemplateResponse(
        request,
        "settings.html",
        {
            "settings": current,
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
    custom_prompt: Annotated[str, Form()] = "",
):
    """Save settings and switch provider."""
    state.db.set_setting("ai_provider", ai_provider)
    state.db.set_setting("claude_model", claude_model)
    state.db.set_setting("ollama_url", ollama_url)
    state.db.set_setting("ollama_model", ollama_model)
    state.db.set_setting("custom_prompt", custom_prompt.strip())

    # Rebuild provider at runtime
    provider = _build_provider(state.db)
    state.classifier.provider = provider
    state.classifier.custom_prompt = custom_prompt.strip()

    current = {
        "ai_provider": ai_provider,
        "claude_model": claude_model,
        "ollama_url": ollama_url,
        "ollama_model": ollama_model,
        "custom_prompt": custom_prompt.strip(),
    }
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    port = int(os.environ.get("PAPERLY_PORT", 8002))
    host = os.environ.get("PAPERLY_HOST", "0.0.0.0")
    uvicorn.run("paperly.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
