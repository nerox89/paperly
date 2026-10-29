import asyncio
import copy
import importlib
from importlib.util import find_spec
import json
import shutil
import socket
import subprocess
import unittest
from contextlib import asynccontextmanager
from unittest.mock import patch

import httpx
import uvicorn

from paperly.classifier import BaseProvider, Classifier
from paperly.cleanup import MergeAction, _reassign_and_delete
from paperly.database import Database
from paperly.paperless import PaperlessClient

web = importlib.import_module("paperly.app")


class FixtureProvider(BaseProvider):
    name = "fixture"
    model = "synthetic"

    def __init__(self, response=None, error=None):
        self.response = response or {
            "title": "Synthetic invoice",
            "created": "2024-03-15",
            "correspondent_id": None,
            "correspondent_name": "New sender",
            "document_type_id": None,
            "storage_path_id": None,
            "tag_ids": [2],
            "new_tags": ["New tag"],
            "confidence": 0.95,
            "reasoning": "Synthetic fixture",
        }
        self.error = error

    async def generate(self, system, user_message):
        if self.error:
            raise self.error
        return copy.deepcopy(self.response)


class PaperlessFixture:
    def __init__(self):
        self.docs = {
            1: {
                "id": 1, "title": "Original", "content": "Synthetic OCR",
                "created": "2020-01-01", "correspondent": 10,
                "document_type": 20, "storage_path": 30, "tags": [1, 2],
                "archive_serial_number": None, "original_file_name": "fixture.pdf",
            },
        }
        self.collections = {
            "tags": [
                {"id": 1, "name": "INBOX", "is_inbox_tag": True, "document_count": 1},
                {"id": 2, "name": "Existing", "is_inbox_tag": False, "document_count": 1},
            ],
            "correspondents": [{"id": 10, "name": "Existing sender", "document_count": 1}],
            "document_types": [{"id": 20, "name": "Invoice", "document_count": 1}],
            "storage_paths": [{"id": 30, "name": "Archive", "document_count": 1}],
        }
        self.requests = []
        self.fail_docs = set()
        self.page_limit = 1000

    def __call__(self, request):
        self.requests.append(request)
        parts = request.url.path.strip("/").split("/")
        params = request.url.params
        resource = parts[1]
        if resource == "statistics":
            inbox_ids = {t["id"] for t in self.collections["tags"] if t.get("is_inbox_tag")}
            return httpx.Response(200, json={
                "documents_total": len(self.docs),
                "documents_inbox": sum(bool(inbox_ids.intersection(d["tags"])) for d in self.docs.values()),
            })
        if resource == "documents":
            if len(parts) > 2:
                doc_id = int(parts[2])
                if doc_id in self.fail_docs or doc_id not in self.docs:
                    return httpx.Response(404, json={"detail": "Not found"})
                if len(parts) > 3:
                    return httpx.Response(404, json={"detail": "No preview"})
                if request.method == "PATCH":
                    self.docs[doc_id].update(json.loads(request.content))
                if request.method == "DELETE":
                    del self.docs[doc_id]
                    return httpx.Response(204)
                return httpx.Response(200, json=self.docs[doc_id])
            rows = list(self.docs.values())
            if "is_in_inbox" in params:
                inbox_ids = {t["id"] for t in self.collections["tags"] if t.get("is_inbox_tag")}
                wanted = params["is_in_inbox"].lower() == "true"
                rows = [d for d in rows if bool(inbox_ids.intersection(d["tags"])) == wanted]
            for param, field in (
                ("correspondent__id", "correspondent"),
                ("document_type__id", "document_type"),
            ):
                if param in params:
                    rows = [d for d in rows if d[field] == int(params[param])]
            if "tags__id__all" in params:
                rows = [d for d in rows if int(params["tags__id__all"]) in d["tags"]]
            if "tags__id__none" in params:
                rows = [d for d in rows if int(params["tags__id__none"]) not in d["tags"]]
        else:
            rows = self.collections[resource]
            if request.method == "POST":
                item = json.loads(request.content)
                item["id"] = max((x["id"] for x in rows), default=0) + 1
                rows.append(item)
                return httpx.Response(201, json=item)
            if request.method == "DELETE":
                item_id = int(parts[2])
                self.collections[resource] = [x for x in rows if x["id"] != item_id]
                return httpx.Response(204)
            if "name__iexact" in params:
                rows = [x for x in rows if x["name"].casefold() == params["name__iexact"].casefold()]
        size = min(int(params.get("page_size", 25)), self.page_limit)
        page = int(params.get("page", 1))
        end = page * size
        next_url = str(request.url.copy_set_param("page", page + 1)) if end < len(rows) else None
        return httpx.Response(200, json={
            "count": len(rows), "next": next_url, "previous": None,
            "results": rows[(page - 1) * size:end],
        })


@asynccontextmanager
async def browser_page():
    from playwright.async_api import async_playwright

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    origin = f"http://127.0.0.1:{sock.getsockname()[1]}"
    server = uvicorn.Server(uvicorn.Config(web.app, lifespan="off", log_level="error"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        while not server.started:
            if task.done():
                await task
                raise RuntimeError("Test server did not start")
            await asyncio.sleep(0.01)
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            try:
                page = await browser.new_page(viewport={"width": 1440, "height": 1000})
                await page.route("**/*", lambda route: route.continue_()
                                 if route.request.url.startswith(origin + "/") else route.abort())
                yield page, origin
            finally:
                await browser.close()
    finally:
        server.should_exit = True
        await task
        sock.close()


class RegressionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.backend = PaperlessFixture()
        self.paperless = PaperlessClient("http://paperless.test", "synthetic-token")
        self.paperless._http = httpx.AsyncClient(
            base_url="http://paperless.test",
            transport=httpx.MockTransport(self.backend),
        )
        self.db = Database(":memory:")
        self.db.open()
        self.state = web.AppState()
        self.state.paperless = self.paperless
        self.state.db = self.db
        self.state.taxonomy = await self.paperless.get_taxonomy()
        self.state.taxonomy_refreshed_at = web.time.time()
        self.state.classifier = Classifier(FixtureProvider())
        self.state_patch = patch.object(web, "state", self.state)
        self.state_patch.start()
        self.client = httpx.AsyncClient(
            base_url="http://paperly.test",
            transport=httpx.ASGITransport(app=web.app, raise_app_exceptions=False),
        )

    async def asyncTearDown(self):
        task = getattr(self.state, "batch_task", None)
        if task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await self.client.aclose()
        if self.state.paperless is not self.paperless:
            await self.state.paperless.close()
        await self.paperless.close()
        self.db.close()
        self.state_patch.stop()

    async def cache_suggestion(self, doc_id=1, **overrides):
        result = await self.state.classifier.classify("Synthetic OCR", self.state.taxonomy)
        for key, value in overrides.items():
            setattr(result, key, value)
        self.db.set_suggestion(doc_id, result)
        return result

    async def test_native_inbox_includes_all_flagged_tags_not_name_matches(self):
        self.backend.collections["tags"] = [
            {"id": 1, "name": "Eingang", "is_inbox_tag": True},
            {"id": 2, "name": "Review", "is_inbox_tag": True},
            {"id": 3, "name": "INBOX", "is_inbox_tag": False},
        ]
        for doc_id in (2, 3):
            self.backend.docs[doc_id] = {**self.backend.docs[1], "id": doc_id, "tags": [doc_id]}
        self.backend.docs[1]["tags"] = [1]
        self.state.taxonomy = await self.paperless.get_taxonomy()
        response = await self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn('href="/document/1"', response.text)
        self.assertIn('href="/document/2"', response.text)
        self.assertNotIn('href="/document/3"', response.text)
        self.assertEqual(set(self.state.taxonomy.inbox_tag_ids), {1, 2})

    async def test_no_inbox_tags_is_empty_not_server_error(self):
        self.backend.collections["tags"][0]["is_inbox_tag"] = False
        self.backend.collections["tags"][0]["name"] = "Regular"
        self.state.taxonomy = await self.paperless.get_taxonomy()
        response = await self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Inbox ist leer", response.text)

    async def test_index_does_not_report_unstarted_batch_as_completed(self):
        response = await self.client.get("/")
        self.assertNotIn("Klassifizierung abgeschlossen", response.text)

    async def test_empty_audit_status_page_keeps_pagination(self):
        for doc_id in range(2, 27):
            self.backend.docs[doc_id] = {**self.backend.docs[1], "id": doc_id}
        response = await self.client.get("/audit?scope=all&show=changes&page=2")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Seite 2 / 2", response.text)

    async def test_audit_pagination_keeps_all_scope(self):
        from html.parser import HTMLParser

        class Links(HTMLParser):
            def __init__(self):
                super().__init__()
                self.hrefs = []

            def handle_starttag(self, tag, attrs):
                if tag == "a":
                    self.hrefs.extend(value for key, value in attrs if key == "href")

        for doc_id in range(2, 27):
            self.backend.docs[doc_id] = {**self.backend.docs[1], "id": doc_id}
        response = await self.client.get("/audit?scope=all")
        links = Links()
        links.feed(response.text)
        next_page = next(url for url in links.hrefs if "page=2" in url)
        response = await self.client.get(next_page)
        self.assertIn('value="all" selected', response.text)
        self.assertIn("Original", response.text)

    async def test_taxonomy_follows_next_pages(self):
        self.backend.page_limit = 1
        taxonomy = await self.paperless.get_taxonomy()
        self.assertEqual([t.id for t in taxonomy.tags], [1, 2])

    async def test_taxonomy_raises_upstream_status(self):
        await self.paperless.close()
        self.paperless._http = httpx.AsyncClient(
            base_url="http://paperless.test",
            transport=httpx.MockTransport(lambda request: httpx.Response(401, json={"detail": "Invalid token"})),
        )
        with self.assertRaises(httpx.HTTPStatusError):
            await self.paperless.get_taxonomy()

    async def test_explicit_none_clears_metadata_but_omitted_fields_remain(self):
        await self.paperless.update_document(1, correspondent=None, document_type=None)
        self.assertIsNone(self.backend.docs[1]["correspondent"])
        self.assertIsNone(self.backend.docs[1]["document_type"])
        self.assertEqual(self.backend.docs[1]["storage_path"], 30)

    async def test_classification_failure_renders_retry_not_500(self):
        self.state.classifier = Classifier(FixtureProvider(error=RuntimeError("Model unavailable")))
        response = await self.client.get("/document/1/classify")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Erneut versuchen", response.text)
        self.assertIsNone(self.db.get_suggestion(1))

    async def test_failed_result_does_not_enter_batch_cache(self):
        self.state.classifier = Classifier(FixtureProvider(error=RuntimeError("Unavailable")))
        await web._run_batch([1])
        self.assertEqual(self.state.batch_done, 1)
        self.assertEqual(self.state.batch_errors, 1)
        self.assertIsNone(self.db.get_suggestion(1))

    async def test_batch_fetch_failure_is_counted_and_remaining_docs_continue(self):
        self.backend.fail_docs.add(99)
        await web._run_batch([99, 1])
        self.assertEqual(self.state.batch_done, 2)
        self.assertEqual(self.state.batch_errors, 1)
        self.assertFalse(self.state.batch_running)
        self.assertIsNotNone(self.db.get_suggestion(1))

    async def test_batch_cancel_before_worker_starts_is_not_reset(self):
        self.state.batch_cancel = True
        await web._run_batch([1])
        self.assertEqual(self.state.batch_done, 0)
        self.assertIsNone(self.db.get_suggestion(1))

    async def test_batch_start_reserves_job_before_document_fetch(self):
        entered, release = asyncio.Event(), asyncio.Event()
        original = self.paperless.get_inbox_documents

        async def blocked(*args, **kwargs):
            entered.set()
            await release.wait()
            return await original(*args, **kwargs)

        with patch.object(self.paperless, "get_inbox_documents", side_effect=blocked):
            first = asyncio.create_task(self.client.post("/batch/start"))
            await entered.wait()
            second = asyncio.create_task(self.client.post("/batch/start"))
            await asyncio.sleep(0)
            reserved = self.state.batch_running
            release.set()
            responses = await asyncio.gather(first, second)
        if getattr(self.state, "batch_task", None):
            await self.state.batch_task
        else:
            await asyncio.sleep(0.1)
        self.assertTrue(reserved)
        self.assertEqual(sorted(r.json()["status"] for r in responses), ["already_running", "started"])

    async def test_bulk_apply_resolves_new_metadata_and_records_actual_values(self):
        await self.cache_suggestion()
        response = await self.client.post("/review/apply-all", data={"min_confidence": "0.85"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.backend.docs[1]["correspondent"], 11)
        self.assertEqual(self.backend.docs[1]["tags"], [2, 3])
        self.assertEqual(self.backend.docs[1]["created"], "2024-03-15")
        self.assertEqual(self.backend.docs[1]["storage_path"], 30)
        feedback = self.db.get_recent_feedback()[0]
        self.assertFalse(feedback["correspondent_changed"])
        self.assertFalse(feedback["tags_changed"])

    async def test_audit_apply_uses_same_complete_metadata_as_review(self):
        await self.cache_suggestion()
        response = await self.client.post("/audit/1/apply-suggestion")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.backend.docs[1]["created"], "2024-03-15")
        self.assertEqual(self.backend.docs[1]["correspondent"], 11)
        self.assertEqual(self.backend.docs[1]["tags"], [2, 3])
        self.assertEqual(self.db.get_history()[0]["action"], "apply")

    async def test_failed_suggestion_cannot_be_applied_by_audit_or_bulk(self):
        await self.cache_suggestion(confidence=0.0, title="Failed result", tag_ids=[])
        original = copy.deepcopy(self.backend.docs[1])
        response = await self.client.post("/audit/1/apply-suggestion")
        self.assertIn(response.status_code, (200, 409))
        self.assertEqual(self.backend.docs[1], original)
        await self.client.post("/review/apply-all", data={"min_confidence": "0"})
        self.assertEqual(self.backend.docs[1], original)

    async def test_bulk_confidence_out_of_range_is_rejected(self):
        response = await self.client.post("/review/apply-all", data={"min_confidence": "-1"})
        self.assertEqual(response.status_code, 422)

    async def test_apply_returns_to_encoded_audit_context(self):
        await self.cache_suggestion()
        response = await self.client.post(
            "/document/1/apply", params={"ref": "audit:/audit?scope=all&search=A&B"},
            data={"title": "Edited", "correspondent_id": "10", "tag_ids": "2"},
        )
        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/audit?scope=all&search=A&B")

    async def test_manual_apply_can_clear_metadata(self):
        response = await self.client.post("/document/1/apply", data={"title": "Edited"})
        self.assertEqual(response.status_code, 303)
        self.assertIsNone(self.backend.docs[1]["correspondent"])
        self.assertIsNone(self.backend.docs[1]["document_type"])
        self.assertIsNone(self.backend.docs[1]["storage_path"])

    async def test_audit_confirmation_discards_rejected_suggestion(self):
        await self.cache_suggestion()
        response = await self.client.post("/audit/1/confirm")
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(self.db.get_suggestion(1))
        response = await self.client.get("/audit?scope=all")
        self.assertIn("Bestätigt", response.text)
        self.assertNotIn("Synthetic invoice", response.text)

    async def test_batch_confirmation_does_not_silently_overwrite_conflicting_inbox(self):
        await self.cache_suggestion(title="Conflicting title")
        original = copy.deepcopy(self.backend.docs[1])
        response = await self.client.post("/audit/batch-confirm", data={"doc_ids": "1"})
        self.assertEqual(response.json()["confirmed"], 0)
        self.assertEqual(self.backend.docs[1], original)

    async def test_batch_and_single_audit_use_same_diff_definition(self):
        await self.cache_suggestion(
            title="Original", created="2020-01-01",
            correspondent_id=10, correspondent_name="Existing sender",
            document_type_id=20, new_tags=[],
        )
        response = await self.client.post("/audit/batch-classify", data={"doc_ids": "1"})
        self.assertEqual(response.json()["diffs"], 0)

    async def test_audit_date_change_is_reviewed_before_batch_confirmation(self):
        await self.cache_suggestion(
            title="Original", created="2024-03-15",
            correspondent_id=10, correspondent_name="Existing sender",
            document_type_id=20, new_tags=[],
        )
        response = await self.client.post("/audit/batch-confirm", data={"doc_ids": "1"})
        self.assertEqual(response.json()["confirmed"], 0)
        self.assertEqual(self.backend.docs[1]["created"], "2020-01-01")

    @unittest.skipUnless(shutil.which("node"), "Node is needed for the keyboard regression")
    async def test_one_keyboard_skip_sends_one_action(self):
        await self.cache_suggestion()
        response = await self.client.get("/document/1")
        script = r"""
const vm = require('node:vm');
const fs = require('node:fs');
const html = fs.readFileSync(0, 'utf8');
const listeners = [];
let skips = 0;
const context = {
  tailwind: {},
  document: {
    addEventListener: (name, fn) => { if (name === 'keydown') listeners.push(fn); },
    getElementById: id => id === 'skip-btn' ? {click: () => skips++} : {value: '', style: {}},
    body: {addEventListener: () => {}}
  },
  window: {addEventListener: () => {}, matchMedia: () => ({matches: false}), location: {}}
};
vm.createContext(context);
for (const match of html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/g)) {
  if (match[1].trim()) vm.runInContext(match[1], context);
}
for (const listener of listeners) {
  listener({key: 's', target: {tagName: 'DIV'}, preventDefault: () => {}});
}
console.log(skips);
"""
        result = subprocess.run(["node", "-e", script], input=response.text, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "1")

    async def test_failed_skip_does_not_record_successful_feedback(self):
        await self.cache_suggestion()
        self.backend.fail_docs.add(1)
        await self.client.post("/document/1/skip")
        self.assertEqual(self.db.feedback_count(), 0)
        self.assertIsNotNone(self.db.get_suggestion(1))

    async def test_proxy_preserves_missing_document_status(self):
        response = await self.client.get("/proxy/preview/99")
        self.assertEqual(response.status_code, 404)

    async def test_paperless_outage_is_inline_not_unhandled_500(self):
        async def unavailable(*args, **kwargs):
            raise httpx.ConnectError("offline")
        with patch.object(self.paperless, "get_inbox_documents", side_effect=unavailable):
            response = await self.client.get("/")
        self.assertEqual(response.status_code, 502)
        self.assertIn("Paperless", response.text)

    async def test_invalid_settings_preserve_working_connection_and_database(self):
        self.db.set_setting("ai_provider", "fixture")
        original_http = self.paperless._http

        async def open_failed(client):
            client._http = httpx.AsyncClient(
                base_url="http://invalid.test",
                transport=httpx.MockTransport(lambda r: httpx.Response(401, json={"detail": "Invalid"})),
            )

        with patch.object(PaperlessClient, "open", open_failed):
            response = await self.client.post("/settings", data={
                "ai_provider": "openai", "paperless_url": "http://invalid.test",
                "paperless_token": "invalid", "openai_model": "synthetic",
            })
        self.assertEqual(response.status_code, 400)
        self.assertIs(self.state.paperless, self.paperless)
        self.assertFalse(original_http.is_closed)
        self.assertEqual(self.db.get_setting("ai_provider"), "fixture")
        self.assertIsNone(self.db.get_setting("paperless_url"))

    async def test_settings_cannot_replace_clients_during_batch(self):
        self.state.batch_running = True
        response = await self.client.post("/settings", data={"ai_provider": "openai"})
        self.assertEqual(response.status_code, 409)
        self.assertIsNone(self.db.get_setting("ai_provider"))

    async def test_model_change_keeps_connection_and_valid_review_cache(self):
        self.db.set_settings({
            "paperless_url": "http://paperless.test", "paperless_token": "synthetic-token",
            "ai_provider": "openai", "openai_model": "old-model",
        })
        await self.cache_suggestion()
        original_http = self.paperless._http
        response = await self.client.post("/settings", data={
            "ai_provider": "openai", "openai_model": "replacement-model",
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.db.get_setting("openai_model"), "replacement-model")
        self.assertIs(self.paperless._http, original_http)
        self.assertFalse(original_http.is_closed)
        self.assertIsNotNone(self.db.get_suggestion(1))

    async def test_classification_cannot_cache_across_connection_changes(self):
        self.db.set_settings({
            "paperless_url": "http://paperless.test", "paperless_token": "synthetic-token",
        })
        entered, release = asyncio.Event(), asyncio.Event()
        provider = self.state.classifier.provider
        generate = provider.generate

        async def delayed(*args):
            entered.set()
            await release.wait()
            return await generate(*args)

        other = PaperlessFixture()
        other.docs[1]["title"] = "Unrelated document on server B"

        async def open_other(client):
            client._http = httpx.AsyncClient(
                base_url="http://server-b.test", transport=httpx.MockTransport(other),
            )

        with patch.object(provider, "generate", side_effect=delayed):
            classification = asyncio.create_task(self.client.get("/document/1/classify"))
            await entered.wait()
            with patch.object(PaperlessClient, "open", open_other):
                response = await self.client.post("/settings", data={
                    "ai_provider": "ollama", "paperless_url": "http://server-b.test",
                    "paperless_token": "server-b-token",
                })
            self.assertEqual(response.status_code, 200)
            release.set()
            await classification
        self.assertIsNone(self.db.get_suggestion(1))
        await self.client.post("/review/apply-all")
        self.assertEqual(other.docs[1]["title"], "Unrelated document on server B")

    async def test_cleanup_invalidates_suggestions_referencing_merged_entities(self):
        self.backend.collections["document_types"] = [
            {"id": 20, "name": "Rechnung", "document_count": 10},
            {"id": 21, "name": "Rechnungen", "document_count": 1},
        ]
        self.backend.docs[1]["document_type"] = 21
        self.state.taxonomy = await self.paperless.get_taxonomy()
        await self.cache_suggestion(document_type_id=21)
        response = await self.client.post("/cleanup/execute")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.backend.docs[1]["document_type"], 20)
        self.assertIsNone(self.db.get_suggestion(1))

    async def test_cleanup_reassigns_all_pages_before_deleting_source(self):
        self.backend.page_limit = 1
        self.backend.collections["document_types"].append({"id": 21, "name": "Invoices", "document_count": 2})
        for doc_id in (1, 2):
            self.backend.docs[doc_id] = {**self.backend.docs[1], "id": doc_id, "document_type": 21}
        action = MergeAction("document_type", 20, "Invoice", 21, "Invoices", 2)
        await _reassign_and_delete(self.paperless, self.state.taxonomy, action)
        self.assertEqual([d["document_type"] for d in self.backend.docs.values()], [20, 20])
        self.assertNotIn(21, [d["id"] for d in self.backend.collections["document_types"]])

    async def test_invalid_classification_output_is_safe_failure(self):
        cases = [
            {"confidence": 0.95},
            {"title": "", "confidence": 0.95},
            {"title": "Unsafe", "confidence": 2},
            {"title": "Unsafe", "confidence": "nan"},
            {"title": "Unsafe", "confidence": 0.95, "tag_ids": "2"},
            {"title": "Unsafe", "confidence": 0.95, "new_tags": "New tag"},
        ]
        for output in cases:
            with self.subTest(output=output):
                result = await Classifier(FixtureProvider(output)).classify(
                    "Synthetic OCR", self.state.taxonomy, original_title="Original",
                )
                self.assertEqual(result.confidence, 0.0)
                self.assertEqual(result.title, "Original")

    async def test_invalid_calendar_date_is_not_sent_to_paperless(self):
        output = {"title": "Valid", "confidence": 0.9, "created": "2024-02-31"}
        result = await Classifier(FixtureProvider(output)).classify("Synthetic OCR", self.state.taxonomy)
        self.assertIsNone(result.created)

    async def test_public_paperless_url_environment_is_used_for_document_links(self):
        with patch.dict(web.os.environ, {"PAPERLY_PAPERLESS_PUBLIC_URL": "https://paperless.example.test"}):
            response = await self.client.get("/document/1")
        self.assertIn('href="https://paperless.example.test/documents/1"', response.text)

    @unittest.skipUnless(find_spec("playwright"), "Install the test extra and Chromium for browser checks")
    async def test_ui_remains_visible_and_styled_without_external_cdns(self):
        async with browser_page() as (page, origin):
            await page.goto(origin + "/", wait_until="networkidle")
            panels = page.locator(".animate-in")
            hidden = await panels.evaluate_all(
                '(els) => els.filter(e => getComputedStyle(e).opacity === "0").length'
            )
            self.assertEqual(hidden, 0)
            self.assertEqual(await page.locator(".grid").first.evaluate(
                "(el) => getComputedStyle(el).display"
            ), "grid")

    @unittest.skipUnless(find_spec("playwright"), "Install the test extra and Chromium for browser checks")
    async def test_htmx_displays_retry_after_paperless_fetch_failure(self):
        original = self.paperless.get_document
        calls = 0

        async def flaky(doc_id):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise httpx.ConnectError("Temporary outage")
            return await original(doc_id)

        with patch.object(self.paperless, "get_document", side_effect=flaky):
            async with browser_page() as (page, origin):
                await page.goto(origin + "/document/1", wait_until="networkidle")
                retry = page.get_by_role("button", name="Erneut versuchen")
                await retry.wait_for(state="visible", timeout=2500)
                self.assertIn("Paperless", await page.locator("#suggestion-area").inner_text())

    @unittest.skipUnless(find_spec("playwright"), "Install the test extra and Chromium for browser checks")
    async def test_browser_can_create_metadata_and_apply_one_document(self):
        await self.cache_suggestion()
        async with browser_page() as (page, origin):
            await page.goto(origin + "/document/1", wait_until="networkidle")
            await page.locator("#corr-select").select_option("__new__")
            await page.locator("#corr-new").fill("UI sender")
            await page.locator("#dt-select").select_option("__new__")
            await page.locator("#dt-new").fill("UI type")
            await page.locator("#tag-new-input").fill("UI tag")
            await page.locator('#apply-form button[type="submit"]').click()
            await page.wait_for_url(origin + "/")
            self.assertEqual(self.backend.docs[1]["correspondent"], 11)
            self.assertEqual(self.backend.docs[1]["document_type"], 21)
            self.assertEqual(self.backend.docs[1]["tags"], [2, 3, 4])
            self.assertEqual(self.backend.docs[1]["created"], "2024-03-15")
            self.assertIsNone(self.db.get_suggestion(1))

    async def test_learning_rule_lifecycle_invalidates_suggestions(self):
        await self.cache_suggestion()
        response = await self.client.post("/learning/rules/add", data={
            "description": "Synthetic rule", "prompt_text": "Prefer synthetic titles",
        })
        self.assertEqual(response.status_code, 303)
        self.assertIsNone(self.db.get_suggestion(1))
        rule = self.db.get_all_rules()[0]
        await self.client.post(f"/learning/rules/{rule['id']}/toggle")
        self.assertEqual(self.db.get_active_rules(), [])
        await self.client.post(f"/learning/rules/{rule['id']}/edit", data={
            "description": "Updated", "prompt_text": "Updated instruction",
        })
        self.assertEqual(self.db.get_all_rules()[0]["description"], "Updated")
        await self.client.post(f"/learning/rules/{rule['id']}/delete")
        self.assertEqual(self.db.get_all_rules(), [])

    async def test_successful_skip_and_delete_update_history(self):
        await self.cache_suggestion()
        response = await self.client.post("/document/1/skip", headers={"HX-Request": "true"})
        self.assertEqual(response.headers["hx-redirect"], "/")
        self.assertEqual(self.backend.docs[1]["tags"], [2])
        self.assertEqual(self.db.get_history()[0]["action"], "skip")
        response = await self.client.post("/document/1/delete")
        self.assertEqual(response.status_code, 303)
        self.assertNotIn(1, self.backend.docs)
        self.assertEqual(self.db.get_history()[0]["action"], "delete")

    async def test_empty_batch_and_progress_routes_are_consistent(self):
        await self.cache_suggestion()
        response = await self.client.post("/batch/start")
        self.assertEqual(response.json()["status"], "nothing_to_do")
        self.assertFalse((await self.client.get("/batch/status")).json()["running"])
        self.assertEqual((await self.client.get("/batch/progress")).status_code, 200)
        self.assertEqual((await self.client.post("/batch/cancel")).json()["status"], "not_running")

    async def test_bulk_apply_failure_keeps_suggestion_for_retry(self):
        await self.cache_suggestion(correspondent_id=10, correspondent_name="", new_tags=[])
        original = self.backend.__call__

        def fail_patch(request):
            if request.method == "PATCH":
                return httpx.Response(503, json={"detail": "Temporary outage"})
            return original(request)

        self.paperless._http._transport = httpx.MockTransport(fail_patch)
        response = await self.client.post("/review/apply-all")
        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(self.db.get_suggestion(1))
        self.assertEqual(self.db.feedback_count(), 0)
        self.assertEqual(self.backend.docs[1]["tags"], [1, 2])

    async def test_inbox_tag_is_never_an_ai_target(self):
        output = {"title": "Valid", "confidence": 0.9, "tag_ids": [1, 2]}
        result = await Classifier(FixtureProvider(output)).classify("Synthetic OCR", self.state.taxonomy)
        self.assertEqual(result.tag_ids, [2])

    async def test_all_main_pages_render_with_isolated_data(self):
        await self.cache_suggestion()
        for path in ("/", "/review", "/audit", "/history", "/learning", "/cleanup", "/settings", "/document/1"):
            with self.subTest(path=path):
                response = await self.client.get(path)
                self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
