"""AI classifier for Paperless-NGX documents — multi-provider."""

from __future__ import annotations

import abc
import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field

import httpx

from paperly.paperless import Taxonomy

logger = logging.getLogger(__name__)

MAX_CONTENT_CHARS = 6000
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # seconds

SYSTEM_PROMPT = """\
Du bist ein Assistent zur Klassifizierung von gescannten Dokumenten in einem deutschen \
Dokumentenmanagementsystem (Paperless-NGX).

Deine Aufgabe ist es, anhand des OCR-Textes eines Dokuments folgende Metadaten vorzuschlagen:
- Ein Datum im Format YYYY-MM-DD (aus dem Dokumentinhalt extrahiert, z.B. Rechnungsdatum, Briefdatum, \
Vertragsdatum). Bevorzuge das inhaltlich wichtigste Datum (Rechnungsdatum > Druckdatum > Eingangsdatum).
- Einen prägnanten, beschreibenden Titel (max. 80 Zeichen, auf Deutsch)
- Den passenden Absender/Korrespondenten aus der vorhandenen Liste (oder einen neuen Namen). \
Der Korrespondent ist immer der ABSENDER/VERFASSER des Dokuments (z.B. Firma, Behörde, Bank), \
NIEMALS der Empfänger. Der Dokumenteneigentümer heißt "Maximilian Faure" — sein Name erscheint \
auf fast allen Dokumenten als Empfänger/Adressat und darf NICHT als Korrespondent gewählt werden. \
Achte auf ähnliche Schreibweisen (z.B. "Sparkasse" vs "Sparkasse Lübeck") und wähle den existierenden \
Eintrag wenn er passt.
- Passende Tags aus der vorhandenen Liste. Du kannst auch neue Tags vorschlagen wenn kein passender existiert.
- Den passenden Dokumenttyp aus der vorhandenen Liste

Antworte ausschließlich mit validem JSON. Keine Erklärungen außerhalb des JSON-Objekts.

JSON-Format:
{
  "title": "Kurzer beschreibender Titel",
  "created": "2024-03-15",
  "correspondent_id": 42,
  "correspondent_name": "Name des Korrespondenten",
  "tag_ids": [5, 12],
  "new_tags": ["Neuer Tag"],
  "document_type_id": 3,
  "confidence": 0.85,
  "reasoning": "Kurze Begründung auf Deutsch"
}

Regeln:
- Nutze vorhandene Korrespondenten wenn möglich; setze correspondent_id auf null wenn kein passender existiert
- Gib dann correspondent_name mit dem vorgeschlagenen neuen Namen an
- tag_ids: nur relevante Tags aus der vorhandenen Liste; lasse INBOX-Tag weg
- new_tags: optional, Liste von neuen Tag-Namen die erstellt werden sollten (nur wenn kein passender Tag existiert)
- document_type_id: null wenn kein passender Typ existiert
- created: Datum aus dem Dokument extrahieren; null wenn nicht erkennbar. \
Unterstützte Formate: "15.03.2024" → "2024-03-15", "März 2024" → "2024-03-01"
- confidence: 0.0–1.0 (wie sicher du dir bei der Klassifizierung bist)
- Titel: kein Datum, kein Absendername (diese sind separat gespeichert)
"""

# Few-shot example appended for local models that benefit from concrete examples
FEWSHOT_EXAMPLE = """
## Beispiel

Eingabe (OCR-Text): "Sparkasse Lübeck Kontoauszug Nr. 3/2024 Kontonummer 123456789 \
Auszugsdatum: 15.03.2024 Alter Saldo: 1.234,56 EUR Neuer Saldo: 1.456,78 EUR"
Vorhandene Korrespondenten: 42: Sparkasse Lübeck
Vorhandene Tags: 5: Finanzen, 12: Bank
Vorhandene Dokumenttypen: 3: Kontoauszug

Korrekte Antwort:
{"title": "Kontoauszug Nr. 3/2024", "created": "2024-03-15", "correspondent_id": 42, \
"correspondent_name": "Sparkasse Lübeck", "tag_ids": [5, 12], "new_tags": [], \
"document_type_id": 3, "confidence": 0.95, "reasoning": "Kontoauszug der Sparkasse Lübeck, \
Auszugsdatum 15.03.2024, Korrespondent und Dokumenttyp eindeutig zuordenbar."}

Jetzt klassifiziere das folgende Dokument:
"""


@dataclass
class ClassificationResult:
    title: str
    created: str | None
    correspondent_id: int | None
    correspondent_name: str | None
    tag_ids: list[int]
    document_type_id: int | None
    confidence: float
    reasoning: str
    raw_content_preview: str = ""
    new_tags: list[str] = field(default_factory=list)
    storage_path_id: int | None = None
    provider_name: str = ""
    provider_model: str = ""


# ---------------------------------------------------------------------------
# Provider abstraction
# ---------------------------------------------------------------------------

class BaseProvider(abc.ABC):
    """Abstract base for LLM providers — returns raw parsed JSON dict."""

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @property
    @abc.abstractmethod
    def model(self) -> str: ...

    @abc.abstractmethod
    async def generate(self, system: str, user_message: str) -> dict:
        """Send prompt to LLM and return parsed JSON dict."""
        ...


class AnthropicProvider(BaseProvider):
    """Claude via the Anthropic SDK (synchronous under the hood)."""

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5") -> None:
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    @property
    def name(self) -> str:
        return "claude"

    @property
    def model(self) -> str:
        return self._model

    async def generate(self, system: str, user_message: str) -> dict:
        # Anthropic SDK is synchronous — run in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._call, system, user_message)

    def _call(self, system: str, user_message: str) -> dict:
        import anthropic
        last_error: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=600,
                    system=system,
                    messages=[{"role": "user", "content": user_message}],
                )
                raw = response.content[0].text.strip()
                return _parse_json_response(raw)
            except json.JSONDecodeError as e:
                logger.warning("Claude returned invalid JSON (attempt %d/%d): %s", attempt, MAX_RETRIES, e)
                last_error = e
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BASE_DELAY * attempt)
                    continue
                raise
            except anthropic.APIError as e:
                logger.error("Anthropic API error (attempt %d/%d): %s", attempt, MAX_RETRIES, e)
                last_error = e
                if attempt < MAX_RETRIES:
                    # Rate limit (429): wait longer using retry-after or 60s default
                    if getattr(e, 'status_code', 0) == 429:
                        retry_after = 60
                        if hasattr(e, 'response') and e.response is not None:
                            retry_after = int(e.response.headers.get("retry-after", 60))
                        logger.info("Rate limited — waiting %ds before retry", retry_after)
                        time.sleep(retry_after)
                    else:
                        time.sleep(RETRY_BASE_DELAY * attempt)
                    continue
                raise
        raise last_error  # type: ignore[misc]


class OllamaProvider(BaseProvider):
    """Local Ollama instance via REST API."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "gemma4:e4b") -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model(self) -> str:
        return self._model

    async def generate(self, system: str, user_message: str) -> dict:
        last_error: Exception | None = None

        # Try with thinking first (better quality), then fall back to no-think (reliable JSON)
        for use_think in (True, False):
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    raw = await self._ollama_call(system, user_message, think=use_think)
                    parsed = _parse_json_response(raw)
                    _validate_schema(parsed)
                    return parsed
                except (json.JSONDecodeError, ValueError) as e:
                    phase = "think" if use_think else "no-think"
                    logger.warning(
                        "Ollama %s bad response (attempt %d/%d): %s — raw: %.300s",
                        phase, attempt, MAX_RETRIES, e, raw if 'raw' in dir() else '(no raw)',
                    )
                    last_error = e
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(RETRY_BASE_DELAY * attempt)
                        continue
                    if use_think:
                        logger.info("Thinking mode failed, retrying without think mode")
                except httpx.HTTPError as e:
                    logger.error("Ollama HTTP error: %s", e)
                    last_error = e
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(RETRY_BASE_DELAY * attempt)
                        continue
        # Last resort: try to salvage the last wrong-schema response
        if isinstance(last_error, ValueError) and 'parsed' in dir():
            salvaged = _salvage_wrong_schema(parsed)  # type: ignore[possibly-undefined]
            if salvaged:
                logger.info("Salvaged wrong-schema response as low-confidence result")
                return salvaged
        raise last_error  # type: ignore[misc]

    async def _ollama_call(self, system: str, user_message: str, *, think: bool) -> str:
        """Make a single Ollama API call and return the raw content string."""
        async with httpx.AsyncClient(timeout=240.0) as client:
            payload: dict = {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_message},
                ],
                "format": "json",
                "stream": False,
                "options": {
                    "num_predict": 4096 if think else 1024,
                    "temperature": 0.1,
                },
            }
            if think:
                payload["think"] = True

            resp = await client.post(f"{self._base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            msg = data.get("message", {})
            thinking = (msg.get("thinking", "") or "").strip()
            if thinking:
                logger.debug("Ollama thinking (%d chars): %.300s…", len(thinking), thinking)
            raw = (msg.get("content", "") or "").strip()

            # Fallback: extract JSON from thinking field
            if not raw and thinking:
                logger.info("Content empty, extracting JSON from thinking field")
                raw = _extract_json_from_text(thinking)

            if not raw:
                logger.warning("Ollama empty response (think=%s), keys=%s", think, list(msg.keys()))
                raise json.JSONDecodeError("Empty response from Ollama", "", 0)

            logger.info("Ollama raw (think=%s, %d chars): %.200s", think, len(raw), raw)
            return raw

    async def test_connection(self) -> dict:
        """Test connectivity and return available model info."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{self._base_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = [m["name"] for m in data.get("models", [])]
            return {"ok": True, "models": models, "url": self._base_url}


# ---------------------------------------------------------------------------
# Classifier (orchestrator)
# ---------------------------------------------------------------------------

class Classifier:
    def __init__(self, provider: BaseProvider, custom_prompt: str = "") -> None:
        self._provider = provider
        self.custom_prompt = custom_prompt

    @property
    def provider(self) -> BaseProvider:
        return self._provider

    @provider.setter
    def provider(self, p: BaseProvider) -> None:
        self._provider = p

    async def classify(
        self,
        content: str,
        taxonomy: Taxonomy,
        *,
        original_title: str = "",
        filename: str = "",
        db: object | None = None,
    ) -> ClassificationResult:
        """Classify a document based on OCR content and existing taxonomy."""
        truncated = _smart_truncate(content, MAX_CONTENT_CHARS)

        # Few-shot examples from learning history
        examples = None
        if db and hasattr(db, "get_similar_examples"):
            try:
                examples = db.get_similar_examples(truncated[:500], limit=3)
                if examples:
                    logger.info("Injecting %d few-shot examples from learning history", len(examples))
            except Exception as e:
                logger.warning("Failed to load few-shot examples: %s", e)

        user_message = _build_user_message(truncated, taxonomy, original_title, filename, examples)

        # Build system prompt: default + optional custom instructions
        system = SYSTEM_PROMPT
        if self.custom_prompt:
            system += f"\n\nZusätzliche Anweisungen:\n{self.custom_prompt}\n"

        # Inject active correction rules
        rule_ids_used: list[int] = []
        if db and hasattr(db, "get_active_rules"):
            try:
                rules = db.get_active_rules()
                if rules:
                    rule_texts = []
                    for r in rules:
                        rule_texts.append(f"- {r['prompt_text']}")
                        rule_ids_used.append(r["id"])
                    system += "\n\nGelernte Korrekturregeln:\n" + "\n".join(rule_texts) + "\n"
                    logger.info("Injecting %d correction rules into prompt", len(rules))
            except Exception as e:
                logger.warning("Failed to load correction rules: %s", e)

        if self._provider.name == "ollama":
            system += FEWSHOT_EXAMPLE

        try:
            data = await self._provider.generate(system, user_message)
        except Exception as e:
            logger.error("Classification failed (%s/%s): %s", self._provider.name, self._provider.model, e)
            return _fallback_result(original_title, f"Fehler ({self._provider.name}): {e}")

        # Track rule usage
        if rule_ids_used and db and hasattr(db, "increment_rule_hits"):
            try:
                db.increment_rule_hits(rule_ids_used)
            except Exception:
                pass

        created = _normalise_date(data.get("created"))

        result = ClassificationResult(
            title=data.get("title", original_title),
            created=created,
            correspondent_id=data.get("correspondent_id"),
            correspondent_name=data.get("correspondent_name"),
            tag_ids=data.get("tag_ids") or [],
            document_type_id=data.get("document_type_id"),
            confidence=float(data.get("confidence", 0.5)),
            reasoning=data.get("reasoning", ""),
            raw_content_preview=truncated[:500],
            new_tags=data.get("new_tags") or [],
            storage_path_id=data.get("storage_path_id"),
        )

        result = _validate_ids(result, taxonomy)
        result = _fuzzy_match_correspondent(result, taxonomy)
        result.provider_name = self._provider.name
        result.provider_model = self._provider.model
        return result


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _validate_schema(data: dict) -> None:
    """Ensure the LLM response has the expected classification schema keys."""
    required = {"title", "confidence"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"Response missing required keys: {missing}. Got keys: {list(data.keys())}")
    # Coerce confidence to float (Gemma sometimes returns it as string)
    try:
        data["confidence"] = float(data["confidence"])
    except (TypeError, ValueError):
        raise ValueError(f"confidence not numeric, got: {data.get('confidence')!r}")


def _salvage_wrong_schema(data: dict) -> dict | None:
    """Try to convert a wrong-schema LLM response into our expected format."""
    # Only attempt if it has some recognizable classification info
    if not isinstance(data, dict):
        return None
    has_useful_info = any(k in data for k in ("document_type", "keywords", "entities", "kategorie"))
    if not has_useful_info:
        return None

    logger.info("Attempting to salvage wrong-schema response: %s", list(data.keys()))
    result = {
        "title": data.get("title", data.get("document_type", "")),
        "created": data.get("created", data.get("date")),
        "correspondent_id": data.get("correspondent_id"),
        "correspondent_name": data.get("correspondent_name"),
        "tag_ids": data.get("tag_ids", []),
        "new_tags": data.get("new_tags", data.get("keywords", [])[:5]),
        "document_type_id": data.get("document_type_id"),
        "confidence": float(data.get("confidence", 0.4)),
        "reasoning": data.get("reasoning", data.get("analysis", "Automatisch rekonstruiert aus unvollständiger KI-Antwort")),
    }

    if not result["title"]:
        return None
    return result


def _extract_json_from_text(text: str) -> str:
    """Try to extract a JSON object from free-form text (e.g. thinking output)."""
    # Look for ```json ... ``` blocks first
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    # Find the outermost { ... } block
    start = text.find("{")
    if start == -1:
        return ""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return ""


def _parse_json_response(raw: str) -> dict:
    """Parse JSON from LLM response, handling markdown fences and free text."""
    # Try direct JSON parse first
    cleaned = raw.strip()

    # Strip markdown fences
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.rstrip("`").strip()

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: extract JSON object from free-form text
        extracted = _extract_json_from_text(raw)
        if not extracted:
            raise json.JSONDecodeError("No JSON object found in response", raw[:200], 0)
        logger.info("Extracted JSON from free-form response: %.200s", extracted)
        result = json.loads(extracted)

    # Some models wrap the result in an array — unwrap it
    if isinstance(result, list):
        if len(result) > 0 and isinstance(result[0], dict):
            return result[0]
        raise json.JSONDecodeError("LLM returned array without dict", raw[:200], 0)
    if not isinstance(result, dict):
        raise json.JSONDecodeError("LLM returned non-object JSON", raw[:200], 0)
    return result


def _smart_truncate(content: str, max_chars: int) -> str:
    """Truncate content keeping beginning and end (important info is usually there)."""
    if len(content) <= max_chars:
        return content
    head_size = int(max_chars * 0.7)
    tail_size = max_chars - head_size - 50
    head = content[:head_size]
    tail = content[-tail_size:]
    return f"{head}\n\n[... {len(content) - head_size - tail_size} Zeichen übersprungen ...]\n\n{tail}"


def _build_user_message(
    content: str,
    taxonomy: Taxonomy,
    original_title: str,
    filename: str = "",
    examples: list[dict] | None = None,
) -> str:
    corr_list = "\n".join(
        f"  {c.id}: {c.name}" for c in sorted(taxonomy.correspondents, key=lambda x: x.name)
    )
    tag_list = "\n".join(
        f"  {t.id}: {t.name}"
        for t in sorted(taxonomy.tags, key=lambda x: x.name)
        if t.id != taxonomy.inbox_tag_id
    )
    dt_list = "\n".join(
        f"  {dt.id}: {dt.name}"
        for dt in sorted(taxonomy.document_types, key=lambda x: x.name)
    )

    sp_list = "\n".join(
        f"  {sp.id}: {sp.name}"
        for sp in sorted(taxonomy.storage_paths, key=lambda x: x.name)
    )

    parts = [
        f"## Vorhandene Korrespondenten\n{corr_list}",
        f"## Vorhandene Tags\n{tag_list}",
        f"## Vorhandene Dokumenttypen\n{dt_list}",
        f"## Vorhandene Speicherpfade\n{sp_list}",
        f"## Aktueller Titel des Dokuments\n{original_title or '(kein Titel)'}",
    ]

    if filename:
        parts.append(f"## Dateiname\n{filename}")

    # Few-shot examples from confirmed history (self-learning)
    if examples:
        example_parts = []
        for i, ex in enumerate(examples, 1):
            preview = (ex.get("content_preview") or "")[:200]
            corr_name = ""
            if ex.get("correspondent_id"):
                c = taxonomy.correspondent_by_id(ex["correspondent_id"])
                corr_name = f" ({c.name})" if c else ""
            dt_name = ""
            if ex.get("document_type_id"):
                dt = taxonomy.document_type_by_id(ex["document_type_id"])
                dt_name = dt.name if dt else ""
            sp_name = ""
            if ex.get("storage_path_id"):
                sp = taxonomy.storage_path_by_id(ex["storage_path_id"])
                sp_name = sp.name if sp else ""
            tag_names = []
            for tid in ex.get("tag_ids") or []:
                t = taxonomy.tag_by_id(tid)
                if t:
                    tag_names.append(t.name)

            lines = [f"Beispiel {i}:"]
            lines.append(f'  Text: "{preview}..."')
            lines.append(f'  → Titel: "{ex.get("title", "")}"')
            if corr_name:
                lines.append(f"  → Absender: ID {ex['correspondent_id']}{corr_name}")
            if dt_name:
                lines.append(f"  → Dokumenttyp: {dt_name}")
            if sp_name:
                lines.append(f"  → Speicherpfad: {sp_name}")
            if tag_names:
                lines.append(f"  → Tags: [{', '.join(tag_names)}]")
            example_parts.append("\n".join(lines))

        parts.append(
            "## Bereits korrekt klassifizierte ähnliche Dokumente\n"
            + "\n\n".join(example_parts)
        )

    parts.append(f"## OCR-Text des Dokuments\n{content}")
    parts.append(
        "Klassifiziere dieses Dokument. Antworte NUR mit diesem exakten JSON-Format:\n"
        '{"title": "Kurzer Titel", "created": "YYYY-MM-DD", '
        '"correspondent_id": null, "correspondent_name": "Name", '
        '"tag_ids": [], "new_tags": [], "document_type_id": null, '
        '"storage_path_id": null, '
        '"confidence": 0.85, "reasoning": "Begründung"}\n\n'
        "WICHTIG: Verwende exakt diese Schlüssel. Kein anderes Format."
    )

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Date normalisation
# ---------------------------------------------------------------------------

_GERMAN_MONTHS = {
    "januar": "01", "februar": "02", "märz": "03", "april": "04",
    "mai": "05", "juni": "06", "juli": "07", "august": "08",
    "september": "09", "oktober": "10", "november": "11", "dezember": "12",
    "jan": "01", "feb": "02", "mär": "03", "apr": "04",
    "jun": "06", "jul": "07", "aug": "08", "sep": "09",
    "okt": "10", "nov": "11", "dez": "12",
}


def _normalise_date(raw: str | None) -> str | None:
    """Normalise date to YYYY-MM-DD. Handles various German date formats."""
    if not raw or not isinstance(raw, str):
        return None

    raw = raw.strip()

    if re.match(r"^\d{4}-\d{2}-\d{2}$", raw):
        return raw

    m = re.match(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$", raw)
    if m:
        return f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}"

    for month_name, month_num in _GERMAN_MONTHS.items():
        if month_name in raw.lower():
            year_m = re.search(r"(\d{4})", raw)
            day_m = re.search(r"(\d{1,2})\.", raw)
            if year_m:
                day = day_m.group(1).zfill(2) if day_m else "01"
                return f"{year_m.group(1)}-{month_num}-{day}"

    m = re.match(r"^(\d{1,2})/(\d{4})$", raw)
    if m:
        return f"{m.group(2)}-{m.group(1).zfill(2)}-01"

    return raw if re.match(r"^\d{4}-\d{2}-\d{2}", raw) else None


# ---------------------------------------------------------------------------
# Fuzzy correspondent matching
# ---------------------------------------------------------------------------

def _fuzzy_match_correspondent(result: ClassificationResult, taxonomy: Taxonomy) -> ClassificationResult:
    """If the LLM suggested a new correspondent name, check if a similar one already exists."""
    if result.correspondent_id is not None or not result.correspondent_name:
        return result

    suggested = result.correspondent_name.lower().strip()

    for c in taxonomy.correspondents:
        existing = c.name.lower().strip()
        if suggested == existing:
            result.correspondent_id = c.id
            return result
        if suggested in existing or existing in suggested:
            logger.info(
                "Fuzzy matched correspondent: '%s' → '%s' (id=%d)",
                result.correspondent_name, c.name, c.id,
            )
            result.correspondent_id = c.id
            result.correspondent_name = c.name
            return result

    return result


def _fallback_result(title: str, reason: str = "Klassifizierung fehlgeschlagen.") -> ClassificationResult:
    return ClassificationResult(
        title=title,
        created=None,
        correspondent_id=None,
        correspondent_name=None,
        tag_ids=[],
        document_type_id=None,
        confidence=0.0,
        reasoning=reason,
        new_tags=[],
    )


def _validate_ids(result: ClassificationResult, taxonomy: Taxonomy) -> ClassificationResult:
    """Ensure all returned IDs actually exist in the taxonomy."""
    if result.correspondent_id is not None:
        if not taxonomy.correspondent_by_id(result.correspondent_id):
            logger.warning(
                "LLM returned unknown correspondent_id=%d, clearing",
                result.correspondent_id,
            )
            result.correspondent_id = None

    if result.document_type_id is not None:
        if not taxonomy.document_type_by_id(result.document_type_id):
            logger.warning(
                "LLM returned unknown document_type_id=%d, clearing",
                result.document_type_id,
            )
            result.document_type_id = None

    valid_tag_ids = {t.id for t in taxonomy.tags}
    invalid_tags = [tid for tid in result.tag_ids if tid not in valid_tag_ids]
    if invalid_tags:
        logger.warning("LLM returned unknown tag_ids=%s, removing", invalid_tags)
        result.tag_ids = [tid for tid in result.tag_ids if tid in valid_tag_ids]

    if result.storage_path_id is not None:
        if not taxonomy.storage_path_by_id(result.storage_path_id):
            logger.warning(
                "LLM returned unknown storage_path_id=%d, clearing",
                result.storage_path_id,
            )
            result.storage_path_id = None

    return result
