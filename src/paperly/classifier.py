"""Claude AI classifier for Paperless-NGX documents."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

import anthropic

from paperly.paperless import Taxonomy

logger = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5"
MAX_CONTENT_CHARS = 6000
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # seconds

SYSTEM_PROMPT = """\
Du bist ein Assistent zur Klassifizierung von gescannten Dokumenten in einem deutschen \
Dokumentenmanagementsystem (Paperless-NGX).

Deine Aufgabe ist es, anhand des OCR-Textes eines Dokuments folgende Metadaten vorzuschlagen:
- Ein Datum im Format YYYY-MM-DD (aus dem Dokumentinhalt extrahiert, z.B. Rechnungsdatum, Briefdatum)
- Einen prägnanten, beschreibenden Titel (max. 80 Zeichen, auf Deutsch)
- Den passenden Absender/Korrespondenten aus der vorhandenen Liste (oder einen neuen Namen)
- Passende Tags aus der vorhandenen Liste
- Den passenden Dokumenttyp aus der vorhandenen Liste

Antworte ausschließlich mit validem JSON. Keine Erklärungen außerhalb des JSON-Objekts.

JSON-Format:
{
  "title": "Kurzer beschreibender Titel",
  "created": "2024-03-15",
  "correspondent_id": 42,
  "correspondent_name": "Name des Korrespondenten",
  "tag_ids": [5, 12],
  "document_type_id": 3,
  "confidence": 0.85,
  "reasoning": "Kurze Begründung auf Deutsch"
}

Regeln:
- Nutze vorhandene Korrespondenten wenn möglich; setze correspondent_id auf null wenn kein passender existiert
- Gib dann correspondent_name mit dem vorgeschlagenen neuen Namen an
- tag_ids: nur relevante Tags aus der Liste; lasse INBOX-Tag weg
- document_type_id: null wenn kein passender Typ existiert
- created: Datum aus dem Dokument extrahieren (Rechnungsdatum, Briefdatum, etc.); null wenn nicht erkennbar
- confidence: 0.0–1.0 (wie sicher du dir bei der Klassifizierung bist)
- Titel: kein Datum, kein Absendername (diese sind separat gespeichert)
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


class Classifier:
    def __init__(self, api_key: str) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)

    async def classify(
        self,
        content: str,
        taxonomy: Taxonomy,
        *,
        original_title: str = "",
    ) -> ClassificationResult:
        """Classify a document based on OCR content and existing taxonomy."""
        truncated = content[:MAX_CONTENT_CHARS]
        if len(content) > MAX_CONTENT_CHARS:
            truncated += "\n[... Text gekürzt ...]"

        user_message = _build_user_message(truncated, taxonomy, original_title)

        last_error: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._client.messages.create(
                    model=MODEL,
                    max_tokens=512,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_message}],
                )
                raw = response.content[0].text.strip()
                # Strip markdown code fences if Claude wraps the JSON
                if raw.startswith("```"):
                    raw = raw.split("```", 2)[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                    raw = raw.rstrip("`").strip()
                data = json.loads(raw)
                break
            except json.JSONDecodeError as e:
                logger.warning("Claude returned invalid JSON (attempt %d/%d): %s", attempt, MAX_RETRIES, e)
                last_error = e
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BASE_DELAY * attempt)
                    continue
                return _fallback_result(original_title, "Ungültiges JSON von Claude.")
            except anthropic.APIError as e:
                logger.error("Anthropic API error (attempt %d/%d): %s", attempt, MAX_RETRIES, e)
                last_error = e
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BASE_DELAY * attempt)
                    continue
                return _fallback_result(original_title, f"API-Fehler: {e}")

        result = ClassificationResult(
            title=data.get("title", original_title),
            created=data.get("created"),
            correspondent_id=data.get("correspondent_id"),
            correspondent_name=data.get("correspondent_name"),
            tag_ids=data.get("tag_ids") or [],
            document_type_id=data.get("document_type_id"),
            confidence=float(data.get("confidence", 0.5)),
            reasoning=data.get("reasoning", ""),
            raw_content_preview=truncated[:500],
        )

        # Validate returned IDs against taxonomy
        result = _validate_ids(result, taxonomy)
        return result


def _build_user_message(content: str, taxonomy: Taxonomy, original_title: str) -> str:
    # Build taxonomy reference lists for Claude
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

    return f"""\
## Vorhandene Korrespondenten
{corr_list}

## Vorhandene Tags
{tag_list}

## Vorhandene Dokumenttypen
{dt_list}

## Aktueller Titel des Dokuments
{original_title or '(kein Titel)'}

## OCR-Text des Dokuments
{content}

Bitte klassifiziere dieses Dokument anhand des OCR-Textes und der vorhandenen Taxonomie.
"""


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
    )


def _validate_ids(result: ClassificationResult, taxonomy: Taxonomy) -> ClassificationResult:
    """Ensure all returned IDs actually exist in the taxonomy."""
    if result.correspondent_id is not None:
        if not taxonomy.correspondent_by_id(result.correspondent_id):
            logger.warning(
                "Claude returned unknown correspondent_id=%d, clearing",
                result.correspondent_id,
            )
            result.correspondent_id = None

    if result.document_type_id is not None:
        if not taxonomy.document_type_by_id(result.document_type_id):
            logger.warning(
                "Claude returned unknown document_type_id=%d, clearing",
                result.document_type_id,
            )
            result.document_type_id = None

    valid_tag_ids = {t.id for t in taxonomy.tags}
    invalid_tags = [tid for tid in result.tag_ids if tid not in valid_tag_ids]
    if invalid_tags:
        logger.warning("Claude returned unknown tag_ids=%s, removing", invalid_tags)
        result.tag_ids = [tid for tid in result.tag_ids if tid in valid_tag_ids]

    return result
