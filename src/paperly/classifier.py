"""Claude AI classifier for Paperless-NGX documents."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import anthropic

from paperly.paperless import Taxonomy

logger = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5"
MAX_CONTENT_CHARS = 3500

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
        except json.JSONDecodeError as e:
            logger.warning("Claude returned invalid JSON: %s", e)
            return _fallback_result(original_title)
        except anthropic.APIError as e:
            logger.error("Anthropic API error: %s", e)
            return _fallback_result(original_title)

        return ClassificationResult(
            title=data.get("title", original_title),
            created=data.get("created"),
            correspondent_id=data.get("correspondent_id"),
            correspondent_name=data.get("correspondent_name"),
            tag_ids=data.get("tag_ids") or [],
            document_type_id=data.get("document_type_id"),
            confidence=float(data.get("confidence", 0.5)),
            reasoning=data.get("reasoning", ""),
            raw_content_preview=truncated[:200],
        )


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


def _fallback_result(title: str) -> ClassificationResult:
    return ClassificationResult(
        title=title,
        created=None,
        correspondent_id=None,
        correspondent_name=None,
        tag_ids=[],
        document_type_id=None,
        confidence=0.0,
        reasoning="Klassifizierung fehlgeschlagen.",
    )
