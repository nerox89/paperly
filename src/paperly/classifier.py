"""Claude AI classifier for Paperless-NGX documents."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field

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
- Ein Datum im Format YYYY-MM-DD (aus dem Dokumentinhalt extrahiert, z.B. Rechnungsdatum, Briefdatum, \
Vertragsdatum). Bevorzuge das inhaltlich wichtigste Datum (Rechnungsdatum > Druckdatum > Eingangsdatum).
- Einen prägnanten, beschreibenden Titel (max. 80 Zeichen, auf Deutsch)
- Den passenden Absender/Korrespondenten aus der vorhandenen Liste (oder einen neuen Namen). \
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


class Classifier:
    def __init__(self, api_key: str) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)

    async def classify(
        self,
        content: str,
        taxonomy: Taxonomy,
        *,
        original_title: str = "",
        filename: str = "",
    ) -> ClassificationResult:
        """Classify a document based on OCR content and existing taxonomy."""
        truncated = _smart_truncate(content, MAX_CONTENT_CHARS)
        user_message = _build_user_message(truncated, taxonomy, original_title, filename)

        last_error: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._client.messages.create(
                    model=MODEL,
                    max_tokens=600,
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

        # Normalise created date
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
        )

        # Validate returned IDs and fuzzy-match correspondents
        result = _validate_ids(result, taxonomy)
        result = _fuzzy_match_correspondent(result, taxonomy)
        return result


def _smart_truncate(content: str, max_chars: int) -> str:
    """Truncate content keeping beginning and end (important info is usually there)."""
    if len(content) <= max_chars:
        return content
    # Keep 70% from start, 30% from end
    head_size = int(max_chars * 0.7)
    tail_size = max_chars - head_size - 50  # 50 chars for separator
    head = content[:head_size]
    tail = content[-tail_size:]
    return f"{head}\n\n[... {len(content) - head_size - tail_size} Zeichen übersprungen ...]\n\n{tail}"


def _build_user_message(content: str, taxonomy: Taxonomy, original_title: str, filename: str = "") -> str:
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

    parts = [
        f"## Vorhandene Korrespondenten\n{corr_list}",
        f"## Vorhandene Tags\n{tag_list}",
        f"## Vorhandene Dokumenttypen\n{dt_list}",
        f"## Aktueller Titel des Dokuments\n{original_title or '(kein Titel)'}",
    ]

    if filename:
        parts.append(f"## Dateiname\n{filename}")

    parts.append(f"## OCR-Text des Dokuments\n{content}")
    parts.append("Bitte klassifiziere dieses Dokument anhand des OCR-Textes und der vorhandenen Taxonomie.")

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

    # Already YYYY-MM-DD
    if re.match(r"^\d{4}-\d{2}-\d{2}$", raw):
        return raw

    # DD.MM.YYYY
    m = re.match(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$", raw)
    if m:
        return f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}"

    # "März 2024" or "15. März 2024"
    for month_name, month_num in _GERMAN_MONTHS.items():
        if month_name in raw.lower():
            year_m = re.search(r"(\d{4})", raw)
            day_m = re.search(r"(\d{1,2})\.", raw)
            if year_m:
                day = day_m.group(1).zfill(2) if day_m else "01"
                return f"{year_m.group(1)}-{month_num}-{day}"

    # MM/YYYY
    m = re.match(r"^(\d{1,2})/(\d{4})$", raw)
    if m:
        return f"{m.group(2)}-{m.group(1).zfill(2)}-01"

    return raw if re.match(r"^\d{4}-\d{2}-\d{2}", raw) else None


# ---------------------------------------------------------------------------
# Fuzzy correspondent matching
# ---------------------------------------------------------------------------

def _fuzzy_match_correspondent(result: ClassificationResult, taxonomy: Taxonomy) -> ClassificationResult:
    """If Claude suggested a new correspondent name, check if a similar one already exists."""
    if result.correspondent_id is not None or not result.correspondent_name:
        return result

    suggested = result.correspondent_name.lower().strip()

    for c in taxonomy.correspondents:
        existing = c.name.lower().strip()
        # Exact match (case-insensitive)
        if suggested == existing:
            result.correspondent_id = c.id
            return result
        # One contains the other (e.g., "Sparkasse" matches "Sparkasse Lübeck")
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
