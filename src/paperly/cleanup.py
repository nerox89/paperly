"""Taxonomy cleanup CLI — merge duplicates, remove orphans."""

from __future__ import annotations

import asyncio
import os
import sys
from typing import NamedTuple

from dotenv import load_dotenv

from paperly.paperless import PaperlessClient, Taxonomy

load_dotenv()


class MergeAction(NamedTuple):
    kind: str  # "document_type" or "correspondent"
    keep_id: int
    keep_name: str
    remove_id: int
    remove_name: str
    doc_count: int


class DeleteAction(NamedTuple):
    kind: str
    item_id: int
    name: str


async def run_cleanup(dry_run: bool = False) -> None:
    client = PaperlessClient(os.environ["PAPERLESS_URL"], os.environ["PAPERLESS_TOKEN"])
    taxonomy = await client.get_taxonomy()

    merge_actions, delete_actions = _analyse(taxonomy)

    if not merge_actions and not delete_actions:
        print("✅ Keine Probleme gefunden. Taxonomie ist sauber!")
        return

    print(f"\n{'[DRY-RUN] ' if dry_run else ''}Analyse abgeschlossen:\n")

    if merge_actions:
        print("── ZU ZUSAMMENFÜHREN ──────────────────────────")
        for a in merge_actions:
            kind_label = "Dokumenttyp" if a.kind == "document_type" else "Korrespondent"
            print(
                f"  [{kind_label}] '{a.remove_name}' ({a.doc_count} Docs) → '{a.keep_name}'"
            )

    if delete_actions:
        print("\n── ZU LÖSCHEN (0 Dokumente) ────────────────────")
        for d in delete_actions:
            kind_label = "Dokumenttyp" if d.kind == "document_type" else "Korrespondent"
            print(f"  [{kind_label}] '{d.name}'")

    if dry_run:
        print("\n(Dry-run: keine Änderungen vorgenommen)")
        return

    # Interactive confirmation
    print(f"\n{len(merge_actions)} Zusammenführungen, {len(delete_actions)} Löschungen geplant.")
    answer = input("Fortfahren? [j/N] ").strip().lower()
    if answer not in ("j", "ja", "y", "yes"):
        print("Abgebrochen.")
        return

    # Execute merges
    for action in merge_actions:
        print(f"  Zusammenführen: '{action.remove_name}' → '{action.keep_name}'…", end=" ")
        try:
            await _reassign_and_delete(client, taxonomy, action)
            print("✓")
        except Exception as e:
            print(f"✗ Fehler: {e}")

    # Execute deletes (only 0-doc items)
    for action in delete_actions:
        print(f"  Löschen: '{action.name}'…", end=" ")
        try:
            if action.kind == "document_type":
                await client.delete_document_type(action.item_id)
            else:
                await client.delete_correspondent(action.item_id)
            print("✓")
        except Exception as e:
            print(f"✗ Fehler: {e}")

    print("\n✅ Fertig!")


def _analyse(taxonomy: Taxonomy) -> tuple[list[MergeAction], list[DeleteAction]]:
    merge_actions: list[MergeAction] = []
    delete_actions: list[DeleteAction] = []

    # Document type duplicates: "Rechnungen" / "Rechnung" etc.
    # Strategy: keep the one with more documents; merge others into it
    seen_dt: dict[str, tuple[int, int]] = {}  # normalised_name -> (id, count)
    for dt in sorted(taxonomy.document_types, key=lambda x: x.document_count, reverse=True):
        normalised = _normalise(dt.name)
        if normalised in seen_dt:
            keep_id, keep_count = seen_dt[normalised]
            keep_name = next(d.name for d in taxonomy.document_types if d.id == keep_id)
            if dt.document_count > 0:
                merge_actions.append(
                    MergeAction("document_type", keep_id, keep_name, dt.id, dt.name, dt.document_count)
                )
            else:
                delete_actions.append(DeleteAction("document_type", dt.id, dt.name))
        else:
            seen_dt[normalised] = (dt.id, dt.document_count)
            if dt.document_count == 0:
                delete_actions.append(DeleteAction("document_type", dt.id, dt.name))

    # Correspondents with 0 documents → delete
    for c in taxonomy.correspondents:
        if c.document_count == 0:
            delete_actions.append(DeleteAction("correspondent", c.id, c.name))

    return merge_actions, delete_actions


def _normalise(name: str) -> str:
    """Normalise a name for duplicate detection (lowercase, strip plural 'n')."""
    n = name.lower().strip()
    # German plural ending -n/-en
    if n.endswith("en"):
        n = n[:-2]
    elif n.endswith("n"):
        n = n[:-1]
    return n


async def _reassign_and_delete(
    client: PaperlessClient, taxonomy: Taxonomy, action: MergeAction
) -> None:
    """Fetch all docs with remove_id, reassign to keep_id, then delete remove_id."""
    import httpx

    # Fetch documents with the entity to remove
    base = os.environ["PAPERLESS_URL"]
    token = os.environ["PAPERLESS_TOKEN"]
    field = "document_type" if action.kind == "document_type" else "correspondent"
    param = f"{field}__id" if action.kind == "correspondent" else "document_type__id"

    async with httpx.AsyncClient(
        base_url=base, headers={"Authorization": f"Token {token}"}, timeout=30.0
    ) as http:
        r = await http.get(f"/api/documents/?{param}={action.remove_id}&page_size=500")
        r.raise_for_status()
        doc_ids = [d["id"] for d in r.json()["results"]]

    if doc_ids:
        kwargs = {field: action.keep_id}
        await client.bulk_update_documents(doc_ids, **kwargs)  # type: ignore[arg-type]

    # Delete the now-empty entity
    if action.kind == "document_type":
        await client.delete_document_type(action.remove_id)
    else:
        await client.delete_correspondent(action.remove_id)


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    asyncio.run(run_cleanup(dry_run=dry_run))


if __name__ == "__main__":
    main()
