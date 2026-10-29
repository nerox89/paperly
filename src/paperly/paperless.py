"""Async Paperless-NGX API client."""

from __future__ import annotations

import asyncio
import httpx
from dataclasses import dataclass, field
from types import EllipsisType


@dataclass
class Tag:
    id: int
    name: str
    document_count: int = 0
    is_inbox_tag: bool = False


@dataclass
class Correspondent:
    id: int
    name: str
    document_count: int = 0


@dataclass
class DocumentType:
    id: int
    name: str
    document_count: int = 0


@dataclass
class Document:
    id: int
    title: str
    content: str
    created: str
    correspondent: int | None
    document_type: int | None
    tags: list[int]
    storage_path: int | None
    archive_serial_number: int | None
    original_file_name: str | None


@dataclass
class StoragePath:
    id: int
    name: str
    document_count: int = 0


@dataclass
class Taxonomy:
    tags: list[Tag] = field(default_factory=list)
    correspondents: list[Correspondent] = field(default_factory=list)
    document_types: list[DocumentType] = field(default_factory=list)
    storage_paths: list[StoragePath] = field(default_factory=list)
    inbox_tag_id: int | None = None

    @property
    def inbox_tag_ids(self) -> set[int]:
        return {t.id for t in self.tags if t.is_inbox_tag}

    def tag_by_id(self, id: int) -> Tag | None:
        return next((t for t in self.tags if t.id == id), None)

    def correspondent_by_id(self, id: int) -> Correspondent | None:
        return next((c for c in self.correspondents if c.id == id), None)

    def document_type_by_id(self, id: int) -> DocumentType | None:
        return next((dt for dt in self.document_types if dt.id == id), None)

    def storage_path_by_id(self, id: int) -> StoragePath | None:
        return next((sp for sp in self.storage_paths if sp.id == id), None)


class PaperlessClient:
    def __init__(self, base_url: str, token: str) -> None:
        self._base = base_url.rstrip("/")
        self._headers = {"Authorization": f"Token {token}"}
        self._http: httpx.AsyncClient | None = None
        self._update_lock = asyncio.Lock()

    async def open(self) -> None:
        """Create the shared HTTP client. Call once at startup."""
        self._http = httpx.AsyncClient(
            base_url=self._base,
            headers=self._headers,
            timeout=30.0,
        )

    async def close(self) -> None:
        """Close the shared HTTP client. Call at shutdown."""
        if self._http:
            await self._http.aclose()
            self._http = None

    @property
    def _c(self) -> httpx.AsyncClient:
        """Return the shared client (raises if not opened)."""
        if self._http is None:
            raise RuntimeError("PaperlessClient not opened — call .open() first")
        return self._http

    async def get_statistics(self) -> dict:
        r = await self._c.get("/api/statistics/")
        r.raise_for_status()
        return r.json()

    async def get_taxonomy(self) -> Taxonomy:
        tag_data, corr_data, dt_data, sp_data = await _gather(
            self.get_all("/api/tags/"),
            self.get_all("/api/correspondents/"),
            self.get_all("/api/document_types/"),
            self.get_all("/api/storage_paths/"),
        )

        tags = [
            Tag(id=t["id"], name=t["name"], document_count=t.get("document_count", 0),
                is_inbox_tag=t.get("is_inbox_tag", False))
            for t in tag_data
        ]
        correspondents = [
            Correspondent(id=c["id"], name=c["name"], document_count=c.get("document_count", 0))
            for c in corr_data
        ]
        document_types = [
            DocumentType(id=dt["id"], name=dt["name"], document_count=dt.get("document_count", 0))
            for dt in dt_data
        ]
        storage_paths = [
            StoragePath(id=sp["id"], name=sp["name"], document_count=sp.get("document_count", 0))
            for sp in sp_data
        ]

        inbox_tag = next(
            (t for t in tags if t.is_inbox_tag),
            None,
        )

        return Taxonomy(
            tags=tags,
            correspondents=correspondents,
            document_types=document_types,
            storage_paths=storage_paths,
            inbox_tag_id=inbox_tag.id if inbox_tag else None,
        )

    async def get_all(self, path: str, **params: str | int) -> list[dict]:
        """Read every page before callers mutate the collection."""
        results: list[dict] = []
        url: str | None = path
        while url:
            response = await self._c.get(url, params={"page_size": 500, **params} if not results else None)
            response.raise_for_status()
            data = response.json()
            results.extend(data["results"])
            url = data.get("next")
        return results

    async def get_inbox_documents(
        self,
        *,
        page: int = 1,
        page_size: int = 25,
    ) -> tuple[list[Document], int]:
        """Returns (documents, total_count)."""
        r = await self._c.get(
            "/api/documents/",
            params={
                "is_in_inbox": "true",
                "page": page,
                "page_size": page_size,
                "ordering": "created,id",
            },
        )
        r.raise_for_status()
        data = r.json()
        return [_parse_document(d) for d in data["results"]], data["count"]

    async def get_documents(
        self,
        *,
        page: int = 1,
        page_size: int = 25,
        ordering: str = "-created",
        correspondent_id: int | None = None,
        document_type_id: int | None = None,
        storage_path_id: int | None = None,
        tag_id: int | None = None,
        exclude_tag_id: int | None = None,
        in_inbox: bool | None = None,
        search: str | None = None,
    ) -> tuple[list[Document], int]:
        """Generic document query with flexible filters."""
        params: dict[str, str | int] = {
            "page": page,
            "page_size": page_size,
            "ordering": ordering,
        }
        if correspondent_id:
            params["correspondent__id"] = correspondent_id
        if document_type_id:
            params["document_type__id"] = document_type_id
        if storage_path_id:
            params["storage_path__id"] = storage_path_id
        if tag_id:
            params["tags__id__all"] = tag_id
        if exclude_tag_id:
            params["tags__id__none"] = exclude_tag_id
        if in_inbox is not None:
            params["is_in_inbox"] = str(in_inbox).lower()
        if search:
            params["query"] = search
        r = await self._c.get("/api/documents/", params=params)
        r.raise_for_status()
        data = r.json()
        return [_parse_document(d) for d in data["results"]], data["count"]

    async def get_document(self, doc_id: int) -> Document:
        r = await self._c.get(f"/api/documents/{doc_id}/")
        r.raise_for_status()
        return _parse_document(r.json())

    async def get_document_thumb_url(self, doc_id: int) -> str:
        """Returns the thumbnail URL path (client must know base URL)."""
        return f"/api/documents/{doc_id}/thumb/"

    async def update_document(
        self,
        doc_id: int,
        *,
        title: str | None = None,
        created: str | None = None,
        correspondent: int | None | EllipsisType = ...,
        document_type: int | None | EllipsisType = ...,
        storage_path: int | None | EllipsisType = ...,
        tags: list[int] | None = None,
        remove_tag: int | None = None,
    ) -> Document:
        """Patch document metadata. Pass remove_tag to strip inbox tag after processing."""
        payload: dict = {}
        if title is not None:
            payload["title"] = title
        if created is not None:
            payload["created"] = created
        if correspondent is not ...:
            payload["correspondent"] = correspondent
        if document_type is not ...:
            payload["document_type"] = document_type
        if storage_path is not ...:
            payload["storage_path"] = storage_path
        if tags is not None:
            payload["tags"] = tags

        if remove_tag is not None and tags is None:
            doc = await self.get_document(doc_id)
            payload["tags"] = [t for t in doc.tags if t != remove_tag]

        async with self._update_lock:
            r = await self._c.patch(f"/api/documents/{doc_id}/", json=payload)
            r.raise_for_status()
            return _parse_document(r.json())

    async def create_correspondent(self, name: str) -> Correspondent:
        # Check for existing correspondent with same name (case-insensitive)
        r = await self._c.get("/api/correspondents/", params={"name__iexact": name})
        r.raise_for_status()
        results = r.json().get("results", [])
        if results:
            d = results[0]
            return Correspondent(id=d["id"], name=d["name"])
        r = await self._c.post("/api/correspondents/", json={"name": name})
        r.raise_for_status()
        d = r.json()
        return Correspondent(id=d["id"], name=d["name"])

    async def create_document_type(self, name: str) -> DocumentType:
        r = await self._c.get("/api/document_types/", params={"name__iexact": name})
        r.raise_for_status()
        results = r.json().get("results", [])
        if results:
            d = results[0]
            return DocumentType(id=d["id"], name=d["name"])
        r = await self._c.post("/api/document_types/", json={"name": name})
        r.raise_for_status()
        d = r.json()
        return DocumentType(id=d["id"], name=d["name"])

    async def create_tag(self, name: str) -> Tag:
        r = await self._c.get("/api/tags/", params={"name__iexact": name})
        r.raise_for_status()
        results = r.json().get("results", [])
        if results:
            d = results[0]
            return Tag(id=d["id"], name=d["name"])
        r = await self._c.post("/api/tags/", json={"name": name})
        r.raise_for_status()
        d = r.json()
        return Tag(id=d["id"], name=d["name"])

    async def delete_correspondent(self, correspondent_id: int) -> None:
        r = await self._c.delete(f"/api/correspondents/{correspondent_id}/")
        r.raise_for_status()

    async def delete_document(self, doc_id: int) -> None:
        r = await self._c.delete(f"/api/documents/{doc_id}/")
        r.raise_for_status()

    async def delete_document_type(self, document_type_id: int) -> None:
        r = await self._c.delete(f"/api/document_types/{document_type_id}/")
        r.raise_for_status()

    async def bulk_update_documents(
        self,
        doc_ids: list[int],
        *,
        correspondent: int | None = None,
        document_type: int | None = None,
    ) -> None:
        """Reassign correspondent/document_type for multiple docs (used during cleanup)."""
        for doc_id in doc_ids:
            payload: dict = {}
            if correspondent is not None:
                payload["correspondent"] = correspondent
            if document_type is not None:
                payload["document_type"] = document_type
            if payload:
                await self.update_document(doc_id, **payload)


def _parse_document(d: dict) -> Document:
    return Document(
        id=d["id"],
        title=d.get("title", ""),
        content=d.get("content", "") or "",
        created=d.get("created", ""),
        correspondent=d.get("correspondent"),
        document_type=d.get("document_type"),
        tags=d.get("tags", []),
        storage_path=d.get("storage_path"),
        archive_serial_number=d.get("archive_serial_number"),
        original_file_name=d.get("original_file_name"),
    )


async def _gather(*coros):
    """Run multiple coroutines concurrently and return results in order."""
    import asyncio
    return await asyncio.gather(*coros)
