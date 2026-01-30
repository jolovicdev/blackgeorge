import json
import logging
from typing import Any

import chromadb
from chromadb.config import DEFAULT_DATABASE, DEFAULT_TENANT

from blackgeorge.memory.base import MemoryScope, MemoryStore
from blackgeorge.utils import utc_now

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 8000
DEFAULT_CHUNK_OVERLAP = 200


def _chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    if chunk_size <= 0:
        return [text]
    chunks: list[str] = []
    start = 0
    step = max(chunk_size - overlap, 1)
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += step
    return chunks


def _serialize_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True, default=str)


def _deserialize_value(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


class VectorMemoryStore(MemoryStore):
    def __init__(
        self,
        path: str,
        collection_name: str = "memories",
        tenant: str = DEFAULT_TENANT,
        database: str = DEFAULT_DATABASE,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self._path = path
        self._client = chromadb.PersistentClient(
            path=path,
            tenant=tenant,
            database=database,
        )
        self._collection_name = collection_name
        self._chunk_size = max(chunk_size, 1)
        self._chunk_overlap = max(chunk_overlap, 0)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _resolve_chunk_config(self, metadata: dict[str, Any]) -> tuple[int, int]:
        chunk_size_raw = metadata.get("chunk_size", self._chunk_size)
        chunk_overlap_raw = metadata.get("chunk_overlap", self._chunk_overlap)
        try:
            chunk_size = int(chunk_size_raw)
        except (TypeError, ValueError):
            chunk_size = self._chunk_size
        try:
            chunk_overlap = int(chunk_overlap_raw)
        except (TypeError, ValueError):
            chunk_overlap = self._chunk_overlap
        if chunk_size <= 0:
            chunk_size = self._chunk_size
        if chunk_overlap < 0:
            chunk_overlap = 0
        return chunk_size, chunk_overlap

    def write(self, key: str, value: Any, scope: MemoryScope) -> None:
        text = _serialize_value(value)
        chunks = _chunk_text(text, chunk_size=self._chunk_size, overlap=self._chunk_overlap)
        now = utc_now().isoformat()
        self._delete_by_key(key, scope)
        for i, chunk in enumerate(chunks):
            doc_id = f"{scope}:{key}:{i}"
            self._collection.add(
                ids=[doc_id],
                documents=[chunk],
                metadatas=[
                    {
                        "scope": scope,
                        "key": key,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "chunk_size": self._chunk_size,
                        "chunk_overlap": self._chunk_overlap,
                        "created_at": now,
                    }
                ],
            )

    def read(self, key: str, scope: MemoryScope) -> Any | None:
        results = self._collection.get(
            where={"$and": [{"scope": {"$eq": scope}}, {"key": {"$eq": key}}]},  # type: ignore[dict-item]
            include=["documents", "metadatas"],
        )
        if not results["ids"]:
            return None
        docs = results["documents"] or []
        metadatas = results["metadatas"] or []
        pairs: list[tuple[dict[str, Any], str]] = []
        for meta, doc in zip(metadatas, docs, strict=False):
            pairs.append((dict(meta), str(doc)))
        sorted_chunks = sorted(pairs, key=lambda x: int(x[0].get("chunk_index", 0)))
        chunk_meta = sorted_chunks[0][0] if sorted_chunks else {}
        chunk_size, chunk_overlap = self._resolve_chunk_config(chunk_meta)
        step = max(chunk_size - chunk_overlap, 1)
        overlap_size = chunk_size - step
        text_parts: list[str] = []
        for idx, (_, doc) in enumerate(sorted_chunks):
            if idx == 0:
                text_parts.append(doc)
            else:
                new_len = len(doc) - overlap_size
                if new_len > 0:
                    text_parts.append(doc[-new_len:])
        full_text = "".join(text_parts)
        return _deserialize_value(full_text)

    def search(
        self,
        query: str,
        scope: MemoryScope,
        top_k: int = 5,
    ) -> list[tuple[str, Any]]:
        results = self._collection.query(
            query_texts=[query],
            where={"scope": {"$eq": scope}},  # type: ignore[dict-item]
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        if not results["ids"] or not results["ids"][0]:
            return []
        seen_keys: set[str] = set()
        matches: list[tuple[str, Any]] = []
        ids = results["ids"][0]
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        for i, doc_id in enumerate(ids):
            meta = dict(metadatas[i]) if i < len(metadatas) else {}
            key = str(meta.get("key", doc_id))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            value = self.read(key, scope)
            if value is not None:
                matches.append((key, value))
        return matches

    def reset(self, scope: MemoryScope) -> None:
        try:
            results = self._collection.get(
                where={"scope": {"$eq": scope}},  # type: ignore[dict-item]
                include=[],
            )
            if results["ids"]:
                self._collection.delete(ids=results["ids"])
        except Exception as exc:
            logger.debug("Failed to reset memory scope %s: %s", scope, exc)

    def _delete_by_key(self, key: str, scope: MemoryScope) -> None:
        try:
            results = self._collection.get(
                where={"$and": [{"scope": {"$eq": scope}}, {"key": {"$eq": key}}]},  # type: ignore[dict-item]
                include=[],
            )
            if results["ids"]:
                self._collection.delete(ids=results["ids"])
        except Exception as exc:
            logger.debug("Failed to delete memory key %s in scope %s: %s", key, scope, exc)
