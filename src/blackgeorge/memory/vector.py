import hashlib
import json
import logging
import re
from typing import Any, cast

import chromadb
import numpy as np
from chromadb.api.types import Documents, Embeddable, EmbeddingFunction, Embeddings, Metadatas
from chromadb.config import DEFAULT_DATABASE, DEFAULT_TENANT
from chromadb.errors import NotFoundError
from numpy.typing import NDArray

from blackgeorge.memory.base import MemoryScope, MemoryStore
from blackgeorge.utils import utc_now

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 8000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_EMBEDDING_DIMENSIONS = 256
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


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


class DeterministicEmbeddingFunction(EmbeddingFunction[Embeddable]):
    def __init__(self, dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS) -> None:
        self._dimensions = max(int(dimensions), 1)

    def __call__(self, input: Embeddable) -> Embeddings:
        if not input:
            return []
        first_item = input[0]
        if isinstance(first_item, str):
            documents = cast(Documents, input)
            return [self._embed_document(document) for document in documents]
        raise TypeError("DeterministicEmbeddingFunction only supports text documents")

    @staticmethod
    def name() -> str:
        return "blackgeorge_deterministic_embedding"

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "DeterministicEmbeddingFunction":
        dimensions_raw = config.get("dimensions", DEFAULT_EMBEDDING_DIMENSIONS)
        try:
            dimensions = int(dimensions_raw)
        except (TypeError, ValueError):
            dimensions = DEFAULT_EMBEDDING_DIMENSIONS
        return DeterministicEmbeddingFunction(dimensions=dimensions)

    def get_config(self) -> dict[str, Any]:
        return {"dimensions": self._dimensions}

    def _embed_document(self, document: str) -> NDArray[np.float32]:
        vector = np.zeros(self._dimensions, dtype=np.float32)
        tokens = TOKEN_PATTERN.findall(document.lower())
        if not tokens:
            tokens = [document.lower()] if document else [""]
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            hashed = int.from_bytes(digest, byteorder="big", signed=False)
            index = hashed % self._dimensions
            direction = 1.0 if ((hashed >> 1) & 1) == 0 else -1.0
            vector[index] += direction
        magnitude = float(np.linalg.norm(vector))
        if magnitude == 0.0:
            vector[0] = 1.0
            return vector
        return vector / magnitude


def _deterministic_embedding_from_collection(
    collection: Any,
) -> DeterministicEmbeddingFunction | None:
    model = getattr(collection, "_model", None)
    if model is None:
        return None
    configuration_json = getattr(model, "configuration_json", None)
    if not isinstance(configuration_json, dict):
        return None
    embedding_function_config = configuration_json.get("embedding_function")
    if not isinstance(embedding_function_config, dict):
        return None
    name = embedding_function_config.get("name")
    if name != DeterministicEmbeddingFunction.name():
        return None
    config = embedding_function_config.get("config", {})
    if not isinstance(config, dict):
        config = {}
    dimensions_raw = config.get("dimensions", DEFAULT_EMBEDDING_DIMENSIONS)
    try:
        dimensions = int(dimensions_raw)
    except (TypeError, ValueError):
        dimensions = DEFAULT_EMBEDDING_DIMENSIONS
    return DeterministicEmbeddingFunction(dimensions=dimensions)


class VectorMemoryStore(MemoryStore):
    def __init__(
        self,
        path: str,
        collection_name: str = "memories",
        tenant: str = DEFAULT_TENANT,
        database: str = DEFAULT_DATABASE,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        embedding_function: EmbeddingFunction[Embeddable] | None = None,
    ) -> None:
        self._path = path
        self._embedding_function = embedding_function
        self._client = chromadb.PersistentClient(
            path=path,
            tenant=tenant,
            database=database,
        )
        self._collection_name = collection_name
        self._chunk_size = max(chunk_size, 1)
        self._chunk_overlap = max(chunk_overlap, 0)
        try:
            existing_collection = self._client.get_collection(name=collection_name)
            if self._embedding_function is None:
                self._embedding_function = _deterministic_embedding_from_collection(
                    existing_collection
                )
            if self._embedding_function is None:
                self._collection = existing_collection
            else:
                self._collection = self._client.get_collection(
                    name=collection_name,
                    embedding_function=self._embedding_function,
                )
        except NotFoundError:
            if self._embedding_function is None:
                self._embedding_function = DeterministicEmbeddingFunction()
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=self._embedding_function,
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
        ids: list[str] = []
        metadatas: Metadatas = []
        for i, _ in enumerate(chunks):
            ids.append(f"{scope}:{key}:{i}")
            metadatas.append(
                {
                    "scope": scope,
                    "key": key,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_size": self._chunk_size,
                    "chunk_overlap": self._chunk_overlap,
                    "created_at": now,
                }
            )
        self._collection.add(
            ids=ids,
            documents=chunks,
            metadatas=metadatas,
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
            self._collection.delete(
                where={"scope": {"$eq": scope}},  # type: ignore[dict-item]
            )
        except Exception as exc:
            logger.debug("Failed to reset memory scope %s: %s", scope, exc)

    def _delete_by_key(self, key: str, scope: MemoryScope) -> None:
        try:
            self._collection.delete(
                where={"$and": [{"scope": {"$eq": scope}}, {"key": {"$eq": key}}]},  # type: ignore[dict-item]
            )
        except Exception as exc:
            logger.debug("Failed to delete memory key %s in scope %s: %s", key, scope, exc)
