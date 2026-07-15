from pathlib import Path

import pytest
from pydantic import BaseModel

from blackgeorge.memory.vector import DeterministicEmbeddingFunction, VectorMemoryStore, _chunk_text


class VectorValue(BaseModel):
    name: str
    count: int


def test_chunk_text_small() -> None:
    text = "hello world"
    chunks = _chunk_text(text, chunk_size=100)
    assert chunks == ["hello world"]


def test_chunk_text_large() -> None:
    text = "a" * 1000
    chunks = _chunk_text(text, chunk_size=400, overlap=50)
    assert len(chunks) >= 3
    assert all(len(chunk) <= 400 for chunk in chunks)


def test_vector_memory_store_write_read(tmp_path: Path) -> None:
    path = tmp_path / "chroma_db"
    store = VectorMemoryStore(str(path))
    store.write("key1", {"value": 42}, "test_scope")
    result = store.read("key1", "test_scope")
    assert result == {"value": 42}


def test_vector_memory_store_serializes_pydantic_models(tmp_path: Path) -> None:
    store = VectorMemoryStore(str(tmp_path / "chroma_db"))

    store.write("model", VectorValue(name="saved", count=2), "scope")

    assert store.read("model", "scope") == {"name": "saved", "count": 2}


def test_vector_memory_store_write_string(tmp_path: Path) -> None:
    path = tmp_path / "chroma_db"
    store = VectorMemoryStore(str(path))
    store.write("key1", "hello world", "test_scope")
    result = store.read("key1", "test_scope")
    assert result == "hello world"


def test_vector_memory_store_read_missing(tmp_path: Path) -> None:
    path = tmp_path / "chroma_db"
    store = VectorMemoryStore(str(path))
    result = store.read("nonexistent", "test_scope")
    assert result is None


def test_vector_memory_store_search(tmp_path: Path) -> None:
    path = tmp_path / "chroma_db"
    store = VectorMemoryStore(str(path))
    store.write("doc1", "The quick brown fox jumps over the lazy dog", "scope1")
    store.write("doc2", "Python programming language is great for AI", "scope1")
    store.write("doc3", "Machine learning models need training data", "scope1")
    results = store.search("artificial intelligence programming", "scope1", top_k=2)
    assert len(results) >= 1
    keys = [r[0] for r in results]
    assert "doc2" in keys or "doc3" in keys


def test_vector_memory_store_search_scoped(tmp_path: Path) -> None:
    path = tmp_path / "chroma_db"
    store = VectorMemoryStore(str(path))
    store.write("doc1", "hello", "scope_a")
    store.write("doc2", "hello", "scope_b")
    results = store.search("hello", "scope_a")
    keys = [r[0] for r in results]
    assert "doc1" in keys
    assert "doc2" not in keys


def test_vector_memory_store_search_limit_validation(tmp_path: Path) -> None:
    store = VectorMemoryStore(str(tmp_path / "chroma_db"))
    store.write("doc", "hello", "scope")

    assert store.search("hello", "scope", top_k=0) == []
    with pytest.raises(ValueError, match="top_k must be non-negative"):
        store.search("hello", "scope", top_k=-1)


def test_vector_memory_store_reset(tmp_path: Path) -> None:
    path = tmp_path / "chroma_db"
    store = VectorMemoryStore(str(path))
    store.write("key1", "value1", "scope1")
    store.write("key2", "value2", "scope1")
    store.write("key3", "value3", "scope2")
    store.reset("scope1")
    assert store.read("key1", "scope1") is None
    assert store.read("key2", "scope1") is None
    assert store.read("key3", "scope2") == "value3"


def test_vector_memory_store_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "chroma_db"
    store = VectorMemoryStore(str(path))
    store.write("key1", "original", "scope1")
    store.write("key1", "updated", "scope1")
    result = store.read("key1", "scope1")
    assert result == "updated"


def test_vector_memory_store_ids_do_not_collide_on_delimiters(tmp_path: Path) -> None:
    store = VectorMemoryStore(str(tmp_path / "chroma_db"))

    store.write("c", "first", "a:b")
    store.write("b:c", "second", "a")

    assert store.read("c", "a:b") == "first"
    assert store.read("b:c", "a") == "second"


def test_vector_memory_store_chunked_document(tmp_path: Path) -> None:
    path = tmp_path / "chroma_db"
    store = VectorMemoryStore(str(path))
    long_text = "word " * 5000
    store.write("long_doc", long_text, "scope1")
    result = store.read("long_doc", "scope1")
    assert result == long_text


def test_chunk_text_overlap_exceeds_size() -> None:
    text = "abc" * 50
    chunks = _chunk_text(text, chunk_size=10, overlap=25)
    assert chunks
    assert len(chunks[0]) <= 10
    assert chunks[-1] == text[-10:]


def test_vector_memory_store_custom_chunking(tmp_path: Path) -> None:
    path = tmp_path / "chroma_db"
    store = VectorMemoryStore(str(path), chunk_size=50, chunk_overlap=10)
    text = "word " * 200
    store.write("doc", text, "scope1")
    result = store.read("doc", "scope1")
    assert result == text

    store_alt = VectorMemoryStore(str(path), chunk_size=10, chunk_overlap=0)
    result_alt = store_alt.read("doc", "scope1")
    assert result_alt == text


def test_vector_memory_store_overwrite_many_chunks(tmp_path: Path) -> None:
    path = tmp_path / "chroma_db"
    store = VectorMemoryStore(str(path), chunk_size=24, chunk_overlap=6)
    original = "abcdefghij " * 500
    updated = "replacement"
    store.write("key1", original, "scope1")
    store.write("key1", updated, "scope1")
    result = store.read("key1", "scope1")
    assert result == updated


def test_vector_memory_store_reset_large_scope(tmp_path: Path) -> None:
    path = tmp_path / "chroma_db"
    store = VectorMemoryStore(str(path), chunk_size=20, chunk_overlap=0)
    for index in range(150):
        store.write(f"key{index}", f"value-{index}", "scope1")
    store.reset("scope1")
    for index in range(150):
        assert store.read(f"key{index}", "scope1") is None


def test_deterministic_embedding_function_dimensions() -> None:
    function = DeterministicEmbeddingFunction(dimensions=12)
    embeddings = function(["hello world", "another document"])
    assert len(embeddings) == 2
    assert all(len(embedding) == 12 for embedding in embeddings)


def test_vector_memory_store_preserves_existing_collection_dimensions(tmp_path: Path) -> None:
    path = tmp_path / "chroma_db"
    legacy_store = VectorMemoryStore(
        str(path),
        embedding_function=DeterministicEmbeddingFunction(dimensions=384),
    )
    legacy_store.write("legacy", "legacy text", "scope1")

    upgraded_store = VectorMemoryStore(str(path))
    upgraded_store.write("new", "new text", "scope1")

    assert upgraded_store.read("legacy", "scope1") == "legacy text"
    assert upgraded_store.read("new", "scope1") == "new text"
