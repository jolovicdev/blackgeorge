# Memory

Blackgeorge defines a simple memory interface that can be used by your own tools or custom components.

## Desk integration

`Desk` always has a memory store. If you do not pass one, it uses `InMemoryMemoryStore`. For worker runs, the desk uses two conventional keys:

- `context`: read before a worker run and inserted as a system message
- `last_output`: written after a completed run

Both use the worker's `memory_scope` as the namespace. You can ignore these conventions or build on them for more advanced memory behavior.

## MemoryStore interface

A memory store supports:

- write(key, value, scope)
- read(key, scope)
- search(query, scope)
- reset(scope)

`scope` is a string namespace such as `worker:Analyst` or `desk`.

## InMemoryMemoryStore

The in-memory store keeps data in a dictionary. It is fast and does not persist data.

## SQLiteMemoryStore

The SQLite store persists memory to a file. Values are serialized as JSON strings and can be searched by key or value substring.

## VectorMemoryStore

The vector store uses ChromaDB for semantic search with embeddings. It persists locally and supports similarity-based retrieval. By default, Blackgeorge uses a deterministic local embedding function so memory operations work in offline or restricted environments. You can pass your own Chroma embedding function for provider-quality embeddings. ChromaDB is not part of the base installation. Install the vector extra before using this store:

```bash
uv add "blackgeorge[vector]"
```

```python
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from blackgeorge.memory import VectorMemoryStore

store = VectorMemoryStore(
    "/path/to/db",
    chunk_size=8000,
    chunk_overlap=200,
    embedding_function=OpenAIEmbeddingFunction(model_name="text-embedding-3-small"),
)

store.write("doc1", "AI is transforming healthcare", "global")
store.write("doc2", "Machine learning predicts outcomes", "global")

results = store.search("artificial intelligence medicine", "global", top_k=5)
for key, value in results:
    print(f"{key}: {value}")

doc = store.read("doc1", "global")
```

Features:

- Configurable chunking for long documents
- Default chunking: `chunk_size=8000`, `chunk_overlap=200`
- Cosine similarity for semantic matching
- Scope-based isolation between workers/runs
- JSON serialization for complex values
- Deterministic default embeddings for reproducible local behavior

## ExternalMemoryStore

`ExternalMemoryStore` is a stub you can implement if you want to integrate with another storage system.
