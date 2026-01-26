# Memory

Blackgeorge defines a simple memory interface that can be used by your own tools or custom components.

## Desk integration

When a `Desk` is configured with a `memory_store`, it uses two conventional keys:

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

The vector store uses ChromaDB for semantic search with embeddings. It persists locally and supports similarity-based retrieval.

```python
from blackgeorge.memory import VectorMemoryStore

store = VectorMemoryStore("/path/to/db", chunk_size=4000, chunk_overlap=200)

store.write("doc1", "AI is transforming healthcare", "global")
store.write("doc2", "Machine learning predicts outcomes", "global")

results = store.search("artificial intelligence medicine", "global", top_k=5)
for key, value in results:
    print(f"{key}: {value}")

doc = store.read("doc1", "global")
```

Features:

- Configurable chunking for long documents
- Cosine similarity for semantic matching
- Scope-based isolation between workers/runs
- JSON serialization for complex values

## ExternalMemoryStore

`ExternalMemoryStore` is a stub you can implement if you want to integrate with another storage system.
