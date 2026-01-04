# Memory

Blackgeorge defines a simple memory interface that can be used by your own tools or custom components.

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

## ExternalMemoryStore

`ExternalMemoryStore` is a stub you can implement if you want to integrate with another storage system.
