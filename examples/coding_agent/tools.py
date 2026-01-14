from pathlib import Path

from blackgeorge.memory import VectorMemoryStore
from blackgeorge.tools import ToolResult, tool

PROJECT_ROOT = Path(__file__).resolve().parent / "project"
MEMORY_PATH = Path(__file__).resolve().parent / ".blackgeorge" / "memory"
_original_files: dict[Path, str] = {}
_memory_store: VectorMemoryStore | None = None


def get_memory_store() -> VectorMemoryStore:
    global _memory_store
    if _memory_store is None:
        MEMORY_PATH.mkdir(parents=True, exist_ok=True)
        _memory_store = VectorMemoryStore(str(MEMORY_PATH))
    return _memory_store


def _resolve_path(path: str) -> Path:
    candidate = (PROJECT_ROOT / path).resolve()
    if PROJECT_ROOT not in candidate.parents and candidate != PROJECT_ROOT:
        raise ValueError("Path outside project")
    return candidate


def _record_original(path: Path) -> None:
    if path not in _original_files:
        _original_files[path] = path.read_text(encoding="utf-8")


def modified_files() -> list[str]:
    return [str(path.relative_to(PROJECT_ROOT)) for path in _original_files]


def restore_modified_files() -> list[str]:
    paths = modified_files()
    for path, content in _original_files.items():
        path.write_text(content, encoding="utf-8")
    _original_files.clear()
    return paths


@tool(timeout=5.0)
def list_files() -> list[str]:
    return [
        str(path.relative_to(PROJECT_ROOT)) for path in PROJECT_ROOT.rglob("*") if path.is_file()
    ]


@tool(timeout=5.0, retries=2, retry_delay=0.5)
def read_file(path: str) -> str:
    resolved = _resolve_path(path)
    content = resolved.read_text(encoding="utf-8")
    store = get_memory_store()
    store.write(f"file:{path}", content, "project")
    return content


@tool(requires_confirmation=True, timeout=10.0)
def write_file(path: str, content: str) -> ToolResult:
    resolved = _resolve_path(path)
    _record_original(resolved)
    resolved.write_text(content, encoding="utf-8")
    store = get_memory_store()
    store.write(f"file:{path}", content, "project")
    return ToolResult(content=f"wrote {path}")


@tool(requires_user_input=True, user_input_prompt="Answer the question")
def ask_user(question: str, user_input: str) -> str:
    return user_input


@tool(timeout=3.0)
def search_docs(query: str) -> list[str]:
    store = get_memory_store()
    results = store.search(query, "project", top_k=3)
    output: list[str] = []
    for key, value in results:
        text = str(value)
        if len(text) > 200:
            output.append(f"{key}: {text[:200]}...")
        else:
            output.append(f"{key}: {text}")
    return output


@tool(timeout=2.0)
def remember(key: str, value: str) -> str:
    store = get_memory_store()
    store.write(key, value, "notes")
    return f"Remembered: {key}"


@tool(timeout=2.0)
def recall(query: str) -> list[str]:
    store = get_memory_store()
    results = store.search(query, "notes", top_k=5)
    return [f"{key}: {value}" for key, value in results]
