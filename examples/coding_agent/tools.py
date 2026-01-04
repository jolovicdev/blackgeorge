from pathlib import Path

from blackgeorge.tools import ToolResult, tool

PROJECT_ROOT = Path(__file__).resolve().parent / "project"
_original_files: dict[Path, str] = {}


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


@tool()
def list_files() -> list[str]:
    return [
        str(path.relative_to(PROJECT_ROOT)) for path in PROJECT_ROOT.rglob("*") if path.is_file()
    ]


@tool()
def read_file(path: str) -> str:
    resolved = _resolve_path(path)
    return resolved.read_text(encoding="utf-8")


@tool(requires_confirmation=True)
def write_file(path: str, content: str) -> ToolResult:
    resolved = _resolve_path(path)
    _record_original(resolved)
    resolved.write_text(content, encoding="utf-8")
    return ToolResult(content=f"wrote {path}")


@tool(requires_user_input=True, user_input_prompt="Answer the question")
def ask_user(question: str, user_input: str) -> str:
    return user_input
