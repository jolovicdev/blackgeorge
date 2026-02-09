import base64
import mimetypes
from pathlib import Path


def encode_file(file_path: str, mime_type: str | None = None) -> str:
    if mime_type is None:
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            ext = Path(file_path).suffix.lower()
            mime_type_map = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp",
                ".mp4": "video/mp4",
                ".avi": "video/x-msvideo",
                ".mov": "video/quicktime",
                ".mp3": "audio/mp3",
                ".wav": "audio/wav",
                ".pdf": "application/pdf",
                ".csv": "text/csv",
                ".doc": "application/msword",
                ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ".xls": "application/vnd.ms-excel",
                ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                ".html": "text/html",
                ".txt": "text/plain",
                ".md": "text/markdown",
            }
            mime_type = mime_type_map.get(ext, "application/octet-stream")

    with open(file_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"
