from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Union
from collections.abc import Mapping

from pydantic import BaseModel



from typing import Any

# Compiled once for speed and consistency
_WS_RE = re.compile(r"\s+")
_MULTI_DASH_RE = re.compile(r"-{2,}")
# Windows forbidden chars + ASCII control chars
_WINDOWS_FORBIDDEN_RE = re.compile(r'[<>:"/\\|?*\x00-\x1F]')


def slugify_short(text: str, max_len: int = 40) -> str:
    """Create a short filename-safe slug from the first max_len characters of text."""
    short = " ".join(text[:max_len].split()).strip()
    if not short:
        return ""

    short = unicodedata.normalize("NFC", short).lower()
    slug = _WS_RE.sub("-", short)
    slug = _WINDOWS_FORBIDDEN_RE.sub("", slug)
    slug = _MULTI_DASH_RE.sub("-", slug)
    slug = slug.strip(" .-_")  # Windows also dislikes trailing dot/space

    return slug or ""


def _coerce_result_payload(data: Any) -> dict[str, Any]:
    """
    Accept:
      - dict-like mappings (e.g. results.data)
      - Pydantic models (anything with .model_dump()), like ExampleSituations
      - objects with a `.data` attribute that is a Mapping (e.g. SequentialAnalysisResult)
    Return a plain dict for JSON writing.
    """
    # Already dict-like
    if isinstance(data, Mapping):
        return dict(data)

    # Pydantic v2 models (and your stage outputs): ExampleSituations, ContextFreeReadings, etc.
    model_dump = getattr(data, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dict(dumped)
        raise TypeError(f"model_dump() did not return a mapping (got {type(dumped).__name__})")

    # SequentialAnalysisResult-style wrapper
    inner = getattr(data, "data", None)
    if isinstance(inner, Mapping):
        return dict(inner)

    raise TypeError(
        "save_as_json expects a dict-like object, a Pydantic model (model_dump), "
        f"or an object with a `.data` mapping (got {type(data).__name__})"
    )


def _json_default(obj: Any) -> Any:
    """Best-effort conversion to something JSON serializable."""
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, set):
        return list(obj)
    return str(obj)


def build_result_filename(
    result: Mapping[str, Any],
    *,
    prefix: str = "sequentialanalysis",
    max_slug_len: int = 40,
) -> str:
    """Build a stable, filesystem-safe filename for a SequentialAnalysis result."""
    inputs = result.get("inputs") or {}
    meta = result.get("meta") or {}

    outer_context = str(inputs.get("outer_context") or "")
    ts = str(meta.get("timestamp") or "no-timestamp")

    slug = slugify_short(outer_context, max_len=max_slug_len) or "no-outer-context"
    return f"{prefix}--{slug}--{ts}.json"


def save_as_json(
    data: Any,
    output_dir: Path | str = ".",
    *,
    filepath: Path | str | None = None,
    filename: str | None = None,
    max_slug_len: int = 40,
    encoding: str = "utf-8",
    indent: int = 2,
    ensure_unique: bool = True,
) -> Path:
    """
    Save a SequentialAnalysis result dict as JSON.

    Path selection precedence:
      1) If filepath is provided, save exactly there (overrides output_dir and filename).
         - If filepath points to an existing directory, an auto-generated filename is used inside it.
         - If filepath has no suffix, ".json" is added.
      2) Else if filename is provided, save to output_dir / filename (adds ".json" if missing).
      3) Else generate a readable filename from the result content and save to output_dir.

    ensure_unique=True appends --2, --3, ... only for auto-generated filenames.
    Explicit filepath/filename overwrites by default.
    """
    payload: dict[str, Any] = _coerce_result_payload(data)

    # Decide where to write
    if filepath is not None:
        path = Path(filepath)

        if path.exists() and path.is_dir():
            out_dir = path
            out_dir.mkdir(parents=True, exist_ok=True)
            auto_name = build_result_filename(payload, max_slug_len=max_slug_len)

            if ensure_unique:
                candidate = out_dir / auto_name
                counter = 2
                while candidate.exists():
                    candidate = out_dir / auto_name.replace(".json", f"--{counter}.json")
                    counter += 1
                path = candidate
            else:
                path = out_dir / auto_name

        else:
            if path.suffix == "":
                path = path.with_suffix(".json")
            path.parent.mkdir(parents=True, exist_ok=True)

    else:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if filename is not None:
            name = filename
            if Path(name).suffix == "":
                name = f"{name}.json"
            path = out_dir / name

        else:
            auto_name = build_result_filename(payload, max_slug_len=max_slug_len)
            path = out_dir / auto_name

            if ensure_unique:
                counter = 2
                while path.exists():
                    path = out_dir / auto_name.replace(".json", f"-{counter}.json")
                    counter += 1

    with path.open("w", encoding=encoding) as f:
        json.dump(payload, f, ensure_ascii=False, indent=indent, default=_json_default)

    return path


def load_and_chunk_protocol(
    txt_file_path: Union[str, Path],
    *,
    sep: str = "[SEP]",
    encoding: str = "utf-8",
    strip_chunks: bool = False,
    drop_empty: bool = True,
) -> list[str]:
    """
    Load a text file and split it into segments by a separator token.

    strip_chunks=True trims whitespace around each chunk.
    drop_empty=True removes empty chunks (common with leading/trailing separators).
    """
    path = Path(txt_file_path)
    text = path.read_text(encoding=encoding)

    chunks = text.split(sep)
    if strip_chunks:
        chunks = [c.strip() for c in chunks]
    if drop_empty:
        chunks = [c for c in chunks if c]

    return chunks