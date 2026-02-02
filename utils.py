# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Union


def slugify_short(text: str, max_len: int = 40) -> str:
    short = " ".join(text[:max_len].split()).strip()
    if not short:
        return ""

    short = unicodedata.normalize("NFC", short).lower()

    slug = re.sub(r"\s+", "-", short)

    # Windows doens't like some symbols
    slug = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "", slug)

    # replace repeating dashes by single dashes
    slug = re.sub(r"-{2,}", "-", slug)

    # Windows don't like some symbols at the end
    slug = slug.strip(" .-_")

    return slug or ""


def remove_responses_meta(analyse):
    d = dict(analyse)
    for r in d['rounds']:
        r.pop("responses_meta", None)
            
    return d


def remove_meta(analyse):
    d = dict(analyse)
    d.pop("model_config", None)
            
    return d


def save_as_json(
    data: Dict[str, Any],
    output_dir: Path | str = ".",
    keep_meta: bool = True,
    keep_responses_meta: bool = True,
    max_len: int = 40,
    encoding: str = "utf-8",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outer_context = data["outer_context"]

    extra = ""
    if not keep_meta:
        data = remove_meta(data)
        extra += "--no-meta"
    if not keep_responses_meta:
        data = remove_responses_meta(data)
        extra += "--no-responses_meta"

    ts = data["meta"]["timestamp"]
    file_name = (
        f"sequentialanalysis--"
        f"{slugify_short(outer_context, max_len=max_len)}"
        f"--{ts}{extra}.json")
    path = output_dir / file_name

    with path.open("w", encoding=encoding) as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return path


def load_and_chunk_protocol(
        txt_file_path: Union[str, Path], 
        sep: str = "[SEP]"
) -> List[str]:
    path = Path(txt_file_path)
    txt_as_str = path.read_text(encoding="utf-8")
    sequences = txt_as_str.split(sep)

    return sequences
