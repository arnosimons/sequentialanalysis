"""Hilfsfunktionen für Dateinamen, Export und Textsequenzierung."""

# -*- coding: utf-8 -*-
from __future__ import annotations

import datetime as _dt
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Optional, List, Union


def make_timestamp(now: Optional[_dt.datetime] = None) -> str:
    """Erzeugt einen Zeitstempel im kompakten Zahlenformat."""
    now = now or _dt.datetime.now()
    parts = [now.year, now.month, now.day, now.hour, now.minute, now.second]
    return "-".join(str(x) for x in parts)


def slugify_short(text: str, max_len: int = 40) -> str:
    """Erstellt einen dateisystemfreundlichen Kurz-Slug aus Freitext."""
    short = " ".join(text[:max_len].split()).strip()
    if not short:
        return "kontext"

    short = unicodedata.normalize("NFC", short).lower()

    slug = re.sub(r"\s+", "-", short)

    # Entferne Zeichen, die in Dateinamen oft Probleme machen (insb. Windows)
    slug = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "", slug)

    # Mehrfachstriche zusammenziehen
    slug = re.sub(r"-{2,}", "-", slug)

    # Windows mag keine endenden Punkte oder Spaces
    slug = slug.strip(" .-_")

    return slug or "kontext"


def remove_responses_meta(analyse):
    """Entfernt responses_meta aus den Analyse-Runden."""
    d = dict(analyse)
    for r in d['runden']:
        if hasattr(r, "responses_meta"):    
            r.pop("responses_meta")
    return d


def analyse_als_json_speichern(
    analyse: Dict[str, Any],
    äußerer_kontext: str,
    output_dir: Path | str = ".",
    remove_responses_meta: Bool = True,
    max_len: int = 40,
    timestamp: Optional[str] = None,
    encoding: str = "utf-8",
) -> Path:
    """Speichert eine Sequenzanalyse als formatiertes JSON im Zielordner."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = timestamp or make_timestamp()
    file_name = f"sequenzanalyse--{slugify_short(äußerer_kontext, max_len=max_len)}--{ts}.json"
    path = output_dir / file_name

    if remove_responses_meta:
        analyse = remove_responses_meta(analyse)

    with path.open("w", encoding=encoding) as f:
        json.dump(analyse, f, ensure_ascii=False, indent=4)

    return path


def txt_sequenzierung(
        txt_file_path: Union[str, Path], 
        sep: str = "[SEP]"
) -> List[str]:
    """Liest eine Textdatei und teilt sie anhand des Trennzeichens in Sequenzen."""
    path = Path(txt_file_path)
    txt_as_str = path.read_text(encoding="utf-8")
    sequenzen = txt_as_str.split(sep)

    return sequenzen
