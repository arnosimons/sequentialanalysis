# -*- coding: utf-8 -*-
from __future__ import annotations

import datetime as _dt
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Optional, List, Union


def make_timestamp(now: Optional[_dt.datetime] = None) -> str:
    """
    Generiert einen aktuellen Zeitstempel
    """
    now = now or _dt.datetime.now()
    parts = [now.year, now.month, now.day, now.hour, now.minute, now.second]
    return "-".join(str(x) for x in parts)


def slugify_short(text: str, max_len: int = 40) -> str:
    """
    Macht aus freiem Text einen dateisystemfreundlichen Kurz-Slug.
    Umlaute bleiben erhalten, problematische Zeichen (Windows + Pfadtrenner) werden entfernt.
    """
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
    """
    Speichert eine Sequenzanalyse als formatiertes JSON in einer Datei.

    Es wird (falls nötig) ein Ausgabeverzeichnis angelegt. Der Dateiname wird aus
    einem kurzen, bereinigten Ausschnitt des äußeren Kontexts sowie einem Zeitstempel
    gebildet, z.B.:
        sequenzanalyse--ein-gespraech-zwischen-mutter--2026-1-28-14-22-05.json

    Parameter:
        analyse: Analyse-Datenstruktur (verschachtelte Dicts/Listen), die als JSON gespeichert werden soll
        äußerer_kontext: Kontext-String, der zur Benennung der Datei herangezogen wird
        output_dir: Zielordner für die JSON-Datei (Default: aktuelles Verzeichnis)
        max_len: Maximale Länge des Kontext-Ausschnitts, der im Dateinamen verwendet wird
        timestamp: Optionaler Zeitstempel (wenn None, wird ein aktueller Zeitstempel erzeugt)
        encoding: Zeichenkodierung für das Schreiben der Datei (Default: "utf-8")

    Rückgabe:
        Pfad zur geschriebenen JSON-Datei (Path)
    """
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
    """
    Liest eine Textdatei ein und zerlegt sie anhand eines Trennzeichens in Sequenzen.

    Vorgehen:
    - Datei wird standardmäßig als UTF-8 gelesen
    - Der Inhalt wird mit `sep` gesplittet

    Parameter:
        txt_file_path: Pfad zur .txt Datei (str oder Path)
        sep: Trennzeichen, an dem die Sequenzgrenzen erkannt werden (Default: "[SEP]")

    Rückgabe:
        Liste von Sequenzen (List[str])
    """
    path = Path(txt_file_path)
    txt_as_str = path.read_text(encoding="utf-8")
    sequenzen = txt_as_str.split(sep)

    return sequenzen
