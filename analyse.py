"""Kernlogik für die Sequenzanalyse inklusive Prompt-Handling."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from openai import OpenAI
import importlib.resources as pkg_resources

from .config import SequenzAnalyseConfig
from .models import Beispielsituationen, KontextfreieLesarten
from .models import KonfrontationMitKontext
from .models import KonfrontationMitKontextErsteRunde
from .models import KonfrontationMitKontextLetzteRunde

from pprint import pprint


def _load_prompt_text(filename: str, encoding: str = "utf-8") -> str:
    """Lädt eine Prompt-Datei aus dem eingebetteten _prompts-Verzeichnis."""
    base = pkg_resources.files("sequenzanalyse").joinpath("_prompts")
    return base.joinpath(filename).read_text(encoding=encoding)


@dataclass(frozen=True)
class _PromptSet:
    """Trägt alle Prompt-Bausteine für die Analyse-Schritte."""
    prompt1_beispielsituationen: str
    prompt2_lesarten: str
    prompt3_konfrontation_template: str

    prompt3_konfrontation_rundeundziel_anfang: str
    prompt3_konfrontation_rundeundziel_mitte: str
    prompt3_konfrontation_rundeundziel_ende: str

    prompt3_konfrontation_eingabe_anfang: str
    prompt3_konfrontation_eingabe_mitte_ende: str

    prompt3_konfrontation_aufgabe_anfang: str
    prompt3_konfrontation_aufgabe_mitte: str
    prompt3_konfrontation_aufgabe_ende: str

    prompt3_konfrontation_ausgabe_anfang: str
    prompt3_konfrontation_ausgabe_mitte: str
    prompt3_konfrontation_ausgabe_ende: str


def _load_default_prompts() -> _PromptSet:
    """Erzeugt den standardisierten Satz an Prompt-Texten."""
    return _PromptSet(
        prompt1_beispielsituationen=_load_prompt_text("_Schritt-1--Beispielsituationen.txt"),
        prompt2_lesarten=_load_prompt_text("_Schritt-2--Lesarten.txt"),
        prompt3_konfrontation_template=_load_prompt_text("_Schritt-3--Konfrontation--TEMPLATE.txt"),

        prompt3_konfrontation_rundeundziel_anfang=_load_prompt_text("_Schritt-3--Konfrontation--RUNDEUNDZIEL--ANFANG.txt"),
        prompt3_konfrontation_rundeundziel_mitte=_load_prompt_text("_Schritt-3--Konfrontation--RUNDEUNDZIEL--MITTE.txt"),
        prompt3_konfrontation_rundeundziel_ende=_load_prompt_text("_Schritt-3--Konfrontation--RUNDEUNDZIEL--ENDE.txt"),

        prompt3_konfrontation_eingabe_anfang=_load_prompt_text("_Schritt-3--Konfrontation--EINGABE--ANFANG.txt"),
        prompt3_konfrontation_eingabe_mitte_ende=_load_prompt_text("_Schritt-3--Konfrontation--EINGABE--MITTE-ENDE.txt"),

        prompt3_konfrontation_aufgabe_anfang=_load_prompt_text("_Schritt-3--Konfrontation--AUFGABE--ANFANG.txt"),
        prompt3_konfrontation_aufgabe_mitte=_load_prompt_text("_Schritt-3--Konfrontation--AUFGABE--MITTE.txt"),
        prompt3_konfrontation_aufgabe_ende=_load_prompt_text("_Schritt-3--Konfrontation--AUFGABE--ENDE.txt"),

        prompt3_konfrontation_ausgabe_anfang=_load_prompt_text("_Schritt-3--Konfrontation--AUSGABE--ANFANG.txt"),
        prompt3_konfrontation_ausgabe_mitte=_load_prompt_text("_Schritt-3--Konfrontation--AUSGABE--MITTE.txt"),
        prompt3_konfrontation_ausgabe_ende=_load_prompt_text("_Schritt-3--Konfrontation--AUSGABE--ENDE.txt"),
    )

def _extract_result_and_meta(response: Any) -> tuple[Any, Dict[str, Any]]:
    """Extrahiert Ergebnisdaten und Metadaten aus einer Responses-API-Antwort."""
    # Meta robust extrahieren
    if hasattr(response, "to_dict") and callable(getattr(response, "to_dict")):
        meta = response.to_dict()
    elif hasattr(response, "model_dump") and callable(getattr(response, "model_dump")):
        meta = response.model_dump()
    elif hasattr(response, "dict") and callable(getattr(response, "dict")):
        meta = response.dict()
    else:
        meta = {}

    # Output rauswerfen (nur wenn vorhanden)
    if isinstance(meta, dict):
        for k in ("output", "text", "output_text"):
            meta.pop(k, None)

    # Result bevorzugt aus output_parsed
    parsed = getattr(response, "output_parsed", None)
    if parsed is not None:
        try:
            return parsed.model_dump(), meta
        except Exception:
            return parsed, meta

    # Fallback: output_text als JSON
    text = getattr(response, "output_text", None)
    if text:
        return json.loads(text), meta

    raise ValueError("Response enthält weder output_parsed noch output_text.")


@dataclass
class SequenzAnalyseErgebnis:
    """Container für das Ergebnis einer Sequenzanalyse."""
    data: Dict[str, Any]


class SequenzAnalyse:
    """Führt die Sequenzanalyse für eine Liste von Sequenzen aus."""

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        config: Optional[SequenzAnalyseConfig] = None,
        verbose: bool = True,
        verbose_outputs: bool = False,
    ) -> None:
        self.client = client or OpenAI()
        self.config = config or SequenzAnalyseConfig()
        self.verbose = verbose
        self.verbose_outputs = verbose_outputs
        self._prompts = _load_default_prompts()

    def _common_parse_args(self) -> Dict[str, Any]:
        """Bündelt Standardparameter für API-Aufrufe."""
        return {
            "model": self.config.model,
            "max_output_tokens": self.config.max_output_tokens,
            "reasoning": {"effort": self.config.reasoning_effort, "summary": self.config.reasoning_summary},
            "store": self.config.store,
            "temperature": self.config.temperature,
            "tool_choice": self.config.tool_choice,
        }

    def analyse(self, sequenzen: List[str], äußerer_kontext: str) -> SequenzAnalyseErgebnis:
        """Analysiert Sequenzen gegen einen äußeren Kontext und sammelt Ergebnisse."""
        laufende_analyse: Dict[str, Any] = {
            "meta": {
                "config": asdict(self.config),
            },
            "sequenzen": sequenzen,
            "äußerer_kontext": äußerer_kontext,
            "runden": [],
        }

        letzte_runde = len(sequenzen)

        print("=== Sequenzanalyze ===\n======================")
        print("\nÄußerer Kontext:")
        pprint(äußerer_kontext)
        print("\nSequenzen:")
        pprint(sequenzen)
        print(f"\nRunden: {len(sequenzen)}")

        for runde, neue_sequenz in enumerate(sequenzen, 1):
            if self.verbose:
                print(f"\n\n=== Runde {runde} ===")
                print("\nNeue Sequenz:")
                pprint(neue_sequenz)

            bisheriges_protokoll = " ".join(sequenzen[: runde - 1])

            ergebnisse_dieser_runde: Dict[str, Any] = {
                "runde": runde,
                "bisheriges_protokoll": bisheriges_protokoll,
                "neue_sequenz": neue_sequenz,
                "ergebnisse": [],
                "responses_meta": [],
            }

            situationenerzählungen, meta1 = self._schritt1(neue_sequenz)
            ergebnisse_dieser_runde["ergebnisse"].append(situationenerzählungen)
            ergebnisse_dieser_runde["responses_meta"].append(meta1)

            kontextfreie_lesarten, meta2 = self._schritt2(situationenerzählungen)
            ergebnisse_dieser_runde["ergebnisse"].append(kontextfreie_lesarten)
            ergebnisse_dieser_runde["responses_meta"].append(meta2)

            konfrontation_mit_kontext, meta3 = self._schritt3(
                runde=runde,
                letzte_runde=letzte_runde,
                neue_sequenz=neue_sequenz,
                bisheriges_protokoll=bisheriges_protokoll,
                äußerer_kontext=äußerer_kontext,
                laufende_analyse=laufende_analyse,
                kontextfreie_lesarten=kontextfreie_lesarten,
            )
            ergebnisse_dieser_runde["ergebnisse"].append(konfrontation_mit_kontext)
            ergebnisse_dieser_runde["responses_meta"].append(meta3)

            laufende_analyse["runden"].append(ergebnisse_dieser_runde)

        if self.verbose:
            print(f"\n\n=== ENDE ===")

        return SequenzAnalyseErgebnis(data=laufende_analyse)

    def _schritt1(self, neue_sequenz: str) -> Dict[str, Any]:
        """Erstellt kontextfreie Beispielsituationen zur neuen Sequenz."""
        if self.verbose:
            print('\nSchritt 1: Beispielsituationen erzählen ("kontextfrei")')

        response = self.client.responses.parse(
            input=[
                {"role": "developer", "content": self._prompts.prompt1_beispielsituationen},
                {"role": "user", "content": neue_sequenz},
            ],
            text_format=Beispielsituationen,
            **self._common_parse_args(),
        )
        result, meta = _extract_result_and_meta(response)
        if self.verbose and self.verbose_outputs:
            pprint(result)

        return result, meta


    def _schritt2(self, situationenerzählungen: Dict[str, Any]) -> Dict[str, Any]:
        """Leitet kontextfreie Lesarten aus den Beispielsituationen ab."""
        if self.verbose:
            print('\nSchritt 2: Lesartenbildung ("kontextfrei")')

        response = self.client.responses.parse(
            input=[
                {"role": "developer", "content": self._prompts.prompt2_lesarten},
                {"role": "user", "content": str(situationenerzählungen)},
            ],
            text_format=KontextfreieLesarten,
            **self._common_parse_args(),
        )
        result, meta = _extract_result_and_meta(response)
        if self.verbose and self.verbose_outputs:
            pprint(result)

        return result, meta
    

    def _schritt3(
        self,
        runde: int,
        letzte_runde: int,
        neue_sequenz: str,
        bisheriges_protokoll: str,
        äußerer_kontext: str,
        laufende_analyse: Dict[str, Any],
        kontextfreie_lesarten: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Konfrontiert Lesarten mit Kontext und erzeugt Fallstrukturhypothesen."""
        if self.verbose:
            print("\nSchritt 3: Kontrastierung mit dem tatsächlichen Kontext")

        vorläufige_fallstrukturhypothese = ""
        erwartete_fortführungen: List[Dict[str, Any]] = []

        if runde > 1:
            prev = laufende_analyse["runden"][runde - 2]["ergebnisse"][2]
            vorläufige_fallstrukturhypothese = (
                prev["erste_fallstrukturhypothese"] if runde == 2
                else prev["neue_fallstrukturhypothese"]
            )
            erwartete_fortführungen = prev.get("neue_fortführungen", [])

        kontext: Dict[str, Any] = {
            "sequenz": neue_sequenz,
            "tatsächlicher_kontext": {
                "äußerer_kontext": äußerer_kontext,
                "innerer_kontext": bisheriges_protokoll,
            },
            "erwartete_fortführungen": erwartete_fortführungen,
            "alte_fallstrukturhypothese": vorläufige_fallstrukturhypothese,
            "kontextfreie_lesarten": kontextfreie_lesarten,
        }

        if runde == 1:
            kontext["tatsächlicher_kontext"].pop("innerer_kontext")
            kontext.pop("erwartete_fortführungen")
            kontext.pop("alte_fallstrukturhypothese")
        
        rundeundziel = (
            self._prompts.prompt3_konfrontation_rundeundziel_anfang if runde == 1
            else self._prompts.prompt3_konfrontation_rundeundziel_mitte if runde < letzte_runde
            else self._prompts.prompt3_konfrontation_rundeundziel_ende
        )

        eingabe = (
            self._prompts.prompt3_konfrontation_eingabe_anfang if runde == 1
            else self._prompts.prompt3_konfrontation_eingabe_mitte_ende
        )

        aufgabe = (
            self._prompts.prompt3_konfrontation_aufgabe_anfang if runde == 1
            else self._prompts.prompt3_konfrontation_aufgabe_mitte if runde < letzte_runde
            else self._prompts.prompt3_konfrontation_aufgabe_ende
        )

        ausgabe = (
            self._prompts.prompt3_konfrontation_ausgabe_anfang if runde == 1
            else self._prompts.prompt3_konfrontation_ausgabe_mitte if runde < letzte_runde
            else self._prompts.prompt3_konfrontation_ausgabe_ende
        )
    
        dev_prompt = (
            self._prompts.prompt3_konfrontation_template.replace("[RUNDE]", str(runde))
            .replace("[RUNDEUNDZIEL]", rundeundziel)
            .replace("[EINGABE]", eingabe)
            .replace("[AUFGABE]", aufgabe)
            .replace("[AUSGABE]", ausgabe)
            .strip()
        )

        schema = (
            KonfrontationMitKontextErsteRunde if runde == 1
            else KonfrontationMitKontext if runde < letzte_runde
            else KonfrontationMitKontextLetzteRunde
        )

        response = self.client.responses.parse(
            input=[
                {"role": "developer", "content": dev_prompt},
                {"role": "user", "content": str(kontext)},
            ],
            text_format=schema,
            **self._common_parse_args(),
        )
        result, meta = _extract_result_and_meta(response)
        if self.verbose and self.verbose_outputs:
            pprint(result)

        return result, meta


def analyse(sequenzen: List[str], äußerer_kontext: str, config: Optional[SequenzAnalyseConfig] = None) -> Dict[str, Any]:
    """Kurzfunktion für die Analyse ohne direkte Klassennutzung."""
    return SequenzAnalyse(config=config, verbose=False).analyse(sequenzen, äußerer_kontext).data
