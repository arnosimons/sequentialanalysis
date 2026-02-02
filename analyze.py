from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Literal

from openai import OpenAI
import importlib.resources as pkg_resources

from .config import SequentialAnalysisConfig
from .models import Beispielsituationen, KontextfreieLesarten
from .models import KonfrontationMitKontext
from .models import KonfrontationMitKontextErsteRunde
from .models import KonfrontationMitKontextLetzteRunde

import datetime as _dt

from pprint import pprint


def _load_prompt_text(filename: str, encoding: str = "utf-8") -> str:
    base = pkg_resources.files("sequentialanalysis").joinpath("_prompts")
    return base.joinpath(filename).read_text(encoding=encoding)


@dataclass(frozen=True)
class _PromptSet:
    prompt1_beispielsituationen: str
    prompt2_lesarten: str
    prompt3_konfrontation_template: str

    prompt3_konfrontation_eingabe_anfang: str
    prompt3_konfrontation_eingabe_mitte_ende: str

    prompt3_konfrontation_aufgabe_anfang: str
    prompt3_konfrontation_aufgabe_mitte: str
    prompt3_konfrontation_aufgabe_ende: str
    prompt3_konfrontation_aufgabe_expertenwissen: str

    prompt3_konfrontation_ausgabe_anfang: str
    prompt3_konfrontation_ausgabe_mitte: str
    prompt3_konfrontation_ausgabe_ende: str


def _load_default_prompts() -> _PromptSet:
    return _PromptSet(
        prompt1_beispielsituationen=_load_prompt_text("_Schritt-1--Beispielsituationen.txt"),
        prompt2_lesarten=_load_prompt_text("_Schritt-2--Lesarten.txt"),
        prompt3_konfrontation_template=_load_prompt_text("_Schritt-3--Konfrontation--TEMPLATE.txt"),

        prompt3_konfrontation_eingabe_anfang=_load_prompt_text("_Schritt-3--Konfrontation--EINGABE--ANFANG.txt"),
        prompt3_konfrontation_eingabe_mitte_ende=_load_prompt_text("_Schritt-3--Konfrontation--EINGABE--MITTE-ENDE.txt"),

        prompt3_konfrontation_aufgabe_anfang=_load_prompt_text("_Schritt-3--Konfrontation--AUFGABE--ANFANG.txt"),
        prompt3_konfrontation_aufgabe_mitte=_load_prompt_text("_Schritt-3--Konfrontation--AUFGABE--MITTE.txt"),
        prompt3_konfrontation_aufgabe_ende=_load_prompt_text("_Schritt-3--Konfrontation--AUFGABE--ENDE.txt"),
        prompt3_konfrontation_aufgabe_expertenwissen=_load_prompt_text("_Schritt-3--Konfrontation--AUFGABE--EXPERTENWISSEN.txt"),

        prompt3_konfrontation_ausgabe_anfang=_load_prompt_text("_Schritt-3--Konfrontation--AUSGABE--ANFANG.txt"),
        prompt3_konfrontation_ausgabe_mitte=_load_prompt_text("_Schritt-3--Konfrontation--AUSGABE--MITTE.txt"),
        prompt3_konfrontation_ausgabe_ende=_load_prompt_text("_Schritt-3--Konfrontation--AUSGABE--ENDE.txt"),
    )

def _extract_result_and_meta(response: Any) -> tuple[Any, Dict[str, Any]]:
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

    raise ValueError("Response includes neither output_parsed nor output_text.")

def _make_timestamp(now: Optional[_dt.datetime] = None) -> str:
    now = now or _dt.datetime.now()
    parts = [now.year, now.month, now.day, now.hour, now.minute, now.second]
    return "-".join(str(x) for x in parts)

@dataclass
class SequentialAnalysisResult:
    data: Dict[str, Any]


class SequentialAnalysis:

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        config: Optional[SequentialAnalysisConfig] = None,
        verbose_rounds: bool = True,
        verbose_outputs: bool = False,
    ) -> None:
        self.client = client or OpenAI()
        self.config = config or SequentialAnalysisConfig()
        self.verbose_rounds = verbose_rounds
        self.verbose_outputs = verbose_outputs
        self._prompts = _load_default_prompts()

    def _common_parse_args(self) -> Dict[str, Any]:
        return {
            "model": self.config.model,
            "max_output_tokens": self.config.max_output_tokens,
            "reasoning": {"effort": self.config.reasoning_effort, "summary": self.config.reasoning_summary},
            "store": self.config.store,
            "temperature": self.config.temperature,
            "tool_choice": self.config.tool_choice,
        }

    def analyze(self, 
                sequences: List[str], 
                outer_context: str,
                expert_context: Optional[str] = None,
                expert_context_enforcement: Literal["low", "medium", "high"] = "medium"
        ) -> SequentialAnalysisResult:
        if expert_context is not None:
            expert_context = (
                f"## Hintergrund: Expertenkontext\n\n{expert_context}")
        else:
            expert_context = ""
        
        analysis: Dict[str, Any] = {
            "meta":{
                "model_config": asdict(self.config),
                "timestamp":_make_timestamp(),
            },        
            "sequences": sequences,
            "outer_context": outer_context,
            "expert_context": expert_context,
            "rounds": [],
        }

        last_round = len(sequences)

        print("=== Sequential Analysis ===\n===========================")
        print("\nOuter context:")
        pprint(outer_context)
        print("\nSequences:")
        pprint(sequences)
        print(f"\nRounds: {len(sequences)}")
        if expert_context:
            print(f'\n-> NOTE: You provided expert context (enforcement="{expert_context_enforcement})')

        for round, new_sequence in enumerate(sequences, 1):
            if self.verbose_rounds:
                print(f"\n\n=== Round {round} ===")
                print("\nNew sequence:")
                pprint(new_sequence)

            inner_context = " ".join(sequences[: round - 1])

            results_current_round: Dict[str, Any] = {
                "round": round,
                "inner_context": inner_context,
                "new_sequence": new_sequence,
                "results": [],
                "responses_meta": [],
            }

            stories, meta1 = self.tell_stories(new_sequence)
            results_current_round["results"].append(stories)
            results_current_round["responses_meta"].append(meta1)

            context_free_readings, meta2 = self.form_readings(stories)
            results_current_round["results"].append(context_free_readings)
            results_current_round["responses_meta"].append(meta2)

            confrontation, meta3 = self.confront_with_context(
                round=round,
                last_round=last_round,
                new_sequence=new_sequence,
                inner_context=inner_context,
                outer_context=outer_context,
                analysis=analysis,
                context_free_readings=context_free_readings,
                expert_context=expert_context,
                expert_context_enforcement=expert_context_enforcement,
            )
            results_current_round["results"].append(confrontation)
            results_current_round["responses_meta"].append(meta3)

            analysis["rounds"].append(results_current_round)

        print(f"\n\n=== Finish ===")

        return SequentialAnalysisResult(data=analysis)

    def tell_stories(self, new_sequence: str) -> Dict[str, Any]:
        if self.verbose_rounds:
            print('\nStage 1: Storytelling ("context-free")')

        response = self.client.responses.parse(
            input=[
                {"role": "developer", "content": self._prompts.prompt1_beispielsituationen},
                {"role": "user", "content": new_sequence},
            ],
            text_format=Beispielsituationen,
            **self._common_parse_args(),
        )
        result, meta = _extract_result_and_meta(response)
        if self.verbose_rounds and self.verbose_outputs:
            pprint(result)

        return result, meta


    def form_readings(self, stories: Dict[str, Any]) -> Dict[str, Any]:
        if self.verbose_rounds:
            print('\nStage 2: Forming readings ("context-fre")')

        response = self.client.responses.parse(
            input=[
                {"role": "developer", "content": self._prompts.prompt2_lesarten},
                {"role": "user", "content": str(stories)},
            ],
            text_format=KontextfreieLesarten,
            **self._common_parse_args(),
        )
        result, meta = _extract_result_and_meta(response)
        if self.verbose_rounds and self.verbose_outputs:
            pprint(result)

        return result, meta
    

    def confront_with_context(
        self,
        round: int,
        last_round: int,
        new_sequence: str,
        inner_context: str,
        outer_context: str,
        analysis: Dict[str, Any],
        context_free_readings: Dict[str, Any],
        expert_context: Optional[str] = None,
        expert_context_enforcement: Literal["low", "medium", "high"] = "soll"
    ) -> Dict[str, Any]:
        if self.verbose_rounds:
            print("\nStage 3: Confrontation")

        hypothesis = ""
        expectations: List[Dict[str, Any]] = []

        if round > 1:
            prev = analysis["rounds"][round - 2]["results"][2]
            hypothesis = (
                prev["erste_fallstrukturhypothese"] if round == 2
                else prev["neue_fallstrukturhypothese"]
            )
            expectations = prev.get("neue_fortführungen", [])

        context: Dict[str, Any] = {
            "sequenz": new_sequence,
            "tatsächlicher_kontext": {
                "äußerer_kontext": outer_context,
                "innerer_kontext": inner_context,
            },
            "erwartete_fortführungen": expectations,
            "alte_fallstrukturhypothese": hypothesis,
            "kontextfreie_lesarten": context_free_readings,
        }

        if round == 1:
            context["tatsächlicher_kontext"].pop("innerer_kontext")
            context.pop("erwartete_fortführungen")
            context.pop("alte_fallstrukturhypothese")

        inputs = (
            self._prompts.prompt3_konfrontation_eingabe_anfang if round == 1
            else self._prompts.prompt3_konfrontation_eingabe_mitte_ende
        )

        tasks = (
            self._prompts.prompt3_konfrontation_aufgabe_anfang if round == 1
            else self._prompts.prompt3_konfrontation_aufgabe_mitte if round < last_round
            else self._prompts.prompt3_konfrontation_aufgabe_ende
        )

        outputs = (
            self._prompts.prompt3_konfrontation_ausgabe_anfang if round == 1
            else self._prompts.prompt3_konfrontation_ausgabe_mitte if round < last_round
            else self._prompts.prompt3_konfrontation_ausgabe_ende
        )
        if expert_context.strip():
            exconenf_mapping = {
                "low":"darfst",
                "medium":"sollst",
                "high":"musst",
            }
            exconusage = self._prompts.prompt3_konfrontation_aufgabe_expertenwissen
            exconusage = exconusage.replace(
                "[EXCONENF]", 
                exconenf_mapping[expert_context_enforcement])
        else:
            exconusage = ""
                
        dev_prompt = (
            self._prompts.prompt3_konfrontation_template
                .replace("[ROUND]", str(round))
                .replace("[INPUTS]", inputs)
                .replace("[EXCON]", expert_context)
                .replace("[TASKS]", tasks)
                .replace("[EXCONUSAGE]", exconusage)
                .replace("[OUTPUTS]", outputs)
                .strip()
        )

        pprint(dev_prompt)

        schema = (
            KonfrontationMitKontextErsteRunde if round == 1
            else KonfrontationMitKontext if round < last_round
            else KonfrontationMitKontextLetzteRunde
        )

        response = self.client.responses.parse(
            input=[
                {"role": "developer", "content": dev_prompt},
                {"role": "user", "content": str(context)},
            ],
            text_format=schema,
            **self._common_parse_args(),
        )
        result, meta = _extract_result_and_meta(response)
        if self.verbose_rounds and self.verbose_outputs:
            pprint(result)

        return result, meta
