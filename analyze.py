from __future__ import annotations

import datetime as _dt
import json
import warnings
from pprint import pprint
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, Literal
from typing import Callable, Protocol, cast, get_args, get_origin
from pydantic import BaseModel, ValidationError

try:
    # Pydantic v2 sentinel for "no default"
    from pydantic.fields import PydanticUndefined  # type: ignore
except Exception:  # pragma: no cover
    PydanticUndefined = object()  # type: ignore

from .config import SequentialAnalysisConfig
from .prompts import PackagePromptProvider
from .models import (
    ConfrontationInput,
    ConfrontationInputFirstRound,
    ContextConfrontation,
    ContextConfrontationFinalRound,
    ContextConfrontationFirstRound,
    ContextFreeReadings,
    ExampleSituations,
    Prediction,
)

"""Core SequentialAnalysis class and related"""

T = TypeVar("T", bound=BaseModel)


def _timestamp(now: Optional[_dt.datetime] = None) -> str:
    now = now or _dt.datetime.now()
    parts = [now.year, now.month, now.day, now.hour, now.minute, now.second]
    return "-".join(str(x) for x in parts)


def _to_json(data: Any) -> str:
    # default=str prevents crashes if an unexpected sentinel slips through
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def _empty_model(model_cls: Type[T]) -> T:
    """Create an "empty but schema-shaped" instance for fallbacks."""

    def empty_for(annotation: Any) -> Any:
        origin = get_origin(annotation)
        args = get_args(annotation)

        if origin is Union and args:
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                return empty_for(non_none[0])
            return None

        if origin is Literal and args:
            return args[0]

        if origin in (list, List):
            return []
        if origin in (dict, Dict):
            return {}

        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return _empty_model(annotation)

        if annotation is str:
            return ""
        if annotation is int:
            return 0
        if annotation is float:
            return 0.0
        if annotation is bool:
            return False

        return None

    data: Dict[str, Any] = {}
    for name, field in model_cls.model_fields.items():
        default_val = getattr(field, "default", PydanticUndefined)
        if default_val not in (PydanticUndefined, ...):
            data[name] = default_val
            continue

        default_factory = getattr(field, "default_factory", None)
        if default_factory is not None:
            data[name] = default_factory()
            continue

        data[name] = empty_for(field.annotation)

    try:
        return model_cls.model_validate(data)
    except Exception:
        return cast(T, model_cls.model_construct(**data))


class LLM(Protocol):
    def parse(
        self,
        *,
        instruction: str,
        user_input: str,
        response_model: Type[T],
        config: SequentialAnalysisConfig,
    ) -> T:
        ...


@dataclass(frozen=True)
class SequentialAnalysisResult:
    data: Dict[str, Any]


@dataclass(frozen=True)
class Stage3PromptBuilder:
    provider: PackagePromptProvider
    language: str

    def build(
            self, 
            *, 
            round: int, 
            total_rounds: int, 
            expert_knowledge: Optional[str] = None
    ) -> str:
        if total_rounds < 1:
            raise ValueError("total_rounds must be >= 1")
        if round < 1 or round > total_rounds:
            raise ValueError("round out of range")

        is_first = round == 1
        is_final = round == total_rounds

        base_text = self.provider.read_text("stage-3--base.txt", language=self.language)

        inputs_file = "stage-3--inputs--first.txt" if is_first else "stage-3--inputs--middle-final.txt"
        tasks_file = (
            "stage-3--tasks--first.txt" if is_first
            else "stage-3--tasks--final.txt" if is_final
            else "stage-3--tasks--middle.txt"
        )
        outputs_file = (
            "stage-3--outputs--first.txt" if is_first
            else "stage-3--outputs--final.txt" if is_final
            else "stage-3--outputs--middle.txt"
        )

        inputs_text = self.provider.read_text(inputs_file, language=self.language)
        tasks_text = self.provider.read_text(tasks_file, language=self.language)
        outputs_text = self.provider.read_text(outputs_file, language=self.language)

        expert_knowledge = (expert_knowledge or "")
        if expert_knowledge:
            exconusage_text = (
                self.provider.read_text("stage-3--tasks--expert.txt", language=self.language)
            )
            excon_text = expert_knowledge
        else:
            exconusage_text = ""
            excon_text = ""

        vars_map = {
            "ROUND": round,
            "INPUTS": inputs_text,
            "TASKS": tasks_text,
            "OUTPUTS": outputs_text,
            "EXCON": excon_text,
            "EXCONUSAGE": exconusage_text,
        }

        # Only the base template is formatted. Blocks are inserted as-is.
        # Any literal { or } in stage-3--base.txt must be escaped as {{ or }}.
        try:
            return base_text.format_map(vars_map).strip()
        except KeyError as e:
            raise ValueError(f"Missing template var in stage-3--base.txt: {e.args[0]}") from e


class SequentialAnalysis:
    def __init__(
        self,
        *,
        language: str = "en",
        config: Optional[SequentialAnalysisConfig] = None,
        llm: Optional[LLM] = None,
        prompt_provider: Optional[PackagePromptProvider] = None,
        # Printing control (two switches)
        print_runtime: bool = False,
        print_prompts: bool = False,
        callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        # Warnings (fallback behavior)
        warn_on_fallback: bool = True,
        warn_once: bool = True,
        # Legacy (currently unused)
        verbose: bool = False,
        verbose_outputs: bool = False,
    ) -> None:
        self.language = language
        self.config = config or SequentialAnalysisConfig()
        self.llm = llm
        self.prompt_provider = prompt_provider or PackagePromptProvider()

        # Printing
        self.print_runtime = print_runtime
        self.print_prompts = bool(print_prompts and print_runtime)
        self.callback = callback

        # Legacy (currently unused)
        self.verbose = verbose
        self.verbose_outputs = verbose_outputs

        # Warnings
        self.warn_on_fallback = warn_on_fallback
        self.warn_once = warn_once
        self._warned: set[str] = set()
        self._active_round: Optional[int] = None

        self._stage3_prompt_builder = Stage3PromptBuilder(
            provider=self.prompt_provider,
            language=self.language,
        )

    @classmethod
    def from_provider(
        cls,
        *,
        provider: str,
        model: str,
        language: str = "en",
        llm_kwargs: Optional[Dict[str, Any]] = None,
        prompt_provider: Optional[PackagePromptProvider] = None,
        print_runtime: bool = False,
        print_prompts: bool = False,
        callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        warn_on_fallback: bool = True,
        warn_once: bool = True,
        # Legacy (currently unused)
        verbose: bool = False,
        verbose_outputs: bool = False,
        **config_kwargs: Any,
    ) -> "SequentialAnalysis":
        """
        segments: List[str],
        *,
        outer_context: str,
        expert_knowledge: str = "",
        strip_whitespace: bool = True, # Strip whitespace of segments for analysis? (default: True)
        strip_linebreaks: bool = True, # Strip linebreaks of segments for analysis? (default: True)
        concat_segments_with: str = " ", # How to concatenate previous segments for inner context (default: single space, set to "" for no separation)

        """
        from .llms.factory import make_llm

        try:
            llm = cast(Optional[LLM], make_llm(provider, **(llm_kwargs or {})))
        except Exception as e:
            llm = None
            warnings.warn(
                f"Could not initialize provider '{provider}'. Returning empty outputs. "
                f"Details: {type(e).__name__}: {e}",
                RuntimeWarning,
                stacklevel=2,
            )

        config = SequentialAnalysisConfig(model=model, **config_kwargs)
        return cls(
            language=language,
            config=config,
            llm=llm,
            prompt_provider=prompt_provider,
            print_runtime=print_runtime,
            print_prompts=print_prompts,
            callback=callback,
            warn_on_fallback=warn_on_fallback,
            warn_once=warn_once,
            verbose=verbose,
            verbose_outputs=verbose_outputs,
        )

    def _emit(self, event: str, payload: Dict[str, Any]) -> None:
        """
        Runtime event hook.

        Controlled by two switches:
          - print_runtime: enable any runtime output at all
          - print_prompts: additionally emit/print assembled prompts (only if print_runtime is True)

        If callback is provided, it receives events instead of the built-in printer.
        """
        if not self.print_runtime:
            return

        # Second-level switch: suppress prompt events entirely unless enabled
        if event == "prompt" and not self.print_prompts:
            return

        if self.callback is not None:
            try:
                self.callback(event, payload)
            except Exception as e:
                warnings.warn(
                    f"Runtime callback failed ({type(e).__name__}: {e}). Continuing without callback output.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return

        self._print_event(event, payload)

    def _print_event(self, event: str, payload: Dict[str, Any]) -> None:
        if event == "analysis_start":
            print("=== Sequential Analysis: start ===")
            print(f'''language: "{payload.get('language')}"''')
            print(f"Model config:")
            pprint(payload.get("model_config"))
            print(f"total_rounds: {payload.get('total_rounds')}")
            print(f"expert_knowledge_provided: {payload.get('expert_knowledge_provided')}")
            print("segments:")
            pprint(payload.get("segments"))
            print("outer_context:")
            print(f'''"{payload.get('outer_context', '')}"''')

        elif event == "round_start":
            print("")
            r = payload.get("round")
            n = payload.get("total_rounds")
            print(f"=== Round {r}/{n} ===")

        elif event == "stage_start":
            stage = payload.get("stage")
            stage_num = {"tell_stories": 1, 
                         "form_readings": 2, 
                         "confront_with_context": 3}.get(stage, "?")
            print(f"Stage {stage_num}: {stage}")

        elif event == "analysis_end":
            print("")
            print("=== Sequential Analysis: done ===")

        elif event == "prompt":
            # Only emitted if print_prompts is True
            print("")
            print("--- Prompt ---")
            pprint(
                {
                    "round": payload.get("round"),
                    "stage": payload.get("stage"),
                    "model": payload.get("model"),
                    "instruction": payload.get("instruction"),
                    "user_input": payload.get("user_input"),
                }
            )
            print("")

    def _warn(self, key: str, message: str) -> None:
        if not self.warn_on_fallback:
            return
        if self.warn_once and key in self._warned:
            return
        self._warned.add(key)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    def _validate_expert_knowledge(self, expert_knowledge: Optional[str]) -> None:
        raw = expert_knowledge or ""
        if not raw.strip():
            return

        for line in raw.splitlines():
            if line.startswith("# ") or line.startswith("## "):
                raise ValueError(
                    "expert_knowledge must not contain Markdown headers level 1 or 2 "
                    "(lines starting with '# ' or '## '). Only '###' or deeper are allowed."
                )
    
    def _prepare_expert_knowledge_for_prompt(self, expert_knowledge: Optional[str]) -> str:
        raw = expert_knowledge or ""
        if not raw.strip():
            return ""

        self._validate_expert_knowledge(raw)

        if self.language == "de":
            prefix = "\n## Hintergrund: Expertenwissen\n\n"
        elif self.language == "en":
            prefix = "\n## Bachground: Expert Knowledge\n\n"
        else:
            prefix = "\n## Background: Expert Knowledge\n\n"

        return prefix + raw.strip() + "\n"
    
    def _prepare_segment(
            self, 
            segment: str, 
            strip_whitespace: bool, 
            strip_linebreaks: bool
    ) -> List[str]:
        strip_all = bool(strip_whitespace) and bool(strip_linebreaks)
        if strip_all:
            segment = segment.strip()
        elif strip_whitespace:
            segment = segment.strip(" \t\f\v")
        elif strip_linebreaks:
            segment = segment.strip("\n\r")
        
        return segment
    
    def _prepare_segments(
            self, 
            segments: List[str], 
            strip_whitespace: bool, 
            strip_linebreaks: bool
    ) -> List[str]:
        return [
            self._prepare_segment(s, strip_whitespace, strip_linebreaks) 
            for s in segments
        ]

    def analyze(
        self,
        segments: List[str],
        *,
        outer_context: str,
        expert_knowledge: str = "",
        strip_whitespace: bool = True, # Strip whitespace of segments for analysis? (default: True)
        strip_linebreaks: bool = True, # Strip linebreaks of segments for analysis? (default: True)
        concat_segments_with: str = " ", # How to concatenate previous segments for inner context (default: single space, set to "" for no separation)
    ) -> SequentialAnalysisResult:
        
        self._validate_expert_knowledge(expert_knowledge)

        total_rounds = len(segments)
        self._emit(
            "analysis_start",
            {
                "language": self.language,
                "segments": segments,
                "outer_context": outer_context,
                "expert_knowledge_provided": bool(expert_knowledge),
                "total_rounds": total_rounds,
                "strip_whitespace": strip_whitespace,
                "strip_linebreaks": strip_linebreaks,
            },
        )

        hypothesis = ""
        predictions_for_next_segment: List[Prediction] = []
        rounds: List[Dict[str, Any]] = []
        original_segments = segments
        prepared_segments = self._prepare_segments(segments, strip_whitespace, strip_linebreaks)
        
        for round, segment in enumerate(prepared_segments, start=1):

            # Build the inner context by concatenating all previous segments
            inner_context = concat_segments_with.join(original_segments[: round - 1])            
            self._active_round = round
            self._emit(
                "round_start", 
                {"round": round, "total_rounds": total_rounds}
            )

            # Stage 1: Tell stories about the current segment
            self._emit(
                "stage_start", 
                {"round": round, "stage": "tell_stories"}
            )
            stories = self.tell_stories(segment)

            # Stage 2: Form context-free readings based on the stories
            self._emit(
                "stage_start", 
                {"round": round, "stage": "form_readings"}
            )
            readings = self.form_readings(stories)

            # Stage 3: Confront the segment, stories, and readings with the context
            self._emit(
                "stage_start", 
                {"round": round, "stage": "confront_with_context"}
            )
            if round == 1:
                confrontation_input = ConfrontationInputFirstRound(
                    current_segment=segment,
                    outer_context=outer_context,
                    context_free_readings=readings,
                )
            else:
                confrontation_input = ConfrontationInput(
                    current_segment=segment,
                    outer_context=outer_context,
                    inner_context=inner_context,
                    hypothesis_from_the_previous_round=hypothesis,
                    context_free_readings=readings,
                    predictions_for_next_segment_from_the_previous_round=predictions_for_next_segment,
                )
            confrontation = self.confront_with_context(
                total_rounds=total_rounds,
                round=round,
                context=confrontation_input,
                expert_knowledge=expert_knowledge,
            )

            if round == 1:
                hypothesis = (
                    cast(ContextConfrontationFirstRound, confrontation)
                    .initial_case_structure_hypothesis
                )
            elif round < total_rounds:
                hypothesis = (
                    cast(ContextConfrontation, confrontation)
                    .updated_case_structure_hypothesis
                )
            else:
                hypothesis = (
                    cast(ContextConfrontationFinalRound, confrontation)
                    .final_case_structure_hypothesis
                )

            if round < total_rounds:
                predictions_for_next_segment = self._extract_predictions_for_next_segment(confrontation)
            else:
                predictions_for_next_segment = []

            rounds.append(
                {
                    "round": round,
                    "segment": segment,
                    "inner_context": inner_context,
                    "stories": stories.model_dump(),
                    "readings": readings.model_dump(),
                    "confrontation": confrontation.model_dump(),
                }
            )

        self._active_round = None
        self._emit("analysis_end", {"total_rounds": total_rounds})

        data: Dict[str, Any] = {
            "meta": {"timestamp": _timestamp()},
            "language": self.language,
            "model_config": asdict(self.config),
            "settings": {
                "strip_whitespace": strip_whitespace,
                "strip_linebreaks": strip_linebreaks,
                "concat_segments_with": concat_segments_with,
            },
            "inputs": {
                "segments": original_segments,
                "outer_context": outer_context,
                "expert_knowledge": bool(expert_knowledge),
            },
            "rounds": rounds,
        }
        return SequentialAnalysisResult(data=data)

    def _parse_or_empty(
        self,
        *,
        stage: str,
        instruction: str,
        user_input: str,
        response_model: Type[T],
    ) -> T:
        self._emit(

            "prompt",
            {
                "round": getattr(self, "_active_round", None),
                "stage": stage,
                "model": getattr(self.config, "model", None),
                "instruction": instruction,
                "user_input": user_input,
            },
        )
        if self.llm is None:
            self._warn(
                key="no_llm",
                message=(
                    "No LLM configured. Returning empty outputs. "
                    "Pass llm=... or use SequentialAnalysis.from_provider(...)."
                ),
            )
            return _empty_model(response_model)

        try:
            return self.llm.parse(
                instruction=instruction,
                user_input=user_input,
                response_model=response_model,
                config=self.config,
            )
        except Exception as e:
            model_name = getattr(self.config, "model", "<unknown>")
            self._warn(
                key=f"llm_call_failed:{stage}",
                message=(
                    f"LLM call failed in {stage} for model '{model_name}'. Returning empty outputs. "
                    f"Details: {type(e).__name__}: {e}"
                ),
            )
            return _empty_model(response_model)

    # Stage 1: Tell stories about the current segment
    def tell_stories(
            self, 
            segment: str,
            strip_whitespace: bool = True,
            strip_linebreaks: bool = True,
    ) -> ExampleSituations:
        segment = self._prepare_segment(segment, strip_whitespace, strip_linebreaks)
        prompt_text = (
            self.prompt_provider
                .read_text("stage-1.txt", language=self.language)
            )
        return self._parse_or_empty(
            stage="tell_stories",
            instruction=prompt_text.strip(),
            user_input=segment,
            response_model=ExampleSituations,
        )

    # Stage 2: Form context-free readings based on the stories
    def form_readings(
            self, 
            stories: Union[ExampleSituations, Dict[str, Any]]
    ) -> ContextFreeReadings:
        try:
            stories_obj = (
                stories if isinstance(stories, ExampleSituations) 
                else ExampleSituations.model_validate(stories)
            )
        except ValidationError:
            self._warn(
                key="invalid_stage2_intput",
                message="Stage 2 received invalid stories input. Returning empty outputs.",
            )
            return _empty_model(ContextFreeReadings)

        prompt_text = self.prompt_provider.read_text("stage-2.txt", 
                                                     language=self.language)
        user_input = _to_json(stories_obj.model_dump())

        return self._parse_or_empty(
            stage="form_readings",
            instruction=prompt_text.strip(),
            user_input=user_input,
            response_model=ContextFreeReadings,
        )

    # Stage 3: Confront the segment, stories, and readings with the context
    def confront_with_context(
        self,
        *,
        total_rounds: int,
        round: int,
        context: Union[ConfrontationInput, 
                       ConfrontationInputFirstRound,
                       Dict[str, Any]],
        expert_knowledge: Optional[str] = None,
    ) -> Union[ContextConfrontationFirstRound, 
               ContextConfrontation, 
               ContextConfrontationFinalRound]:
        
        InputModel = (
            ConfrontationInputFirstRound if round == 1 
            else ConfrontationInput
        )
        try:
            if isinstance(context, InputModel):
                context_obj = context
            elif isinstance(context, BaseModel):
                # In case a wrong Pydantic-Model is passed
                context_obj = InputModel.model_validate(context.model_dump())
            else:
                # In case a dict / mapping is passed
                context_obj = InputModel.model_validate(context)
        except ValidationError:
            self._warn(
                key="invalid_stage3_input",
                message="Stage 3 received invalid context input. Returning empty outputs.",
            )
            if round == 1:
                return _empty_model(ContextConfrontationFirstRound)
            if round == total_rounds:
                return _empty_model(ContextConfrontationFinalRound)
            return _empty_model(ContextConfrontation)
        
        expert_knowledge_for_prompt = self._prepare_expert_knowledge_for_prompt(expert_knowledge)

        instruction = self._stage3_prompt_builder.build(
            round=round,
            total_rounds=total_rounds,
            expert_knowledge=expert_knowledge_for_prompt,
        )

        if round == 1:
            schema: Type[BaseModel] = ContextConfrontationFirstRound
        elif round == total_rounds:
            schema = ContextConfrontationFinalRound
        else:
            schema = ContextConfrontation

        user_input = _to_json(context_obj.model_dump())

        return cast(
            Union[ContextConfrontationFirstRound, 
                  ContextConfrontation, 
                  ContextConfrontationFinalRound],
            self._parse_or_empty(
                stage="confront_with_context",
                instruction=instruction,
                user_input=user_input,
                response_model=cast(Type[T], schema),
            ),
        )

    @staticmethod
    def _extract_predictions_for_next_segment(
        confrontation: Union[ContextConfrontationFirstRound, 
                             ContextConfrontation, 
                             ContextConfrontationFinalRound]
    ) -> List[str]:
        if isinstance(confrontation, ContextConfrontationFinalRound):
            return []

        preds: List[Prediction] = []
        for reading in confrontation.context_informed_readings_with_predictions_for_next_round:
            preds.append(reading.most_plausible_prediction.next_segment)
            preds.append(reading.only_just_plausible_prediction.next_segment)
        return preds