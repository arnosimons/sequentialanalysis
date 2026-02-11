# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set

import importlib.resources as pkg_resources
import string as _string

import re


_DEFAULT_SECTION_LABELS: Dict[str, Dict[str, str]] = {
    "en": {
        "role": "Role",
        "background": "Background",
        "tasks": "Tasks",
        "inputs": "Inputs",
        "outputs": "Outputs",
    },
    "de": {
        "role": "Rolle",
        "background": "Hintergrund",
        "tasks": "Aufgaben",
        "inputs": "Eingaben",
        "outputs": "Ausgaben",
    },
}


def _labels_for_language(language: str) -> Dict[str, str]:
    language = (language or "en").lower()
    return _DEFAULT_SECTION_LABELS.get(language, _DEFAULT_SECTION_LABELS["en"])


@dataclass(frozen=True)
class Prompt:
    """
    Structured prompt sections that compose into a single instruction string.

    Provider-agnostic:
      - no chat message roles
      - no provider-specific formatting
    """

    language: str = "en"

    role: Optional[str] = None
    background: Optional[str] = None
    tasks: Optional[str] = None
    inputs: Optional[str] = None
    outputs: Optional[str] = None

    # Optional extra sections (heading -> body). Headings are used verbatim.
    extras: Mapping[str, str] = field(default_factory=dict)

    def compose(
        self,
        *,
        heading_level: int = 2,
        include_extras: bool = True,
        strip: bool = True,
    ) -> str:
        if heading_level < 1:
            raise ValueError("heading_level must be >= 1")

        labels = _labels_for_language(self.language)
        h = "#" * heading_level

        parts: List[str] = []

        def add(label: str, body: Optional[str]) -> None:
            if body is None:
                return
            body2 = body.strip()
            if not body2:
                return
            parts.append(f"{h} {label}\n\n{body2}".strip())

        add(labels["role"], self.role)
        add(labels["background"], self.background)
        add(labels["tasks"], self.tasks)
        add(labels["inputs"], self.inputs)
        add(labels["outputs"], self.outputs)

        if include_extras and self.extras:
            for k, v in self.extras.items():
                if v is None:
                    continue
                v2 = str(v).strip()
                if not v2:
                    continue
                parts.append(f"{h} {k}\n\n{v2}".strip())

        out = "\n\n".join(parts)
        return out.strip() if strip else out


class PromptTemplate:
    """
    Strict renderer for Python-format templates like "{ROUND}".

    Clean-slate design:
      - single placeholder style (Python format fields)
      - strict missing-variable errors
      - optional enforcement against unused or extra vars
    """

    def __init__(
        self,
        template: str,
        *,
        required_vars: Optional[Iterable[str]] = None,
        forbid_extra_vars: bool = False,
    ) -> None:
        self.template = template
        self.forbid_extra_vars = forbid_extra_vars

        if required_vars is None:
            self.required_vars = self._infer_required_vars(template)
        else:
            self.required_vars = set(required_vars)

    @staticmethod
    def _infer_required_vars(template: str) -> Set[str]:
        fmt = _string.Formatter()
        required: Set[str] = set()
        for _, field_name, _, _ in fmt.parse(template):
            if not field_name:
                continue
            # field_name can contain attribute/index access like "foo.bar" or "foo[0]"
            root = re.split(r"[.\[]", field_name, maxsplit=1)[0]
            if root:
                required.add(root)
        return required

    def render(self, vars: Mapping[str, Any]) -> str:
        missing = [k for k in sorted(self.required_vars) if k not in vars]
        if missing:
            raise ValueError(f"Missing template vars: {missing}")

        if self.forbid_extra_vars:
            extra = [k for k in sorted(vars.keys()) if k not in self.required_vars]
            if extra:
                raise ValueError(f"Extra template vars not used by template: {extra}")

        try:
            return self.template.format_map(vars)
        except KeyError as e:
            raise ValueError(f"Missing template var: {e.args[0]}") from e


class PackagePromptProvider:
    """
    Load prompt files from package resources.

    Default layout:
      sequentialanalysis/_prompts/<language>/<filename>

    If you want a single-language prototype:
      sequentialanalysis/_prompts/<filename>
    """

    def __init__(self, *, package: str = __name__.split(".", 1)[0], base_dir: str = "_prompts") -> None:
        self.package = package
        self.base_dir = base_dir

    def read_text(
        self,
        filename: str,
        *,
        language: Optional[str] = None,
        encoding: str = "utf-8",
    ) -> str:
        base = pkg_resources.files(self.package).joinpath(self.base_dir)
        path = base.joinpath(language, filename) if language else base.joinpath(filename)
        return path.read_text(encoding=encoding)