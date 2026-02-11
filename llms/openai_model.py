from __future__ import annotations

import os
import json
from typing import Any, Optional, Type, TypeVar, cast

from pydantic import BaseModel

from ..config import SequentialAnalysisConfig

try:
    from openai import OpenAI as OpenAIClient  # type: ignore
except Exception:  # pragma: no cover
    OpenAIClient = None  # type: ignore


T = TypeVar("T", bound=BaseModel)


class OpenAIModel:
    """
    OpenAI LLM wrapper implementing the `parse(...)` contract expected by SequentialAnalysis.

    Responsibilities:
      - owns auth, base_url, and SDK client details
      - formats messages for OpenAI
      - uses config.model and other config fields to make the request
      - returns a validated Pydantic instance of response_model
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        client: Optional[Any] = None,
    ) -> None:
        if client is not None:
            self.client = client
            return

        if OpenAIClient is None:
            raise ImportError("openai is not available. Install it to use OpenAIModel.")

        kwargs: dict[str, Any] = {}
        if api_key is not None:
            kwargs["api_key"] = api_key
        if base_url is not None:
            kwargs["base_url"] = base_url

        self.client = OpenAIClient(**kwargs)

    @classmethod
    def from_env(cls, *, api_key_env: str = "OPENAI_API_KEY", base_url_env: str = "OPENAI_BASE_URL") -> "OpenAIModel":
        api_key = os.getenv(api_key_env)
        base_url = os.getenv(base_url_env) or None
        return cls(api_key=api_key, base_url=base_url)

    def parse(
        self,
        *,
        instruction: str,
        user_input: str,
        response_model: Type[T],
        config: SequentialAnalysisConfig,
    ) -> T:
        # Prefer Responses API structured parsing if available.
        if not hasattr(self.client, "responses") or not hasattr(self.client.responses, "parse"):
            raise RuntimeError("OpenAI client does not support responses.parse. Update the openai package.")

        response = self.client.responses.parse(
            input=[
                {"role": "developer", "content": instruction},
                {"role": "user", "content": user_input},
            ],
            text_format=response_model,
            model=config.model,
            max_output_tokens=config.max_output_tokens,
            reasoning={"effort": config.reasoning_effort, "summary": config.reasoning_summary},
            temperature=config.temperature,
            tool_choice=config.tool_choice,
            store=config.store,
        )

        parsed = getattr(response, "output_parsed", None)
        if parsed is not None:
            return cast(T, parsed)

        text = getattr(response, "output_text", None)
        if text:
            data = json.loads(text)
            return response_model.model_validate(data)

        raise ValueError("OpenAI response has neither output_parsed nor output_text.")
