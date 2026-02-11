from __future__ import annotations

from typing import Any, Optional

from .openai_model import OpenAIModel


def make_llm(provider: str, **kwargs: Any) -> Any:
    """
    Create an LLM wrapper instance from a simple provider string.

    Examples:
      make_llm("openai")
      make_llm("openai", api_key="...", base_url="...")
    """
    p = (provider or "").strip().lower()

    if p == "openai":
        api_key: Optional[str] = kwargs.pop("api_key", None)
        base_url: Optional[str] = kwargs.pop("base_url", None)
        if kwargs:
            raise TypeError(f"Unknown OpenAIModel kwargs: {sorted(kwargs.keys())}")
        if api_key is None and base_url is None:
            return OpenAIModel.from_env()
        return OpenAIModel(api_key=api_key, base_url=base_url)

    raise ValueError(f"Unknown provider: {provider}")
