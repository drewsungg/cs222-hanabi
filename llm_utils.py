"""LLM utilities — unified interface for OpenAI and Anthropic."""

import json
import re
from typing import Dict, List

from openai import OpenAI
from anthropic import Anthropic

from config import OPENAI_API_KEY, ANTHROPIC_API_KEY, LLM_PROVIDER, LLM_MODEL

# Default models per provider
_DEFAULTS = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-5-20250929",
}


def create_client(provider: str | None = None, api_key: str | None = None):
    """Return an OpenAI or Anthropic client."""
    provider = (provider or LLM_PROVIDER).lower()
    if provider == "openai":
        return OpenAI(api_key=api_key or OPENAI_API_KEY)
    elif provider == "anthropic":
        client = Anthropic(api_key=api_key or ANTHROPIC_API_KEY)
        return client
    raise ValueError(f"Unknown provider: {provider}")


def generate(
    client,
    provider: str,
    messages: List[Dict],
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
) -> str:
    """Send messages to the LLM and return the text response."""
    provider = provider.lower()
    model = model or LLM_MODEL or _DEFAULTS.get(provider, "gpt-4o")

    if provider == "openai":
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    elif provider == "anthropic":
        # Anthropic expects system in a separate param
        system_msg = ""
        user_msgs = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                user_msgs.append(m)
        kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=user_msgs,
        )
        if system_msg:
            kwargs["system"] = system_msg
        resp = client.messages.create(**kwargs)
        return resp.content[0].text

    raise ValueError(f"Unknown provider: {provider}")


# ---- prompt / parsing helpers (kept from original) ----

def parse_json(response: str, target_keys=None) -> dict:
    """Best-effort JSON extraction from LLM output."""
    json_start = response.find("{")
    json_end = response.rfind("}") + 1
    if json_start == -1 or json_end == 0:
        return {}
    cleaned = response[json_start:json_end]
    try:
        parsed = json.loads(cleaned)
        if target_keys:
            parsed = {k: parsed.get(k, "") for k in target_keys}
        return parsed
    except json.JSONDecodeError:
        # Regex fallback
        parsed = {}
        for m in re.finditer(r'"(\w+)":\s*"(.*?)"', cleaned):
            key, val = m.group(1), m.group(2)
            if target_keys is None or key in target_keys:
                parsed[key] = val
        return parsed
