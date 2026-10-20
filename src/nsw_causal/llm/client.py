"""Thin client for the local Strata-served LLM (OpenAI-compatible API)."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from ..config import Settings, get_settings


def extract_json(text: str) -> Any:
    """Best-effort extraction of a JSON value from an LLM reply.

    Tries, in order: whole text, ```json fenced block, first balanced
    ``[...]`` or ``{...}`` span.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = re.search(r"```(?:json)?\s*(.+?)```", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            c = text[i]
            if c == opener:
                depth += 1
            elif c == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"No parseable JSON in LLM reply: {text[:200]!r}...")


class LLMClient:
    """Chat client for any OpenAI-compatible endpoint (Strata, vLLM, ...)."""

    def __init__(self, settings: Settings | None = None) -> None:
        from openai import OpenAI

        self.settings = settings or get_settings()
        self.client = OpenAI(
            base_url=self.settings.llm_base_url,
            api_key=self.settings.llm_api_key or "local",
            timeout=self.settings.llm_timeout_s,
        )
        self.model = self.settings.llm_model

    def chat(self, system: str, user: str, temperature: float | None = None,
             max_retries: int = 2) -> str:
        last_err: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=(self.settings.llm_temperature
                                 if temperature is None else temperature),
                    max_tokens=self.settings.llm_max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001 - retry any transport/API error
                last_err = exc
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"LLM request failed after retries: {last_err}") from last_err

    def chat_json(self, system: str, user: str, temperature: float | None = None) -> Any:
        """Chat and parse a JSON payload; retries once with a repair nudge."""
        raw = self.chat(system, user, temperature)
        try:
            return extract_json(raw)
        except ValueError:
            raw = self.chat(
                system,
                user + "\n\nIMPORTANT: reply with ONLY valid JSON, no prose.",
                temperature=0.0,
            )
            return extract_json(raw)

    def health(self) -> bool:
        try:
            self.chat("ping", "Reply with the single word: pong", max_retries=0)
            return True
        except Exception:  # noqa: BLE001
            return False
