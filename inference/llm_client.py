"""
inference/llm_client.py
Unified LLM client for GPT-4o, Claude Sonnet, and LLaMA-3.1-70B.

Usage:
    from inference.llm_client import LLMClient
    client = LLMClient("gpt-4o")
    text = client.generate(system_prompt="...", user_prompt="...")
"""

from __future__ import annotations
import os
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=False)  # env vars set at process level take precedence
logger = logging.getLogger(__name__)

# ── Model registry ────────────────────────────────────────────────────────────
MODELS = {
    # 2026 models
    "gpt-4o":               {"provider": "openai",    "slug": "gpt4o"},
    "claude-sonnet-4-6":    {"provider": "anthropic", "slug": "claude"},
    "llama-3.3-70b":        {"provider": "groq",      "slug": "llama33"},
    # Keep old slug alias so any cached references still resolve
    "llama-3.1-70b":        {"provider": "groq",      "slug": "llama31"},
    # 2023 baselines (kept for reference; original notebooks used different SDKs)
    "gpt-3.5-turbo":        {"provider": "openai",    "slug": "gpt35"},
    "llama-2-7b":           {"provider": "groq",      "slug": "llama2"},
}

GROQ_MODEL_IDS = {
    "llama-3.3-70b": "llama-3.3-70b-versatile",   # current production model
    "llama-3.1-70b": "llama-3.3-70b-versatile",   # decommissioned → redirect
    "llama-2-7b":    "llama-3.3-70b-versatile",   # reference only
}


def _load_groq_backup_keys() -> list[str]:
    """Return all GROQ_API_KEY_N values (N=2,3,...) from the environment."""
    keys = []
    for i in range(2, 20):
        k = os.environ.get(f"GROQ_API_KEY_{i}", "")
        if k:
            keys.append(k)
        else:
            break
    return keys


class LLMClient:
    """Single interface for all three LLM providers."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        seed: int = 42,
        max_tokens: int = 512,
    ):
        if model not in MODELS:
            raise ValueError(f"Unknown model '{model}'. Choose from: {list(MODELS)}")
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.provider = MODELS[model]["provider"]
        self.slug = MODELS[model]["slug"]
        self._groq_keys = ([os.environ.get("GROQ_API_KEY", "")] + _load_groq_backup_keys()
                           if self.provider == "groq" else [])
        self._groq_key_idx = 0
        self._client = self._init_client()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_client(self):
        if self.provider == "openai":
            from openai import OpenAI
            return OpenAI()                        # reads OPENAI_API_KEY
        if self.provider == "anthropic":
            import anthropic
            return anthropic.Anthropic()           # reads ANTHROPIC_API_KEY
        if self.provider == "groq":
            from groq import Groq
            key = self._groq_keys[self._groq_key_idx] if self._groq_keys else None
            return Groq(api_key=key) if key else Groq()
        raise ValueError(f"Unknown provider: {self.provider}")

    def _rotate_groq_key(self) -> bool:
        """Switch to the next available Groq API key. Returns True if rotated."""
        if self.provider != "groq":
            return False
        next_idx = self._groq_key_idx + 1
        if next_idx < len(self._groq_keys):
            self._groq_key_idx = next_idx
            from groq import Groq
            self._client = Groq(api_key=self._groq_keys[next_idx])
            logger.info("Rotated to Groq key #%d.", next_idx + 1)
            return True
        return False

    # ── Core method ──────────────────────────────────────────────────────────

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        retries: int = 6,
        backoff: float = 5.0,
    ) -> str:
        """Call the model and return the response text.

        Retries up to `retries` times on transient errors (rate limits,
        timeouts). For daily TPD Groq errors, rotates to the next key
        immediately. For TPM / other 429s, parses the suggested retry delay.
        Raises after all retries are exhausted.
        """
        import re as _re
        for attempt in range(retries):
            try:
                return self._call(system_prompt, user_prompt)
            except Exception as exc:
                exc_str = str(exc)
                # Daily token quota (TPD) → rotate key immediately, no sleep
                if "tokens per day" in exc_str and self._rotate_groq_key():
                    logger.warning(
                        "Groq TPD limit on key #%d; rotated to key #%d.",
                        self._groq_key_idx, self._groq_key_idx + 1,
                    )
                    continue   # retry immediately with new key
                # Parse "Please try again in Xs/Xms/Xm" from Groq/OpenAI 429 messages
                m = _re.search(r"try again in (\d+(?:\.\d+)?)(\w+)", exc_str)
                if m:
                    val, unit = float(m.group(1)), m.group(2).lower()
                    if unit == "ms":
                        wait = max(val / 1000, 1)   # milliseconds → seconds
                    elif unit.startswith("m"):
                        wait = val * 60              # minutes → seconds
                    else:
                        wait = val                   # already seconds
                    wait = max(wait + 5, backoff)    # add 5s buffer
                else:
                    wait = backoff * (2 ** attempt)
                logger.warning(
                    "Attempt %d/%d failed for %s: %s. Retrying in %.0fs.",
                    attempt + 1, retries, self.model, exc, wait,
                )
                if attempt < retries - 1:
                    time.sleep(wait)
                else:
                    logger.error("All retries exhausted for %s.", self.model)
                    raise

    def _call(self, system_prompt: str, user_prompt: str) -> str:
        if self.provider == "openai":
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=self.temperature,
                seed=self.seed,
                max_tokens=self.max_tokens,
            )
            return resp.choices[0].message.content.strip()

        if self.provider == "anthropic":
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return resp.content[0].text.strip()

        if self.provider == "groq":
            groq_model = GROQ_MODEL_IDS.get(self.model, self.model)
            resp = self._client.chat.completions.create(
                model=groq_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return resp.choices[0].message.content.strip()

        raise ValueError(f"Unhandled provider: {self.provider}")

    # ── Metadata ─────────────────────────────────────────────────────────────

    def metadata(self) -> dict:
        """Return a dict suitable for logging alongside inference outputs."""
        return {
            "model":       self.model,
            "provider":    self.provider,
            "slug":        self.slug,
            "temperature": self.temperature,
            "seed":        self.seed,
            "max_tokens":  self.max_tokens,
            "inference_date": datetime.now().isoformat(),
        }
