"""
GigaChat client wrapper that mimics the OpenAI SDK interface.

timed_llm_call() expects client.chat.completions.create() returning an object
with .choices[0].message.content and .usage.{prompt_tokens, completion_tokens}.
This module wraps langchain-gigachat to satisfy that contract.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class _Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class _Message:
    content: Optional[str] = None
    role: str = "assistant"


@dataclass
class _Choice:
    message: _Message = field(default_factory=_Message)
    index: int = 0
    finish_reason: str = "stop"


@dataclass
class _ChatCompletion:
    choices: list = field(default_factory=list)
    usage: _Usage = field(default_factory=_Usage)
    model: str = ""


class _Completions:
    """Namespace that exposes .create() matching openai.chat.completions.create()."""

    def __init__(self, gc_instance, default_model: str):
        self._gc = gc_instance
        self._default_model = default_model

    def create(self, *, model=None, messages=None, temperature=0.0, **kwargs):
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

        lc_messages = []
        for msg in (messages or []):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        result = self._gc.invoke(lc_messages)

        usage = _Usage()
        token_usage = getattr(result, "response_metadata", {}).get("token_usage", None)
        if token_usage:
            usage.prompt_tokens = token_usage.get("prompt_tokens", 0) or 0
            usage.completion_tokens = token_usage.get("completion_tokens", 0) or 0
        elif hasattr(result, "usage_metadata") and result.usage_metadata:
            usage.prompt_tokens = getattr(result.usage_metadata, "input_tokens", 0) or 0
            usage.completion_tokens = getattr(result.usage_metadata, "output_tokens", 0) or 0

        return _ChatCompletion(
            choices=[_Choice(message=_Message(content=result.content, role="assistant"))],
            usage=usage,
            model=model or self._default_model,
        )


class _ChatNamespace:
    """Mimics openai.OpenAI().chat  so that client.chat.completions.create() works."""

    def __init__(self, completions: _Completions):
        self.completions = completions


class GigaChatClient:
    """Drop-in replacement for openai.OpenAI() that routes to GigaChat via langchain."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str = "GigaChat-2-Max",
        verify_ssl_certs: bool = False,
        cert_file: Optional[str] = None,
        key_file: Optional[str] = None,
        credentials: Optional[str] = None,
        scope: Optional[str] = None,
        timeout: float = 120,
        profanity_check: bool = False,
        repetition_penalty: float = 1,
        temperature: float = 0.0,
        **extra_kwargs,
    ):
        from langchain_gigachat.chat_models import GigaChat

        gc_kwargs: dict[str, Any] = {
            "base_url": base_url,
            "model": model,
            "verify_ssl_certs": verify_ssl_certs,
            "timeout": timeout,
            "streaming": False,
            "temperature": temperature,
            "profanity_check": profanity_check,
            "repetition_penalty": repetition_penalty,
        }
        if cert_file and key_file:
            gc_kwargs["cert_file"] = cert_file
            gc_kwargs["key_file"] = key_file
        if credentials:
            gc_kwargs["credentials"] = credentials
        if scope:
            gc_kwargs["scope"] = scope
        gc_kwargs.update(extra_kwargs)

        self._gc = GigaChat(**gc_kwargs)
        self._model = model
        self.chat = _ChatNamespace(_Completions(self._gc, model))

    def close(self):
        pass


def make_gigachat_client(gigachat_config: dict) -> "GigaChatClient":
    """Create a GigaChatClient from a config dict (as read from YAML)."""
    cfg = dict(gigachat_config)

    if "credentials_env" in cfg:
        env_key = cfg.pop("credentials_env")
        cfg["credentials"] = os.getenv(env_key, "")
    if "auth_key_env" in cfg:
        env_key = cfg.pop("auth_key_env")
        cfg["credentials"] = os.getenv(env_key, "")

    return GigaChatClient(**cfg)
