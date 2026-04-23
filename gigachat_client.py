"""
GigaChat client wrapper that mimics the OpenAI SDK interface.

timed_llm_call() expects client.chat.completions.create() returning an object
with .choices[0].message.content and .usage.{prompt_tokens, completion_tokens}.
This module wraps langchain-gigachat to satisfy that contract.

When response_format={"type": "json_object"} is passed (i.e. use_json_mode=True
in ACE), we use GigaChat's function_calling to force structured JSON output.
The role is inferred from the prompt content so the correct schema is applied.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic schemas for GigaChat function-calling structured output
# ---------------------------------------------------------------------------

class GeneratorOutput(BaseModel):
    """Generator response for ACE system."""
    reasoning: str = Field(description="Chain of thought / reasoning")
    considered_bullet_ids: List[str] = Field(description="Playbook bullet IDs considered")
    used_bullet_ids: List[str] = Field(description="Subset of considered IDs that shaped the answer")
    final_answer: str = Field(description="Concise final answer")


class _BulletTag(BaseModel):
    """Single bullet tag entry."""
    id: str = Field(description="Bullet ID")
    tag: str = Field(description="helpful / harmful / neutral")


class ReflectorOutput(BaseModel):
    """Reflector response for ACE system."""
    reasoning: str = Field(description="Detailed analysis of the error")
    error_identification: str = Field(description="What went wrong")
    root_cause_analysis: str = Field(description="Why the error occurred")
    correct_approach: str = Field(description="What should have been done")
    key_insight: str = Field(description="Principle to remember")
    bullet_tags: List[_BulletTag] = Field(description="Tags for each considered bullet")


class _CuratorOperation(BaseModel):
    """Single curator operation."""
    type: str = Field(description="Operation type: ADD, UPDATE, MERGE, or ARCHIVE")
    section: Optional[str] = Field(default=None, description="Target section for ADD/MERGE")
    content: Optional[str] = Field(default=None, description="Bullet content for ADD/UPDATE/MERGE")
    bullet_id: Optional[str] = Field(default=None, description="Bullet ID for UPDATE/ARCHIVE")
    source_ids: Optional[List[str]] = Field(default=None, description="Source bullet IDs for MERGE")
    reason: Optional[str] = Field(default=None, description="Reason for ARCHIVE")


class CuratorOutput(BaseModel):
    """Curator response for ACE system."""
    reasoning: str = Field(description="Analysis of playbook state")
    operations: List[_CuratorOperation] = Field(description="List of ADD/UPDATE/MERGE/ARCHIVE operations")


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

    _ROLE_SCHEMAS = {
        "generator": GeneratorOutput,
        "reflector": ReflectorOutput,
        "curator": CuratorOutput,
    }

    def __init__(self, gc_instance, default_model: str):
        self._gc = gc_instance
        self._default_model = default_model

    @staticmethod
    def _detect_role(messages: list[dict]) -> str | None:
        """Infer ACE role from prompt content to select the right schema."""
        full = " ".join(m.get("content", "") for m in (messages or []))
        text = (full[:1200] + " " + full[-600:]).lower()
        if "operations" in text and ("curator" in text or "playbook" in text):
            return "curator"
        if "error_identification" in text or "root_cause" in text or "bullet_tags" in text:
            return "reflector"
        if "final_answer" in text or "considered_bullet_ids" in text:
            return "generator"
        return None

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

        response_format = kwargs.pop("response_format", None)
        kwargs.pop("max_tokens", None)
        kwargs.pop("max_completion_tokens", None)
        kwargs.pop("extra_body", None)
        kwargs.pop("reasoning_effort", None)
        kwargs.pop("reasoning", None)

        use_fc = (
            response_format is not None
            and isinstance(response_format, dict)
            and response_format.get("type") == "json_object"
        )

        role_hint = kwargs.pop("role_hint", None)

        if use_fc:
            detected = role_hint or self._detect_role(messages)
            schema = self._ROLE_SCHEMAS.get(detected) if detected else None
            if schema:
                bound = self._gc.bind_tools([schema], tool_choice=schema.__name__)
                result = bound.invoke(lc_messages)
                content = self._extract_fc_content(result)
            else:
                result = self._gc.invoke(lc_messages)
                content = result.content
        else:
            result = self._gc.invoke(lc_messages)
            content = result.content

        usage = self._extract_usage(result)

        return _ChatCompletion(
            choices=[_Choice(message=_Message(content=content, role="assistant"))],
            usage=usage,
            model=model or self._default_model,
        )

    @staticmethod
    def _extract_fc_content(result) -> str:
        """Extract structured content from a function-calling response."""
        if result.tool_calls:
            return json.dumps(result.tool_calls[0]["args"], ensure_ascii=False)
        fc = (getattr(result, "additional_kwargs", None) or {}).get("function_call")
        if fc and "arguments" in fc:
            args = fc["arguments"]
            return args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
        return result.content or ""

    @staticmethod
    def _extract_usage(result) -> "_Usage":
        usage = _Usage()
        token_usage = getattr(result, "response_metadata", {}).get("token_usage", None)
        if token_usage:
            usage.prompt_tokens = token_usage.get("prompt_tokens", 0) or 0
            usage.completion_tokens = token_usage.get("completion_tokens", 0) or 0
        elif hasattr(result, "usage_metadata") and result.usage_metadata:
            usage.prompt_tokens = getattr(result.usage_metadata, "input_tokens", 0) or 0
            usage.completion_tokens = getattr(result.usage_metadata, "output_tokens", 0) or 0
        return usage


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
