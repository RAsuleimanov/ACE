"""
Generator agent for ACE system.
Generates answers to questions using playbook and reflection.
"""

import json
import re
from typing import Any

from llm import timed_llm_call

from ..prompts.generator import GENERATOR_PROMPT


class Generator:
    """
    Generator agent that produces answers to questions using knowledge
    from a playbook and previous reflections.
    """

    def __init__(self, api_client, api_provider, model: str, max_tokens: int = 4096):
        """
        Initialize the Generator agent.

        Args:
            api_client: OpenAI client for LLM calls
            api_provider: API provider for LLM calls
            model: Model name to use for generation
            max_tokens: Maximum tokens for generation
        """
        self.api_client = api_client
        self.api_provider = api_provider
        self.model = model
        self.max_tokens = max_tokens

    def generate(
        self,
        question: str,
        playbook: str,
        context: str = "",
        reflection: str = "(empty)",
        use_json_mode: bool = False,
        call_id: str = "gen",
        log_dir: str | None = None,
    ) -> tuple[str, list[str], list[str], dict[str, Any]]:
        """
        Generate an answer to a question using the playbook.

        Args:
            question: The question to answer
            playbook: The current playbook content
            context: Additional context for the question
            reflection: Previous reflection content
            use_json_mode: Whether to use JSON mode
            call_id: Unique identifier for this call
            log_dir: Directory for logging

        Returns:
            Tuple of (full_response, considered_bullet_ids, used_bullet_ids, call_info)
        """
        # Format the prompt
        prompt = GENERATOR_PROMPT.format(playbook, reflection, question, context)

        response, call_info = timed_llm_call(
            self.api_client,
            self.api_provider,
            self.model,
            prompt,
            role="generator",
            call_id=call_id,
            max_tokens=self.max_tokens,
            log_dir=log_dir,
            use_json_mode=use_json_mode,
        )

        considered_bullet_ids, used_bullet_ids = self._extract_bullet_tracking(
            response, use_json_mode
        )

        return response, considered_bullet_ids, used_bullet_ids, call_info

    def _extract_bullet_tracking(
        self, response: str, use_json_mode: bool
    ) -> tuple[list[str], list[str]]:
        """
        Extract considered and used bullet IDs from generator response.

        Args:
            response: The generator's response
            use_json_mode: Whether JSON mode was used

        Returns:
            Tuple of (considered_bullet_ids, used_bullet_ids)
        """
        considered_bullet_ids: list[str] = []
        used_bullet_ids: list[str] = []

        # Generator prompt always asks for JSON output (considered_bullet_ids +
        # used_bullet_ids fields). Try json.loads first regardless of
        # use_json_mode flag — the flag only controls OpenAI response_format,
        # not the shape of the prompt. Fallback to regex only if parsing fails.
        parsed_json = False
        # Strip markdown code fences (```json ... ```) that some models wrap JSON in.
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```\s*$", "", cleaned)
        try:
            response_json = json.loads(cleaned)
            if isinstance(response_json, dict):
                considered_bullet_ids = self._normalize_bullet_ids(
                    response_json.get("considered_bullet_ids")
                )
                used_bullet_ids = self._normalize_bullet_ids(
                    response_json.get("used_bullet_ids")
                )
                if not considered_bullet_ids and "bullet_ids" in response_json:
                    legacy_bullet_ids = self._normalize_bullet_ids(response_json.get("bullet_ids"))
                    considered_bullet_ids = legacy_bullet_ids
                    if "used_bullet_ids" not in response_json:
                        used_bullet_ids = legacy_bullet_ids
                if considered_bullet_ids and "used_bullet_ids" not in response_json:
                    used_bullet_ids = considered_bullet_ids
                parsed_json = True
        except (json.JSONDecodeError, TypeError):
            pass

        if not parsed_json:
            # Regex fallback — bullet IDs may use non-ASCII prefixes (e.g. Cyrillic
            # инст-/пд-/то-), so match any word-character run of length 3+.
            considered_bullet_ids = self._extract_bullet_ids_regex(response)
            used_bullet_ids = considered_bullet_ids

        return considered_bullet_ids, used_bullet_ids

    def _normalize_bullet_ids(self, raw_bullet_ids: Any) -> list[str]:
        """Normalize model output into a list of bullet ID strings."""
        if not isinstance(raw_bullet_ids, list):
            return []
        return [bullet_id for bullet_id in raw_bullet_ids if isinstance(bullet_id, str)]

    def _extract_bullet_ids_regex(self, text: str) -> list[str]:
        """
        Extract bullet IDs using regex pattern matching.

        Args:
            text: Text to extract bullet IDs from

        Returns:
            List of bullet IDs
        """
        # Pattern matches [инст-00001], [пд-00042], [abc-00007], etc.
        # \w with re.UNICODE covers Cyrillic + ASCII letters.
        pattern = r"\[(\w{2,}-\d{5})\]"
        matches = re.findall(pattern, text, flags=re.UNICODE)
        return matches
