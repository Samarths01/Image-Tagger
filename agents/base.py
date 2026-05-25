"""
base.py — Shared Anthropic client and agent call utilities.

All agents import create_client() and call_agent() from here.
Single place to manage auth and model configuration.
"""

import json
import os
from typing import Any

import anthropic
from dotenv import load_dotenv

load_dotenv()

# Model used across all agents — change here to update everywhere
MODEL = "claude-sonnet-4-5"


def create_client() -> anthropic.Anthropic:
    """
    Creates an Anthropic client using the public API.
    Reads credentials from environment — never hardcoded.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. "
            "Copy .env.example to .env and fill in your key."
        )

    return anthropic.Anthropic(api_key=api_key)


def call_agent(
    client: anthropic.Anthropic,
    system_prompt: str,
    messages: list[dict],
    max_tokens: int = 2048,
) -> dict[str, Any]:
    """
    Makes a single agent call and returns parsed JSON output.

    All agents in this pipeline respond ONLY with valid JSON.
    This helper enforces that contract and raises on violations.

    Args:
        client:        Authenticated Anthropic client
        system_prompt: The agent's identity and instructions
        messages:      The conversation messages (user turn with data)
        max_tokens:    Response length limit

    Returns:
        Parsed JSON dict from the agent's response

    Raises:
        ValueError: If the agent response is not valid JSON
        anthropic.APIError: On API-level failures
    """
    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=messages,
    )

    raw_text = response.content[0].text.strip()

    # Strip markdown code fences if Claude wraps the JSON (defensive)
    if raw_text.startswith("```"):
        lines = raw_text.split("\n")
        raw_text = "\n".join(lines[1:-1])

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Agent returned non-JSON response.\n"
            f"Parse error: {e}\n"
            f"Raw response:\n{raw_text}"
        ) from e
