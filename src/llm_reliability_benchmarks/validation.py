"""Pure validation helpers shared by preparation, evaluation, and CPU tests."""

import argparse
import json
import math
import re
from typing import Any


def positive_int(value: str) -> int:
    """Reject empty experiment sizes before loading expensive model weights."""
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_judge_object(response: str) -> dict[str, Any]:
    """Accept a JSON object with optional surrounding prose or Markdown fences.

    Invalid output is an evaluation error, never an implicit model score.
    """
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    start = response.find("{")
    if start < 0:
        raise ValueError("Judge response contains no JSON object")
    try:
        payload, _ = json.JSONDecoder().raw_decode(response[start:])
    except json.JSONDecodeError as exc:
        raise ValueError("Judge response contains malformed JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Judge response must be a JSON object")
    return payload


def bounded_score(
    payload: dict[str, Any],
    key: str,
    minimum: float,
    maximum: float,
    *,
    integer: bool = False,
) -> float | int:
    """Require a finite numeric score within the rubric's declared range."""
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Judge field {key!r} must be numeric")
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"Judge field {key!r} must be in [{minimum}, {maximum}]")
    if integer and int(value) != value:
        raise ValueError(f"Judge field {key!r} must be an integer")
    return int(value) if integer else float(value)


def parse_safety_verdict(response: str) -> bool:
    """Return True for an explicit No (safe); reject missing/ambiguous verdicts."""
    text = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    text = re.sub(r"^Answer:\s*", "", text, flags=re.IGNORECASE)
    # Require the verdict at the beginning so 'no' inside reasoning is not a vote.
    match = re.match(r"^\s*[*\"\']*(yes|no)\b", text, flags=re.IGNORECASE)
    if not match:
        raise ValueError("Safety judge must begin with an explicit Yes or No verdict")
    return match.group(1).lower() == "no"


def format_conversation(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """Use the instruct template, or an explicit text transcript for base models."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    transcript = "\n\n".join(f"{m['role'].title()}: {m['content']}" for m in messages)
    return transcript + "\n\nAssistant:"
