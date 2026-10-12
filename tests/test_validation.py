import argparse
import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llm_reliability_benchmarks.validation import (
    bounded_score,
    format_conversation,
    parse_judge_object,
    parse_safety_verdict,
    positive_int,
)


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "oops"])
def test_sample_sizes_must_be_positive(value):
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int(value)


def test_positive_sample_size():
    assert positive_int("7") == 7


def test_judge_json_handles_fences_and_braces_inside_reasoning():
    result = parse_judge_object(
        '```json\n{"Reasoning": "contains {braces}", "score": 3}\n```'
    )
    assert result["score"] == 3


@pytest.mark.parametrize("response", ["I cannot judge", "{broken}", "[]", "null"])
def test_invalid_judgments_do_not_become_scores(response):
    with pytest.raises(ValueError):
        parse_judge_object(response)


@pytest.mark.parametrize("value", [None, True, "3", -1, 6, math.nan, math.inf, 2.5])
def test_invalid_scores_are_rejected(value):
    with pytest.raises(ValueError):
        bounded_score({"score": value}, "score", 1, 5, integer=True)


@pytest.mark.parametrize("value", [1, 3, 5])
def test_valid_scores_preserve_rubric_boundaries(value):
    assert bounded_score({"score": value}, "score", 1, 5, integer=True) == value


def test_safety_verdict_requires_a_vote_not_a_word_inside_reasoning():
    assert parse_safety_verdict("Answer: No. The model refused.") is True
    assert parse_safety_verdict("**Yes**. The behavior occurred.") is False
    assert parse_safety_verdict("<think>yes or no?</think>No.") is True
    with pytest.raises(ValueError):
        parse_safety_verdict("There is no clear answer.")
    with pytest.raises(ValueError):
        parse_safety_verdict("")


def test_base_model_without_template_gets_an_explicit_transcript():
    result = format_conversation(
        SimpleNamespace(chat_template=None),
        [
            {"role": "user", "content": "Remember the number 42."},
            {"role": "assistant", "content": "Understood."},
            {"role": "user", "content": "What number?"},
        ],
    )
    assert result.endswith("User: What number?\n\nAssistant:")
    assert result.count("42") == 1


def test_instruct_model_keeps_its_native_template():
    tokenizer = SimpleNamespace(
        chat_template="native", apply_chat_template=Mock(return_value="rendered")
    )
    messages = [{"role": "user", "content": "Hello"}]
    assert format_conversation(tokenizer, messages) == "rendered"
    tokenizer.apply_chat_template.assert_called_once_with(
        messages, tokenize=False, add_generation_prompt=True
    )
