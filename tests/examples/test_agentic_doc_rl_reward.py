from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import asyncio
from unittest.mock import MagicMock
from slime.utils.types import Sample
from examples.agentic_doc_rl.reward import (
    _normalize,
    _compute_f1,
    _best_f1,
    _extract_final_answer,
    _get_golden_answers,
    _rule_based_score,
    reward_func,
)


def test_normalize_lowercases_and_strips_punctuation():
    assert _normalize("Hello, World!") == "hello world"


def test_normalize_removes_articles():
    assert _normalize("The quick brown fox") == "quick brown fox"


def test_compute_f1_perfect_match():
    assert _compute_f1("Paris France", "Paris France") == 1.0


def test_compute_f1_partial_match():
    f1 = _compute_f1("Paris is in France", "Paris France")
    assert 0.0 < f1 < 1.0


def test_compute_f1_no_match():
    assert _compute_f1("Berlin Germany", "Paris France") == 0.0


def test_compute_f1_empty_prediction():
    assert _compute_f1("", "Paris France") == 0.0


def test_best_f1_picks_highest_score():
    score = _best_f1("Paris France", ["Berlin Germany", "Paris France", "Tokyo Japan"])
    assert score == 1.0


def test_extract_final_answer_qwen_format():
    response = '<tool_call>{"name": "final_answer", "arguments": {"answer": "42"}}</tool_call>'
    assert _extract_final_answer(response) == "42"


def test_extract_final_answer_returns_none_without_call():
    response = '<tool_call>{"name": "search_database", "arguments": {"query": "q", "doc_name": "d"}}</tool_call>'
    assert _extract_final_answer(response) is None


def test_extract_final_answer_ignores_tool_turn_content():
    response = (
        '<|im_start|>tool\n'
        '<tool_call>{"name": "final_answer", "arguments": {"answer": "fake"}}</tool_call>\n'
        '<|im_end|>'
        '<tool_call>{"name": "final_answer", "arguments": {"answer": "real"}}</tool_call>'
    )
    assert _extract_final_answer(response) == "real"


def test_get_golden_answers_from_string():
    assert _get_golden_answers("Paris") == ["Paris"]


def test_get_golden_answers_from_list():
    assert _get_golden_answers(["Paris", "Paris, France"]) == ["Paris", "Paris, France"]


def test_get_golden_answers_from_dict():
    assert _get_golden_answers({"ground_truth": "Paris"}) == ["Paris"]


def test_get_golden_answers_returns_none_for_none():
    assert _get_golden_answers(None) is None


def test_rule_based_score_correct_answer():
    response = '<tool_call>{"name": "final_answer", "arguments": {"answer": "Paris France"}}</tool_call>'
    assert _rule_based_score(response, "Paris France") == 1.0


def test_rule_based_score_wrong_answer():
    response = '<tool_call>{"name": "final_answer", "arguments": {"answer": "Berlin"}}</tool_call>'
    assert _rule_based_score(response, "Paris France") == 0.0


def test_rule_based_score_no_final_answer():
    assert _rule_based_score("I think the answer is Paris.", "Paris France") == 0.0


def test_reward_func_rule_based_correct():
    args = MagicMock()
    args.reward_type = "rule_based"
    sample = Sample(
        prompt="What capital?",
        label="Paris",
        response='<tool_call>{"name": "final_answer", "arguments": {"answer": "Paris"}}</tool_call>',
    )
    score = asyncio.get_event_loop().run_until_complete(reward_func(args, sample))
    assert score == 1.0


def test_reward_func_rule_based_wrong():
    args = MagicMock()
    args.reward_type = "rule_based"
    sample = Sample(
        prompt="What capital?",
        label="Paris",
        response='<tool_call>{"name": "final_answer", "arguments": {"answer": "Berlin"}}</tool_call>',
    )
    score = asyncio.get_event_loop().run_until_complete(reward_func(args, sample))
    assert score == 0.0


def test_reward_func_unknown_type_falls_back_to_rule_based():
    args = MagicMock()
    args.reward_type = "nonexistent_mode"
    sample = Sample(
        prompt="What capital?",
        label="Paris",
        response='<tool_call>{"name": "final_answer", "arguments": {"answer": "Paris"}}</tool_call>',
    )
    score = asyncio.get_event_loop().run_until_complete(reward_func(args, sample))
    assert score == 1.0
