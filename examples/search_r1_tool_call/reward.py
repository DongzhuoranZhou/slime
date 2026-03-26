"""
Reward function for search_r1_tool_call.

Exact reproduction of the search-r1 reward (examples/search-r1/qa_em_format.py:compute_score_em)
adapted for the native tool-call format.

Original reward is purely sparse 0/1:
- 1.0  : EM match on the final answer
- 0.0  : everything else (no answer, wrong answer, no tool call)

Note: compute_score_em in qa_em_format.py accepts a `format_score` kwarg but never uses it
in the function body — all partial-credit parameters default to 0.  The original
generate_with_search.py therefore produces strictly sparse 0/1 rewards, and this
function reproduces that behaviour exactly.

The only difference from the original is the answer extraction: instead of
    <answer>...</answer>
we parse the native Qwen tool-call format:
    <tool_call>{"name": "answer", "arguments": {"response": "..."}}</tool_call>
"""

from __future__ import annotations

import json
import re
import string
from typing import Any

from slime.utils.types import Sample

# Strips injected tool observation turns so retrieved documents containing
# answer-like JSON don't produce false positives.
_TOOL_TURN_RE = re.compile(r"<\|im_start\|>tool\n.*?<\|im_end\|>", re.DOTALL)

# Extracts the answer tool call JSON payload.
_ANSWER_TOOL_RE = re.compile(r'<tool_call>\s*(\{.*?"name":\s*"answer".*?\})\s*</tool_call>', re.DOTALL)


# ── EM helpers (identical to qa_em_format.py) ────────────────────────────────

def _normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _em_check(prediction: str, golden_answers: list[str] | str) -> bool:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    norm_pred = _normalize_answer(prediction)
    return any(_normalize_answer(g) == norm_pred for g in golden_answers)


# ─────────────────────────────────────────────────────────────────────────────

def _extract_final_answer(response: str) -> str | None:
    """Extract the answer string from the last answer tool call in the response."""
    cleaned = _TOOL_TURN_RE.sub("", response)
    matches = list(_ANSWER_TOOL_RE.finditer(cleaned))
    if not matches:
        return None
    raw = matches[-1].group(1).strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    arguments = payload.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    return (arguments.get("response") or "").strip() or None


def _get_golden_answers(label: Any) -> list[str] | None:
    if label is None:
        return None
    if isinstance(label, dict):
        gt = label.get("ground_truth", label)
        if isinstance(gt, dict):
            return gt.get("target") or gt.get("answers")
        if isinstance(gt, list):
            return gt
        return [str(gt)]
    if isinstance(label, list):
        return label
    return [str(label)]


async def reward_func(args: Any, sample: Sample, **kwargs) -> float:
    """
    Sparse 0/1 reward — exact reproduction of the original search-r1 reward.

    Returns:
        1.0  if the extracted answer is an EM match against the ground truth
        0.0  otherwise (no answer tool call, wrong answer, missing ground truth)
    """
    response = sample.response or ""
    answer = _extract_final_answer(response)
    golden = _get_golden_answers(sample.label)

    if answer is None or golden is None:
        return 0.0

    return 1.0 if _em_check(answer, golden) else 0.0
