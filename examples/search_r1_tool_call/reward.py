"""
Reward function for search_r1_tool_call.

Scoring logic:
- Full score (reward_score)                : EM match + tool was used
- Slightly reduced (reward_score - format) : EM match but no tool call made
- Partial (reward_tool_use_score)          : tool used but no EM match / no answer
- Zero                                     : no answer + no tool

The model is instructed to wrap its final answer in <answer>...</answer> tags
(same as the original search-r1 convention), which keeps answer extraction simple
and avoids re-parsing Qwen's role tokens.

normalize_answer and em_check are inlined from examples/search-r1/qa_em_format.py
(Apache-2.0, adapted from Bytedance / Search-R1).
"""

from __future__ import annotations

import random
import re
import string
from typing import Any

from slime.utils.types import Sample

# Matches the model's standard tool-call output
TOOL_CALL_RE = re.compile(r"<tool_call>\s*\{.*?\}\s*</tool_call>", re.DOTALL)

# Strips injected tool observation turns (Qwen role tokens added by apply_chat_template)
# e.g. <|im_start|>tool\nDoc 1 ...<|im_end|>
_TOOL_TURN_RE = re.compile(r"<\|im_start\|>tool\n.*?<\|im_end\|>", re.DOTALL)

# Extracts the final answer the model placed between <answer> tags
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


# ── EM helpers (inlined from examples/search-r1/qa_em_format.py) ─────────────

def _normalize_answer(s: str) -> str:
    """Lower, strip punctuation and articles, collapse whitespace."""
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
    """
    Extract the last <answer>...</answer> from the model's response, ignoring
    content inside injected tool observation turns.

    The full response (sample.response) contains both model-generated tokens
    and injected observation tokens. Tool turns look like:
        <|im_start|>tool\n...search results...<|im_end|>
    We strip those before looking for <answer> so web content can't accidentally
    match the answer pattern.
    """
    cleaned = _TOOL_TURN_RE.sub("", response)
    matches = list(_ANSWER_RE.finditer(cleaned))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def _get_golden_answers(label: Any) -> list[str] | None:
    """Resolve ground-truth answers from the sample label."""
    if label is None:
        return None
    if isinstance(label, dict):
        # Search-R1 format: {"ground_truth": {"target": [...]}}
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
    Reward function for standard tool-calling search training.

    Args:
        args: training args (includes reward_score, reward_format_score,
              reward_tool_use_score set from config.yaml)
        sample: completed rollout sample

    Returns:
        float reward in [0, reward_score]
    """
    reward_score = getattr(args, "reward_score", 1.0)
    reward_format_score = getattr(args, "reward_format_score", 0.1)
    reward_tool_use_score = getattr(args, "reward_tool_use_score", 0.05)

    response = sample.response or ""
    tool_used = bool(TOOL_CALL_RE.search(response))
    answer = _extract_final_answer(response)
    golden = _get_golden_answers(sample.label)

    do_print = random.randint(1, 64) == 1
    if do_print:
        print("─" * 40)
        print(f"[reward] golden: {golden}")
        print(f"[reward] extracted answer: {answer!r}")
        print(f"[reward] tool_used: {tool_used}")

    if answer is None:
        return reward_tool_use_score if tool_used else 0.0

    if golden is None:
        return 0.0

    if _em_check(answer, golden):
        return reward_score if tool_used else reward_score - reward_format_score

    return reward_tool_use_score if tool_used else 0.0
