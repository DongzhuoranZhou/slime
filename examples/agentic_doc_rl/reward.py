# examples/agentic_doc_rl/reward.py
from __future__ import annotations

import json
import logging
import re
import string
from typing import Any

from slime.utils.types import Sample

logger = logging.getLogger(__name__)

_TOOL_TURN_RE = re.compile(r"<\|im_start\|>tool\n.*?<\|im_end\|>", re.DOTALL)
_FINAL_ANSWER_RE = re.compile(
    r'<tool_call>\s*(\{.*?"name":\s*"final_answer".*?\})\s*</tool_call>', re.DOTALL
)


def _normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize(prediction).split()
    gt_tokens = _normalize(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def _best_f1(prediction: str, golden_answers: list[str]) -> float:
    return max(_compute_f1(prediction, g) for g in golden_answers)


def _extract_final_answer(response: str) -> str | None:
    cleaned = _TOOL_TURN_RE.sub("", response)
    matches = list(_FINAL_ANSWER_RE.finditer(cleaned))
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
    return (arguments.get("answer") or arguments.get("response") or "").strip() or None


def _get_golden_answers(label: Any) -> list[str] | None:
    if label is None:
        return None
    if isinstance(label, dict):
        gt = label.get("ground_truth") or label.get("answer") or label.get("target")
        if isinstance(gt, list):
            return [str(g) for g in gt]
        if gt is not None:
            return [str(gt)]
        return None
    if isinstance(label, list):
        return [str(g) for g in label]
    return [str(label)]


def _rule_based_score(response: str, label: Any) -> float:
    answer = _extract_final_answer(response)
    golden = _get_golden_answers(label)
    if answer is None or not golden:
        return 0.0
    return _best_f1(answer, golden)


def _llm_judge_score(response: str, label: Any, question: str, args: Any) -> float:
    answer = _extract_final_answer(response)
    if answer is None:
        return 0.0
    golden = _get_golden_answers(label)
    ground_truth = golden[0] if golden else ""
    try:
        # utils.py lives in AgenticMemory/ — added to PYTHONPATH by the launch script.
        from utils import evaluate_response  # noqa: PLC0415
        from openai import OpenAI

        client = OpenAI(
            api_key=getattr(args, "judge_api_key", "EMPTY"),
            base_url=getattr(args, "judge_url", "http://localhost:30000") + "/v1",
        )
        result = evaluate_response(
            client,
            getattr(args, "judge_model", "Qwen3-VL-30B-A3B-Instruct"),
            answer,
            ground_truth,
            question,
        )
        score = result.get("score", 0)
        return float(score) if score not in (-1, None) else 0.0
    except Exception as exc:
        logger.warning("LLM judge failed: %s", exc)
        return 0.0


async def reward_func(args: Any, sample: Sample, **kwargs) -> float:
    """Pluggable reward function called per sample by slime's rollout.

    reward_type (from args / config.yaml):
        rule_based  — F1 token overlap (default)
        llm_judge   — LLM evaluator via OpenAI-compatible API
        both        — rule_based drives training; llm_judge logged to wandb
    """
    response = sample.response or ""
    reward_type = getattr(args, "reward_type", "rule_based")
    question = sample.label.get("question", "") if isinstance(sample.label, dict) else ""

    if reward_type == "rule_based":
        return _rule_based_score(response, sample.label)

    if reward_type == "llm_judge":
        return _llm_judge_score(response, sample.label, question, args)

    if reward_type == "both":
        rb_score = _rule_based_score(response, sample.label)
        lj_score = _llm_judge_score(response, sample.label, question, args)
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({"reward/rule_based": rb_score, "reward/llm_judge": lj_score})
        except Exception:
            pass
        return rb_score

    logger.warning("Unknown reward_type %r; falling back to rule_based.", reward_type)
    return _rule_based_score(response, sample.label)
