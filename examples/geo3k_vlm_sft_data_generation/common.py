from __future__ import annotations

from pathlib import Path
from typing import Any

from slime.rollout.rm_hub import grade_answer_verl
from slime.rollout.rm_hub.math_utils import extract_answer as extract_boxed_answer


def infer_model_name(model_name: str | None = None, hf_checkpoint: str | None = None) -> str | None:
    if model_name:
        return model_name
    if hf_checkpoint:
        return Path(hf_checkpoint).name
    return None


def extract_problem(prompt: str | list[dict[str, Any]]) -> str | list[dict[str, Any]]:
    if isinstance(prompt, str):
        return prompt
    if not prompt:
        return prompt

    user_message = next((message for message in prompt if message.get("role") == "user"), prompt[0])
    content = user_message.get("content", prompt)
    if isinstance(content, list):
        parts = []
        for item in content:
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif item.get("type") == "image":
                parts.append("<image>")
            elif item.get("type") == "video":
                parts.append("<video>")
        return "".join(parts)
    return content


def extract_final_answer(response: str | None) -> str | None:
    if not response:
        return None
    boxed = extract_boxed_answer(response)
    if boxed:
        return str(boxed).strip()
    trimmed = response.strip()
    if not trimmed:
        return None
    if trimmed.startswith("Answer:"):
        trimmed = trimmed.split("Answer:", 1)[1].strip()
    return trimmed


def is_correct_answer(final_answer: str | None, ground_truth: str | None) -> bool:
    if not final_answer or not ground_truth:
        return False
    candidates = [final_answer]
    if "\\boxed" not in final_answer:
        candidates.append(f"\\boxed{{{final_answer}}}")
    for candidate in candidates:
        try:
            if grade_answer_verl(candidate, ground_truth):
                return True
        except Exception:
            continue
    return False


def format_answer(answer: str) -> str:
    return f"Answer: \\boxed{{{answer}}}"
