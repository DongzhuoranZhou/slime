# Adapt from https://github.com/PeterGriffinJin/Search-R1/blob/ceee7b89655ed52f205b9beb98e1190c3eedcfb0/verl/utils/reward_score/qa_em_format.py
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import random
import re
import string


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def _extract_tool_calls(text: str) -> list[dict]:
    pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    matches = list(re.finditer(pattern, text, re.DOTALL))
    calls = []
    for match in matches:
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, dict):
            return []
        payload["_start"] = match.start()
        payload["_end"] = match.end()
        calls.append(payload)
    return calls


def _extract_tool_responses(text: str) -> list[dict]:
    pattern = r"<tool_response>\s*(\{.*?\})\s*</tool_response>"
    matches = list(re.finditer(pattern, text, re.DOTALL))
    responses = []
    for match in matches:
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, dict):
            return []
        payload["_start"] = match.start()
        payload["_end"] = match.end()
        responses.append(payload)
    return responses


def is_valid_sequence(text):
    assistant_pattern = r"<\|im_start\|>assistant\s*"
    assistant_match = re.search(assistant_pattern, text)

    if not assistant_match:
        return False, "Missing assistant marker"

    content = text[assistant_match.end() :]
    calls = _extract_tool_calls(content)
    if not calls:
        return False, "Missing tool_call"

    last_call = calls[-1]
    if last_call.get("name") != "final_answer":
        return False, "Final tool call must be final_answer"

    responses = _extract_tool_responses(content)
    response_positions = [resp.get("_start") for resp in responses]

    for idx, call in enumerate(calls):
        call_name = call.get("name")
        call_end = call.get("_end", -1)
        next_call_start = calls[idx + 1].get("_start", None) if idx + 1 < len(calls) else None

        if call_name == "search":
            has_response = False
            for resp in responses:
                resp_name = resp.get("name")
                resp_start = resp.get("_start", -1)
                if resp_start is None:
                    continue
                if resp_name == "search" and resp_start > call_end:
                    if next_call_start is None or resp_start < next_call_start:
                        has_response = True
                        break
            if not has_response:
                return False, "Missing tool_response for search"
        elif call_name == "final_answer":
            for pos in response_positions:
                if pos is not None and pos > call_end:
                    return False, "tool_response after final_answer"

    return True, "Valid sequence format"


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    calls = _extract_tool_calls(solution_str)
    if not calls:
        return None

    for call in reversed(calls):
        if call.get("name") == "final_answer":
            arguments = call.get("arguments") or {}
            if isinstance(arguments, dict):
                answer = arguments.get("answer")
                return answer.strip() if isinstance(answer, str) else None

    return None


def extract_information_blocks(text: str) -> list[str]:
    responses = _extract_tool_responses(text)
    blocks = []
    for resp in responses:
        if resp.get("name") != "search":
            continue
        if resp.get("status") != "success":
            continue
        content = resp.get("content")
        if isinstance(content, str):
            blocks.append(content.strip())
    return blocks


def is_retrieval_correct(text: str, golden_answers: list[str]) -> bool:
    seqs = extract_information_blocks(text)
    for seq in seqs:
        for golden_answer in golden_answers:
            if normalize_answer(golden_answer) in normalize_answer(seq):
                return True
    return False


def compute_score_em(
    solution_str,
    ground_truth,
    method="strict",
    structure_format_score=0,
    final_format_score=0,
    retrieval_score=0,
    format_score=0,
    score=1.0,
):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    if not isinstance(ground_truth, dict):
        ground_truth = {"target": []}
    if "target" not in ground_truth:
        ground_truth["target"] = []

    is_valid_format, _ = is_valid_sequence(solution_str)
    retrieval_correct = False
    if is_valid_format:
        retrieval_correct = is_retrieval_correct(solution_str, ground_truth["target"])
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    if answer is None:
        if is_valid_format:
            if retrieval_correct:
                return structure_format_score + retrieval_score  # 0.3
            else:
                return structure_format_score  # 0.2
        else:
            return 0
    else:
        if em_check(answer, ground_truth["target"]):
            if is_valid_format:
                return score  # 1
            else:
                return score - structure_format_score  # 0.8
        elif is_valid_format:
            if retrieval_correct:
                return structure_format_score + retrieval_score  # 0.3
            else:
                return structure_format_score  # 0.2
        else:
            return final_format_score  # 0.1
