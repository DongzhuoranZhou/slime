# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/ceee7b89655ed52f205b9beb98e1190c3eedcfb0/search_r1/llm_agent/generation.py
# This is a unified version supporting both local search and Google search, with optional log probability collection

import asyncio
import json
import re
from typing import Any, Optional, Tuple, cast

from qa_em_format import compute_score_em

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

# Configuration for Search-R1
SEARCH_R1_CONFIGS = {
    # ============== General Configuration ==============
    "max_turns": 2,
    "topk": 3,
    "search_concurrency": 256,
    # ============== Search Backend Selection ==============
    "search_backend": "local",  # Options: "local" or "google"
    # ============== Local Search Configuration ==============
    # (Only used when search_backend="local")
    "local": {
        "search_url": "http://127.0.0.1:8000/retrieve",  # URL of your local retrieval server
        "proxy": None,  # Set to your proxy if needed
    },
    # ============== Google Search Configuration ==============
    # (Only used when search_backend="google")
    "google": {
        "api_key": "your_api_key_here",  # Replace with your actual API key
        "snippet_only": True,  # Set to True to only return snippets
        "proxy": None,  # Set to your proxy if needed
    },
    # ============== Log Probability Collection ==============
    "return_logprob": True,  # Set to True to collect log probabilities for TIS metrics
    # ============== Reward Model Configuration ==============
    "format_score": 0.2,
}


SEMAPHORE = asyncio.Semaphore(SEARCH_R1_CONFIGS["search_concurrency"])

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(\{.*?\})\s*</tool_response>", re.DOTALL)
TOOL_INSTRUCTIONS = (
    "\nYou can call tools by emitting a JSON object inside <tool_call>...</tool_call>.\n"
    "Available tools:\n"
    "- search: {\"query\": \"...\"}\n"
    "- final_answer: {\"answer\": \"...\"}\n"
    "For search results, you will receive a <tool_response>{\"name\":\"search\",...}</tool_response>.\n"
    "Only use <tool_call> for actions; do not use <search> or <answer> tags.\n"
)


def _passages2string(retrieval_result):
    """
    Convert retrieval results to a formatted string.
    This function works with both google_search and local_search results.
    """
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

    return format_reference


async def search(query: str) -> str:
    """
    Perform search using either local search engine or Google search.
    The search backend is determined by SEARCH_R1_CONFIGS["search_backend"].
    """
    backend = SEARCH_R1_CONFIGS["search_backend"]

    if backend == "local":
        from local_search_server import local_search

        local_config = SEARCH_R1_CONFIGS["local"]
        result = await local_search(
            local_config["search_url"],
            query,
            SEARCH_R1_CONFIGS["topk"],
            proxy=local_config["proxy"],
        )
    elif backend == "google":
        from google_search_server import google_search

        google_config = SEARCH_R1_CONFIGS["google"]
        result = await google_search(
            google_config["api_key"],
            query,
            SEARCH_R1_CONFIGS["topk"],
            snippet_only=google_config["snippet_only"],
            proxy=google_config["proxy"],
        )
    else:
        raise ValueError(f"Unknown search backend: {backend}. " f"Must be either 'local' or 'google'.")

    return _passages2string(result)


# IMPORTANT: When we need to collect log probabilities (logp), we CANNOT do any postprocessing
# on the strings returned from the inference engine (sglang). This is because:
# 1. We don't know how to truncate the corresponding tokens/logp arrays to match the modified string
# 2. Re-tokenizing the postprocessed string may produce different tokens than what the engine generated,
#    leading to misalignment between tokens and their log probabilities
# Therefore, postprocess_responses is only used when return_logprob=False.
def postprocess_responses(resp: str) -> str:
    """
    Post-process response to ensure tag completeness.
    Only used when SEARCH_R1_CONFIGS["return_logprob"] is False.
    """
    matches = list(TOOL_CALL_RE.finditer(resp)) + list(TOOL_RESPONSE_RE.finditer(resp))
    if matches:
        last_match = max(matches, key=lambda m: m.end())
        return resp[: last_match.end()]
    return resp


def postprocess_predictions(prediction: str) -> Tuple[Optional[str], dict[str, Any]]:
    matches = list(TOOL_CALL_RE.finditer(prediction))
    if not matches:
        return None, {}

    json_str = matches[-1].group(1)
    try:
        payload = json.loads(json_str)
    except json.JSONDecodeError:
        return None, {}

    if not isinstance(payload, dict):
        return None, {}

    tool_name = payload.get("name")
    arguments = payload.get("arguments") or {}
    if not isinstance(arguments, dict):
        return None, {}

    return tool_name, arguments


async def execute_predictions(prediction: str) -> tuple[str, bool]:
    tool_name, arguments = postprocess_predictions(prediction)
    if arguments is None:
        arguments = {}

    if tool_name == "search":
        search_query = (arguments.get("query") or "").strip()
        if not search_query:
            next_obs = (
                "\nMy previous action is invalid. "
                "Use <tool_call>{\"name\":\"search\",\"arguments\":{\"query\":\"...\"}}</tool_call> "
                "or <tool_call>{\"name\":\"final_answer\",\"arguments\":{\"answer\":\"...\"}}</tool_call>.\n"
            )
            return next_obs, False

        try:
            async with SEMAPHORE:
                search_results = await search(search_query)
            response_payload = {
                "name": "search",
                "status": "success",
                "content": search_results.strip(),
            }
        except Exception as exc:
            response_payload = {
                "name": "search",
                "status": "error",
                "error": str(exc),
            }
        response_json = json.dumps(response_payload, ensure_ascii=True)
        next_obs = f"\n\n<tool_response>{response_json}</tool_response>\n\n"
        done = False
    elif tool_name == "final_answer":
        answer_text = (arguments.get("answer") or "").strip()
        if not answer_text:
            next_obs = (
                "\nMy previous action is invalid. "
                "Use <tool_call>{\"name\":\"final_answer\",\"arguments\":{\"answer\":\"...\"}}</tool_call>.\n"
            )
            return next_obs, False
        next_obs = ""
        done = True
    else:
        next_obs = (
            "\nMy previous action is invalid. "
            "Use <tool_call>{\"name\":\"search\",\"arguments\":{\"query\":\"...\"}}</tool_call> "
            "or <tool_call>{\"name\":\"final_answer\",\"arguments\":{\"answer\":\"...\"}}</tool_call>.\n"
        )
        done = False

    return next_obs, done


def _inject_tool_instructions(prompt_text, tokenizer) -> str:
    if not isinstance(prompt_text, str):
        prompt_text = tokenizer.apply_chat_template(prompt_text, tokenize=False, add_generation_prompt=True)

    system_start = prompt_text.find("<|im_start|>system")
    if system_start == -1:
        return f"<|im_start|>system\n{TOOL_INSTRUCTIONS}<|im_end|>\n{prompt_text}"

    system_end = prompt_text.find("<|im_end|>", system_start)
    if system_end == -1:
        return prompt_text

    return prompt_text[:system_end] + TOOL_INSTRUCTIONS + prompt_text[system_end:]


async def generate(args, sample: Sample, sampling_params) -> Sample:
    assert not args.partial_rollout, "Partial rollout is not supported for this function at the moment."

    state = GenerateState(args)

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # Handle partial rollout samples: continue generation from existing response
    prompt_text = _inject_tool_instructions(sample.prompt, state.tokenizer)
    prompt_tokens_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    response = ""
    response_token_ids = []
    loss_mask = []
    rollout_log_probs = [] if SEARCH_R1_CONFIGS["return_logprob"] else None

    output: Optional[dict[str, Any]] = None
    for _turn_idx in range(SEARCH_R1_CONFIGS["max_turns"]):
        cur_response_log_probs = []
        payload = {
            "text": prompt_text + response,
            "sampling_params": sampling_params,
        }
        # Add log probability collection if enabled
        if SEARCH_R1_CONFIGS["return_logprob"]:
            payload["return_logprob"] = True

        raw_output = await post(url, payload)
        output = cast(dict[str, Any], raw_output) if isinstance(raw_output, dict) else {}

        # abort
        if output.get("meta_info", {}).get("finish_reason", {}).get("type") == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        output_text = output.get("text", "")
        cur_response = output_text

        # Extract tokens and log probs based on configuration
        if SEARCH_R1_CONFIGS["return_logprob"]:
            # Extract log probs from output - required for TIS metrics
            if "output_token_logprobs" not in output.get("meta_info", {}):
                raise RuntimeError(
                    "output_token_logprobs not found in output meta_info. "
                    "Make sure 'return_logprob': True is set in the payload."
                )

            # Use token IDs and log probs directly from output_token_logprobs
            # This ensures perfect alignment between tokens and log probs
            # output_token_logprobs format: [[log_prob, token_id, ...], ...]
            cur_response_token_ids = [item[1] for item in output.get("meta_info", {}).get("output_token_logprobs", [])]
            cur_response_log_probs = [item[0] for item in output.get("meta_info", {}).get("output_token_logprobs", [])]
        else:
            # When not collecting log probs, we can safely postprocess the response
            cur_response = postprocess_responses(cur_response)
            # Tokenize the (possibly postprocessed) response
            cur_response_token_ids = state.tokenizer(cur_response, add_special_tokens=False)["input_ids"]

        response += cur_response
        response_token_ids += cur_response_token_ids
        loss_mask += [1] * len(cur_response_token_ids)

        # Add log probs if enabled
        if SEARCH_R1_CONFIGS["return_logprob"]:
            if rollout_log_probs is None:
                rollout_log_probs = []
            rollout_log_probs += cur_response_log_probs

        finish_reason = output.get("meta_info", {}).get("finish_reason", {}).get("type")
        if finish_reason == "length":
            break

        next_obs, done = await execute_predictions(cur_response)
        if done:
            break

        assert next_obs != "", "Next observation should not be empty."
        obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
        response += next_obs
        response_token_ids += obs_tokens_ids
        loss_mask += [0] * len(obs_tokens_ids)

        # Add dummy log probs for observation tokens if enabled (they won't be used due to loss_mask=0)
        if SEARCH_R1_CONFIGS["return_logprob"]:
            if rollout_log_probs is None:
                rollout_log_probs = []
            rollout_log_probs += [0.0] * len(obs_tokens_ids)

            # Verify alignment when collecting log probs
            assert len(response_token_ids) == len(
                rollout_log_probs
            ), f"Token/logp length mismatch: {len(response_token_ids)} tokens vs {len(rollout_log_probs)} logps"

    # Store statistics for wandb logging
    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_mask
    sample.prompt = prompt_text

    # Store log probs if enabled
    if SEARCH_R1_CONFIGS["return_logprob"]:
        sample.rollout_log_probs = rollout_log_probs if rollout_log_probs else None

    if output is not None:
        finish_type = output.get("meta_info", {}).get("finish_reason", {}).get("type")
        if finish_type == "length":
            sample.status = Sample.Status.TRUNCATED
        elif finish_type == "abort":
            sample.status = Sample.Status.ABORTED
        elif finish_type == "stop":
            sample.status = Sample.Status.COMPLETED

    return sample


async def reward_func(args, sample, **kwargs):
    """The reward function for retrieval-based question answering.

    Args:
        args: the arguments
        sample: the sample to evaluate
    """
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    prompt_text = sample.prompt if isinstance(sample.prompt, str) else ""
    label = sample.label if isinstance(sample.label, dict) else {}
    ground_truth = label.get("ground_truth", {"target": []})
    score = compute_score_em(
        solution_str=prompt_text + sample.response,
        ground_truth=ground_truth,
        format_score=SEARCH_R1_CONFIGS["format_score"],
    )

    return score
