"""
SearchEnv: standard tool-calling environment for retrieval-augmented RL training.

The model uses two native Qwen tool calls:
    <tool_call>{"name": "search", "arguments": {"query": "..."}}</tool_call>
    <tool_call>{"name": "answer", "arguments": {"response": "..."}}</tool_call>

The search tool is backend-agnostic: config.yaml selects "local" (dense retrieval via
local_search_server.py) or "google" (Google Search API via google_search_server.py).
Search results are returned as a tool observation turn via apply_chat_template.
The answer tool ends the episode immediately with no observation.

Reuses local_search_server.py and google_search_server.py from examples/search-r1/
(exposed via PYTHONPATH in the launch script).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from copy import deepcopy
from typing import Any

from examples.geo3k_vlm_multi_turn.base_env import BaseInteractionEnv

from slime.utils.types import Sample

logger = logging.getLogger(__name__)

# Matches <tool_call>{...}</tool_call> — same pattern as env_geo3k.py
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

# OpenAI-compatible tool schemas.
# Passed to tokenizer.apply_chat_template(tools=TOOLS) so Qwen embeds both
# function definitions in the system prompt automatically.
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search for information relevant to the question.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string.",
                }
            },
            "required": ["query"],
        },
    },
}

ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": "answer",
        "description": "Submit your final answer to the question.",
        "parameters": {
            "type": "object",
            "properties": {
                "response": {
                    "type": "string",
                    "description": "Your final answer.",
                }
            },
            "required": ["response"],
        },
    },
}

TOOLS = [SEARCH_TOOL, ANSWER_TOOL]

# Module-level semaphore — created lazily on the background async loop (L2)
# so it belongs to the same event loop that runs all search coroutines.
_SEARCH_SEMAPHORE: asyncio.Semaphore | None = None


def _passages2string(retrieval_result: list[dict]) -> str:
    """Format retrieval results into a readable string (mirrors search-r1)."""
    out = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        out += f"Doc {idx + 1}(Title: {title}) {text}\n"
    return out


async def _dispatch_search(query: str, config: dict) -> str:
    """Run the appropriate search backend and return a formatted result string."""
    global _SEARCH_SEMAPHORE
    if _SEARCH_SEMAPHORE is None:
        _SEARCH_SEMAPHORE = asyncio.Semaphore(config["search_concurrency"])

    async with _SEARCH_SEMAPHORE:
        backend = config["search_backend"]
        topk = config["topk"]

        if backend == "local":
            from local_search_server import local_search

            result = await local_search(
                config["local_search_url"],
                query,
                topk,
                proxy=config.get("local_search_proxy"),
            )
        elif backend == "google":
            from google_search_server import google_search

            result = await google_search(
                config["google_api_key"],
                query,
                topk,
                snippet_only=config.get("google_snippet_only", True),
                proxy=config.get("google_proxy"),
            )
        else:
            raise ValueError(f"Unknown search_backend: {backend!r}. Must be 'local' or 'google'.")

    return _passages2string(result)


class SearchEnv(BaseInteractionEnv):
    """
    Interaction environment for multi-turn web-search RL.

    On each turn the rollout calls step(response_text). The model uses two tools:
    - search: execute a search, return results as the next observation.
    - answer: submit final answer, end the episode immediately (done=True, empty obs).
    If no tool call is found the episode also ends (model gave free-form response).
    """

    def __init__(self, *, ground_truth: Any = None, max_turns: int, config: dict):
        self.ground_truth = ground_truth
        self.max_turns = max_turns
        self._config = config
        self.turn = 0

    def reset(self):
        self.turn = 0
        return {}, {"ground_truth_available": self.ground_truth is not None}

    def close(self):
        pass

    def _extract_tool_call(self, text: str) -> dict | None:
        """Parse the last <tool_call>JSON</tool_call> from the model response."""
        matches = list(TOOL_CALL_RE.finditer(text))
        if not matches:
            return None
        raw = matches[-1].group(1).strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse tool call JSON: %s", exc)
            return None

        # Qwen native format:    {"name": "search", "arguments": {"query": "..."}}
        # OpenAI/fallback format: {"function": {"name": "search", "arguments": {...}}}
        # Qwen's apply_chat_template produces the first format; the second is kept
        # as a defensive fallback in case an OpenAI-style wrapper wraps the payload.
        name = payload.get("name") or payload.get("function", {}).get("name")
        arguments = payload.get("arguments") or payload.get("function", {}).get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                logger.warning("Tool call arguments are not valid JSON; ignoring.")
                return None
        if not name:
            return None
        return {"name": name, "arguments": arguments}

    async def step(self, response_text: str):
        """
        Process one model response turn.

        Returns:
            (observation_dict, done_bool, info_dict)

        The observation_dict contains:
            "obs_str": text to show the model as the tool result
            "role":    "tool" (so format_observation returns the right role)
        """
        self.turn += 1
        is_final = self.turn >= self.max_turns
        info: dict[str, Any] = {}

        tool_call = self._extract_tool_call(response_text)
        info["tool_call"] = deepcopy(tool_call)

        if tool_call is None:
            # No tool call → model gave free-form response without using tools.
            info["tool_executed"] = False
            obs = {"obs_str": "", "role": "tool"}
            return obs, True, info

        name = (tool_call.get("name") or "").strip()

        if name == "answer":
            # Model submitted its final answer — end episode immediately.
            response = (tool_call["arguments"].get("response") or "").strip()
            info["tool_executed"] = True
            info["final_answer"] = response
            obs = {"obs_str": "", "role": "tool"}
            return obs, True, info

        if name != "search":
            info["tool_executed"] = False
            obs = {
                "obs_str": (
                    f"Tool `{name}` is not supported. "
                    'Use <tool_call>{"name": "search", "arguments": {"query": "your query"}}</tool_call> '
                    'to search, or <tool_call>{"name": "answer", "arguments": {"response": "your answer"}}</tool_call> '
                    "to submit your final answer."
                ),
                "role": "tool",
            }
            return obs, is_final, info

        query = (tool_call["arguments"].get("query") or "").strip()
        if not query:
            info["tool_executed"] = False
            obs = {
                "obs_str": (
                    "No query provided. "
                    'Use <tool_call>{"name": "search", "arguments": {"query": "your query"}}</tool_call>'
                ),
                "role": "tool",
            }
            return obs, is_final, info

        result_str = await _dispatch_search(query, self._config)
        info["tool_executed"] = True
        info["query"] = query

        obs = {"obs_str": result_str.strip(), "role": "tool"}
        return obs, is_final, info

    def format_observation(self, observation: dict) -> dict:
        """
        Wrap the observation as a tool-role chat message.

        Returns {"role": "tool", "content": "..."} so that
        tokenizer.apply_chat_template produces Qwen's native
        <|im_start|>tool\n...<|im_end|> tokens.
        """
        return {"role": "tool", "content": observation.get("obs_str", "")}


def _build_config(args: Any) -> dict:
    return {
        "search_backend": getattr(args, "search_backend", "local"),
        "search_concurrency": getattr(args, "search_concurrency", 256),
        "topk": getattr(args, "topk", 3),
        "local_search_url": getattr(args, "local_search_url", "http://127.0.0.1:8000/retrieve"),
        "local_search_proxy": getattr(args, "local_search_proxy", None),
        "google_api_key": getattr(args, "google_api_key", ""),
        "google_snippet_only": getattr(args, "google_snippet_only", True),
        "google_proxy": getattr(args, "google_proxy", None),
    }


def _extract_ground_truth(sample: Sample | None) -> Any:
    if sample is None:
        return None
    if sample.label is not None:
        return sample.label
    return None


def build_env(sample: Sample | None = None, args: Any | None = None, **_: Any) -> SearchEnv:
    """
    Factory function called by geo3k_vlm_multi_turn/rollout.py.

    Injects both tool schemas (web_search + answer) into sample.metadata["tools"]
    so that _encode_observation_for_generation() uses the correct tools parameter
    when calling apply_chat_template for subsequent observation turns.
    """
    if args is None or args.max_turns is None:
        raise ValueError("max_turns must be set via --custom-config-path.")

    if sample is not None:
        sample.metadata = sample.metadata or {}
        sample.metadata["tools"] = TOOLS

    ground_truth = _extract_ground_truth(sample)
    if ground_truth is None:
        logger.warning("Ground truth missing; EM reward will always be 0.")

    return SearchEnv(
        ground_truth=ground_truth,
        max_turns=args.max_turns,
        config=_build_config(args),
    )
