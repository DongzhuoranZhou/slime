"""
SearchEnv: standard tool-calling environment for web search RL training.

The model is expected to emit:
    <tool_call>{"name": "web_search", "arguments": {"query": "..."}}</tool_call>

Search results are returned as a tool observation turn, injected via
apply_chat_template by the geo3k_vlm_multi_turn rollout infrastructure.

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

from slime.utils.async_utils import run
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

# Matches <tool_call>{...}</tool_call> — same pattern as env_geo3k.py
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

# OpenAI-compatible tool schema for the web search tool.
# Passed to tokenizer.apply_chat_template(tools=[WEB_SEARCH_TOOL]) so Qwen
# embeds the function definition in the system prompt automatically.
WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for information relevant to the question.",
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

    On each turn the rollout calls step(response_text). If the model emitted a
    <tool_call> for web_search, we execute the search synchronously (bridging
    to the async search function via slime.utils.async_utils.run) and return
    the results as the next observation. If no tool call is found we treat the
    response as the model's final answer and set done=True.
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

        # Support both {"name":..., "arguments":...} and {"function":{"name":...}}
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

    def step(self, response_text: str):
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
            # No tool call → model has given its final answer (or errored).
            info["tool_executed"] = False
            obs = {"obs_str": "", "role": "tool"}
            return obs, True, info

        name = (tool_call.get("name") or "").strip()
        if name != "web_search":
            info["tool_executed"] = False
            obs = {
                "obs_str": (
                    f"Tool `{name}` is not supported. "
                    'Use <tool_call>{"name": "web_search", "arguments": {"query": "your query"}}</tool_call>'
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
                    'Use <tool_call>{"name": "web_search", "arguments": {"query": "your query"}}</tool_call>'
                ),
                "role": "tool",
            }
            return obs, is_final, info

        # Execute search synchronously by submitting to the background async loop (L2).
        # slime.utils.async_utils.run() uses run_coroutine_threadsafe + .result(),
        # which blocks only this thread — not the main asyncio event loop running
        # the outer rollout.py generate() coroutine.
        result_str = run(_dispatch_search(query, self._config))
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

    Also injects the WEB_SEARCH_TOOL schema into sample.metadata["tools"] so
    that _encode_observation_for_generation() uses the correct tools parameter
    when calling apply_chat_template for subsequent observation turns.
    """
    if args is None or args.max_turns is None:
        raise ValueError("max_turns must be set via --custom-config-path.")

    if sample is not None:
        sample.metadata = sample.metadata or {}
        sample.metadata["tools"] = [WEB_SEARCH_TOOL]

    ground_truth = _extract_ground_truth(sample)
    if ground_truth is None:
        logger.warning("Ground truth missing; EM reward will always be 0.")

    return SearchEnv(
        ground_truth=ground_truth,
        max_turns=args.max_turns,
        config=_build_config(args),
    )
