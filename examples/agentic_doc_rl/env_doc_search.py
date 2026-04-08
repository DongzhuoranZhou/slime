# examples/agentic_doc_rl/env_doc_search.py
from __future__ import annotations

import json
import logging
import re
from typing import Any

from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

from examples.geo3k_vlm_multi_turn.base_env import BaseInteractionEnv
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

# Matches Qwen's native tool-call format: <tool_call>{...}</tool_call>
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

# ── Tool schemas (OpenAI function-calling format) ─────────────────────────────
# Passed to apply_chat_template(tools=TOOLS) so Qwen embeds them in the system
# prompt. The model learns to call them by name during SFT and RL rewards good calls.

SEARCH_DATABASE_TOOL = {
    "type": "function",
    "function": {
        "name": "search_database",
        "description": (
            "Search the document vector database for pages semantically relevant to the query. "
            "Returns the top matching page images."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string.",
                },
                "doc_name": {
                    "type": "string",
                    "description": "The exact document name to search within (from the task description).",
                },
                "excluded_pages": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Page numbers to exclude from results (pages already seen).",
                },
            },
            "required": ["query", "doc_name"],
        },
    },
}

GET_SPECIFIC_PAGES_TOOL = {
    "type": "function",
    "function": {
        "name": "get_specific_pages",
        "description": (
            "Retrieve specific page numbers directly from the document. "
            "Use when you know which pages to examine. Maximum 5 pages per call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "doc_name": {
                    "type": "string",
                    "description": "The exact document name.",
                },
                "page_numbers": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Page numbers to retrieve.",
                },
            },
            "required": ["doc_name", "page_numbers"],
        },
    },
}

TOOLS = [SEARCH_DATABASE_TOOL, GET_SPECIFIC_PAGES_TOOL]

# ── Module-level singletons ───────────────────────────────────────────────────
# QdrantClient and JinaV4Model are initialised once per worker process.
# JinaV4 is ~1.5 GB — reloading it per sample would be prohibitively slow.
_qdrant_client: QdrantClient | None = None
_embed_model: JinaV4Model | None = None


def _get_qdrant_client(qdrant_url: str) -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=qdrant_url)
    return _qdrant_client


def _get_embed_model(embed_device: str):
    global _embed_model
    if _embed_model is None:
        # search_models.py lives in AgenticMemory/ — added to PYTHONPATH by the launch script.
        from search_models import JinaV4Model  # noqa: PLC0415
        _embed_model = JinaV4Model(device=embed_device, multivector=True)
    return _embed_model


def _extract_tool_call(text: str) -> dict | None:
    """Parse the last <tool_call>JSON</tool_call> from the model response.

    Supports both Qwen native format {"name": ..., "arguments": ...} and
    OpenAI fallback {"function": {"name": ..., "arguments": ...}}.
    Returns None if no valid tool call is found.
    """
    matches = list(_TOOL_CALL_RE.finditer(text))
    if not matches:
        return None
    raw = matches[-1].group(1).strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    name = payload.get("name") or payload.get("function", {}).get("name")
    arguments = payload.get("arguments") or payload.get("function", {}).get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not name:
        return None
    return {"name": name, "arguments": arguments}


class DocSearchEnv(BaseInteractionEnv):
    """Multi-turn document search environment for RL training.

    The model issues tool calls to retrieve page images from Qdrant.
    Each call to step() processes one model response turn.
    """

    def __init__(
        self,
        *,
        ground_truth: Any = None,
        max_turns: int,
        qdrant_client: QdrantClient,
        embed_model: Any,
        collection_name: str,
        embed_model_name: str = "jinav4_multivector",
        topk: int = 3,
    ):
        self.ground_truth = ground_truth
        self.max_turns = max_turns
        self._client = qdrant_client
        self._embed_model = embed_model
        self._collection_name = collection_name
        self._embed_model_name = embed_model_name
        self._topk = topk
        self.turn = 0
        self.final_answer: str | None = None

    def reset(self):
        self.turn = 0
        self.final_answer = None
        return {}, {}

    def close(self):
        pass

    # ── Qdrant helpers ────────────────────────────────────────────────────────

    def _search_database(
        self, query: str, doc_name: str, excluded_pages: list[int]
    ) -> tuple[list[Image.Image], list[int]]:
        query_vector = self._embed_model.embed_text(query)
        must_not = (
            [FieldCondition(key="page_num", match=MatchAny(any=excluded_pages))]
            if excluded_pages
            else []
        )
        query_filter = Filter(
            must=[FieldCondition(key="document_name", match=MatchValue(value=doc_name))],
            must_not=must_not,
        )
        results = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector.tolist() if hasattr(query_vector, "tolist") else query_vector,
            using=self._embed_model_name,
            limit=self._topk,
            with_payload=True,
            query_filter=query_filter,
        )
        images, pages = [], []
        for point in results.points:
            with Image.open(point.payload["full_img_path"]) as img:
                img.load()
                images.append(img.copy())
            pages.append(point.payload["page_num"])
        return images, pages

    def _get_specific_pages(
        self, doc_name: str, page_numbers: list[int]
    ) -> tuple[list[Image.Image], list[int]]:
        page_numbers = page_numbers[:5]
        query_filter = Filter(
            must=[
                FieldCondition(key="document_name", match=MatchValue(value=doc_name)),
                FieldCondition(key="page_num", match=MatchAny(any=page_numbers)),
            ]
        )
        scroll_results = self._client.scroll(
            collection_name=self._collection_name,
            scroll_filter=query_filter,
            limit=len(page_numbers),
            with_payload=True,
        )
        images, pages = [], []
        for point in scroll_results[0]:
            with Image.open(point.payload["full_img_path"]) as img:
                img.load()
                images.append(img.copy())
            pages.append(point.payload["page_num"])
        return images, pages

    # ── Core env interface ────────────────────────────────────────────────────

    async def step(self, response_text: str):
        """Process one model response turn.

        Returns:
            (observation_dict, done_bool, info_dict)
            observation_dict keys: "images" (list[PIL.Image]), "obs_str" (str)
        """
        self.turn += 1
        is_final = self.turn >= self.max_turns
        info: dict[str, Any] = {}

        tool_call = _extract_tool_call(response_text)
        info["tool_call"] = tool_call

        if tool_call is None:
            info["tool_executed"] = False
            return {"images": [], "obs_str": ""}, True, info

        name = (tool_call.get("name") or "").strip()
        args = tool_call.get("arguments") or {}

        if name == "final_answer":
            self.final_answer = (args.get("answer") or args.get("response") or "").strip()
            info["tool_executed"] = True
            info["final_answer"] = self.final_answer
            return {"images": [], "obs_str": ""}, True, info

        if name == "search_database":
            query = (args.get("query") or "").strip()
            doc_name = (args.get("doc_name") or "").strip()
            excluded_pages = args.get("excluded_pages") or []
            if not query or not doc_name:
                info["tool_executed"] = False
                obs = {"images": [], "obs_str": "search_database requires 'query' and 'doc_name'."}
                return obs, is_final, info
            try:
                images, pages = self._search_database(query, doc_name, excluded_pages)
                pages_str = ", ".join(map(str, pages))
                obs = {"images": images, "obs_str": f"Found relevant content on pages {pages_str}."}
                info["tool_executed"] = True
                info["pages"] = pages
            except Exception as exc:
                logger.warning("search_database failed: %s", exc)
                obs = {"images": [], "obs_str": f"Search failed: {exc}"}
                info["tool_executed"] = False
            return obs, is_final, info

        if name == "get_specific_pages":
            doc_name = (args.get("doc_name") or "").strip()
            page_numbers = args.get("page_numbers") or []
            if not doc_name or not page_numbers:
                info["tool_executed"] = False
                obs = {"images": [], "obs_str": "get_specific_pages requires 'doc_name' and 'page_numbers'."}
                return obs, is_final, info
            try:
                images, pages = self._get_specific_pages(doc_name, page_numbers)
                pages_str = ", ".join(map(str, pages))
                obs = {"images": images, "obs_str": f"Retrieved pages {pages_str}."}
                info["tool_executed"] = True
                info["pages"] = pages
            except Exception as exc:
                logger.warning("get_specific_pages failed: %s", exc)
                obs = {"images": [], "obs_str": f"Page retrieval failed: {exc}"}
                info["tool_executed"] = False
            return obs, is_final, info

        # Unknown tool — return helpful error and continue the episode.
        info["tool_executed"] = False
        supported = ", ".join(t["function"]["name"] for t in TOOLS)
        obs = {
            "images": [],
            "obs_str": f"Tool '{name}' is not supported. Available tools: {supported}.",
        }
        return obs, is_final, info

    def format_observation(self, observation: dict) -> dict:
        """Convert observation dict to a tool-role chat message with images.

        The rollout's _encode_observation_for_generation() calls
        process_vision_info() on this message to extract PIL images and
        encode them for SGLang. Observation tokens receive loss_mask=0.
        """
        images = observation.get("images") or []
        obs_str = observation.get("obs_str", "")
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": obs_str})
        return {"role": "tool", "content": content}


def build_env(sample: Sample | None = None, args: Any | None = None, **_: Any) -> DocSearchEnv:
    """Factory function called by geo3k_vlm_multi_turn/rollout.py."""
    if args is None:
        raise ValueError("args must be provided to build_env.")

    max_turns = getattr(args, "max_turns", None)
    if max_turns is None:
        raise ValueError("max_turns must be set via --custom-config-path.")

    qdrant_url = getattr(args, "qdrant_url", "http://localhost:6333")
    collection_name = getattr(args, "collection_name", "mmlongdoc")
    embed_device = getattr(args, "embed_device", "cuda:0")
    topk = getattr(args, "topk", 3)

    qdrant_client = _get_qdrant_client(qdrant_url)
    embed_model = _get_embed_model(embed_device)

    if sample is not None:
        sample.metadata = sample.metadata or {}
        sample.metadata["tools"] = TOOLS

    ground_truth = None
    if sample is not None and sample.label is not None:
        label = sample.label
        if isinstance(label, dict):
            ground_truth = label.get("ground_truth") or label.get("answer") or label.get("target")
        else:
            ground_truth = str(label)

    return DocSearchEnv(
        ground_truth=ground_truth,
        max_turns=max_turns,
        qdrant_client=qdrant_client,
        embed_model=embed_model,
        collection_name=collection_name,
        topk=topk,
    )
