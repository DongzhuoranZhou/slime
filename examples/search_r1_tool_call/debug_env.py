"""
Standalone debug script for SearchEnv — no SGLang server or search backend needed.

Run directly:
    cd /workspace/src/clean_code_for_rl/slime_0224_2026/slime
    PYTHONPATH=. python examples/search_r1_tool_call/debug_env.py

Or use the VSCode launch config: "Debug: SearchEnv step-through"

Set breakpoints anywhere in:
    examples/search_r1_tool_call/env_search.py  (step(), _extract_tool_call(), format_observation())
    examples/geo3k_vlm_multi_turn/rollout.py    (_encode_observation_for_generation())
    this file                                   (main())

What this script exercises (without needing a real server):
    Turn 1 — model issues a search tool call
              → SearchEnv.step() parses JSON, calls (mocked) _dispatch_search
              → obs = {"obs_str": "Doc 1...", "role": "tool"}
              → format_observation() → {"role": "tool", "content": "Doc 1..."}
              → _encode_observation_for_generation() trims chat-template prefix → obs token ids

    Turn 2 — model issues an answer tool call
              → SearchEnv.step() parses JSON, sets done=True
              → obs = {"obs_str": "", "role": "tool"}
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock

# ── PYTHONPATH: make examples.* and slime.* importable ──────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from examples.search_r1_tool_call.env_search import SearchEnv, TOOLS
from examples.geo3k_vlm_multi_turn.rollout import _encode_observation_for_generation
from slime.utils.types import Sample
from slime.utils.processing_utils import load_tokenizer

# ── Config (mirrors config.yaml defaults) ───────────────────────────────────
CONFIG = {
    "search_backend": "local",       # irrelevant — _dispatch_search is mocked
    "search_concurrency": 1,
    "topk": 3,
    "local_search_url": "http://127.0.0.1:8000/retrieve",
    "local_search_proxy": None,
}
MAX_TURNS = 2
HF_CHECKPOINT = "Qwen/Qwen3-4B"    # HF Hub ID — auto-downloads if not cached locally

# ── Fake model responses ─────────────────────────────────────────────────────
# Turn 1: model thinks, then calls the search tool
TURN1_RESPONSE = (
    "<think>\nI need to look up the capital of France.\n</think>\n"
    '<tool_call>{"name": "search", "arguments": {"query": "capital of France"}}</tool_call>'
)

# Turn 2: model thinks, then submits the final answer
TURN2_RESPONSE = (
    "<think>\nThe search results confirm it is Paris.\n</think>\n"
    '<tool_call>{"name": "answer", "arguments": {"response": "Paris"}}</tool_call>'
)

# ── Fake search results (replaces live local/google search) ──────────────────
FAKE_PASSAGES = [
    {"title": "France", "contents": "Paris is the capital and most populous city of France."},
    {"title": "Capital city", "contents": "A capital city is where a government is located."},
    {"title": "Paris", "contents": "Paris has been the capital of France since the 10th century."},
]

async def _fake_dispatch_search(query: str, config: dict) -> str:
    """Replaces the real _dispatch_search — returns formatted fake passages."""
    return "\n".join(
        f"Doc {i+1}(Title: {p['title']}) {p['contents']}"
        for i, p in enumerate(FAKE_PASSAGES)
    )


async def main():
    # ── Build SearchEnv directly (no args/sample needed) ────────────────────
    env = SearchEnv(ground_truth="Paris", max_turns=MAX_TURNS, config=CONFIG)
    env.reset()

    # ── Load real tokenizer for _encode_observation_for_generation() demo ───
    print(f"Loading tokenizer from {HF_CHECKPOINT} ...")
    tokenizer = load_tokenizer(HF_CHECKPOINT, trust_remote_code=True, cache_dir="/workspace/.cache/huggingface/hub")
    print("Tokenizer loaded.\n")

    # metadata mirrors what build_env() injects into sample.metadata["tools"]
    metadata = {"tools": TOOLS}
    apply_chat_template_kwargs = {"enable_thinking": True}

    # ════════════════════════════════════════════════════════════════════════
    # TURN 1 — search
    # ════════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("TURN 1  model response:")
    print(TURN1_RESPONSE)
    print()

    with patch(
        "examples.search_r1_tool_call.env_search._dispatch_search",
        new=AsyncMock(side_effect=_fake_dispatch_search),
    ):
        # ← SET BREAKPOINT HERE to inspect tool_call, result_str, obs inside step()
        obs, done, info = env.step(TURN1_RESPONSE)

    print(f"obs        = {obs!r}")       # {"obs_str": "Doc 1...", "role": "tool"}
    print(f"done       = {done!r}")      # False — episode continues
    print(f"info       = {info!r}")      # {"tool_call": {...}, "tool_executed": True, "query": "..."}
    print()

    # format_observation() renames obs_str → content
    chat_msg = env.format_observation(obs)
    print(f"chat_msg   = {chat_msg!r}")  # {"role": "tool", "content": "Doc 1..."}
    print()

    # _encode_observation_for_generation() applies chat template and trims prefix
    obs_token_ids, image_data, _, _ = _encode_observation_for_generation(
        tokenizer=tokenizer,
        processor=None,
        message=chat_msg,
        metadata=metadata,
        apply_chat_template=True,
        apply_chat_template_kwargs=apply_chat_template_kwargs,
    )
    decoded = tokenizer.decode(obs_token_ids, skip_special_tokens=False)
    print(f"obs_token_ids (len={len(obs_token_ids)}):")
    print(f"  decoded = {decoded!r}")
    print()

    # ════════════════════════════════════════════════════════════════════════
    # TURN 2 — answer (ends episode)
    # ════════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("TURN 2  model response:")
    print(TURN2_RESPONSE)
    print()

    # ← SET BREAKPOINT HERE to inspect tool_call, done, info["final_answer"]
    obs2, done2, info2 = env.step(TURN2_RESPONSE)

    print(f"obs        = {obs2!r}")      # {"obs_str": "", "role": "tool"}
    print(f"done       = {done2!r}")     # True — episode ended
    print(f"info       = {info2!r}")     # {"tool_call": {...}, "tool_executed": True, "final_answer": "Paris"}
    print()
    print("Episode complete. Final answer extracted:", info2.get("final_answer"))


if __name__ == "__main__":
    asyncio.run(main())
