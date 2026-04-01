"""
Full end-to-end debug script: real SGLang inference + real search backend.

Calls geo3k_vlm_multi_turn/rollout.py:generate() directly — the same function
used in production — with a minimal args namespace pointing at your node's servers.

Prerequisites on your node
───────────────────────────
1. Start the local search server (e.g. on port 8000):
       cd examples/search-r1
       python local_search_server.py --port 8000

2. Start SGLang server (e.g. on port 30000):
       python -m sglang.launch_server \
           --model-path /workspace/.cache/huggingface/hub/Qwen3-4B \
           --port 30000 --host 0.0.0.0
       # 'python -m sglang.launch_server' runs from the installed package —
       # no file path needed. Binary also available at /usr/local/bin/sglang.

3. Edit SGLANG_HOST / SGLANG_PORT / SEARCH_URL below to match your node.

Set breakpoints anywhere in:
    examples/search_r1_tool_call/env_search.py   — step(), _dispatch_search()
    examples/geo3k_vlm_multi_turn/rollout.py     — generate(), _run_inference_step(),
                                                   _process_env_step(),
                                                   _encode_observation_for_generation()
    this file                                    — main()

What this script exercises:
    • Renders the initial prompt with apply_chat_template (as the data loader does)
    • Calls generate() — which creates GenerateState, builds SearchEnv, loops turns
    • Each turn: SGLang generates a response → SearchEnv.step() parses tool call
                 → real HTTP call to local_search_server → observation tokens appended
    • Prints the final Sample fields so you can inspect tokens / loss_mask / response
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

# ── PYTHONPATH ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
# Needed so _dispatch_search can do: from local_search_server import local_search
sys.path.insert(0, str(REPO_ROOT / "examples" / "search-r1"))

from examples.geo3k_vlm_multi_turn.rollout import generate
from examples.search_r1_tool_call.env_search import TOOLS
from slime.utils.http_utils import init_http_client
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

# ── Node profiles — switch by setting TARGET env var or changing ACTIVE_PROFILE
# Usage:  TARGET=training python debug_generate.py
#         TARGET=dev     python debug_generate.py
#         (default: dev)
_PROFILES = {
    "dev": {
        "sglang_host":    "172.20.112.104",
        "sglang_port":    30000,
        "search_url":     "http://172.20.112.104:8000/retrieve",
        "hf_checkpoint":  "/workspace/.cache/huggingface/hub/Qwen3-4B",
    },
    "training": {
        "sglang_host":    "127.0.0.1",
        "sglang_port":    30000,
        "search_url":     "http://127.0.0.1:8000/retrieve",
        "hf_checkpoint":  "/workspace/.cache/huggingface/hub/Qwen3-4B",
    },
}
ACTIVE_PROFILE = os.environ.get("TARGET", "training")
_profile = _PROFILES[ACTIVE_PROFILE]

SGLANG_HOST   = _profile["sglang_host"]
SGLANG_PORT   = _profile["sglang_port"]
SEARCH_URL    = _profile["search_url"]
HF_CHECKPOINT = _profile["hf_checkpoint"]

# ── Sample source: parquet row OR hardcoded fallback ─────────────────────────
# Set PARQUET_PATH to load a real training sample.
# Set to None to use the hardcoded QUESTION/GROUND_TRUTH below.
PARQUET_PATH = "/workspace/src/clean_code_for_rl/Search-R1/data/nq_hotpotqa_train_standard/train_standard.parquet"
PARQUET_ROW  = 10     # which row to debug (change to try different questions)

# Fallback if PARQUET_PATH is None
QUESTION     = "What is the capital of France?"
GROUND_TRUTH = "Paris"

# ── Minimal args namespace — satisfies all getattr() calls in generate() ─────
#
#   Sources:
#     GenerateState.__init__        slime/rollout/sglang_rollout.py:63
#     generate()                    examples/geo3k_vlm_multi_turn/rollout.py:309
#     _build_config() / build_env() examples/search_r1_tool_call/env_search.py:260
#
args = SimpleNamespace(
    # ── SGLang routing ────────────────────────────────────────────────────────
    sglang_router_ip=SGLANG_HOST,
    sglang_router_port=SGLANG_PORT,

    # ── Tokenizer / model ─────────────────────────────────────────────────────
    hf_checkpoint=HF_CHECKPOINT,

    # ── Multi-turn env ────────────────────────────────────────────────────────
    rollout_interaction_env_path="examples.search_r1_tool_call.env_search",
    max_turns=3,
    partial_rollout=False,

    # ── Sampling ──────────────────────────────────────────────────────────────
    rollout_temperature=0.7,
    rollout_top_p=1.0,
    rollout_top_k=-1,
    rollout_max_response_len=2048,
    rollout_max_context_len=None,
    rollout_stop=[],
    rollout_stop_token_ids=[],
    rollout_skip_special_tokens=False,

    # ── Chat template ─────────────────────────────────────────────────────────
    apply_chat_template=True,
    apply_chat_template_kwargs={"enable_thinking": True},

    # ── Search backend (forwarded to _build_config → SearchEnv) ──────────────
    search_backend="local",
    search_concurrency=1,
    topk=3,
    local_search_url=SEARCH_URL,
    local_search_proxy=None,
    google_api_key="",
    google_snippet_only=True,
    google_proxy=None,

    # ── GenerateState internals ───────────────────────────────────────────────
    # Semaphore size = sglang_server_concurrency * rollout_num_gpus // rollout_num_gpus_per_engine
    sglang_server_concurrency=1,
    rollout_num_gpus=1,
    rollout_num_gpus_per_engine=1,
    sglang_dp_size=1,
    sglang_enable_deterministic_inference=False,

    # ── HTTP client ───────────────────────────────────────────────────────────
    use_distributed_post=False,

    # ── Misc ──────────────────────────────────────────────────────────────────
    ci_test=False,
)


async def main():
    print(f"Loading tokenizer from {HF_CHECKPOINT} ...")
    tokenizer = load_tokenizer(
        HF_CHECKPOINT,
        trust_remote_code=True,
        cache_dir="/workspace/.cache/huggingface/hub",
    )

    if PARQUET_PATH:
        # ── Load a real row from the training parquet ─────────────────────────
        # The parquet "prompt" column is a list of message dicts (not yet rendered).
        # Slime's data loader calls apply_chat_template on it — we do the same here.
        df = pd.read_parquet(PARQUET_PATH)
        row = df.iloc[PARQUET_ROW]
        messages   = row["prompt"]           # list of {"role": ..., "content": ...}
        ground_truth = row["reward_model"]   # string or list — SearchEnv handles both
        tools_col  = json.loads(row["tools"]) if isinstance(row["tools"], str) else row["tools"]
        print(f"Loaded row {PARQUET_ROW} from {PARQUET_PATH}")
        print(f"  ground_truth = {ground_truth!r}")
        print(f"  messages[0]  = {messages[0]!r}\n")
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tools=tools_col,
            tokenize=False,
            add_generation_prompt=True,
            **args.apply_chat_template_kwargs,
        )
    else:
        # ── Fallback: hardcoded question ──────────────────────────────────────
        tools_col = TOOLS
        ground_truth = GROUND_TRUTH
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": QUESTION}],
            tools=tools_col,
            tokenize=False,
            add_generation_prompt=True,
            **args.apply_chat_template_kwargs,
        )

    print(f"Rendered prompt ({len(prompt_text)} chars):\n{prompt_text}\n")

    # ── Build Sample — mirrors what slime's data loader populates ─────────────
    sample = Sample(
        group_index=0,
        index=0,
        prompt=prompt_text,
        label=ground_truth,   # SearchEnv uses this as ground_truth for EM scoring
        metadata={"tools": tools_col},
        status=Sample.Status.PENDING,
    )

    # sampling_params are usually built by GenerateState and passed through;
    # generate() does sampling_params.copy() at the top so this dict is safe.
    sampling_params = {
        "temperature": args.rollout_temperature,
        "top_p":       args.rollout_top_p,
        "top_k":       args.rollout_top_k,
        "max_new_tokens": args.rollout_max_response_len,
        "stop":           args.rollout_stop,
        "stop_token_ids": args.rollout_stop_token_ids,
        "skip_special_tokens": args.rollout_skip_special_tokens,
        "no_stop_trim": True,
        "spaces_between_special_tokens": False,
    }

    # Initialize the shared httpx.AsyncClient (normally done by GenerateState)
    init_http_client(args)

    print(f"Profile: {ACTIVE_PROFILE} | SGLang at {SGLANG_HOST}:{SGLANG_PORT}")
    print(f"Search backend     → {SEARCH_URL}")
    print("─" * 60)

    # ← SET BREAKPOINT HERE to step into generate() and watch the turn loop
    result = await generate(args, sample, sampling_params)

    # ── Print finished Sample ─────────────────────────────────────────────────
    print("=" * 60)
    print(f"status        = {result.status}")
    print(f"response_len  = {result.response_length} tokens")
    print(f"loss_mask     = {result.loss_mask[:40]} ...")
    print(f"  (1=model output, 0=search observation — no gradient through obs)")
    print(f"reward        = {result.reward}")
    print()
    print("── response (decoded) ──")
    print(result.response)


if __name__ == "__main__":
    asyncio.run(main())
