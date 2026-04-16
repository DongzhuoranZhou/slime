"""
eval_async.py

Parallel evaluation script — same worker/SGLang/lock skeleton as
collect_trajectories_async.py, but eval-only: no trajectory steps, no image
encoding. Output JSONL is small and fast to produce.

Shared across workers (thread-safe):
  - OpenAIModel / LiteLLMModel  (stateless HTTP)
  - QdrantClient                 (thread-safe)
  - JinaV4Model                  (serialised via threading.Lock in LockedEmbedModel)

Per-sample (fresh every run):
  - SearchMemoryTool, ReflectionTool, SearchDatabaseTool, SearchAgent

Output fields per line:
  id, doc_name, question, ground_truth, model_answer, score, num_steps

Output paths default to $AGENTIC_MEMORY_LOG_DIR (default /lc3T/AgenticMemory/logs).

Usage:
    # Baseline (student base weights on SGLang):
    python sft/eval_async.py \\
        --dataset longdocurl \\
        --backend sglang \\
        --sglang-url http://localhost:30000 \\
        --model-name Qwen3-VL-8B-Instruct \\
        --embed-device cuda:7 \\
        --workers 4 \\
        --num-samples 100 \\
        --output /lc3T/AgenticMemory/logs/eval_baseline_train100.jsonl

    # After SFT — swap SGLang checkpoint, then:
    python sft/eval_async.py \\
        ... same args ... \\
        --output /lc3T/AgenticMemory/logs/eval_sft_train100.jsonl

    # Compare:
    python sft/compare_eval.py \\
        --teacher  /lc3T/AgenticMemory/logs/trajectories_overfit100.jsonl \\
        --baseline /lc3T/AgenticMemory/logs/eval_baseline_train100.jsonl \\
        --sft      /lc3T/AgenticMemory/logs/eval_sft_train100.jsonl
"""

import logging
import os
import sys
import json
import importlib
import threading
import yaml
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Path setup (same as collect_trajectories_async.py)
# ---------------------------------------------------------------------------
_AGENTIC_ROOT = Path(__file__).resolve().parent.parent
if str(_AGENTIC_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENTIC_ROOT))

_LOCAL_SRC = _AGENTIC_ROOT / "src"
if str(_LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(_LOCAL_SRC))

for _mod in list(sys.modules):
    if _mod == "smolagents" or _mod.startswith("smolagents."):
        _file = getattr(sys.modules[_mod], "__file__", "")
        if _file and not _file.startswith(str(_LOCAL_SRC)):
            sys.modules.pop(_mod, None)

from typing import Any, List, Dict
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor as _InnerThreadPoolExecutor
from contextvars import copy_context

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from rich.panel import Panel
from rich.text import Text

from smolagents import ToolCallingAgent, ToolOutput, OpenAIModel, LiteLLMModel, TransformersModel
from smolagents.memory import ActionStep, ToolCall
from smolagents.models import ChatMessage, REMOVE_PARAMETER
from smolagents.monitoring import LogLevel
from src.data_utils import PathManager
from search_models import JinaV4Model
from tools import (
    SearchDatabaseTool,
    GetSpecificPagesTool,
    SearchMemoryTool,
    ReflectionTool,
    SearchResults,
    SpecificPagesResults,
)
from utils import evaluate_response, filter_existing_results, log_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thread-safe embed model wrapper (identical to collect_trajectories_async.py)
# ---------------------------------------------------------------------------

class LockedEmbedModel:
    """Serialises JinaV4Model GPU calls across worker threads."""

    def __init__(self, model: JinaV4Model, lock: threading.Lock) -> None:
        self._model = model
        self._lock = lock

    def embed_text(self, *args, **kwargs):
        with self._lock:
            return self._model.embed_text(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._model, name)


# ---------------------------------------------------------------------------
# SearchAgent (identical to collect_trajectories_async.py)
# ---------------------------------------------------------------------------

class SearchAgent(ToolCallingAgent):
    """Overrides process_tool_calls to capture images from tool outputs."""

    def process_tool_calls(
        self, chat_message: ChatMessage, memory_step: ActionStep
    ) -> Generator[ToolCall | ToolOutput]:
        parallel_calls: dict[str, ToolCall] = {}
        assert chat_message.tool_calls is not None
        for chat_tool_call in chat_message.tool_calls:
            tool_call = ToolCall(
                name=chat_tool_call.function.name,
                arguments=chat_tool_call.function.arguments,
                id=chat_tool_call.id,
            )
            yield tool_call
            parallel_calls[tool_call.id] = tool_call

        def process_single_tool_call(tool_call: ToolCall) -> ToolOutput:
            tool_name = tool_call.name
            tool_arguments = tool_call.arguments or {}
            self.logger.log(
                Panel(Text(f"Calling tool: '{tool_name}' with arguments: {tool_arguments}")),
                level=LogLevel.INFO,
            )
            tool_call_result = self.execute_tool_call(tool_name, tool_arguments)
            if type(tool_call_result) in [SearchResults, SpecificPagesResults]:
                observation = tool_call_result.to_string()
            else:
                observation = str(tool_call_result).strip()
            self.logger.log(
                f"Observations: {observation.replace('[', '|')}",
                level=LogLevel.INFO,
            )
            return ToolOutput(
                id=tool_call.id,
                output=tool_call_result,
                is_final_answer=(tool_name == "final_answer"),
                observation=observation,
                tool_call=tool_call,
            )

        outputs = {}
        if len(parallel_calls) == 1:
            tool_call = list(parallel_calls.values())[0]
            tool_output = process_single_tool_call(tool_call)
            outputs[tool_output.id] = tool_output
            yield tool_output
        else:
            with _InnerThreadPoolExecutor(self.max_tool_threads) as executor:
                futures = []
                for tool_call in parallel_calls.values():
                    ctx = copy_context()
                    futures.append(executor.submit(ctx.run, process_single_tool_call, tool_call))
                for future in as_completed(futures):
                    tool_output = future.result()
                    outputs[tool_output.id] = tool_output
                    yield tool_output

        memory_step.tool_calls = [parallel_calls[k] for k in sorted(parallel_calls.keys())]
        memory_step.observations = memory_step.observations or ""
        for tool_output in [outputs[k] for k in sorted(outputs.keys())]:
            memory_step.observations += tool_output.observation + "\n"
        memory_step.observations = (
            memory_step.observations.rstrip("\n") if memory_step.observations else memory_step.observations
        )


# ---------------------------------------------------------------------------
# Per-sample worker
# ---------------------------------------------------------------------------

def process_sample(
    d: Dict,
    agent_model,
    eval_client: OpenAI,
    eval_model: str,
    qdrant_client: QdrantClient,
    locked_embed_model: LockedEmbedModel,
    args: argparse.Namespace,
    prompt_templates: dict,
    write_lock: threading.Lock,
) -> None:
    """Run one sample end-to-end. Write score + metadata to JSONL (no trajectory)."""
    question_id = d["id"]
    doc_name = d["doc_name"]
    question = d["question"]
    ground_truth = d["answer"]
    task = f"Based on Document <{doc_name}>, answer the following question: [{question}]."

    # Fresh per-sample tools (SearchMemoryTool is stateful)
    search_mem = SearchMemoryTool()
    search_mem.original_question = question  # set directly — no longer a tool parameter
    agent = SearchAgent(
        tools=[
            SearchDatabaseTool(
                client=qdrant_client,
                embed_model=locked_embed_model,
                collection_name=args.dataset,
                embed_model_name="jinav4_multivector",
                max_image_size=args.max_image_size,
            ),
            GetSpecificPagesTool(client=qdrant_client, collection_name=args.dataset,
                                 max_image_size=args.max_image_size),
            search_mem,
            ReflectionTool(search_memory=search_mem),
        ],
        model=agent_model,
        prompt_templates=prompt_templates,
        max_steps=15,
    )

    try:
        response = agent.run(task, return_full_result=True)
    except Exception as exc:
        print(f"[ERROR] question_id={question_id}: {exc}", file=sys.stderr)
        return

    model_answer = str(response.output)
    num_steps = len(response.dict()["steps"])

    score = -1
    try:
        eval_result = evaluate_response(
            eval_client, eval_model, model_answer, ground_truth, question,
            max_tokens=1024, temperature=0.0,
        )
        score = eval_result.get("score", -1)
    except Exception as exc:
        print(f"[WARN] eval failed for question_id={question_id}: {exc}", file=sys.stderr)

    record = {
        "id":           question_id,
        "doc_name":     doc_name,
        "question":     question,
        "ground_truth": ground_truth,
        "model_answer": model_answer,
        "score":        score,
        "num_steps":    num_steps,
    }

    with write_lock:
        with open(args.output, "a", encoding="utf-8") as f:
            json.dump(record, f)
            f.write("\n")

    print(f"[INFO] question_id={question_id} | steps={num_steps} | score={score} | answer={str(model_answer)[:80]}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel agent evaluation (eval-only, no trajectory storage).")
    parser.add_argument("--dataset",      type=str, default="longdocurl")
    parser.add_argument("--embed-device", type=str, default="cuda:7")
    parser.add_argument("--output",       type=str, default=log_path("eval.jsonl"))
    parser.add_argument("--num-samples",  type=int, default=None)
    parser.add_argument("--offset", type=int, default=0,
        help="Skip the first N questions in the dataset. Use to select held-out ranges (e.g. --offset 100 --num-samples 100 evaluates questions 100-199).")
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Concurrent agent workers. Each is a Python thread. SGLang batches their requests. Start at 4.",
    )
    parser.add_argument("--backend",    type=str, default="sglang", choices=["sglang", "api", "transformers"])
    parser.add_argument("--sglang-url", type=str, default="http://localhost:30000")
    parser.add_argument("--model-name", type=str, default="Qwen3-VL-8B-Instruct",
                        help="Model name SGLang was launched with (sglang backend)")
    parser.add_argument("--api-model",  type=str, default="openai/gemini-2.5-flash",
                        help="LiteLLM model ID (api backend)")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                        help="HF model ID or local checkpoint path (transformers backend)")
    parser.add_argument("--max-image-size", type=int, default=None,
        help="Resize page images so long edge ≤ this many pixels. Use 768 for 30B teacher.")
    parser.add_argument("--model-device", type=str, default="cuda:0",
                        help="Device for in-process model (transformers backend)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    load_dotenv()

    # ---- dataset ----
    jsonl_path = PathManager.get_dataset_jsonl(args.dataset)
    data: List[Dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    if args.offset:
        data = data[args.offset:]
    if args.num_samples is not None:
        data = data[: args.num_samples]
    data = filter_existing_results(data, args.output)

    print(f"[INFO] {len(data)} samples to evaluate with {args.workers} workers")

    # ---- agent model ----
    # sglang / api: stateless HTTP, thread-safe, shared across workers
    # transformers: in-process weights on one GPU. Not safe for parallel generate()
    #   calls, so we force workers=1 — the ThreadPoolExecutor degrades to sequential.
    if args.backend == "sglang":
        agent_model = OpenAIModel(
            model_id=args.model_name,
            api_base=f"{args.sglang_url}/v1",
            api_key="EMPTY",
            tools=REMOVE_PARAMETER,
            tool_choice=REMOVE_PARAMETER,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    elif args.backend == "transformers":
        if args.workers != 1:
            print(f"[WARN] transformers backend runs in-process on one GPU; forcing workers=1 (was {args.workers})")
            args.workers = 1
        agent_model = TransformersModel(
            model_id=args.model_path,
            device_map=args.model_device,
            torch_dtype="bfloat16",
            trust_remote_code=True,
            max_new_tokens=2048,
        )
    else:
        agent_model = LiteLLMModel(
            model_id=args.api_model,
            api_base=os.getenv("API_BASE"),
            api_key=os.getenv("API_KEY"),
        )

    # ---- eval client — always uses GLM gateway (EVAL_API_BASE), not SGLang ----
    eval_api_base = os.getenv("EVAL_API_BASE", os.getenv("API_BASE"))
    eval_api_key  = os.getenv("EVAL_API_KEY",  os.getenv("API_KEY", "EMPTY"))
    eval_model    = os.getenv("EVAL_MODEL", "gpt-4o-2024-11-20")
    eval_client   = OpenAI(api_key=eval_api_key, base_url=eval_api_base)

    # ---- shared Qdrant + embed model ----
    os.environ["NO_PROXY"] = "localhost"
    qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"))
    raw_embed     = JinaV4Model(device=args.embed_device, multivector=True)
    locked_embed  = LockedEmbedModel(raw_embed, threading.Lock())

    # ---- prompt templates ----
    prompt_templates = yaml.safe_load(
        importlib.resources.files("smolagents.prompts")
        .joinpath("deepresearch_agent.yaml")
        .read_text()
    )

    os.makedirs(Path(args.output).parent, exist_ok=True)
    write_lock = threading.Lock()

    # ---- parallel execution ----
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_sample,
                d, agent_model, eval_client, eval_model,
                qdrant_client, locked_embed, args, prompt_templates, write_lock,
            ): d["id"]
            for d in data
        }
        for future in as_completed(futures):
            question_id = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"[ERROR] question_id={question_id} raised: {exc}", file=sys.stderr)

    # ---- final summary ----
    records = []
    with open(args.output, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    valid_scores = [r["score"] for r in records if r.get("score", -1) != -1]
    acc = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    avg_steps = sum(r["num_steps"] for r in records) / len(records) if records else 0.0
    print(f"\n[DONE] {len(records)} samples | accuracy={acc:.1%} | avg_steps={avg_steps:.1f}")
    print(f"       output: {args.output}")


if __name__ == "__main__":
    main()
