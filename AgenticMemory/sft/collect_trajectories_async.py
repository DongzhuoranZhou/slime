"""
collect_trajectories_async.py

Async (parallel) version of collect_trajectories.py.

Runs --workers samples concurrently via ThreadPoolExecutor so that SGLang's
continuous batching is fully utilised — multiple requests are in-flight at
once instead of the server idling between sequential steps.

Shared across workers (thread-safe):
  - OpenAIModel / LiteLLMModel  (stateless HTTP)
  - QdrantClient                 (thread-safe)
  - JinaV4Model                  (serialised via threading.Lock in LockedEmbedModel)

Per-sample (fresh every run):
  - SearchMemoryTool, ReflectionTool, SearchDatabaseTool, SearchAgent

Usage:
    python sft/collect_trajectories_async.py \\
        --dataset longdocurl \\
        --backend sglang \\
        --sglang-url http://localhost:30000 \\
        --model-name Qwen3-VL-30B-A3B-Instruct \\
        --embed-device cuda:7 \\
        --workers 4 \\
        --num-samples 20 \\
        --output /lc3T/AgenticMemory/logs/trajectories_async.jsonl   # override via $AGENTIC_MEMORY_LOG_DIR
"""

import logging
import os
import sys
import json
import base64
import importlib
import io
import threading
import yaml
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Path setup (identical to collect_trajectories.py)
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

from PIL import Image
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from rich.panel import Panel
from rich.text import Text

from smolagents import ToolCallingAgent, ToolOutput, OpenAIModel, LiteLLMModel
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
from utils import evaluate_response, filter_existing_results, log_path, SYSTEM_PROMPT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thread-safe embed model wrapper
# ---------------------------------------------------------------------------

class LockedEmbedModel:
    """Wraps JinaV4Model and serialises GPU calls with a threading.Lock.

    Multiple worker threads share one JinaV4Model instance. Without locking,
    concurrent embed_text() calls would corrupt GPU state. The lock makes
    embedding serial while everything else (SGLang inference, Qdrant queries,
    result processing) runs in parallel across workers.
    """

    def __init__(self, model: JinaV4Model, lock: threading.Lock) -> None:
        self._model = model
        self._lock = lock

    def embed_text(self, *args, **kwargs):
        with self._lock:
            return self._model.embed_text(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._model, name)


# ---------------------------------------------------------------------------
# SearchAgent (identical to collect_trajectories.py)
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
            tool_call_result_type = type(tool_call_result)
            if tool_call_result_type in [SearchResults, SpecificPagesResults]:
                observation = tool_call_result.to_string()
                observations_images = tool_call_result.to_raw()
                print(f"sucessfully loaded {len(observations_images)} images.")
                memory_step.observations_images = observations_images
            else:
                observation = str(tool_call_result).strip()
            self.logger.log(
                f"Observations: {observation.replace('[', '|')}",
                level=LogLevel.INFO,
            )
            is_final_answer = tool_name == "final_answer"
            return ToolOutput(
                id=tool_call.id,
                output=tool_call_result,
                is_final_answer=is_final_answer,
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
# Helpers (identical to collect_trajectories.py)
# ---------------------------------------------------------------------------

def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _safe_json(obj: Any) -> str:
    if isinstance(obj, bytes):
        return f"<bytes:{len(obj)}>"
    raise TypeError(type(obj))


def _tool_to_schema(tool) -> dict:
    properties = {}
    required = []
    for param, info in tool.inputs.items():
        prop = {"type": info["type"], "description": info.get("description", "")}
        if info["type"] == "array" and "items" in info:
            prop["items"] = info["items"]
        properties[param] = prop
        if not info.get("nullable", False):
            required.append(param)
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def extract_steps(raw_steps: List[Dict]) -> List[Dict]:
    IMAGE_TOOLS = {"search_database", "get_specific_pages"}
    structured = []
    for i, step in enumerate(raw_steps):
        tool_calls_raw = step.get("tool_calls", []) or []
        tool_calls = []
        for tc in tool_calls_raw:
            func = tc.get("function") or {}
            tc_name = func.get("name", "")
            arguments = func.get("arguments", {})
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments)
            tool_calls.append({"id": tc.get("id", ""), "name": tc_name, "arguments": arguments})

        is_final_answer_step = any(tc["name"] == "final_answer" for tc in tool_calls)
        is_last_step = i + 1 >= len(raw_steps)
        if is_last_step and not is_final_answer_step:
            logger.warning("Last step does not call final_answer — agent likely hit max_steps.")

        obs_images: List[str] = step.get("observations_images") or []
        obs_text: str = step.get("observations") or ""

        tool_responses = []
        images_assigned = False
        for tc in tool_calls_raw:
            func = tc.get("function") or {}
            tc_name = func.get("name", "")
            if tc_name == "final_answer":
                continue
            if tc_name in IMAGE_TOOLS and not images_assigned:
                imgs_b64 = obs_images
                images_assigned = True
            else:
                imgs_b64 = []
            tool_responses.append(
                {"tool_call_id": tc.get("id", ""), "image_paths": imgs_b64, "content": obs_text}
            )

        structured.append({"tool_calls": tool_calls, "tool_responses": tool_responses})

    return structured


# ---------------------------------------------------------------------------
# Per-sample worker
# ---------------------------------------------------------------------------

def process_sample(
    d: Dict,
    agent_model,
    qdrant_client: QdrantClient,
    locked_embed_model: LockedEmbedModel,
    args: argparse.Namespace,
    tools_json: str,
    prompt_templates: dict,
    write_lock: threading.Lock,
) -> None:
    """Run one sample end-to-end. Called from a thread pool worker."""
    question_id = d["id"]
    doc_name = d["doc_name"]
    question = d["question"]
    ground_truth = d["answer"]
    task = f"Based on Document <{doc_name}>, answer the following question: [{question}]."

    # Fresh per-sample tools (SearchMemoryTool and ReflectionTool are stateful)
    search_mem_tool = SearchMemoryTool()
    search_mem_tool.original_question = question  # set directly — no longer a tool parameter
    reflection_tool = ReflectionTool(search_memory=search_mem_tool)
    search_tool = SearchDatabaseTool(
        client=qdrant_client,
        embed_model=locked_embed_model,
        collection_name=args.dataset,
        embed_model_name="jinav4_multivector",
        max_image_size=args.max_image_size,
    )
    get_pages_tool = GetSpecificPagesTool(
        client=qdrant_client,
        collection_name=args.dataset,
        max_image_size=args.max_image_size,
    )

    agent = SearchAgent(
        tools=[search_tool, get_pages_tool, search_mem_tool, reflection_tool],
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
    response_dict = response.dict()
    num_steps = len(response_dict["steps"])

    # LLM eval
    score = -1
    try:
        from openai import OpenAI

        if args.backend == "api":
            eval_api_base = os.getenv("API_BASE")
            eval_api_key = os.getenv("API_KEY", "EMPTY")
            eval_model = os.getenv("EVAL_MODEL", args.api_model.split("/")[-1])
        else:
            eval_api_base = os.getenv("API_BASE", f"{args.sglang_url}/v1")
            eval_api_key = os.getenv("API_KEY", "EMPTY")
            eval_model = os.getenv("EVAL_MODEL", "gpt-4o-2024-11-20")

        eval_client = OpenAI(api_key=eval_api_key, base_url=eval_api_base)
        eval_result = evaluate_response(
            eval_client, eval_model, model_answer, ground_truth, question,
            max_tokens=1024, temperature=0.0,
        )
        score = eval_result.get("score", -1)
    except Exception as exc:
        print(f"[WARN] eval failed for question_id={question_id}: {exc}", file=sys.stderr)

    steps = extract_steps(response_dict["steps"])
    trajectory = {
        "id": question_id,
        "question_id": question_id,
        "doc_name": doc_name,
        "question": question,
        "ground_truth": ground_truth,
        "model_answer": model_answer,
        "score": score,
        "num_steps": num_steps,
        "system_prompt": SYSTEM_PROMPT,
        "task": task,
        "tools_json": tools_json,
        "steps": steps,
    }

    try:
        with write_lock:
            with open(args.output, "a", encoding="utf-8") as out_f:
                json.dump(trajectory, out_f, default=_safe_json)
                out_f.write("\n")
    except Exception as exc:
        print(f"[ERROR] failed to write trajectory for question_id={question_id}: {exc}", file=sys.stderr)
        return

    print(
        f"[INFO] question_id={question_id} | steps={num_steps} | score={score} | "
        f"answer={str(model_answer)[:80]}"
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect agent trajectories (parallel workers).")
    parser.add_argument("--dataset", type=str, default="longdocurl")
    parser.add_argument("--embed-device", type=str, default="cuda:7", help="Device for JinaV4 embedding model")
    parser.add_argument("--output", type=str, default=log_path("trajectories_async.jsonl"))
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0,
        help="Skip the first N questions. Use with --num-samples to select a range (e.g. --offset 100 --num-samples 100 collects questions 100-199).")
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of concurrent agent workers. Each worker runs one sample at a time. "
             "SGLang batches their requests together. Start with 4 and increase if GPU-Util stays low.",
    )
    parser.add_argument("--backend", type=str, default="sglang", choices=["sglang", "api"])
    parser.add_argument("--sglang-url", type=str, default="http://localhost:30000")
    parser.add_argument("--model-name", type=str, default="Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--api-model", type=str, default="openai/gemini-2.5-flash")
    parser.add_argument(
        "--max-image-size", type=int, default=None,
        help="Resize document page images so the long edge is at most this many pixels before "
             "sending to the VLM. Reduces vision token count and prevents GPU OOM on long "
             "multi-search trajectories. Recommended: 768 for 30B on 4×80GB, None (no resize) "
             "for smaller models. Example: --max-image-size 768",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ---- load dataset ----
    jsonl_path = PathManager.get_dataset_jsonl(args.dataset)
    data: List[Dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    if args.offset:
        data = data[args.offset:]
    if args.num_samples is not None:
        data = data[: args.num_samples]
    data = filter_existing_results(data, args.output)

    print(f"[INFO] {len(data)} samples to process with {args.workers} workers")

    # ---- shared resources ----
    load_dotenv()

    if args.backend == "sglang":
        agent_model = OpenAIModel(
            model_id=args.model_name,
            api_base=f"{args.sglang_url}/v1",
            api_key="EMPTY",
            tools=REMOVE_PARAMETER,
            tool_choice=REMOVE_PARAMETER,
        )
    else:
        agent_model = LiteLLMModel(
            model_id=args.api_model,
            api_base=os.getenv("API_BASE"),
            api_key=os.getenv("API_KEY"),
        )

    os.environ["NO_PROXY"] = "localhost"
    qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"))

    raw_embed_model = JinaV4Model(device=args.embed_device, multivector=True)
    embed_lock = threading.Lock()
    locked_embed_model = LockedEmbedModel(raw_embed_model, embed_lock)

    # ---- prompt templates (loaded once) ----
    prompt_yaml = "deepresearch_agent.yaml"
    prompt_templates = yaml.safe_load(
        importlib.resources.files("smolagents.prompts").joinpath(prompt_yaml).read_text()
    )

    # ---- tools json (serialised once using a throwaway tool set) ----
    _tmp_search_mem = SearchMemoryTool()
    _tmp_tools = [
        SearchDatabaseTool(client=qdrant_client, embed_model=locked_embed_model,
                           collection_name=args.dataset, embed_model_name="jinav4_multivector"),
        GetSpecificPagesTool(client=qdrant_client, collection_name=args.dataset),
        _tmp_search_mem,
        ReflectionTool(search_memory=_tmp_search_mem),
    ]
    tools_json = json.dumps([_tool_to_schema(t) for t in _tmp_tools], default=_safe_json)

    # ---- output dir ----
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    write_lock = threading.Lock()

    # ---- parallel execution ----
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_sample,
                d, agent_model, qdrant_client, locked_embed_model,
                args, tools_json, prompt_templates, write_lock,
            ): d["id"]
            for d in data
        }
        for future in as_completed(futures):
            question_id = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"[ERROR] question_id={question_id} raised unhandled exception: {exc}", file=sys.stderr)

    print(f"[DONE] trajectories written to {args.output}")


if __name__ == "__main__":
    main()
