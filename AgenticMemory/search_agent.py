#%%
"""
Search Agent implementation using smolagent framework.

This module implements a custom agent that can search a Qdrant vector database
using embedding models and reflect on search results to decide next actions.
"""

import os
import json
import yaml
import importlib
from openai import OpenAI
from typing import Optional, Any, Dict, List
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from rich.panel import Panel
from rich.text import Text

from smolagents import ToolCallingAgent, ToolOutput, LiteLLMModel
from smolagents.memory import ActionStep, ToolCall
from smolagents.models import ChatMessage

from utils import log_path
from smolagents.monitoring import LogLevel
from src.data_utils import PathManager
from search_models import JinaV4Model, ColPaliModel

from tools import (
    SearchDatabaseTool,
    GetSpecificPagesTool,
    SearchMemoryTool,
    ReflectionTool,
    SearchResults,
    SpecificPagesResults
)
from utils import evaluate_response, filter_existing_results
import argparse

class SearchAgent(ToolCallingAgent):
    """
    overwrite process_tool_calls to observe images from tool calling outputs
    """
    def process_tool_calls(
        self, chat_message: ChatMessage, memory_step: ActionStep
    ) -> Generator[ToolCall | ToolOutput]:
        """Process tool calls from the model output and update agent memory.
        Args:
            chat_message (`ChatMessage`): Chat message containing tool calls from the model.
            memory_step (`ActionStep)`: Memory ActionStep to update with results.
        Yields:
            `ToolCall | ToolOutput`: The tool call or tool output.
        """
        parallel_calls: dict[str, ToolCall] = {}
        assert chat_message.tool_calls is not None
        for chat_tool_call in chat_message.tool_calls:
            tool_call = ToolCall(
                name=chat_tool_call.function.name, arguments=chat_tool_call.function.arguments, id=chat_tool_call.id
            )
            yield tool_call
            parallel_calls[tool_call.id] = tool_call

        # Helper function to process a single tool call
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
                
                print(f'sucessfully loaded {len(observations_images)} images.')
                observation = observation
                memory_step.observations_images = observations_images
            else:
                observation = str(tool_call_result).strip()
            self.logger.log(
                f"Observations: {observation.replace('[', '|')}",  # escape potential rich-tag-like components
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

        # Process tool calls in parallel
        outputs = {}
        if len(parallel_calls) == 1:
            # If there's only one call, process it directly
            tool_call = list(parallel_calls.values())[0]
            tool_output = process_single_tool_call(tool_call)
            outputs[tool_output.id] = tool_output
            yield tool_output
        else:
            # If multiple tool calls, process them in parallel
            with ThreadPoolExecutor(self.max_tool_threads) as executor:
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

# =========== User setup ===============
parser = argparse.ArgumentParser(description="Search Agent Configuration")
parser.add_argument("--dataset", type=str, default="mmlongdoc", help="Dataset name")
parser.add_argument("--agent_model_name", type=str, default="openai/gemini-2.5-pro", help="Agent model name")
parser.add_argument("--embed_device", type=str, default="cuda:2", help="Embedding device")

args = parser.parse_args()

dataset = args.dataset
agent_model_name = args.agent_model_name
embed_device = args.embed_device

num_samples = None
sample_indices = None
prompt_yaml = "deepresearch_agent.yaml"
agent_max_steps = 15
embed_model_name = "jinav4_multivector"


results_jsonl_path = log_path(f"{dataset}_{agent_model_name.split('/')[1]}.jsonl")
# results_jsonl_path = log_path("debug.jsonl")

eval_model_name = "gpt-4o-2024-11-20"

# =========== code ===============
# load dataset
jsonl_path = PathManager.get_dataset_jsonl(dataset)
data = []
with open(jsonl_path, 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

data = filter_existing_results(data, results_jsonl_path)

# check sample_indices and num_samples are mutually exclusive
assert not (sample_indices is not None and num_samples is not None), "sample_indices and num_samples are mutually exclusive"

if sample_indices is not None:
    data = [data[i] for i in sample_indices]
elif num_samples is not None:
    data = data[:num_samples]

# build model
load_dotenv()
agent_model = LiteLLMModel(
    model_id=agent_model_name,
    api_base=os.getenv("API_BASE"),
    api_key=os.getenv("API_KEY"),
    do_sample=False,
)

eval_client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("API_BASE")
)

## setup vector database
# Setup Qdrant client
os.environ["NO_PROXY"] = "localhost"
qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"))

if embed_model_name == "jinav4_dense":
    embed_model = JinaV4Model(device=embed_device, multivector=False)
elif embed_model_name == "jinav4_multivector":
    embed_model = JinaV4Model(device=embed_device, multivector=True)
elif embed_model_name == "colpali_multivector":
    embed_model = ColPaliModel(device=embed_device)
else:
    raise ValueError(f"Unknown embedding model: {embed_model_name}")

## build agent
# load prompt templates
prompt_templates = yaml.safe_load(
    importlib.resources.files("smolagents.prompts").joinpath(prompt_yaml).read_text()
)

# initialize tools
search_tool = SearchDatabaseTool(
            client=qdrant_client,
            embed_model=embed_model,
            collection_name=dataset,
            embed_model_name=embed_model_name
        )
get_pages_tool = GetSpecificPagesTool(
            client=qdrant_client,
            collection_name=dataset
        )
search_memory_tool = SearchMemoryTool()
reflection_tool = ReflectionTool(search_memory=search_memory_tool)

# initialize agent
agent = SearchAgent(
    tools=[
        search_tool,
        get_pages_tool,
        search_memory_tool,
        reflection_tool
    ],
    model=agent_model,
    prompt_templates=prompt_templates,
    max_steps=agent_max_steps
)

# =============== main loop to process each document ===============
for d in data:
    # Reset search memory for each new question
    search_memory_tool.reset()

    doc_id = d['id']
    doc_name = d['doc_name']
    question = d['question']
    answer = d['answer']

    # Seed the original question directly on the tool — it is NOT a schema input
    # (see LESSONS_LEARNED §5/§7; async runners in sft/ do the same).
    search_memory_tool.original_question = question

    query = f"""
    Based on Document {doc_name},
    answer the following question: {question}.
    """

    # try:
    response = agent.run(query, return_full_result=True)

    eval_result = evaluate_response(eval_client, 
                    eval_model_name, 
                    response.output, 
                    answer, 
                    question, 
                    max_tokens=1024, 
                    temperature=0.0)

    print(f"================= Document ID: {doc_id} =================")
    print(f"Question: {question}")
    print(f"Final Answer: {response.output}")
    print(f"Ground Truth: {answer}")
    print("--------------------------------")
    print(f"LLM Evaluation: {eval_result}")
    print(f"Input Tokens Used: {response.token_usage.input_tokens}")
    print(f"Output Tokens Used: {response.token_usage.output_tokens}")
    print(f"Duration (s): {response.timing.end_time - response.timing.start_time:.1f}\n")

    results = response.dict()
    results["id"] = doc_id
    results["doc_name"] = doc_name
    results["question"] = question
    results["score"] = eval_result["score"]
        
    # except Exception as e:
    #     print(f"Error processing document ID {doc_id}: {e}")
    #     results = {
    #         "id": doc_id,
    #         "doc_name": doc_name,
    #         "question": question,
    #         "error": str(e)
    #     }

    def safe_serialize(obj: Any) -> str:
        if isinstance(obj, bytes):
            return f"<bytes_object_length_{len(obj)}>"
        raise TypeError(f"Type {type(obj)} is not JSON serializable")
    
    with open(results_jsonl_path, "a", encoding="utf-8") as f:
        json.dump(results, f, default=safe_serialize)
        f.write("\n")


# try:
#     log_agent_response(log_path("search_agent_debug_log.log"), d, response)
# except Exception as e:
#     # save the entire response object for debugging
#     with open(log_path("search_agent_debug_log_full_response.json"), "w") as f:
#         json.dump(response.to_dict(), f, indent=4)

# %%
