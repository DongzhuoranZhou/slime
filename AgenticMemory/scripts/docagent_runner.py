from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import yaml
from pathlib import Path
from typing import Any

from smolagents import DocAgent

from src.data_utils import load_jsonl, get_subset_results, PathManager
from src.docagent_tools import DocumentPreviewTool, DocumentReaderTool, PageLoaderTool
from src.evaluation_utils import evaluation


DEFAULT_SYSTEM_PROMPT = """
**ROLE & CONTEXT:**
You are an advanced Visual Document Assistant designed for "Progressive Disclosure" analysis.
**Input Context:** You will be given a list of document image paths (retrieved by a search engine) and a specific user question.
**Mission:** Your goal is to answer the question by **progressively loading relevant information**. You must balance **efficiency** (minimizing token usage) with **deep reasoning** (verifying sufficiency) to ensure the final answer is accurate.

**CORE OPERATING PROTOCOL:**

1.  **PHASE 1: SCOUT (Mandatory First Step)**
    * **Action:** ALWAYS start by calling `preview_document` on the provided document paths.
    * **Goal:** Understand the document structure. Look for:
        * **Page Numbers:** Where is the relevant content?
        * **Content Type:** Is the page Text-Heavy (paragraphs) or Visual-Heavy (tables, charts, slides)?

2.  **PHASE 2: STRATEGIZE (Decision Point)**
    * Analyze the `preview_document` output to decide your next move.
    * **Decision A: Text Retrieval**
        * *Condition:* The user asks about general content, summaries, terms, or specific clauses, AND the preview shows "Paragraphs" or "Titles" on the relevant page.
        * *Tool:* Call `read_document_text` with `page_indices` set to the specific page(s).
    * **Decision B: Visual Inspection**
        * *Condition:* The user asks about data trends, specific figures, layout, or information likely contained in a Chart, or Slide.
        * *Tool:* Call `load_specific_pages` with `page_indices` set to the specific page(s).

3.  **PHASE 3: VERIFICATION & SELF-CORRECTION (Critical)**
* **Assess Sufficiency:** Before answering, analyze the tool output. Ask yourself: *"Is this evidence explicit and unambiguous enough to fully answer the user's request?"*
* **Reasoning Loop (If information is missing/unclear):**
    * **Identify the Deficit:** Is the current information sufficient? Did I look at the wrong page?
    * **Switch Modality:** If the current tool failed to capture the data, switch to the `load_specific_pages` tool for that same page.
    * **Follow Leads:** If the content refers to another section (e.g., "See Annex A"), use the Preview map to find and load that new section.
* **Final Answer:** Only output the final answer when you have verified the facts.

**CRITICAL RULES:**
* **NEVER** call `read_document_text` or `load_specific_pages` without first calling `preview_document`.
* **NEVER** load the entire document if the user's question is specific to a section (e.g., "What is the revenue on page 5?"). Use the `page_indices` argument strictly.
* **ALWAYS** cite the page number where you found the information.
"""

QUESTION_INSTRUCTIONS = """
You are given a document with text and images, and a question.
Answer the question as concisely as you can, using a single phrase or sentence if possible.
The final output should look like this:
Action:
{
    "name": "final_answer",
    "arguments": {"answer": "[answer]."}
}
If the question cannot be answered based on the information in the article,
Action:
{
    "name": "final_answer",
    "arguments": {"answer": "Not answerable."}
}
"""

@dataclass(frozen=True)
class BaseDocAgentConfig:
    @classmethod
    def from_yaml(cls, path: str | Path) -> "BaseDocAgentConfig":
        raise NotImplementedError("Subclasses must implement from_yaml")


@dataclass(frozen=True)
class OpenSourceDocAgentConfig(BaseDocAgentConfig):
    model_id: str = "Qwen/Qwen2-VL-7B-Instruct"
    device_map: str | None = None
    torch_dtype: str | None = None
    trust_remote_code: bool = True
    max_new_tokens: int = 4096
    do_sample: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "OpenSourceDocAgentConfig":
        config_path = Path(path)
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(config, dict):
            raise ValueError(f"YAML config must be a mapping, got {type(config)}")

        return cls(
            model_id=config.get("model_id", cls.model_id),
            device_map=config.get("device_map", cls.device_map),
            torch_dtype=config.get("torch_dtype", cls.torch_dtype),
            trust_remote_code=config.get("trust_remote_code", cls.trust_remote_code),
            max_new_tokens=config.get("max_new_tokens", cls.max_new_tokens),
            do_sample=config.get("do_sample", cls.do_sample)
        )


@dataclass(frozen=True)
class APIDocAgentConfig(BaseDocAgentConfig):
    model_id: str = "openai/gemini-3-pro-preview"
    api_base: str | None = os.getenv("API_BASE")
    api_key: str | None = os.getenv("API_KEY")
    do_sample: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "APIDocAgentConfig":
        config_path = Path(path)
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(config, dict):
            raise ValueError(f"YAML config must be a mapping, got {type(config)}")

        return cls(
            model_id=config.get("model_id", cls.model_id),
            api_base=config.get("api_base", cls.api_base),
            api_key=config.get("api_key", cls.api_key),
            do_sample=config.get("do_sample", cls.do_sample),
        )
    

class BaseDocAgentRunner:
    def __init__(
        self,
        config: BaseDocAgentConfig,
        tools: list,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        question_instructions: str = QUESTION_INSTRUCTIONS,
    ) -> None:
        self.config = config
        self.tools = tools
        self.system_prompt = system_prompt
        self.question_instructions = question_instructions

    @staticmethod
    def safe_serialize(obj: Any) -> str:
        if isinstance(obj, bytes):
            return f"<bytes_object_length_{len(obj)}>"
        raise TypeError(f"Type {type(obj)} is not JSON serializable")

    def build_model(self) -> Any:
        raise NotImplementedError

    def build_agent(self) -> DocAgent:
        return DocAgent(
            tools=self.tools,
            model=self.build_model(),
            instructions=self.system_prompt,
        )

    def _build_query(self, doc_paths: list[str], question: str) -> str:
        query = f"""
        Based on the document provided in following paths:
        {doc_paths},
        answer the question: {question}.
        """
        return self.question_instructions + "\n" + query

    # def single_run(self, 
    #                agent: DocAgent, 
    #                index: int, 
    #                top_k: int | None = None, 
    #                dataset: str | None = None
    # ) -> dict:
    #     dataset = dataset or self.config.dataset
    #     top_k = top_k or self.config.top_k
    #     data = load_jsonl(PathManager.get_retrieval_jsonl("jina_multi", dataset))
    #     current_data = data[index]

    #     doc_paths = PathManager.get_retrieval_paths(current_data, top_k)
    #     encode_doc_paths = [str(PathManager.encode_doc_path(p)) for p in doc_paths]

    #     query = self._build_query(encode_doc_paths, current_data["question"])
    #     response = agent.run(query, return_full_result=True)

    #     results = response.dict()
    #     results["acc"] = evaluation(response.output, current_data)
    #     results["data"] = current_data

    #     with open(self.config.debug_pickle_path, "wb") as f:
    #         import pickle

    #         pickle.dump(results, f)

    #     print(f"Final Answer: {response.output}")
    #     print(f"Ground Truth: {current_data['answer']}")
    #     print(f"Input Tokens Used: {response.token_usage.input_tokens}")
    #     print(f"Output Tokens Used: {response.token_usage.output_tokens}")
    #     print(f"Duration (s): {response.timing.end_time - response.timing.start_time:.1f}")
    #     print(f"Accuracy: {results['acc']}")

    #     return results

    def run(
        self,
        data: list,
        agent: DocAgent,
        top_k: int = 10,
        results_jsonl_path: str = "results/test.jsonl",
    ) -> None:

        result_list = []
        for current_data in data:
            doc_paths = PathManager.get_retrieval_paths(current_data, top_k)
            encode_doc_paths = [str(PathManager.encode_doc_path(p)) for p in doc_paths]

            query = self._build_query(encode_doc_paths, current_data["question"])
            response = agent.run(query, return_full_result=True)

            results = response.dict()
            results["acc"] = evaluation(response.output, current_data)
            results["data"] = current_data

            result_list.append(results)

            with open(results_jsonl_path, "a", encoding="utf-8") as f:
                json.dump(results, f, default=self.safe_serialize)
                f.write("\n")

            print(f"Final Answer: {response.output}")
            print(f"Ground Truth: {current_data['answer']}")
            print(f"Input Tokens Used: {response.token_usage.input_tokens}")
            print(f"Output Tokens Used: {response.token_usage.output_tokens}")
            print(f"Duration (s): {response.timing.end_time - response.timing.start_time:.1f}")
            print(f"Accuracy: {results['acc']}")

        accs = [x["acc"] for x in result_list]
        input_tokens = [x["token_usage"]["input_tokens"] for x in result_list]
        output_tokens = [x["token_usage"]["output_tokens"] for x in result_list]
        durations = [x["timing"]["end_time"] - x["timing"]["start_time"] for x in result_list]
        print(f"Average Input Tokens Used: {sum(input_tokens)/len(input_tokens)}")
        print(f"Average Output Tokens Used: {sum(output_tokens)/len(output_tokens)}")
        print(f"Average Duration (s): {sum(durations)/len(durations):.1f}")
        print(f"Average Accuracy: {sum(accs)/len(accs) if accs else 0}")

    # def subset_run(
    #     self,
    #     agent: DocAgent,
    #     subset_size: int,
    #     top_k: int | None = None,
    #     dataset: str | None = None,
    # ) -> None:
    #     dataset = dataset or self.config.dataset
    #     top_k = top_k or self.config.top_k

    #     results = get_subset_results(dataset, subset_size=subset_size)
    #     eval_result_path = Path("results") / f"subset_results_{dataset}_size{subset_size}_top{top_k}.pkl"

    #     eval_results = []
    #     for result in results:
    #         doc_paths = PathManager.get_retrieval_paths(result, top_k)
    #         encode_doc_paths = [str(PathManager.encode_doc_path(p)) for p in doc_paths]

    #         query = self._build_query(encode_doc_paths, result["question"])
    #         response = agent.run(query, return_full_result=True)

    #         eval_result = response.dict()
    #         eval_result["acc"] = evaluation(response.output, result)
    #         eval_result["data"] = result

    #         eval_results.append(eval_result)

    #         with open(eval_result_path, "wb") as f:
    #             import pickle

    #             pickle.dump(eval_results, f)

    #         accs = [x["acc"] for x in eval_results]
    #         input_tokens = [x["token_usage"]["input_tokens"] for x in eval_results]
    #         output_tokens = [x["token_usage"]["output_tokens"] for x in eval_results]
    #         durations = [x["timing"]["end_time"] - x["timing"]["start_time"] for x in eval_results]

    #         print(f"Final Answer: {response.output}")
    #         print(f"Ground Truth: {result['answer']}")
    #         print(f"Input Tokens Used: {sum(input_tokens)}")
    #         print(f"Output Tokens Used: {sum(output_tokens)}")
    #         print(f"Duration (s): {sum(durations):.1f}")
    #         print(f"Accuracy: {sum(accs)/len(accs) if accs else 0}")
