#%%
from __future__ import annotations

from src.data_utils import load_jsonl, PathManager
from dotenv import load_dotenv
from smolagents import LiteLLMModel
from smolagents import TransformersModel

from docagent_runner import APIDocAgentConfig, BaseDocAgentRunner, OpenSourceDocAgentConfig
from src.docagent_tools import DocumentPreviewTool, DocumentReaderTool, PageLoaderTool

class APIDocAgentRunner(BaseDocAgentRunner):
    def build_model(self) -> LiteLLMModel:
        return LiteLLMModel(
            model_id=self.config.model_id,
            api_base=self.config.api_base,
            api_key=self.config.api_key,
            do_sample=self.config.do_sample,
        )

class OpenSourceDocAgentRunner(BaseDocAgentRunner):
    def build_model(self) -> TransformersModel:
        return TransformersModel(
            model_id=self.config.model_id,
            device_map=self.config.device_map,
            torch_dtype=self.config.torch_dtype,
            trust_remote_code=self.config.trust_remote_code,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=self.config.do_sample,
        )
    
if __name__ == "__main__":
    load_dotenv()
    
    # config = OpenSourceDocAgentConfig.from_yaml("configs/opensource_config.yaml")

    # tools = [
    #     DocumentPreviewTool(),
    #     DocumentReaderTool(),
    #     PageLoaderTool()
    # ]

    # runner = OpenSourceDocAgentRunner(config, tools)
    # agent = runner.build_agent()

    config = APIDocAgentConfig.from_yaml("configs/api_config.yaml")

    tools = [
        DocumentPreviewTool(),
        DocumentReaderTool(),
        PageLoaderTool()
    ]

    runner = APIDocAgentRunner(config, tools)
    agent = runner.build_agent()

    # evaluation
    dataset = "mmlongdoc"
    data = load_jsonl(PathManager.get_retrieval_jsonl("jina_multi", dataset))
    runner.run([data[0]], agent)

# %%
