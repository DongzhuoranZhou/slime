#%%
# load jsonl
import json
from typing import List, Dict, Any
from src.data_utils import PathManager
from utils import log_path

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

dataset = "mmlongdoc"

jsonl_path = PathManager.get_dataset_jsonl(dataset)
data = []
with open(jsonl_path, 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

results = load_jsonl(log_path(f"{dataset}_gemini-2.5-pro.jsonl"))
scores = [x["score"] for x in results]
input_tokens = [x["token_usage"]["input_tokens"] for x in results]
output_tokens = [x["token_usage"]["output_tokens"] for x in results]
step_nums = [len(x["steps"]) for x in results]
print(f"{len(results)} / 961")
print(f"Average score: {sum(scores) / len(scores)}")
print(f"Average input tokens: {sum(input_tokens) / len(input_tokens)}")
print(f"Average output tokens: {sum(output_tokens) / len(output_tokens)}")
print(f"Average step nums: {sum(step_nums) / len(step_nums)}")
# %%
results = load_jsonl(log_path(f"{dataset}_gemini-3-pro-preview.jsonl"))
scores = [x["score"] for x in results]
input_tokens = [x["token_usage"]["input_tokens"] for x in results]
output_tokens = [x["token_usage"]["output_tokens"] for x in results]
step_nums = [len(x["steps"]) for x in results]
print(f"{len(results)} / 961")
print(f"Average score: {sum(scores) / len(scores)}")
print(f"Average input tokens: {sum(input_tokens) / len(input_tokens)}")
print(f"Average output tokens: {sum(output_tokens) / len(output_tokens)}")
print(f"Average step nums: {sum(step_nums) / len(step_nums)}")
# %%
