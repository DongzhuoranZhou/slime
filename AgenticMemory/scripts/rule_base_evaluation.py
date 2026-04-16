#%%
import json
from pathlib import Path

def get_result_path(config):
    base_dir = Path("/workspace/VLM-Memory/results")
    result_dir = base_dir / config["retrieve"] / config["embed_model"] / config["vlm_model"]

    for json_path in result_dir.glob("*.json"):
        if f"K128_N{config['K']}_" in json_path.name and f"{config['dataset']}_" in json_path.name:
            return str(json_path)
    
    raise FileNotFoundError("No matching result file found.")

def rule_based_evaluation(method, result_path):
    if method == "RAG":
        with open(result_path, "r") as f:
            results = json.load(f)
        accs = []
        for result in results["data"][:sample_num]:
            accs.append(result["doc_qa"])
        accuracy = sum(accs) / len(accs)
        print("---- Evaluation Results of RAG ----")
        print(f"Accuracy over {sample_num} samples: {accuracy:.4f}")
    elif method == "agent":
        accs = []
        with open(result_path, "r") as f:
            for line in f.readlines():
                result = json.loads(line)
                accs.append(result["acc"])
        accuracy = sum(accs) / len(accs)
        print("---- Evaluation Results of Agent ----")
        print(f"Accuracy over {len(accs)} samples: {accuracy:.4f}")

config = {
    "retrieve": "RAG",
    "K": 10,
    "embed_model": "jina_multi",
    "vlm_model": "gemini-2.5-flash",
    "dataset": "mmlongdoc"  # mmlongdoc, longdocurl
}

# evaluate accuracy
sample_num = 50

result_path = get_result_path(config)
agent_result_path = "/workspace/smolagent_dev/results/top50_run.jsonl"

rule_based_evaluation("RAG", result_path)
rule_based_evaluation("agent", agent_result_path)
# %%
