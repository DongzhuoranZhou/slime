#%%
import json
from pathlib import Path

def load_and_group_datapoints(dataset):
    """
    Load results from JSON and group datapoints by their answer sources.
    
    Args:
        dataset: Dataset name
        result_dir: Directory containing the results file
        result_filename: Name of the results JSON file
        
    Returns:
        Dictionary mapping each unique source to list of datapoints containing that source
    """
    result_dir = "/workspace/VLM-Memory/results/RAG/jina_multi/gemini-2.5-flash/"
    result_path = f"{dataset}_{dataset}_K128_N10_in131072_sizeNone_sampFalsemax32000min0t1.0p1.0_chatTrue_42.json"

    with open(Path(result_dir) / result_path, "r") as f:
        results = json.load(f)
    
    unique_sources = []
    for x in results["data"]:
        unique_sources.extend(x["answer_sources"])
    unique_sources = set(unique_sources)
    
    source_to_datapoints = {source: [] for source in unique_sources}
    for source in unique_sources:
        for x in results["data"]:
            if source in x["answer_sources"]:
                source_to_datapoints[source].append(x)
    
    return source_to_datapoints

# for each subset randomly sample a subset of datapoints
def create_source_subsets(source_to_datapoints, subset_size=15, seed=42):
    """
    Create random subsets of datapoints for each source with controlled performance distribution.
    
    Args:
        source_to_datapoints: Dictionary mapping sources to their datapoints
        subset_size: Maximum number of datapoints per source
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping sources to their sampled datapoint subsets
        Subsets contain 1/3 perfect (doc_qa==1.0), 1/3 partial (0<doc_qa<1.0), 1/3 failed (doc_qa==0.0)
    """
    import random
    random.seed(seed)
    source_to_subset = {}
    sampled_datapoints = set()
    
    for source, datapoints in source_to_datapoints.items():
        available_datapoints = [dp for dp in datapoints if id(dp) not in sampled_datapoints]
        
        # Group datapoints by performance
        perfect = [dp for dp in available_datapoints if dp.get("doc_qa") == 1.0]
        partial = [dp for dp in available_datapoints if 0 < dp.get("doc_qa", 0) < 1.0]
        failed = [dp for dp in available_datapoints if dp.get("doc_qa") == 0.0]
        
        # Calculate target sizes for each performance group
        target_per_group = subset_size // 3
        remainder = subset_size % 3
        
        # Sample from each group
        sampled_perfect = random.sample(perfect, min(len(perfect), target_per_group + (1 if remainder > 0 else 0)))
        sampled_partial = random.sample(partial, min(len(partial), target_per_group + (1 if remainder > 1 else 0)))
        sampled_failed = random.sample(failed, min(len(failed), target_per_group))
        
        # Combine samples
        subset = sampled_perfect + sampled_partial + sampled_failed
        
        # If we don't have enough datapoints matching the criteria, fill with any available
        if len(subset) < subset_size and len(available_datapoints) > len(subset):
            remaining = [dp for dp in available_datapoints if dp not in subset]
            additional_needed = min(subset_size - len(subset), len(remaining))
            subset.extend(random.sample(remaining, additional_needed))
        
        source_to_subset[source] = subset
        sampled_datapoints.update(id(dp) for dp in subset)
    
    return source_to_subset

dataset = "longdocurl" # mmlongdoc, longdocurl
subset_size = 3

source_to_datapoints = load_and_group_datapoints(dataset)
source_to_subset = create_source_subsets(source_to_datapoints, subset_size=subset_size, seed=42)

for source, subset in source_to_subset.items():
    print(f"Source: {source}, Subset size: {len(subset)}")
    #evaluate accuracy
    total = len(subset)
    correct = sum(1 for dp in subset if dp.get("doc_qa") == 1.0)
    accuracy = sum(dp["doc_qa"] for dp in subset) / total * 100 if total > 0 else 0.0
    print(f"  Accuracy: {accuracy:.1f}")

#totoal accuracy
accs = []
for _, subset in source_to_subset.items():
    for dp in subset:
        accs.append(dp["doc_qa"])
acc = sum(accs) / len(accs) * 100 if len(accs) > 0 else 0.0
print(f"Overall Subset Accuracy: {acc:.1f} over {len(accs)} samples")

# %%
