import json
from pathlib import Path
from typing import Dict, Tuple, Union
from functools import lru_cache

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_eval_results(dataset: str, emd_model: str, 
                            inference_model: str) -> dict:
    """Load evaluation results from file."""
    result_dir = Path("/workspace/VLM-Memory/results/RAG") / emd_model / inference_model
    result_paths = [f for f in result_dir.glob("*.json") if dataset in f.name and "_K128_N10_" in f.name]
    # result_path = f"{dataset}_{dataset}_K128_N10_in131072_sizeNone_sampFalsemax32000min0t1.0p1.0_chatTrue_42.json"
    
    with open(Path(result_dir) / result_paths[0], "r") as f:
        results = json.load(f)
    
    return results

def group_by_source(results: dict) -> dict[str, list]:
    """Group datapoints by their source documents."""
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

def stratified_sample(source_to_datapoints: dict[str, list], sample_size: int, 
                     seed: int) -> list:
    """Sample datapoints stratified by performance."""
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
        target_per_group = sample_size // 3
        remainder = sample_size % 3
        
        # Sample from each group
        sampled_perfect = random.sample(perfect, min(len(perfect), target_per_group + (1 if remainder > 0 else 0)))
        sampled_partial = random.sample(partial, min(len(partial), target_per_group + (1 if remainder > 1 else 0)))
        sampled_failed = random.sample(failed, min(len(failed), target_per_group))
        
        # Combine samples
        subset = sampled_perfect + sampled_partial + sampled_failed
        
        # If we don't have enough datapoints matching the criteria, fill with any available
        if len(subset) < sample_size and len(available_datapoints) > len(subset):
            remaining = [dp for dp in available_datapoints if dp not in subset]
            additional_needed = min(sample_size - len(subset), len(remaining))
            subset.extend(random.sample(remaining, additional_needed))
        
        source_to_subset[source] = subset
        sampled_datapoints.update(id(dp) for dp in subset)
    
    subset_results = [dp for _, v in source_to_subset.items() for dp in v]

    return subset_results

def get_subset_results(dataset, subset_size,  
                        emd_model="jina_multi", 
                        inference_model="gemini-2.5-flash",
                        seed=42):
    """
    Extracts a stratified subset of RAG evaluation results grouped by document source.
    
    This function loads evaluation results, groups datapoints by their source documents,
    and samples a balanced subset from each source based on performance metrics (perfect,
    partial, and failed predictions).

    Args:
        dataset (str): Dataset identifier (e.g., "mmlongdoc", "longdocurl")
        subset_size (int): Target number of datapoints to sample per source document
        emd_model (str, optional): Embedding model identifier used in retrieval. 
            Defaults to "jina_multi"
        inference_model (str, optional): Inference model identifier used in evaluation.
            Defaults to "gemini-2.5-flash"
        seed (int, optional): Random seed for reproducible sampling. Defaults to 42

    Returns:
        Dict[str, list]: Mapping of source document paths to their sampled datapoint subsets.
            Each source contains up to `subset_size` datapoints stratified by performance:
            - Perfect: doc_qa == 1.0
            - Partial: 0 < doc_qa < 1.0
            - Failed: doc_qa == 0.0
    """
    
    results = load_eval_results(dataset, emd_model, inference_model)
    source_to_datapoints = group_by_source(results)
    subset_results = stratified_sample(source_to_datapoints, subset_size, seed)
    
    return subset_results

class PathManager:
    """Centralizes path logic for reproducibility and ease of updates."""

    # --- Configuration ---
    MOUNT_DIR = Path("/lc")
    BASE_DATA_DIR = MOUNT_DIR / "data"
    
    DATA_DIR = BASE_DATA_DIR / "mmlb_data"
    IMAGE_DIR = BASE_DATA_DIR / "mmlb_image"
    RETR_DIR = BASE_DATA_DIR / "mmlb_retr"
    PARSE_DIR = BASE_DATA_DIR / "parsed_doc"

    DATASET_MAP = {
        "mmlongbench-doc": "mmlongdoc",
        "longdocurl": "longdocurl"
    }

    DATASET2ID = {
        "mmlongbench-doc": "DOC",
        "longdocurl": "URL"
    }

    ID2DATASET = {v: k for k, v in DATASET2ID.items()}
    
    @classmethod
    @lru_cache(maxsize=None)
    def _get_file_maps(cls, dataset_name: str) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Scans the directory once and caches the result. 
        Returns (folder_name_to_id, id_to_folder_name).
        """
        dataset_dir = cls.IMAGE_DIR / dataset_name
        
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

        # Sorted to ensure deterministic ID assignment
        subfolders = sorted([p.name for p in dataset_dir.iterdir() if p.is_dir()])
        
        name2id = {name: str(idx) for idx, name in enumerate(subfolders)}
        id2name = {str(idx): name for idx, name in enumerate(subfolders)}
        
        return name2id, id2name

    @classmethod
    def encode_doc_path(cls, file_path: Union[Path, str]) -> Path:
        """
        Converts a raw file path into an encoded ID-based path.
        Ex: .../mmlongbench-doc/some_doc/img_01.jpg -> DOC/5/5_01.jpg
        """
        path = Path(file_path)
        
        # Extract components
        dataset_name, doc_name, file_name = path.parts
        file_suffix = file_name.split("_")[-1] # Preserves specific suffix logic

        # Validate dataset support
        if dataset_name not in cls.DATASET2ID:
            raise ValueError(f"Unknown dataset directory: {dataset_name}")

        dataset_id = cls.DATASET2ID[dataset_name]
        name2id, _ = cls._get_file_maps(dataset_name)

        if doc_name not in name2id:
            raise KeyError(f"Document '{doc_name}' not found in {dataset_name}")

        doc_id = name2id[doc_name]
        
        # Construct encoded path
        return Path(dataset_id) / doc_id / f"{doc_id}_{file_suffix}"
    
    @classmethod
    def decode_doc_path(cls, encoded_path: Union[Path, str]) -> Path:
        """
        Restores the real file path from an encoded ID-based path.
        Ex: DOC/5/5_01.jpg -> mmlongbench-doc/some_doc/img_01.jpg
        """
        path = Path(encoded_path)
        
        # Extract components
        dataset_id, doc_id, file_name = path.parts
        file_suffix = file_name.split("_")[-1]

        if dataset_id not in cls.ID2DATASET:
             raise ValueError(f"Unknown dataset ID: {dataset_id}")

        dataset_real_name = cls.ID2DATASET[dataset_id]
        _, id2name = cls._get_file_maps(dataset_real_name)

        if doc_id not in id2name:
             raise KeyError(f"Document ID '{doc_id}' not found in {dataset_real_name}")

        doc_name = id2name[doc_id]

        # Construct decoded partial path (relative to IMAGE_DIR usually)
        return Path(dataset_real_name) / doc_name / f"{doc_name}_{file_suffix}"

    @classmethod
    def get_retrieval_jsonl(cls, emd_model: str, dataset: str) -> Path:
        return cls.MOUNT_DIR / "data" / "mmlb_retr" / emd_model / f"{dataset}_K128.jsonl"
    
    @classmethod
    def get_dataset_jsonl(cls, dataset: str) -> Path:
        return cls.MOUNT_DIR / "data" / "mmlb_data" / "documentQA" / f"{dataset}_K128.jsonl"

    @classmethod
    def get_retrieval_paths(cls, data, K):
        """Extracts the top-K retrieval paths from a data entry."""
        if "retrieved" in data:
            return [Path(v) for v, _ in data["retrieved"][:K]]
        else: # for jina_multi
            return [Path(v) for v in data["page_list"][:K]]
        
    @classmethod
    def check_path_list(cls, doc_path_list):
        if isinstance(doc_path_list, list):
            doc_paths = doc_path_list
        else:
            cleaned_str = doc_path_list.strip()
            try:
                import ast
                doc_paths = ast.literal_eval(cleaned_str)
            except:
                raise ValueError("Unable to parse document paths.")
        return doc_paths
    
    @classmethod
    def get_img_path(cls, encode_path: str):
        return cls.IMAGE_DIR / cls.decode_doc_path(encode_path)
    
    @classmethod
    def get_parsed_dir(cls, encode_path: str):
        raw_path = cls.decode_doc_path(encode_path)
        dataset = raw_path.parts[0]
        
        if dataset not in cls.DATASET_MAP:
            raise ValueError("Dataset not supported.")

        relative_structure = Path(*raw_path.parts[1:])
        parsed_folder = (
            cls.PARSE_DIR / 
            cls.DATASET_MAP[dataset] / 
            relative_structure.with_suffix('') /
            "vlm"
        )

        return parsed_folder

    @classmethod
    def get_parsed_text(cls, encode_path: str):
        parse_dir = cls.get_parsed_dir(encode_path)
        if parse_dir is None:
            raise ValueError("Parsed directory not found.")
        
        return parse_dir / f"{parse_dir.parent.name}.md"
    
    @classmethod
    def get_parsed_json(cls, encode_path: str):
        parse_dir = cls.get_parsed_dir(encode_path)
        if parse_dir is None:
            raise ValueError("Parsed directory not found.")
        
        return parse_dir / f"{parse_dir.parent.name}_content_list_v2.json"

    @classmethod
    def get_parsed_img(cls, encode_path: str):
        parse_dir = cls.get_parsed_dir(encode_path)
        if parse_dir is None:
            raise ValueError("Parsed directory not found.")
        
        img_dir = parse_dir / "images"

        if img_dir.exists() and img_dir.is_dir():
            return list(img_dir.glob("*.jpg"))
        else:
            raise ValueError("Image directory not found.")