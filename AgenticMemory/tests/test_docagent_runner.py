from docagent_runner import BaseDocAgentConfig


def test_base_config_from_dict_overrides_defaults():
    config = BaseDocAgentConfig.from_dict(
        {
            "dataset": "custom",
            "top_k": 5,
            "sample_num": 7,
            "results_jsonl_path": "results/custom.jsonl",
            "debug_pickle_path": "results/custom.pkl",
        }
    )

    assert config.dataset == "custom"
    assert config.top_k == 5
    assert config.sample_num == 7
    assert config.results_jsonl_path == "results/custom.jsonl"
    assert config.debug_pickle_path == "results/custom.pkl"


def test_base_config_from_env_uses_prefixes(monkeypatch):
    monkeypatch.setenv("API_DATASET", "env_dataset")
    monkeypatch.setenv("API_TOP_K", "12")
    monkeypatch.setenv("API_SAMPLE_NUM", "3")
    monkeypatch.setenv("API_RESULTS_JSONL_PATH", "results/env.jsonl")
    monkeypatch.setenv("API_DEBUG_PICKLE_PATH", "results/env.pkl")

    config = BaseDocAgentConfig.from_env(
        dataset_env="API_DATASET",
        top_k_env="API_TOP_K",
        sample_num_env="API_SAMPLE_NUM",
        results_jsonl_env="API_RESULTS_JSONL_PATH",
        debug_pickle_env="API_DEBUG_PICKLE_PATH",
    )

    assert config.dataset == "env_dataset"
    assert config.top_k == 12
    assert config.sample_num == 3
    assert config.results_jsonl_path == "results/env.jsonl"
    assert config.debug_pickle_path == "results/env.pkl"
