# Geo3k VLM SFT Data Generation

This example generates raw multi-turn trajectories with Slime in `--debug-rollout-only` mode, then converts the kept correct trajectories into an SFT parquet that matches the `messages` contract used by `examples/geo3k_vlm/run_geo3k_vlm_sft.sh`.

## Workflow

1. Generate rollout dumps:

```bash
python examples/geo3k_vlm_sft_data_generation/run_generate_geo3k_vlm_sft_data.py
```

2. Convert one saved rollout shard into training parquet:

```bash
python examples/geo3k_vlm_sft_data_generation/format_trajectories_to_sft.py \
  --input '/workspace/tmp/Qwen3-VL-4B-Instruct_geo3k_sft_data/rollout_*.pt' \
  --output /workspace/tmp/Qwen3-VL-4B-Instruct_geo3k_sft_data/Qwen3-VL-4B-Instruct_train_formatted.parquet
```

## Output Contract

The final parquet keeps geo3k-friendly columns such as `problem`, `answer`, and `images`, and adds:

```python
sample["messages"] = [
    {"role": "user", "content": sample["problem"]},
    {"role": "assistant", "content": f"Answer: \\boxed{{{sample['answer']}}}"},
]
```

This can be used directly with:

```bash
--rollout-function-path slime.rollout.sft_rollout.generate_rollout
--prompt-data /path/to/Qwen3-VL-4B-Instruct_train_formatted.parquet
--input-key messages
```
