# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**slime** is an LLM post-training framework for Reinforcement Learning (RL) scaling. It powers production models (GLM-4.7, Qwen3, DeepSeek V3) and orchestrates three interacting subsystems: a Megatron-based training module, an SGLang-based rollout module, and a Ray-managed data buffer.

## Commands

### Install
```bash
pip install -r requirements.txt
pip install -e .
```

### Lint & Format
```bash
pre-commit install                                          # one-time setup
pre-commit run --all-files --show-diff-on-failure          # check all files
```
Tools applied in order: autoflake → isort → black (line length 119) → ruff (line length 320).

### Tests
```bash
pytest --verbose --pyargs --durations=0                    # all tests
pytest tests/test_file.py::test_name --durations=0         # single test
pytest -m unit                                             # by marker
```
Available markers: `unit`, `integration`, `system`, `acceptance`, `skipduringci`, `pleasefixme`.

E2E/CI tests run inside Docker (`slimerl/slime:latest`) on a self-hosted GPU cluster via `.github/workflows/pr-test.yml`.

## Architecture

### Three Core Subsystems

**1. Training Module** (`slime/backends/megatron_utils/`)
- Wraps Megatron-LM for distributed training
- Key files: `actor.py`, `model.py`, `loss.py`, `checkpoint.py`, `arguments.py`
- Reads training batches from the Data Buffer, syncs weights to rollout engines after each update

**2. Rollout Module** (`slime/rollout/`, `slime/backends/sglang_utils/`)
- Uses SGLang for efficient inference to generate on-policy data
- `sglang_rollout.py`: primary rollout engine; calls `generate_rollout()`, `async_rm()`, `batched_async_rm()`
- Sub-hubs: `filter_hub/` (sample filtering), `rm_hub/` (reward models), `generate_hub/` (custom generators)
- Extensible via `--rollout-function-path` (custom rollout) and `--custom-rm-path` (custom reward)

**3. Data Buffer / Orchestration** (`slime/ray/`)
- `rollout.py` — `RolloutManager`: coordinates prompt initialization, rollout generation, and data hand-off to training
- `placement_group.py`: GPU resource allocation via Ray placement groups
- `actor_group.py`: manages distributed Ray actors
- `train_actor.py`: training-side Ray actor implementation

### Training Entry Points
- `train.py` — synchronous: rollout → train → sync weights, repeat
- `train_async.py` — asynchronous: prefetches next rollout while training current batch

### Plugin System (`slime_plugins/`)
- `mbridge/`: model-specific parameter mapping bridges (GLM, Qwen, DeepSeek, Llama)
- `megatron_bridge/`: Megatron integration hooks
- `models/`: custom model implementations (e.g., `glm5/`)
- Plugin interface compliance is tested in `tests/plugin_contracts/`

### Configuration
All slime-specific CLI arguments are defined in `slime/utils/arguments.py` (large file, ~77KB). SGLang arguments live in `slime/backends/sglang_utils/arguments.py` and are prefixed with `--sglang-` on the command line. OmegaConf/YAML configs are supported for complex setups.

### Key Utilities (`slime/utils/`)
| File | Purpose |
|------|---------|
| `ppo_utils.py` | KL divergence, GAE, advantage computation |
| `data.py` | Dataset loading and batching |
| `eval_config.py` | Evaluation dataset config parsing |
| `processing_utils.py` | Async data processing pipeline |
| `health_monitor.py` | System health and fault detection |
| `wandb_utils.py` | W&B experiment tracking |

## Key Conventions

- Python 3.10+ required; tested on 3.10, 3.11, 3.12
- Pre-commit must pass before merging (CI enforces this via `pre-commit.yml`)
- Contribution scope: bug fixes, RL optimizations. Large refactors and architecture changes are out of scope per `CONTRIBUTING.md`
- Model support is gated through the plugin system — new models require a bridge in `slime_plugins/mbridge/`
