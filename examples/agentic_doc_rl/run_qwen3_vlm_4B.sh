#!/bin/bash
# Agentic Document QA RL training — Qwen3-VL-4B
#
# Prerequisites:
#   1. Qdrant running:      ./qdrant  (binary) or docker run -p 6333:6333 qdrant/qdrant
#   2. Documents indexed:   see AgenticMemory/scripts/ for indexing pipeline
#   3. SFT checkpoint:      output of AgenticMemory/sft/run_agentic_sft.sh
#
# Usage:
#   ./examples/agentic_doc_rl/run_qwen3_vlm_4B.sh
#   SLIME_SCRIPT_MODEL_NAME=Qwen3-VL-8B-Instruct ./examples/agentic_doc_rl/run_qwen3_vlm_4B.sh

MODEL_NAME=${SLIME_SCRIPT_MODEL_NAME:-"Qwen3-VL-4B-Instruct"}
NUM_GPUS=${SLIME_SCRIPT_NUM_GPUS:-8}
SFT_CKPT=${SLIME_SCRIPT_SFT_CKPT:-"/lc/AgenticMemory/checkpoints/sft"}
PARQUET=${SLIME_SCRIPT_PARQUET:-"/lc/AgenticMemory/logs/train_trajectories_30B.parquet"}
RL_SAVE=${SLIME_SCRIPT_RL_SAVE:-"/lc/AgenticMemory/checkpoints/rl"}

VALID_MODELS="Qwen3-VL-4B-Instruct Qwen3-VL-8B-Instruct Qwen2.5-VL-7B-Instruct"
if ! echo "$VALID_MODELS" | grep -qw "$MODEL_NAME"; then
    echo "Error: MODEL_NAME must be one of: $VALID_MODELS"
    exit 1
fi

pkill -9 sglang; sleep 3
ray stop --force; pkill -9 ray; pkill -9 python; sleep 3

set -ex
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

# Derive model args file (same convention as run_agentic_sft.sh)
MODEL_ARGS_FILE=$(echo "$MODEL_NAME" | sed 's/-Instruct//g; s/Qwen3-VL-/qwen3-/g; s/Qwen2.5-VL-/qwen2.5-/g')
MODEL_ARGS_ROTARY_BASE=5000000 source "${REPO_ROOT}/scripts/models/${MODEL_ARGS_FILE}.sh"

CKPT_ARGS=(
    --hf-checkpoint "${SFT_CKPT}"
    --load          "${SFT_CKPT}"
    --save          "${RL_SAVE}"
    --save-interval 20
)

ROLLOUT_ARGS=(
    --prompt-data        "${PARQUET}"
    --input-key          messages
    --tool-key           tools
    --apply-chat-template
    --apply-chat-template-kwargs '{"enable_thinking": false}'
    --rollout-shuffle
    --num-rollout        2000
    --rollout-batch-size 16
    --n-samples-per-prompt 8
    --rollout-max-response-len 4096
    --rollout-temperature 0.7
    --global-batch-size  128
    --balance-data
    --multimodal-keys '{"image": "images"}'
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.001
    --kl-loss-type low_var_kl
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.01
    --adam-beta1 0.9
    --adam-beta2 0.98
)

BACKEND_ARGS=(
    --train-backend megatron
    --tensor-model-parallel-size 4
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 2048
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --megatron-to-hf-mode bridge
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 2
    --sglang-mem-fraction-static 0.7
)

CUSTOM_ARGS=(
    --custom-generate-function-path examples.geo3k_vlm_multi_turn.rollout.generate
    --custom-rm-path                examples.agentic_doc_rl.reward.reward_func
    --custom-config-path            examples/agentic_doc_rl/config.yaml
)

if [ -n "$WANDB_API_KEY" ]; then
    WANDB_ARGS=(
        --use-wandb
        --wandb-project slime-agentic-doc-rl
        --wandb-group "$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')-grpo"
        --wandb-key "${WANDB_API_KEY}"
        --disable-wandb-random-suffix
    )
else
    WANDB_ARGS=()
fi

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" \
    --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# PYTHONPATH:
#   /workspace/Megatron-LM     — Megatron-LM internals
#   ${REPO_ROOT}               — makes examples.* importable
#   ${REPO_ROOT}/AgenticMemory — exposes search_models.py and utils.py
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/workspace/Megatron-LM:${REPO_ROOT}:${REPO_ROOT}/AgenticMemory\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 "${REPO_ROOT}/train_async.py" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "${NUM_GPUS}" \
    --rollout-num-gpus "${NUM_GPUS}" \
    --colocate \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${BACKEND_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${CUSTOM_ARGS[@]}" \
    "${WANDB_ARGS[@]}"
