#!/bin/bash
# SFT training on agentic trajectories distilled from Qwen3-VL-30B.
#
# Usage:
#   ./run_agentic_sft.sh
#   SLIME_SCRIPT_MODEL_NAME=Qwen3-VL-8B-Instruct ./run_agentic_sft.sh
#   SLIME_SCRIPT_PARQUET=/path/to/parquet ./run_agentic_sft.sh

TRAIN_BACKEND="megatron"
MODEL_NAME=${SLIME_SCRIPT_MODEL_NAME:-"Qwen3-VL-4B-Instruct"}
NUM_GPUS=${SLIME_SCRIPT_NUM_GPUS:-8}
LOG_DIR="${AGENTIC_MEMORY_LOG_DIR:-/lc3T/AgenticMemory/logs}"
PARQUET=${SLIME_SCRIPT_PARQUET:-"${LOG_DIR}/train_trajectories_30B.parquet"}
SAVE_PATH=${SLIME_SCRIPT_SAVE_PATH:-"/lc3T/${MODEL_NAME}_agentic_sft"}

VALID_MODELS="
  Qwen2.5-VL-7B-Instruct
  Qwen3-VL-4B-Instruct
  Qwen3-VL-8B-Instruct
  Qwen3-VL-30B-A3B-Instruct
"
if ! echo "$VALID_MODELS" | grep -qw "$MODEL_NAME"; then
    echo "Error: MODEL_NAME must be one of:$VALID_MODELS"
    exit 1
fi

MODEL_NAME_LOWER=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')

# External Ray flag
if [ -z "$SLIME_SCRIPT_EXTERNAL_RAY" ] || [ "$SLIME_SCRIPT_EXTERNAL_RAY" = "0" ]; then
    USE_EXTERNAL_RAY=0
else
    USE_EXTERNAL_RAY=1
fi

# Cleanup
pkill -9 sglang
sleep 3
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
    ray stop --force
    pkill -9 ray
fi
pkill -9 slime
sleep 3
pkill -9 redis

set -ex

export PYTHONBUFFERED=16

# Detect NVLink
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# Download model if needed
mkdir -p /workspace/.cache/huggingface/hub
if [ ! -d "/workspace/.cache/huggingface/hub/${MODEL_NAME}" ]; then
    hf download Qwen/${MODEL_NAME} --local-dir /workspace/.cache/huggingface/hub/${MODEL_NAME}
fi

CKPT_ARGS=(
    --hf-checkpoint /workspace/.cache/huggingface/hub/${MODEL_NAME}
    --load         /workspace/.cache/huggingface/hub/${MODEL_NAME}
    --save         ${SAVE_PATH}
    --save-interval 50
)

SFT_ARGS=(
    --rollout-function-path slime.rollout.sft_rollout.generate_rollout
    --prompt-data "${PARQUET}"
    --input-key messages
    --tool-key tools
    --rollout-shuffle
    --num-epoch 3
    --rollout-batch-size ${SLIME_SCRIPT_GLOBAL_BATCH_SIZE:-4}
    --global-batch-size ${SLIME_SCRIPT_GLOBAL_BATCH_SIZE:-4}

    --loss-type sft_loss
    --calculate-per-token-loss
    --disable-compute-advantages-and-returns
    --debug-train-only
)

# required for vlm datasets
MULTIMODAL_KEYS='{"image": "images"}'

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-5
    --lr-decay-style cosine
    --min-lr 1e-6
    --lr-warmup-fraction 0.1
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
)

if [ -n "$WANDB_API_KEY" ]; then
    WANDB_ARGS=(
        --use-wandb
        --wandb-project slime-agentic-sft
        --wandb-group ${MODEL_NAME_LOWER}-${TRAIN_BACKEND}
        --wandb-key ${WANDB_API_KEY}
        --wandb-host https://wandb.glm.ai
        --disable-wandb-random-suffix
    )
else
    WANDB_ARGS=()
fi

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
    --max-tokens-per-gpu 4096
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --megatron-to-hf-mode bridge
)

# Derive model-args file name from model name (same convention as geo3k_vlm)
# Qwen3-VL-4B-Instruct -> qwen3-4B, Qwen3-VL-8B-Instruct -> qwen3-8B, etc.
SLIME_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
MODEL_ARGS_FILE=$(echo "$MODEL_NAME" | sed 's/-Instruct//g; s/-Thinking//g; s/Qwen3-VL-/qwen3-/g; s/Qwen2.5-VL-/qwen2.5-/g; s/-2B/-1.7B/g')
# VL models require rotary-base 5000000
MODEL_ARGS_ROTARY_BASE=5000000 source "${SLIME_DIR}/scripts/models/${MODEL_ARGS_FILE}.sh"

# Start Ray if not using external Ray
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
    export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
    export no_proxy="127.0.0.1,${MASTER_ADDR}"
    export NO_PROXY="127.0.0.1,${MASTER_ADDR}"
    ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} \
        --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/workspace/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY}\",
    \"WANDB_BASE_URL\": \"https://wandb.glm.ai\"
  }
}"

# NO_PROXY is exported above but also inlined here so the Ray dashboard
# API calls from `ray job submit` are never routed through http_proxy.
# Without this, log streaming fails with 503 when http_proxy is set.
NO_PROXY="127.0.0.1,${MASTER_ADDR:-127.0.0.1}" \
no_proxy="127.0.0.1,${MASTER_ADDR:-127.0.0.1}" \
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 "${SLIME_DIR}/train_async.py" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node ${NUM_GPUS} \
    --multimodal-keys "${MULTIMODAL_KEYS}" \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${SFT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${BACKEND_ARGS[@]}"
