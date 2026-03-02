#!/bin/bash

# -------------------------
# Workspace Environment
# -------------------------
# export http_proxy="http://httpproxy.glm.ai:8888"
# export https_proxy="http://httpproxy.glm.ai:8888"

export TMPDIR="/workspace/tmp"
export HF_HOME="/workspace/.cache/huggingface"
export XDG_CACHE_HOME="/workspace/.cache"
export XDG_DATA_HOME="/workspace/.local/share"
export XDG_STATE_HOME="/workspace/.local/state"

# Set CUDA_VISIBLE_DEVICES to use only 4 GPUs for development machine
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# Keep local bin; do NOT prepend conda here.
export PATH="/workspace/.local/bin:$PATH"

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16
# Set your wandb key (optional)
export WANDB_KEY="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"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/glm4-9B.sh"

CKPT_ARGS=(
   --hf-checkpoint /workspace/.cache/huggingface/hub/GLM-Z1-9B-0414
   --ref-load /workspace/.cache/huggingface/hub/GLM-Z1-9B-0414_torch_dist
   # If empty or doesn't contain a valid checkpoint, loads from --ref-load instead, so please comment --load
   # --load /workspace/.cache/huggingface/hub/GLM-Z1-9B-0414_slime/ # 
   --save /lc3T/GLM-Z1-9B-0414_slime_dev_machine/
   --save-interval 3
)

ROLLOUT_ARGS=(
   --prompt-data /workspace/.cache/huggingface/datasets/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle

   --rm-type deepscaler

   --num-rollout 5 # 3000
   --rollout-batch-size 2
   --n-samples-per-prompt 2
   --rollout-max-response-len 8192
   --rollout-temperature 1

   --global-batch-size 4
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 3
   --eval-prompt-data aime /workspace/.cache/huggingface/datasets/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 16384
   --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1 # 2 
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4608
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-dev
   --wandb-group glm4-9B-test
   --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# export MASTER_ADDR=$(hostname -I | awk '{print $1}')
# echo "Detected IP: ${MASTER_ADDR}"

# Bypass proxy for all internal/local addresses
export no_proxy="localhost,127.0.0.1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,${MASTER_ADDR}"
export NO_PROXY="${no_proxy}"

ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 4 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/workspace/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://${MASTER_ADDR}:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 2 \
   --rollout-num-gpus 2 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
