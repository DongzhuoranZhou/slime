import os

import slime.utils.misc as U
from slime.utils.external_utils.command_utils import execute_train

MODEL_NAME = os.environ.get("SLIME_SCRIPT_MODEL_NAME", "Qwen3-VL-4B-Instruct")
NUM_GPUS = int(os.environ.get("SLIME_SCRIPT_NUM_GPUS", "4"))
NUM_ROLLOUT = int(os.environ.get("SLIME_SCRIPT_NUM_ROLLOUT", "16"))
DATASET_NAME = os.environ.get("SLIME_SCRIPT_DATASET_NAME", "VeraIsHere/geo3k_imgurl_processed")
DATASET_LOCAL_NAME = os.path.basename(DATASET_NAME)
DATA_ROOT = f"/workspace/.cache/huggingface/datasets/datasets/{DATASET_LOCAL_NAME}"
TRAIN_DATA_PATH = os.path.join(DATA_ROOT, "train.parquet")
OUTPUT_DIR = os.environ.get("SLIME_SCRIPT_OUTPUT_DIR", f"/workspace/tmp/{MODEL_NAME}_geo3k_sft_data")


def get_megatron_model_type(model_name: str) -> str:
    model_type = model_name.replace("-Instruct", "").replace("-Thinking", "")
    model_type = model_type.replace("Qwen3-VL-", "qwen3-")
    return model_type.replace("-2B", "-1.7B")


def prepare():
    U.exec_command("mkdir -p /workspace/.cache/huggingface/hub /workspace/.cache/huggingface/datasets /workspace/tmp")
    if not os.path.exists(TRAIN_DATA_PATH):
        raise FileNotFoundError(f"Dataset not found. Expected local dataset at {TRAIN_DATA_PATH}")


def execute():
    ckpt_args = f"--hf-checkpoint /workspace/.cache/huggingface/hub/{MODEL_NAME} "
    rollout_args = (
        f"--prompt-data {TRAIN_DATA_PATH} "
        "--input-key problem "
        "--label-key answer "
        '--multimodal-keys \'{"image": "images"}\' '
        "--rm-type math "
        "--apply-chat-template "
        "--custom-generate-function-path examples.geo3k_vlm_sft_data_generation.rollout.generate "
        "--custom-config-path examples/geo3k_vlm_sft_data_generation/geo3k_vlm_sft_data_generation_config.yaml "
        "--rollout-shuffle "
        f"--num-rollout {NUM_ROLLOUT} "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 1 "
        "--rollout-max-response-len 4096 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 32 "
        "--debug-rollout-only "
        f"--save-debug-rollout-data {OUTPUT_DIR}/rollout_{{rollout_id}}.pt "
    )
    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.6 "
        "--sglang-cuda-graph-max-bs 32 "
    )
    backend_args = (
        "--train-backend megatron "
        f"--load /workspace/.cache/huggingface/hub/{MODEL_NAME} "
        "--tensor-model-parallel-size 1 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 4096 "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--megatron-to-hf-mode bridge "
    )
    misc_args = f"--actor-num-nodes 1 --actor-num-gpus-per-node {NUM_GPUS} --rollout-num-gpus {NUM_GPUS} --colocate "
    train_args = f"{ckpt_args} {rollout_args} {sglang_args} {backend_args} {misc_args}"
    execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=get_megatron_model_type(MODEL_NAME),
    )


if __name__ == "__main__":
    prepare()
    execute()
