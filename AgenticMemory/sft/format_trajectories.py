"""
format_trajectories.py

Converts raw trajectory JSONL (from VLM long-document QA agent runs) into a
slime-compatible parquet file consumable by the RL training pipeline.

Usage:
    python format_trajectories.py --input trajectories.jsonl \
                                  --output trajectories.parquet \
                                  --min-score 0.5
"""

import argparse
import base64
import io
import json
import logging

from datasets import Dataset
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pil_to_bytes(img: Image.Image) -> bytes:
    """Encode a PIL image to raw PNG bytes.

    This is a utility function for external callers (e.g. collect_trajectories.py).
    It is not used internally by this module.
    """
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def b64_to_data_uri(b64_str: str) -> str:
    """Convert a base64 PNG string to a data URI that qwen_vl_utils can consume.

    qwen_vl_utils.fetch_image expects either a URL string, a data: URI, or a PIL
    Image. Raw bytes cause a TypeError (bytes.startswith() rejects str args).
    """
    return f"data:image/png;base64,{b64_str}"


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def build_messages_and_images(raw: dict) -> tuple[list[dict], list[bytes]]:
    """
    Convert a single raw trajectory dict into (messages, images).

    messages — OpenAI chat format with ``<image>`` placeholders
    images   — flat list of raw PNG bytes in order of appearance
    """
    messages: list[dict] = []
    images: list[str] = []  # data URIs: "data:image/png;base64,..."

    # System + user turn
    messages.append({"role": "system", "content": raw.get("system_prompt", "")})
    messages.append({"role": "user", "content": raw.get("task", "")})

    steps = raw.get("steps", [])
    for step in steps:
        tool_calls_raw = step.get("tool_calls", [])

        # Skip empty steps (no tool calls, no responses) — smolagents inserts
        # these at position 0 (pre-action planning) and at the max-steps
        # boundary. They produce empty assistant messages that, if kept,
        # teach the model to emit nothing.
        if not tool_calls_raw:
            continue

        # Build assistant message with tool calls as "Action:" text in content.
        #
        # WHY NOT structured tool_calls:
        # Qwen2.5-VL's chat template has no tool_calls rendering — the field is
        # silently ignored, so apply_chat_template produces an empty assistant turn
        # (<|im_start|>assistant\n<|im_end|>\n). This results in 0 trainable tokens
        # and the model learns to output EOS immediately (collapse in ~5 steps).
        #
        # At inference time, smolagents uses "Action:" format from the system prompt,
        # and SGLang parses the model's text output into structured tool_calls. So
        # training on "Action:" text format matches the inference format exactly.
        action_parts = []
        for tc in tool_calls_raw:
            arguments = tc.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            action_parts.append(
                "Action:\n" + json.dumps({"name": tc["name"], "arguments": arguments}, ensure_ascii=False)
            )
        action_text = "\n".join(action_parts)

        assistant_msg = {
            "role": "assistant",
            "content": action_text,
        }
        messages.append(assistant_msg)

        # Build tool response messages, skipping final_answer tool calls only
        for tc_data in step.get("tool_calls", []):
            if tc_data["name"] == "final_answer":
                continue  # skip tool message for final_answer only

            matching_resp = next(
                (r for r in step.get("tool_responses", []) if r["tool_call_id"] == tc_data["id"]),
                None,
            )
            if matching_resp is None:
                continue

            imgs_b64 = matching_resp.get("image_paths", [])
            text = matching_resp.get("content", "")

            # Decode images first so placeholder count matches the images list length.
            decoded_images = []
            for b64 in imgs_b64:
                try:
                    decoded_images.append(b64_to_data_uri(b64))
                except Exception:
                    logger.warning(
                        "Failed to convert base64 image for tool_call_id=%s; skipping image.",
                        tc_data["id"],
                    )
            images.extend(decoded_images)

            content = "<image>" * len(decoded_images) + text
            messages.append({"role": "tool", "tool_call_id": tc_data["id"], "content": content})

    return messages, images


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert trajectory JSONL to slime-compatible parquet.")
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument("--output", required=True, help="Path to output parquet file")
    parser.add_argument("--min-score", type=float, default=0.0, help="Minimum score threshold (inclusive)")
    args = parser.parse_args()

    rows = []
    with open(args.input, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON on line %d.", lineno)
                continue

            score = float(raw.get("score", 0))
            if score < args.min_score:
                continue

            tools_raw = raw.get("tools_json", "[]") or "[]"
            try:
                json.loads(tools_raw)
            except json.JSONDecodeError:
                logger.warning("Invalid tools_json on line %d; falling back to [].", lineno)
                tools_raw = "[]"

            messages, images = build_messages_and_images(raw)

            rows.append({
                "messages": messages,
                "images": images,
                "tools": tools_raw,
                "question_id": str(raw.get("question_id", "")),
                "doc_name": str(raw.get("doc_name", "")),
                "question": str(raw.get("question", "")),
                "ground_truth": str(raw.get("ground_truth", "")),
                "model_answer": str(raw.get("model_answer", "")),
                "score": score,
                "num_steps": int(raw.get("num_steps", 0)),
            })

    dataset = Dataset.from_list(rows)
    dataset.to_parquet(args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
