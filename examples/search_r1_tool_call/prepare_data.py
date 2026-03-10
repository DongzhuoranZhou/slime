"""
prepare_data.py — one-time preprocessing for search_r1_tool_call.

Reads the original NQ/HotpotQA parquet produced by Search-R1's data scripts
and outputs a new parquet with:

  1. A "tools" column containing the web_search tool schema as a JSON string.
     slime's data.py reads this with json.loads() and passes it to
     tokenizer.apply_chat_template(tools=...) to embed the tool definition
     in the initial system prompt.

  2. (optional, --add-tool-instructions) Rewrites the system message in each
     row's "prompt" column to instruct the model to use <tool_call> format and
     wrap its final answer in <answer>...</answer> tags.

Usage:
    python examples/search_r1_tool_call/prepare_data.py \\
        --input  /root/Search-R1/data/nq_hotpotqa_train/train.parquet \\
        --output /root/Search-R1/data/nq_hotpotqa_train/train_with_tools.parquet \\
        --add-tool-instructions
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

SYSTEM_MESSAGE_WITH_TOOLS = (
    "You are a helpful assistant with access to a web search tool. "
    "When you need information to answer a question, call the web_search tool using: "
    '<tool_call>{"name": "web_search", "arguments": {"query": "your query here"}}</tool_call>. '
    "After receiving the search results, reason over them carefully. "
    "When you have determined the final answer, provide it between "
    "<answer> and </answer> tags, for example: <answer>Paris</answer>."
)

WEB_SEARCH_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information relevant to the question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]

TOOLS_JSON = json.dumps(WEB_SEARCH_TOOL)


def _rewrite_system_message(prompt_field) -> list:
    """
    Replace (or prepend) the system message in the prompt conversation list.
    The prompt field is expected to be a list of {"role": ..., "content": ...} dicts.
    """
    if isinstance(prompt_field, str):
        try:
            messages = json.loads(prompt_field)
        except json.JSONDecodeError:
            return prompt_field
    else:
        messages = list(prompt_field)

    # Replace existing system message or prepend a new one
    has_system = any(m.get("role") == "system" for m in messages)
    if has_system:
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE_WITH_TOOLS} if m.get("role") == "system" else m
            for m in messages
        ]
    else:
        messages = [{"role": "system", "content": SYSTEM_MESSAGE_WITH_TOOLS}] + messages

    return messages


def add_tools_column(input_path: str, output_path: str, add_tool_instructions: bool = False):
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")
    print(f"Columns: {list(df.columns)}")

    # Add tools column (same JSON string for every row)
    df["tools"] = TOOLS_JSON

    if add_tool_instructions and "prompt" in df.columns:
        df["prompt"] = df["prompt"].apply(_rewrite_system_message)
        print("Rewrote system messages with tool-calling instructions.")

    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")

    # Quick sanity check
    tools = json.loads(df["tools"].iloc[0])
    assert tools[0]["function"]["name"] == "web_search", "tools column sanity check failed"
    print(f"Sanity check passed: tools[0]['function']['name'] = {tools[0]['function']['name']!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add tools column to NQ/HotpotQA parquet for search_r1_tool_call.")
    parser.add_argument("--input", required=True, help="Path to original train.parquet")
    parser.add_argument("--output", required=True, help="Path for output parquet with tools column")
    parser.add_argument(
        "--add-tool-instructions",
        action="store_true",
        help="Rewrite system messages to include tool-call format instructions",
    )
    args = parser.parse_args()
    add_tools_column(args.input, args.output, add_tool_instructions=args.add_tool_instructions)
