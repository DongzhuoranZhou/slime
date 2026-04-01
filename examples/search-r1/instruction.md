# Standard Function Calling — How It Works (Quick Start)

A beginner-friendly guide to standard LLM tool calling, using a **search tool** as the running example.
Written to clarify what YOU must write vs. what `apply_chat_template` handles automatically.

---

## The Big Picture

Standard tool calling is just a **convention on message format** that the model was trained to follow.
There is no magic runtime. The model outputs text, you parse it, you call the tool, you put the result back as a message.

The word "standard" means: the tool schema format (OpenAI spec) and the role names (`"tool"`) are shared conventions, so `tokenizer.apply_chat_template` knows how to format them correctly for each model — without you writing any model-specific formatting code.

**What YOU write:**
- The tool schema (what the tool is named and what arguments it takes)
- The regex/JSON parser for the model's tool call output
- The actual tool execution (calling your search API)
- The content string to send back as the tool result

**What `apply_chat_template` handles automatically:**
- Embedding the tool schema into the system prompt (in Qwen's format, or Llama's format, etc.)
- Wrapping the tool result message with the correct role tokens (`<|im_start|>tool`, etc.)

---

## The Complete Pipeline — Step by Step

```
┌─ YOU define ──────────────────────────────────────────────┐
│  Tool schema (OpenAI JSON format)                         │
└───────────────────────────────────────────────────────────┘
                          ↓
┌─ TEMPLATE formats ────────────────────────────────────────┐
│  apply_chat_template(messages, tools=tools)               │
│  → embeds schema into system prompt with model tokens     │
└───────────────────────────────────────────────────────────┘
                          ↓  (model generates)
┌─ MODEL outputs ───────────────────────────────────────────┐
│  <tool_call>{"name":"web_search",                         │
│    "arguments":{"query":"capital of France"}}</tool_call> │
└───────────────────────────────────────────────────────────┘
                          ↓
┌─ YOU parse ───────────────────────────────────────────────┐
│  regex + json.loads → tool_name, arguments               │
└───────────────────────────────────────────────────────────┘
                          ↓
┌─ YOU execute ─────────────────────────────────────────────┐
│  result_text = await web_search(arguments["query"])       │
└───────────────────────────────────────────────────────────┘
                          ↓
┌─ YOU construct message ───────────────────────────────────┐
│  {"role": "tool", "name": "web_search",                   │
│   "content": result_text}                                 │
└───────────────────────────────────────────────────────────┘
                          ↓
┌─ TEMPLATE wraps ──────────────────────────────────────────┐
│  apply_chat_template([..., tool_result_message], ...)     │
│  → <|im_start|>tool\n...<|im_end|>\n<|im_start|>assistant│
└───────────────────────────────────────────────────────────┘
                          ↓  (model continues generating)
```

---

## Step 1 — Define the Tool Schema

The schema format is **universal** (same for all models — OpenAI, Qwen, Llama, etc.):

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information about a query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string"
                    }
                },
                "required": ["query"]
            }
        }
    }
]
```

YOU write this. It is just a data structure.

---

## Step 2 — Feed into `apply_chat_template` (returns a formatted string)

```python
messages = [
    {"role": "user", "content": "What is the capital of France?"}
]

prompt = tokenizer.apply_chat_template(
    messages,
    tools=tools,               # ← your schema from step 1
    tokenize=False,            # ← returns a string, not token IDs
    add_generation_prompt=True,
)
```

**Yes, it returns a value** — a formatted string. For Qwen2.5 it produces roughly:

```
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "web_search", "description": "Search the web...", ...}}
</tools>
<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant

```

Qwen's Jinja2 template embeds your schema into the system prompt. Llama formats it differently.
You do not write any of this — the tokenizer's built-in template does it.

---

## Step 3 — Model Outputs a Tool Call

The model (trained on this format) generates:

```
<tool_call>{"name": "web_search", "arguments": {"query": "capital of France"}}</tool_call>
```

- `<tool_call>` is Qwen's specific tag format. Other models use different tags or JSON blocks.
- `arguments` contains what the model wants to pass TO your tool — the model is asking you to run `web_search(query="capital of France")`.
- **No framework magic parses this.** You must do it yourself.

---

## Step 4 — YOU Parse the Tool Call

```python
import re, json

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

match = TOOL_CALL_RE.search(model_output)
if match:
    payload = json.loads(match.group(1))
    tool_name = payload["name"]           # "web_search"
    arguments = payload["arguments"]       # {"query": "capital of France"}
    query = arguments["query"]             # "capital of France"
```

This is your code. Always.

---

## Step 5 — YOU Execute the Tool

```python
result_text = await web_search(query)
# → "Paris is the capital of France, located in northern France..."
```

The result is just a Python string. No format is imposed on its content — it is whatever your search API returns.

---

## Step 6 — YOU Construct the Tool Result Message

```python
tool_result_message = {
    "role": "tool",          # standard role name for tool results
    "name": "web_search",    # which tool produced this result
    "content": result_text   # YOUR string — the raw search result text
}
```

**Common confusion:** `arguments` (step 3) is the model's call TO you. `content` here (step 6) is your answer BACK to the model. They are completely separate — `content` is NOT inside `arguments`.

---

## Step 7 — Feed Result Back to Model via `apply_chat_template`

```python
# Append assistant turn (model's tool call) and tool result to history
messages.append({"role": "assistant", "content": model_output})
messages.append(tool_result_message)

# Apply template again → gets next prompt for the model
next_prompt = tokenizer.apply_chat_template(
    messages,
    tools=tools,
    tokenize=False,
    add_generation_prompt=True,
)
```

For Qwen, this produces:

```
... (previous turns) ...
<|im_start|>assistant
<tool_call>{"name": "web_search", "arguments": {"query": "capital of France"}}</tool_call><|im_end|>
<|im_start|>tool
Paris is the capital of France, located in northern France...<|im_end|>
<|im_start|>assistant

```

YOU wrote the content string. The template wrapped it with `<|im_start|>tool` and `<|im_end|>`.

The model then reads this and continues generating its final answer.

---

## Summary Table — Who Does What

| Step | What happens | Who handles it |
|---|---|---|
| Tool schema definition | Write `{"type":"function", "function":{...}}` | **YOU** |
| Embed schema in system prompt | Format schema into model-specific tokens | **`apply_chat_template`** |
| Model output format | `<tool_call>{...}</tool_call>` (Qwen) | **Model** (trained behavior) |
| Parse model tool call | regex + `json.loads` | **YOU** |
| Execute the tool | Call search API, get result string | **YOU** |
| Format result content | Write the string you want the model to see | **YOU** |
| Wrap result as chat turn | `<|im_start|>tool\n...<|im_end|>` tokens | **`apply_chat_template`** |

---

## Comparison: Custom (search-r1) vs. Standard Tool Calling

| Aspect | search-r1 custom | Standard (this guide) |
|---|---|---|
| Tool call tag | `<search>query text</search>` | `<tool_call>{"name":"web_search","arguments":{"query":"..."}}</tool_call>` |
| Payload format | Free-form text | Structured JSON with `name` + `arguments` |
| Tool schema | None — no schema defined | OpenAI-compatible JSON schema |
| Parsing | `re.search(r"<(search\|answer)>(.*?)</\1>")` | `re.compile(r"<tool_call>...<\/tool_call>")` + `json.loads` |
| Result injected as | Raw text: `\n\n<information>...</information>\n\n` | Chat turn: `{"role":"tool","content":"..."}` via `apply_chat_template` |
| Result tokenized via | Direct `tokenizer(next_obs)` re-tokenization | `tokenizer.apply_chat_template([..., tool_msg], ...)` |
| Role tokens in context | None — flat string | Proper `<|im_start|>tool...<|im_end|>` delimiters |
| Model-agnostic? | Yes (any model can learn XML tags) | No — each model has its own `<tool_call>` format |

**Key takeaway:** The custom approach treats the context as a flat string. The standard approach treats each tool result as a proper chat turn with role tokens — which is what the model was fine-tuned to expect.

---

## For RL Training in slime (reference)

When using standard tool calling in a slime rollout:

- Pass `--tool-key <col>` to load tool schemas from the dataset column into `sample.metadata["tools"]`
- Pass `--apply-chat-template` so slime calls `tokenizer.apply_chat_template(prompt, tools=tools, ...)`
- In your custom `generate` function, after getting the model output:
  1. Parse `<tool_call>` manually (step 4 above)
  2. Execute the tool
  3. Construct `{"role": "tool", "content": result_text}`
  4. Tokenize via `apply_chat_template` to get the next input tokens
  5. Set `loss_mask = 0` for tool result tokens (they are environment, not model output)
- The geo3k_vlm_multi_turn example (`examples/geo3k_vlm_multi_turn/`) is the reference implementation for this pattern in slime
