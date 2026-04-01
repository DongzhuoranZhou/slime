  Reading Order (30-45 min total)

  Step 1 — qa_search_train_merge_standard.py (5 min)
  Generates train_standard.parquet from RUC-NLPIR/FlashRAG_datasets.
  Key function: make_prefix(question, model_family="qwen3")
    - Qwen3 (default): returns just "Question: {question}" — no routing instructions,
      no <think>. Qwen3's tool descriptions and built-in thinking handle orchestration.
    - Qwen2.5: adds explicit ReAct routing + <think> instructions.
  Also produces a "tools" column with both search + answer schemas.
  This is the only place where data format diverges from search-r1.

  Step 2 — config.yaml (2 min)
  All tuneable knobs in one place: max_turns, search_backend (local or google),
  reward weights. The rollout_interaction_env_path key is how the rollout
  discovers SearchEnv.

  Step 3 — env_search.py (15 min — the core)
  Read the class bottom-up:
  - _extract_tool_call() — parses <tool_call>{JSON}</tool_call> from model output
  - step() — state machine: "answer" tool → done=True (end episode),
             "search" tool → dispatch to local/Google backend → return obs,
             no tool call → done=True (model gave free-form response)
  - format_observation() — returns {"role": "tool", "content": ...} so
    apply_chat_template generates Qwen's native <|im_start|>tool tokens
  - build_env() — factory that injects TOOLS (both schemas) into
    sample.metadata["tools"] for subsequent observation turns

  Two separate places tools are needed:
    Parquet tools column → data.py at load time → initial prompt apply_chat_template
    sample.metadata["tools"] → rollout.py during rollout → observation turn formatting

  The async bridge (run(_dispatch_search(...))): step() is synchronous but search
  is async. async_utils.run() submits to a background event loop thread.

  Step 4 — reward.py (8 min)
  Reads top-to-bottom. Strips <|im_start|>tool turns first (search results could
  contain JSON matching _ANSWER_TOOL_RE), then finds the last answer tool call
  and extracts arguments["response"]. _SEARCH_RE checks if web search was used
  for reward shaping (full score vs slight penalty for answering without searching).

  Step 5 — run_qwen3.sh (5 min)
  Key flags: --tool-key tools (passes tools column to apply_chat_template) and
  --apply-chat-template-kwargs '{"enable_thinking": true}' (activates Qwen3's
  built-in thinking — no need to instruct <think> in the user prompt).
  Three CUSTOM_ARGS lines wire all five files together.

  Step 6 — geo3k_vlm_multi_turn/rollout.py (reference, skim only)
  The generic multi-turn loop: build_env → reset → [generate → step →
  format_observation] × N → reward. SearchEnv plugs in at build_env and step.
