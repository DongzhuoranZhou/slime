# RL State & Action Definition in slime

> Notes on how classic RL concepts map to LLM training in this codebase.

---

## 1. What is the State?

The **state** at any step is the full token sequence accumulated in `sample.tokens` (`slime/utils/types.py`).

```
state_t = sample.tokens = [prompt_ids] + [response_ids_so_far]
```

- For **standard single-turn RL** (GRPO/PPO): state = prompt tokens + partial response tokens so far.
- For **agentic / multi-turn RL**: state grows monotonically across turns:

```
state_k = [system_prompt]
        + [turn_0_prompt + response_0 + tool_result_0]
        + ...
        + [turn_k_prompt + response_k_so_far]
        = sample.tokens   ← same field, ever-growing
```

Environment feedback (tool results, observations) is **appended to `sample.tokens`** by the custom rollout function (`--rollout-function-path`) before calling `generate()` again. The framework itself is turn-agnostic.

### What is NOT the state

| Field | Role |
|---|---|
| `sample.prompt` (str) | Initial problem text; used only for first-turn tokenization, then discarded in favor of `sample.tokens` |
| `sample.response` (str) | Decoded text of all responses concatenated; for logging only, never fed to the model |
| `sample.metadata` | Auxiliary task metadata; never passed as model context |

---

## 2. What is the Action? (Two Levels)

This is the key subtlety: **action exists at two levels simultaneously**.

### Level 1 — Token level (the math / optimizer)

The policy is autoregressive — it selects **one token at a time**:

```
π_θ(response | prompt) = ∏ π_θ(token_t | token_0 ... token_{t-1})
```

Consequently:
- `sample.rollout_log_probs` is a **list of per-token floats** (`types.py:26`)
- PPO loss in `ppo_utils.py:132` operates **per-token**:

```python
ratio = (-ppo_kl).exp()           # per-token importance ratio
pg_losses = -ratio * advantages   # per-token policy gradient loss
```

At this level: **action = one token from the vocabulary**.

### Level 2 — Sequence level (the reward / environment / human intuition)

The **reward is assigned to the entire response** (or entire turn), not per-token:

```python
sample.reward: float | dict[str, Any] | None = None  # one scalar per response
```

GRPO broadcasts this scalar reward to every token in the response (`ppo_utils.py:201-208`):

```python
def get_grpo_returns(rewards, kl):
    for i in range(len(rewards)):
        returns.append(torch.ones_like(kl[i]) * rewards[i])  # broadcast to all tokens
```

At this level: **action = full response sequence** (a sentence, a tool call, a reasoning block).

### Summary table

| Level | Action granularity | Where used |
|---|---|---|
| Autoregressive (token) | Single token | Policy gradient loss, KL penalty, log-prob computation |
| Environment / agent | Full response or turn | Reward model, reward assignment, agentic step boundary |

---

## 3. The Agentic Hierarchy

For multi-turn agentic RL, both levels form a natural **hierarchical MDP**:

```
High-level MDP (agent reasoning):
  state   = conversation history (sample.tokens)
  action  = one complete turn output (tool call JSON, answer, reasoning step)
  reward  = scalar from RM, assigned at turn end or episode end

Low-level MDP (gradient computation):
  state   = tokens seen so far
  action  = one token
  reward  = 0 for most tokens; broadcasted reward value at designated positions
```

This is why **`loss_mask`** (`sample.loss_mask`, `types.py:24`) is critical in agentic settings:

- `mask = 1` → this token is part of the **agent's action** → train on it
- `mask = 0` → this token is part of the **environment observation / tool result** → do NOT train on it

The mask enforces the boundary between "what the agent chose" and "what the world returned".

---

## 4. What is One "Step" in Agentic RL?

In agentic RL, **"step" refers exclusively to inference turns within one episode** — not training iterations.

"20 steps" = `max_turns=20` in the custom rollout function, meaning the agent can interact with the environment up to 20 times before the episode ends.

### One step = one inference turn

From `examples/search-r1/generate_with_search.py:160`:

```python
for _turn_idx in range(SEARCH_R1_CONFIGS["max_turns"]):   # e.g. max_turns=20
    # 1. Agent generates a response (tool call, search query, answer, etc.)
    output = await post(url, payload)
    cur_response = output["text"]
    loss_mask += [1] * len(cur_response_token_ids)   # agent tokens: train on these

    # 2. Environment executes and returns observation
    next_obs, done = await execute_predictions(cur_response)
    if done:
        break

    # 3. Append observation to context — NOT trained on
    loss_mask += [0] * len(obs_tokens_ids)           # env tokens: skip in loss
```

### What grows during 20 steps (one episode)

```
step 0:  tokens = [prompt] + [agent_response_0]  +  [env_obs_0]
step 1:  tokens = [...   ] + [agent_response_1]  +  [env_obs_1]
...
step 19: tokens = [...   ] + [agent_response_19]
         ↑ this is sample.tokens at episode end — the full trajectory
```

### Where "step" lives in the code

This loop is entirely inside the **custom rollout function** passed via `--rollout-function-path`. The slime framework has no built-in concept of "turn" or `max_turns` — it just calls `generate()` once per episode and gets back a completed `Sample`. How many inference turns happen inside is fully controlled by user code.

---

## 5. What Exactly is One Step? (1 tool call + result)

**1 step = 1 LLM generation + 1 tool call + tool result returned**

From `examples/search-r1/generate_with_search.py`:

```python
# step N:
output = await post(url, payload)                        # LLM generates → "<search>query</search>"
next_obs, done = await execute_predictions(cur_response) # calls search API → gets results
```

One step = one `(action, observation)` pair:

```
action:       LLM outputs  →  <search>what is the capital of France</search>
observation:  search returns → <information>Paris is the capital...</information>
```

The `loss_mask` makes the boundary explicit:
- `loss_mask = 1` on the LLM generation (action) → trained on
- `loss_mask = 0` on the tool result (observation) → not trained on

The episode ends when the agent outputs `<answer>` instead of `<search>` — the final step has no tool call, just the agent giving its answer (`done=True`, line 210).

---

## 6. Key Files

| File | Relevance |
|---|---|
| `slime/utils/types.py` | `Sample` dataclass — the state container |
| `slime/rollout/sglang_rollout.py:199` | `sample.tokens` appended each turn — state growth |
| `slime/utils/ppo_utils.py:132` | Per-token policy loss — token-level action |
| `slime/utils/ppo_utils.py:201` | `get_grpo_returns` — sequence-level reward broadcast |
| `slime/rollout/sglang_rollout.py:204-206` | `loss_mask` update per turn — agent vs env boundary |
| `train.py:69` | Outer training loop — one `rollout_id` = one full iteration |
| `slime/ray/rollout.py:460` | `generate()` — what happens inside one rollout step |
| `slime/utils/arguments.py:1705` | `global_batch_size` formula — inner/outer step relationship |
