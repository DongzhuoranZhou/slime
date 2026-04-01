# Notes: GRPO Training — Algorithm, Loss, and PyTorch Gradients

> Covers: how the rollout trajectory is passed to training, the GRPO/PPO loss formula, what `log_prob_old` and `log_prob_new` are, where the gradient comes from, and the PyTorch autograd chain rule.
> Code references: `slime/backends/megatron_utils/loss.py`, `slime/utils/ppo_utils.py`

---

## Section 1: From Trajectory to Training — Data Flow

After rollout, each `Sample` carries:

| Field | Type | Content |
|---|---|---|
| `sample.tokens` | `list[int]` | Full token sequence: prompt + all turns |
| `sample.loss_mask` | `list[int]` | 1 for model tokens, 0 for prompt/observations |
| `sample.rollout_log_probs` | `list[float]` | Log probabilities saved during generation |
| `sample.response_length` | `int` | Number of model-generated tokens |
| `sample.reward` | `float` | Scalar reward from the reward model |

These are batched into a `RolloutBatch` dict and passed into `compute_advantages_and_returns()` (`loss.py:400`), which computes per-token advantages from rewards, then into `compute_policy_gradient_loss()` for the actual gradient step.

The key asymmetry:
- `sample.rollout_log_probs` → saved from rollout, **no gradient**, used as `log_prob_old`
- log probs recomputed during training forward pass → **has gradient**, used as `log_prob_new`

---

## Section 2: The Loss Formula, Ratio, and Policy Gradient

### 2.1 REINFORCE vs PPO vs GRPO

| Algorithm | Advantage | Critic needed? | Clipping? |
|---|---|---|---|
| REINFORCE | $A = R$ (raw reward) | No | No |
| PPO | $A$ from GAE (value baseline) | Yes (value network) | Yes (ratio clip) |
| GRPO | $A = \frac{R - \mu_g}{\sigma_g}$ (group relative) | **No** | Yes (ratio clip) |

**GRPO** generates $N$ responses per prompt (here `--n-samples-per-prompt 8`), then:

$$A_i = \frac{R_i - \text{mean}(R_{1..N})}{\text{std}(R_{1..N})}$$

No critic network is needed — the group of responses acts as the baseline.

---

### 2.2 The Ratio

$$\text{ratio}_t = \frac{\pi_\theta(\text{token}_t \mid \text{context})}{\pi_{\text{old}}(\text{token}_t \mid \text{context})} = \exp(\log\pi_\theta - \log\pi_{\text{old}})$$

In code (`ppo_utils.py:132`):
```python
# ppo_kl = log_prob_old - log_prob_new  (note the sign)
ratio = (-ppo_kl).exp()   # = exp(log_prob_new - log_prob_old)
```

`ratio > 1` means the current policy assigns higher probability to this token than during rollout.
`ratio < 1` means lower probability.

---

### 2.3 What is `log_prob_old`?

`log_prob_old` is the per-token log probability **saved during rollout** (inference time), stored in `sample.rollout_log_probs`. It is a plain Python float — a historical record, **no gradient attached**.

```
Rollout time:   model generates token_t  →  save log P(token_t | context)  →  sample.rollout_log_probs[t]
Training time:  model re-runs same token_t  →  recompute log P(token_t | context)  →  log_prob_new (has gradient)
```

Together they measure: "how much has the policy drifted since we collected this data?"

---

### 2.4 The PPO Clipped Loss

The unclipped policy gradient loss would be:

$$\mathcal{L}_{\text{PG}} = -\text{ratio}_t \cdot A_t$$

PPO clips the ratio to prevent large updates:

$$\mathcal{L}_{\text{clip}}(t) = -\min\!\Big(\text{ratio}_t \cdot A_t,\ \text{clip}(\text{ratio}_t,\ 1-\varepsilon,\ 1+\varepsilon_{\text{high}}) \cdot A_t\Big)$$

In code (`ppo_utils.py:133`):
```python
pg_losses1 = -ratio * advantages                                    # unclipped
pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages  # clipped
pg_losses  = torch.maximum(pg_losses1, pg_losses2)                 # take the worse (more conservative)
```

The `maximum` of two negative values picks the less negative one — the more conservative update.
In `run_geo3k_qwen35.sh`: `eps_clip=0.2`, `eps_clip_high=0.28`.

---

### 2.5 What is Policy Gradient?

Classic REINFORCE (vanilla policy gradient):

$$\nabla_\theta \mathcal{L} = -R \cdot \nabla_\theta \log \pi_\theta(\text{token})$$

PPO/GRPO replace $R$ with advantage $A$ and $\log\pi$ with the ratio:

$$\nabla_\theta \mathcal{L} = -A \cdot \nabla_\theta \,\text{ratio} = -A \cdot \nabla_\theta \exp(\log\pi_\theta - \log\pi_{\text{old}})$$

Since $\log\pi_{\text{old}}$ is a constant (frozen from rollout):

$$\nabla_\theta \mathcal{L} = -A \cdot \text{ratio} \cdot \nabla_\theta \log\pi_\theta$$

So the policy gradient reduces to: **gradient of the model's log probability, scaled by advantage × ratio**.

High positive advantage → push log probability up (make this token more likely).
Negative advantage → push log probability down (make this token less likely).

---

## Section 3: PyTorch Autograd and the Chain Rule

### 3.1 The Computational Graph

PyTorch's autograd records every differentiable operation during the forward pass into a **computational graph**. Each tensor stores a `grad_fn` — the recipe to compute its gradient.

```
W (model weights, requires_grad=True)
 │
 ▼  forward pass — every operation is recorded
logits  →  log_softmax  →  log_prob_new
                                 │
                    log_prob_old (constant, no grad)
                                 │
                         ratio = exp(new - old)
                                 │
                     loss = -ratio * advantage
                                 │
                         loss.backward()
                                 │
                           ∂loss/∂W  (stored in W.grad)
                                 │
                          optimizer.step()
                                 │
                          W ← W - lr · ∂loss/∂W
```

You never write the gradient formulas. PyTorch computes them automatically by traversing this graph backwards.

---

### 3.2 Step 1: $\frac{\partial \mathcal{L}}{\partial \log\pi_\theta}$

$$\mathcal{L} = -\exp(\log\pi_\theta - \underbrace{\log\pi_{\text{old}}}_{\text{constant}}) \cdot A$$

Let $x = \log\pi_\theta$ and $c = \log\pi_{\text{old}}$:

$$\frac{\partial \mathcal{L}}{\partial x} = -A \cdot \frac{\partial}{\partial x}\exp(x - c) = -A \cdot \exp(x - c) = -A \cdot \text{ratio}$$

**Note:** This equals $\mathcal{L}$ itself (since $\mathcal{L} = -\text{ratio} \cdot A$).
This is **not a typo** — it is because $\frac{d}{dx}e^x = e^x$.
The exponential function is its own derivative. When the loss is linear in $e^x$, the gradient w.r.t. $x$ equals the function value.

---

### 3.3 Step 2: $\frac{\partial \log p_k}{\partial z_i}$ — Through Softmax + Log

The model outputs logits $z$. The log probability of the correct token $k$ is:

$$\log p_k = \log \text{softmax}(z)_k = z_k - \log\!\sum_j \exp(z_j)$$

Taking the derivative with respect to logit $z_i$:

$$\frac{\partial \log p_k}{\partial z_i} = \mathbf{1}[i = k] - p_i$$

where $p_i = \text{softmax}(z)_i$.

The $-p_i$ term comes from differentiating the normalization $-\log\sum_j \exp(z_j)$. This is a direct consequence of the identity:

$$\nabla_z \log\!\left(\sum_j e^{z_j}\right) = \text{softmax}(z)$$

i.e., the gradient of log-sum-exp is the softmax vector. Component-wise:

$$\frac{\partial}{\partial z_i} \log \sum_j e^{z_j} = \frac{e^{z_i}}{\sum_j e^{z_j}} = p_i$$

So differentiating $-\log\sum\exp$ gives $-p_i$.

Explicitly:
- **If $i = k$ (the correct token):** $\frac{\partial}{\partial z_k}(z_k - \log\sum\exp) = 1 - p_k$
- **If $i \neq k$ (any other token):** $\frac{\partial}{\partial z_i}(- \log\sum\exp) = -p_i$

So the gradient is largest for the correct token ($1 - p_k$) and negative for all others ($-p_i$).

> **Note on sign:** In code, `compute_log_probs` returns `-cross\_entropy`, and the cross-entropy gradient is $p_i - \mathbf{1}[i=k]$ (opposite sign). PyTorch handles this correctly through the `grad_fn`.

In practice, `fused_vocab_parallel_cross_entropy` in Megatron computes this analytically (numerically stable, fused kernel) rather than via naive autograd.

---

### 3.4 Step 3: $\frac{\partial z}{\partial W}$ — Through the Transformer

The final language model head is a linear projection:

$$z = W_{\text{vocab}} \cdot h_L$$

where $h_L$ is the last hidden state (output of the transformer layers). So:

$$\frac{\partial z_j}{\partial W_{\text{vocab}}[j, d]} = h_{L,d}$$

For the deeper transformer layers (attention, FFN), each weight matrix $W_\ell$ has:

$$\frac{\partial \mathcal{L}}{\partial W_\ell} = \frac{\partial \mathcal{L}}{\partial h_{\ell+1}} \cdot \frac{\partial h_{\ell+1}}{\partial W_\ell}$$

**One layer deeper — explicit two-step unroll:**

Take two adjacent layers $L$ (last) and $L-1$. The last layer computes $z = W_{\text{vocab}} \cdot h_L$, and the layer before it computes $h_L = f(W_L \cdot h_{L-1})$ (where $f$ is some nonlinearity, e.g. SiLU in the FFN).

Gradient w.r.t. $W_L$:

$$\frac{\partial \mathcal{L}}{\partial W_L} = \underbrace{\frac{\partial \mathcal{L}}{\partial z}}_{\text{from Sec 3.2+3.3}} \cdot \underbrace{\frac{\partial z}{\partial h_L}}_{= W_{\text{vocab}}^T} \cdot \underbrace{\frac{\partial h_L}{\partial W_L}}_{= f'(\cdot)\, h_{L-1}^T}$$

And one step further to $W_{L-1}$:

$$\frac{\partial \mathcal{L}}{\partial W_{L-1}} = \underbrace{\frac{\partial \mathcal{L}}{\partial z}}_{\text{from above}} \cdot \underbrace{\frac{\partial z}{\partial h_L}}_{W_{\text{vocab}}^T} \cdot \underbrace{\frac{\partial h_L}{\partial h_{L-1}}}_{W_L^T \cdot f'(\cdot)} \cdot \underbrace{\frac{\partial h_{L-1}}{\partial W_{L-1}}}_{f'(\cdot)\, h_{L-2}^T}$$

The pattern: each additional layer prepends one more $W_\ell^T \cdot f'(\cdot)$ factor. This is why deep networks suffer from vanishing/exploding gradients — you are multiplying many such matrices together.

This continues recursively all the way to the first layer. PyTorch's autograd does this automatically by following the chain of `grad_fn` objects recorded during the forward pass.

---

### 3.5 The Full Chain

$$\frac{\partial \mathcal{L}}{\partial W_{\text{vocab}}} = \underbrace{\frac{\partial \mathcal{L}}{\partial \log p_k}}_{\text{Sec 3.2: } {-A \cdot \text{ratio}}} \cdot \underbrace{\frac{\partial \log p_k}{\partial z_i}}_{\text{Sec 3.3: }\mathbf{1}[i=k] - p_i} \cdot \underbrace{\frac{\partial z_i}{\partial W_{\text{vocab}}}}_{\text{Sec 3.4: hidden state } h_L}$$

And for a deeper layer $W_{L-1}$, one extra factor is inserted:

$$\frac{\partial \mathcal{L}}{\partial W_{L-1}} = \underbrace{\frac{\partial \mathcal{L}}{\partial \log p_k}}_{-A \cdot \text{ratio}} \cdot \underbrace{\frac{\partial \log p_k}{\partial z_i}}_{\mathbf{1}[i=k] - p_i} \cdot \underbrace{\frac{\partial z_i}{\partial h_L}}_{W_{\text{vocab}}^T} \cdot \underbrace{\frac{\partial h_L}{\partial h_{L-1}}}_{W_L^T \cdot f'} \cdot \underbrace{\frac{\partial h_{L-1}}{\partial W_{L-1}}}_{h_{L-2}^T}$$

PyTorch computes each factor automatically. You only write:

```python
loss = compute_policy_gradient_loss(...)   # the formula from Section 2
loss.backward()                            # autograd handles everything above
optimizer.step()                           # W ← W - lr * W.grad
```

The "policy gradient" is not a special gradient algorithm — it is a specific choice of loss function plugged into the same standard PyTorch training loop used for any neural network.

---

### 3.6 How PyTorch Saves Gradients During the Chain Rule

PyTorch uses a **dynamic computational graph** built during the forward pass. It does not save gradients during the forward pass — it saves the **inputs/outputs needed to compute gradients later**.

#### Forward pass: building the graph

Every tensor operation attaches a `grad_fn` node to the output tensor:

```python
h  = W @ x        # h.grad_fn  = MmBackward0
h2 = h.relu()     # h2.grad_fn = ReluBackward0
loss = h2.sum()   # loss.grad_fn = SumBackward0
```

Each node stores a reference to its parent nodes, forming a directed acyclic graph (DAG). The graph is rebuilt from scratch on every forward pass (dynamic).

#### What each node saves via `save_for_backward()`

Each op saves only what it needs to compute its own gradient — not the full gradient:

| Operation | What is saved | Why |
|---|---|---|
| `W @ x` (MmBackward) | `W`, `x` | needs both to compute `dL/dW = dL/dh · xᵀ` and `dL/dx = Wᵀ · dL/dh` |
| `relu(h)` (ReluBackward) | boolean mask `h > 0` | gradient is `1` where active, `0` elsewhere |
| `exp(x)` (ExpBackward) | **output** `eˣ` | because `d/dx eˣ = eˣ` — cheaper to save the output than recompute |
| `log_softmax` | output probabilities `p` | gradient is `1 - p` for the chosen token, `-p` elsewhere (Sec 3.3) |

#### Backward pass: reverse topological traversal

`loss.backward()` walks the DAG in reverse order, computing and passing gradients downstream:

```
loss.grad_fn (SumBackward)    → sends 1.0 downstream
      ↓
h2.grad_fn  (ReluBackward)    → multiplies by f'(h), passes to h
      ↓
h.grad_fn   (MmBackward)      → dL/dW = incoming_grad · xᵀ  → accumulates into W.grad
                                  dL/dx = Wᵀ · incoming_grad  → passes further back
```

**Intermediate gradients are not stored.** They are computed on the fly and immediately forwarded to the parent node. Only **leaf tensor** `.grad` fields (i.e. `W.grad`) are accumulated and kept after backward.

#### Connection to the GRPO chain rule in Sec 3.5

`log_prob_old` is a plain Python float — detached, no `grad_fn`. When you compute:

```python
ratio = (log_prob_new - log_prob_old).exp()
```

only `log_prob_new` has a `grad_fn`, so the graph only flows through the **current policy weights**. `log_prob_old` is treated as a constant, exactly as the PPO algorithm intends.

#### Why gradients can vanish or explode

Each `MmBackward` node multiplies the incoming gradient by $W_\ell^T$. For a model with $L$ layers, backward traversal multiplies $L$ such matrices in sequence (Sec 3.4). If their spectral radii are all $> 1$ the product explodes; all $< 1$ it vanishes. This is why slime training uses gradient clipping (`--max-grad-norm`) — it caps $\|\nabla W\|$ before `optimizer.step()`.
