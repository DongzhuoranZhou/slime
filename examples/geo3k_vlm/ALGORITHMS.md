# Training Algorithms in geo3k_vlm

Two scripts, two completely different algorithms. Neither uses trajectories from an external teacher model.

---

## 1. SFT — Supervised Fine-Tuning (`run_geo3k_vlm_sft.sh`)

### What is the "trajectory"?

There is no generated trajectory. The dataset is pre-formatted offline:

```python
# from README data-prep step
sample["messages"] = [
    {"role": "user",      "content": sample["problem"]},
    {"role": "assistant", "content": f"Answer: \\boxed{{{sample['answer']}}}"},
]
```

The assistant turn is just the ground-truth answer wrapped in `\boxed{}`. No reasoning chain,
no model generation, no SGLang involved at all (`--debug-train-only` suppresses engine startup).

### The rollout is a no-op

`slime.rollout.sft_rollout.generate_rollout` does **not** call a model. It:

1. Pulls pre-formatted `(prompt, response)` pairs from `data_buffer` — these are the ground-truth parquet rows.
2. Tokenizes the full conversation with `MultiTurnLossMaskGenerator`.
3. Computes `loss_mask`: **1 for assistant response tokens, 0 for everything else** (prompt, system, padding).
4. Sets `sample.reward = 0` — reward is unused; there is no RL signal.
5. Nothing is saved to disk; samples are passed directly in RAM to the trainer.

No SGLang engine is started. The flag `--debug-train-only` suppresses all rollout engine initialization, so the entire inference stack is skipped. The "rollout step" is purely a tokenization pass.

### Algorithm: Negative Log-Likelihood (NLL / MLE)

Given a sample with prompt tokens $x$ and response tokens $y = (y_1, \ldots, y_T)$, the loss is:

$$
\mathcal{L}_{\text{SFT}}(\theta) = -\frac{1}{N} \sum_{n=1}^{N} \frac{1}{|R_n|} \sum_{t \in R_n} \log \pi_\theta(y_t \mid x_n, y_{<t})
$$

where:
- $N$ = number of samples in the mini-batch
- $R_n$ = set of response token positions for sample $n$ (prompt tokens are masked out via `loss_mask`)
- $\pi_\theta(y_t \mid \cdot)$ = the model's predicted probability for the ground-truth token $y_t$

The per-sample mean is summed, not averaged, across the batch (controlled by `calculate_per_token_loss`).

### Code call chain

```
train_async.py
  └── actor_model.async_train(rollout_id, rollout_data_ref)
        └── loss_function(args, batch, num_microbatches, logits)      # loss.py:943
              └── [args.loss_type == "sft_loss"]
                    └── sft_loss_function(args, batch, logits, sum_of_sample_mean)  # loss.py:892
                          └── log_probs = get_log_probs_and_entropy(logits, ...)
                                loss = -sum_of_sample_mean(log_probs)
```

Key argument in the shell script:
```bash
--loss-type sft_loss
--rollout-function-path slime.rollout.sft_rollout.generate_rollout
```

`sft_rollout.generate_rollout` only tokenizes and applies the loss mask — it does **not** call SGLang.

---

## 2. GRPO — Group Relative Policy Optimization (`run_geo3k_vlm.sh`)

### What is the "trajectory" here?

Here SGLang **does** generate trajectories. The model being trained samples $G$ responses
per prompt at inference time. These on-policy responses are then scored by a math verifier
(binary reward $r \in \{0, 1\}$). This is the only place SGLang is involved.

### Algorithm

**Step 1 — Group reward normalization** (`rollout.py:641-660`)

For each prompt, $G$ responses are sampled. Their rewards are normalized within the group:

$$
\hat{r}_i = \frac{r_i - \mu_g}{\sigma_g + \epsilon}, \qquad
\mu_g = \frac{1}{G}\sum_{i=1}^{G} r_i, \quad
\sigma_g = \sqrt{\frac{1}{G}\sum_{i=1}^{G}(r_i - \mu_g)^2}
$$

controlled by `--rewards-normalization` and `--grpo-std-normalization`.

**Step 2 — Advantage broadcast** (`ppo_utils.py:201-208`)

The scalar advantage $\hat{r}_i$ is broadcast to every response token:

$$
A_i(t) = \hat{r}_i \quad \forall\, t \in \text{response}_i
$$

(`get_grpo_returns` ignores KL here; KL penalty is applied separately in the policy loss.)

**Step 3 — PPO-clip policy gradient** (`ppo_utils.py:125-148`, `loss.py:704-706`)

$$
\rho_i(t) = \frac{\pi_\theta(y_t \mid x)}{\pi_{\theta_{\text{old}}}(y_t \mid x)}
= \exp\!\bigl(\log\pi_\theta - \log\pi_{\theta_{\text{old}}}\bigr)
$$

$$
\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{N}\sum_{i=1}^{N} \frac{1}{|R_i|}
\sum_{t \in R_i}
\min\!\Bigl(
  \rho_i(t)\,A_i(t),\;
  \text{clip}\bigl(\rho_i(t),\,1-\varepsilon,\,1+\varepsilon_{\text{high}}\bigr)\,A_i(t)
\Bigr)
$$

where $\varepsilon$ = `--eps-clip`, $\varepsilon_{\text{high}}$ = `--eps-clip-high`.

### Code call chain

```
train_async.py
  ├── rollout_manager.generate(rollout_id)          # generates G responses via SGLang
  │     ├── sglang_rollout.generate_rollout()       # model samples trajectories
  │     ├── reward_model scores each response       # r ∈ {0,1}
  │     └── _post_process_rewards()                 # group mean/std normalization
  │
  └── actor_model.async_train(rollout_id, rollout_data_ref)
        └── loss_function(args, batch, ...)          # loss.py:943
              └── [args.loss_type == "policy_loss"]
                    ├── compute_advantages_and_returns()   # loss.py:400
                    │     └── get_grpo_returns(rewards, kl)
                    └── policy_loss_function()             # loss.py:~600
                          ├── ppo_kl = log π_old - log π_θ
                          └── compute_policy_loss(ppo_kl, advantages, ε, ε_high)
```

Key arguments in the shell script:
```bash
--advantage-estimator grpo
--loss-type policy_loss
--n-samples-per-prompt 8          # G = 8 responses per prompt
--rewards-normalization
--grpo-std-normalization
```

---

## Summary

| | SFT | GRPO |
|---|---|---|
| Source of "trajectory" | Ground-truth answer from dataset | Model generates on-policy via SGLang |
| SGLang used? | No | Yes |
| Loss | $-\log \pi_\theta(y^*)$ (NLL on GT) | PPO-clip on group-normalized advantages |
| Reward signal | None (supervised) | Binary math verifier (0/1) |
| Reference model | None | Yes (KL penalty $\beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})$) |
