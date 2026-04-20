---
layout: blog-post.njk
title: Speculative Decoding
date: 2026-03-24
tags: posts
---

Standard autoregressive decoding generates one token per forward pass of the model. Each forward pass has the same cost regardless of whether the token is a predictable word like "the" or a surprising one. Leviathan et al.[^1] and Chen et al.[^2] independently observed that a small **draft model** can cheaply guess several tokens ahead, and a large **target model** can verify multiple guesses in parallel — turning sequential token generation into a bet on the draft model's accuracy.

The draft model is small and fast. The target model is large and slow — but a single forward pass of the target model can process multiple token positions in parallel (just like prefill). Speculative decoding exploits this:

```
Past sequence: [T1, T2, T3]

Draft model (serial, but cheap):
  A' = draft(T1, T2, T3)
  B' = draft(T1, T2, T3, A')
  C' = draft(T1, T2, T3, A', B')

Target model (one parallel forward pass):
  A = target(T1, T2, T3)         ← verify A'
  B = target(T1, T2, T3, A')     ← verify B' (using draft's A')
  C = target(T1, T2, T3, A', B') ← verify C' (using draft's A', B')
```

The key insight: the target model computes A, B, and C in parallel in a single forward pass. This is the same parallel computation the target model already does during prefill — processing multiple token positions at once. It takes the draft model's guesses (A', B', C') as input and produces the target distribution at each position simultaneously. Three tokens verified for roughly the cost of one.

**Acceptance:** The target model checks each draft token left to right:

- If A = A': accept. Check B.
- If B = B': accept. Check C.
- If C = C': accept. All 3 tokens verified in one target forward pass.

**Rejection:** If any token doesn't match, discard it and everything after it:

- If A ≠ A': reject A', B', C'. Keep only A. (1 token from 1 target forward pass — no worse than standard decoding.)
- If A = A' but B ≠ B': reject B', C'. Keep A, B. (2 tokens from 1 target forward pass.)

In the best case, one expensive target forward pass produces multiple tokens. In the worst case, it produces exactly one — the same as standard decoding. Speculative decoding never makes things worse, it can only help. Note that this only holds when inference is memory-bound — speculative decoding increases total arithmetic operations (the draft model's work plus verifying multiple positions), but since autoregressive decoding is bottlenecked by memory bandwidth, the extra compute fits within what's already being spent on a single forward pass.

**Note on sampling:** The description above uses exact token matching (A = A'), which is how verification works under greedy decoding. More generally, LLM decoding samples from a probability distribution (see [Token Prediction](/blog/flashinfer/#token-prediction)). Leviathan et al.[^1] generalize verification to this setting with *speculative sampling*: accept a draft token with probability `min(1, p(x)/q(x))`, where p is the target distribution and q is the draft distribution. On rejection, sample from the adjusted distribution `norm(max(0, p - q))` — this covers exactly the probability mass not already accounted for by accepted tokens, ensuring the output distribution is identical to sampling from p alone.

## Using a tree model over a draft model

In standard speculative decoding, the draft model generates A', B', C' sequentially — B' depends on A', C' depends on A' and B'. Even though the draft model is small, this sequential dependency is a bottleneck. MEDUSA[^4] eliminates the draft model entirely by adding lightweight **prediction heads** directly to the target model.

Recall from [Token Prediction](/blog/flashinfer/#token-prediction) that the LM head predicts the next token from the final hidden state Z[-1]. MEDUSA adds extra heads alongside the LM head — each is a small feed-forward network (`linear(d→d) → SiLU → residual → linear(d→vocab) → softmax`) with its own independent weights. Each head takes the same Z[-1] as input and outputs the same thing as the LM head — a probability distribution over the vocabulary — but head k's weights are trained to predict the token at position t+k+1 instead of t+1:

```
Single forward pass of [T1, T2, T3] → Z[-1] →
    LM head        → A' (position 4)
    MEDUSA head 1  → B' (position 5)
    MEDUSA head 2  → C' (position 6)
```

All heads use the same Z[-1] — no sequential dependency. But this is an approximation: head 1 predicts B' without knowing A', so it is less accurate than the sequential draft model above.

Because each head predicts independently and is less accurate, the top-1 prediction is often wrong. So we take the top-k predictions from each head. With top-2 from each:

```
- LM head:  A₁, A₂
- Head 1:   B₁, B₂
- Head 2:   C₁, C₂

We create a tree of all possible combinations:

         root
        /    \
      A₁      A₂
     / \     / \
    B₁  B₂  B₁  B₂
   /|  /|  /|  /|
  C₁C₂C₁C₂C₁C₂C₁C₂
```

All candidate tokens are packed into a single forward pass of the target model, using an attention mask that encodes the tree structure. To illustrate, consider a simpler two-level tree (top-2 from LM head, top-2 from head 1):

```
        [T1, T2, T3]  (past context, in KV cache)
        /          \
      A₁            A₂
     / \           / \
    B₁  B₂       B₃  B₄
```

The 6 candidate tokens are packed into a flat sequence: [A₁, A₂, B₁, B₂, B₃, B₄]. The tree attention mask (1 = can attend, 0 = cannot):

```
       A₁  A₂  B₁  B₂  B₃  B₄
  A₁ [  1   0   0   0   0   0 ]
  A₂ [  0   1   0   0   0   0 ]
  B₁ [  1   0   1   0   0   0 ]  ← B₁ attends to A₁ (parent)
  B₂ [  1   0   0   1   0   0 ]  ← B₂ attends to A₁ (parent)
  B₃ [  0   1   0   0   1   0 ]  ← B₃ attends to A₂ (parent)
  B₄ [  0   1   0   0   0   1 ]  ← B₄ attends to A₂ (parent)
```

(All tokens also attend to T1, T2, T3 via the KV cache.)

Each token can only attend to its ancestors in the tree. B₁ and B₂ attend to A₁ (their parent) but not to A₂ or any B on A₂'s branch. This is neither a standard causal mask (which would let B₃ attend to A₁, B₁, B₂) nor a dense mask — it is a custom sparse mask determined by the tree topology.

Recall from [Attention Mechanism](/blog/flashinfer/#attention-mechanism) that attention computes `A = Q · K^T`, then masks, then softmax. The tree mask replaces the standard causal mask: where the mask has a 0, the corresponding attention score is set to -∞ before softmax, so it contributes nothing. The mask is stored as a dense boolean matrix and applied element-wise — it is not stored sparsely, even though it is sparse.

Because of the tree mask, each token's attention output reflects only its ancestral path — B₁ sees [T1, T2, T3, A₁], while B₃ sees [T1, T2, T3, A₂]. So the logits at each position represent the target model's prediction for the *next* token given that specific path. Verification walks the tree:

1. The target model's logits from the past context predict position t+1. Check: does A₁ or A₂ match? Say A₁ matches.
2. The logits at the A₁ position predict position t+2 (given path [T1, T2, T3, A₁]). Check: does B₁ or B₂ match?
3. Continue until a mismatch (truncate) or a leaf (all accepted).

This is the same principle as the earlier verification diagram — the target model processes all candidates in parallel and produces the target distribution at each position. The tree mask ensures each candidate sees only its own ancestral path, so all branches are verified simultaneously in a single forward pass.

## Related

* Google's speculative decoding retrospective [^3].

## Acknowledgements

Thanks to [Tarindu Jayatilaka](https://tarinduj.github.io/) for discussions about Speculative Decoding.

---

[^1]: Leviathan, Y., Kalman, M., & Matias, Y. (2023). [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192). *Proceedings of the 40th International Conference on Machine Learning (ICML)*.

[^2]: Chen, C., Borgeaud, S., Irving, G., Lespiau, J.-B., Sifre, L., & Jumper, J. (2023). [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318). *Technical Report*.

[^3]: Leviathan, Y., Kalman, M., & Matias, Y. (2024). [Looking back at speculative decoding](https://research.google/blog/looking-back-at-speculative-decoding/). *Google Research Blog*.

[^4]: Cai, T., Li, Y., Geng, Z., Peng, H., Lee, J. D., Chen, D., & Dao, T. (2024). [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774). *Proceedings of the 41st International Conference on Machine Learning (ICML)*
