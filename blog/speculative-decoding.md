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

In the best case, one expensive target forward pass produces multiple tokens. In the worst case, it produces exactly one — the same as standard decoding. Speculative decoding never makes things worse, it can only help.

## Related

* Google's speculative decoding retrospective [^3].

---

[^1]: Leviathan, Y., Kalman, M., & Matias, Y. (2023). [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192). *Proceedings of the 40th International Conference on Machine Learning (ICML)*.

[^2]: Chen, C., Borgeaud, S., Irving, G., Lespiau, J.-B., Sifre, L., & Jumper, J. (2023). [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318). *Technical Report*.

[^3]: Leviathan, Y., Kalman, M., & Matias, Y. (2024). [Looking back at speculative decoding](https://research.google/blog/looking-back-at-speculative-decoding/). *Google Research Blog*.
