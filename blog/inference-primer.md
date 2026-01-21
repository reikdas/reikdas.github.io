---
layout: blog-post.njk
title: LLM Inference Primer
description: An introduction to how transformer-based large language models perform inference, covering attention mechanisms, KV caching, and optimization techniques.
date: 2026-01-21
tags: posts
---

We first need to understand how transformers[^2] work (the first part of this document).

## Sequence Representation

A **prompt** is a sequence of text. The sequence is broken into **tokens** (tokenization is implementation-specific). Each token is converted into a vector representation of size `d_model` (an embedding).

Let `seq_len` be the number of tokens in a sequence. The sequence is represented as a dense matrix of shape `(seq_len, d_model)`, where each row is the representation of a token.

Let's call this sequence matrix **X**:

```
     [  ]
     [  ]
X =  [  ]  (seq_len × d_model)
     [  ]
     [  ]
```

## Attention Mechanism

The pre-trained model provides weight matrices: **W_Q**, **W_K**, **W_V**, and **W_vocab**. These are constant for the entire inference. (How are they generated? There will be a training primer later :p)

**Query, Key, and Value projections:**
- Q = X · W_Q
- K = X · W_K  
- V = X · W_V

**Attention computation:**
- A = Q · K^T[^1]
- A = mask(A)  (zero out A[i, j] where j > i - a future token j cannot contribute to a previous token i; this is required for correctness, not just performance)
- A = softmax(A)  (normalize each row to sum to 1)

**Why softmax?** After softmax, each row of A sums to 1, meaning A[i, j] represents the fraction of token j's value vector that contributes to token i's representation.

**Output:**
- Z = A · V

Z is the updated intermediate token representation.

## Token Prediction

For predicting the next token, we use only the last row of Z (the representation of the last token in the existing sequence):

- logits = Z[-1] · W_vocab  (vector of shape `vocab_size` - now each element is a token, instead of each row being a token)
- probabilities = softmax(logits)  (vector of shape `vocab_size` where each element is the probability that the corresponding vocabulary token is the next token; all elements sum to 1)
- next_token = sample(probabilities)  (sample a token from this probability distribution)

## Next Generation Step

For the next round of token prediction:
1. Convert `next_token` to its embedding vector (size `d_model`)
2. Append this new embedding as a new row to X: X_new = [X; next_token_embedding]
3. X_new now has shape `(seq_len + 1, d_model)`
4. Re-run the attention mechanism and token prediction with X_new

This process repeats, generating one token at a time until a stopping condition (e.g., end-of-sequence token or maximum length).

## Query Optimization

For the first step of inference (called the prompt phase), we compute Q = X · W_Q for all tokens. However, we only need the last row of Q in subsequent steps (called the autoregressive generation phase).

Let `X_last` be the last token that was predicted and appended to the end of X. For subsequent steps, we can optimize the calculation:

- Q = X_last · W_Q  (now a vector of shape `d_model` instead of a matrix)

The rest of the computation remains the same:
- A = mask(Q · K^T)
- A = softmax(A)
- Z = A · V

## KV Cache

Since W_Q, W_K, W_V, and W_vocab are constant during inference, we can cache previously computed K and V values. When a new token is appended to X, we only need to:

1. Compute K_new and V_new for the new token
2. Append these to the cached K and V from previous tokens

This avoids recomputing K and V for all previous tokens at each step.

**Why don't we cache Q?** As shown in the query optimization above, we only need Q for the last token, so there's no need to cache the Q matrix. 

## Batched inference

To improve inference performance, often times several batches are decoded at the same time. And often times, these batches have a shared prefix (for eg. a shared system prompt). In this case, several batches can share the KV cache for the prompt phase, but cannot for the autoregressive generation phase.

## Paged Attention[^3]

*More on this topic coming soon...*

---

[^1]: In practice, A = (Q · K^T) / √d_k is used where √d_k is a scaling factor. This scaling is for numerical stability: as the dimension d_k grows, the dot products can become very large, pushing the softmax into regions with very small gradients (saturation). The scaling keeps the variance of the dot products constant, preventing this issue.

[^2]: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). *Advances in Neural Information Processing Systems (NeurIPS)*.

[^3]: Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J., Zhang, H., & Stoica, I. (2023). [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180). *Proceedings of the 29th Symposium on Operating Systems Principles (SOSP)*.
