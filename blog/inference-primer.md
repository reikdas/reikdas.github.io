---
layout: blog-post.njk
title: LLM Inference Primer
date: 2026-01-21
tags: posts
---
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

## Attention Mechanism[^2]

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

For the first step of inference (called the prompt/prefill phase), we compute Q = X · W_Q for all tokens. However, we only need the last row of Q in subsequent steps (called the autoregressive generation phase/decode phase).

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

**Insight:** LLM inference mostly involves vector-matrix multiplication instead of matrix-matrix multiplication.

## Batched Inference

The autoregressive generation phase is **memory-bound**: loading model weights (W_Q, W_K, W_V, W_vocab) from GPU memory dominates the cost, while the actual computation is relatively cheap. Processing one sequence at a time underutilizes the GPU.

**Batching** processes multiple sequences simultaneously. Since all sequences share the same model weights, we load the weights once and apply them to all sequences in the batch. This amortizes the memory bandwidth cost and increases GPU utilization. Each sequence in the batch maintains its own KV cache, as they contain different tokens. The sequences do not need to be related.

**Shared prefix optimization:** When sequences share a common prefix (e.g., a system prompt), they can potentially share the KV cache for that prefix portion, reducing memory usage.

## PagedAttention[^3]

However, sequences cannot share the KV cache during the autoregressive generation phase. The KV cache is typically stored as a contiguous block of memory. Consider two prompts A and B: once they diverge beyond their shared prefix, A and B require different KV cache values. Since the KV cache is stored in contiguous memory, A and B cannot write to the existing shared KV cache and must maintain separate copies.

Additionally, traditional systems suffer from memory fragmentation. In such systems:

- A request arrives, but the system does not know the output length
- The system pre-allocates contiguous memory for the maximum possible length (e.g., 2048 tokens)
- The actual sequence ends up being 500 tokens
- 1548 slots are wasted (internal fragmentation)
- Different requests have different maximum lengths, creating gaps between allocations (external fragmentation). Free memory may exist, but if it is not contiguous, a new KV cache cannot be allocated. One might consider using pointer-based allocation (as CPU allocators like `malloc` do) to utilize scattered free memory, but this degrades GPU performance—GPUs are optimized for coalesced memory access, and random indirection through pointers for each KV cache access is expensive. [^6]
- Result: Very low percentage of KV cache memory actually stores useful data

PagedAttention addresses these issues by splitting the KV cache into fixed-size blocks of non-contiguous memory, with a block table that enables lookup into each block.

PagedAttention:

- Allocates blocks on demand as tokens are generated
- A sequence using 500 tokens allocates exactly ⌈500/B⌉ blocks
- All blocks have the same size, eliminating external fragmentation
- The only waste is unfilled slots in the last block
- Result: High memory utilization

This block-based indirection also enables sequences A and B to share the same KV cache blocks for their common prefix.

## RadixAttention[^4]

PagedAttention's primary motivation was solving memory fragmentation—its contribution is efficient memory management for concurrent requests. It could theoretically retain KV cache after requests complete for future reuse, but its per-sequence block tables don't provide a way to find prefix matches: when a new request arrives, how do you efficiently determine if any cached prefix matches?

RadixAttention solves this with a **radix tree**: a global data structure for cross-request KV cache reuse over time. The tree works as follows:

- Each **edge** is labeled with a sequence of tokens (compressed representation—edges with no branching are merged)
- Each **node** stores a pointer to the KV cache blocks for the token sequence from root to that node
- To find a matching prefix, traverse from the root following edges that match the request's tokens
- The traversal finds the longest matching prefix in O(n) time (where n is the request length)
- An LRU eviction policy removes least-recently-used leaf nodes when memory fills up

**Radix Tree:** When a new sequence partially matches an existing edge, the radix tree splits the edge. For example: if the tree has edge [A,B,C,D,E] and a new request needs [A,B,C,F,G], the edge splits into [A,B,C] → node → [D,E], with a new edge [F,G] branching from that node.

RadixAttention reuses the existing KV cache for the shared prefix [A,B,C] and only computes new KV cache blocks for the divergent suffix [F,G].

Contiguous KV cache storage cannot support this. A radix tree requires sharing and extending prefixes dynamically—if request A caches tokens [1,2,3] and request B needs [1,2,3,4,5], B must extend A's cache without copying it. Contiguous storage would require pre-allocating space or copying the entire prefix.

## Paged Attention[^3]

*More on this topic coming soon...*

---

[^1]: In practice, A = (Q · K^T) / √d_k is used where √d_k is a scaling factor. This scaling is for numerical stability: as the dimension d_k grows, the dot products can become very large, pushing the softmax into regions with very small gradients (saturation). The scaling keeps the variance of the dot products constant, preventing this issue.
    
[^2]: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). *Advances in Neural Information Processing Systems (NeurIPS)*.
    
[^3]: Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J., Zhang, H., & Stoica, I. (2023). [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180). *Proceedings of the 29th Symposium on Operating Systems Principles (SOSP)*.
