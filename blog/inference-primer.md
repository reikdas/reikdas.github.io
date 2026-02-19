---
layout: blog-post.njk
title: LLM Inference Primer (or fastest ramp-up to FlashInfer)
date: 2026-01-21
tags: posts
---
## Sequence Representation

A **prompt** is a sequence of text. The sequence is broken into **tokens** (tokenization is implementation-specific). Each token is converted into a vector representation of size `d_model` (an embedding).

Positional information is also encoded into each token's vector (e.g., via sinusoidal functions or learned embeddings), so the model knows where each token appears in the sequence. The specifics vary by model, but the result is the same: each token's embedding encodes both what the token is and where it appears.

Let `seq_len` be the number of tokens in a sequence. The sequence is represented as a dense matrix of shape `(seq_len, d_model)`, where each row is the embedding of a token.

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
- Z = X + Z  (residual connection — each token's original representation carries through)

Z is the attention output. 

**Implementation detail:** In practice, the attention computation above is repeated multiple times in sequence, each with its own set of W_Q, W_K, W_V weight matrices. The output Z of one repetition becomes the input X to the next. The final repetition's output is used for token prediction. 

## Token Prediction

For predicting the next token, we use only the last row of Z (the representation of the last token in the existing sequence):

- logits = Z[-1] · W_vocab  (vector of shape `vocab_size` - now each element is a token, instead of each row being a token)
- probabilities = softmax(logits)  (vector of shape `vocab_size` where each element is the probability that the corresponding vocabulary token is the next token; all elements sum to 1)
- next_token = sample(probabilities)  (sample a token from this probability distribution)

**Temperature** controls how the probability distribution is shaped before sampling. The logits are divided by a temperature parameter T before softmax: `probabilities = softmax(logits / T)`. With T < 1, the distribution becomes sharper (the model is more likely to pick the highest-probability token). With T > 1, the distribution becomes flatter (more random). At T → 0, sampling always picks the highest-probability token (greedy decoding).

## Next Generation Step

For the next round of token prediction:

1. Convert `next_token` to its embedding vector (size `d_model`)
2. Append this new embedding as a new row to X: X_new = [X; next_token_embedding]
3. X_new now has shape `(seq_len + 1, d_model)`
4. Re-run the attention mechanism and token prediction with X_new

This process repeats, generating one token at a time until a stopping condition (e.g., end-of-sequence token or maximum length).

<!-- Claude: Do we need to briefly mention residual connection here -->

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

**Example:** Consider two prompts with a shared prefix (using page size = 1 token):

- Prompt A: "The cat sat" (3 tokens)
- Prompt B: "The cat ran" (3 tokens)

"The" and "cat" are shared. "sat" is unique to A, "ran" is unique to B.

```
Block tables:
  A: [P0, P1, P2]
  B: [P0, P1, P3]   ← P0, P1 shared via reference counting

Physical pages:
  P0: "The"  (shared)
  P1: "cat"  (shared)
  P2: "sat"  (A-only)
  P3: "ran"  (B-only)
```

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

**Example:** Using the same prompts as the PagedAttention example:

```
Root
  └─"The"─→ [R0]
              └─"cat"─→ [R1]
                         ├─"sat"─→ [R2]
                         └─"ran"─→ [R3]

A uses: R0 → R1 → R2
B uses: R0 → R1 → R3
```

Contiguous KV cache storage cannot support this. A radix tree requires sharing and extending prefixes dynamically—if request A caches tokens [1,2,3] and request B needs [1,2,3,4,5], B must extend A's cache without copying it. Contiguous storage would require pre-allocating space or copying the entire prefix.

PagedAttention and RadixAttention are two points in a much larger space of KV cache optimizations — covering memory efficiency, prefix reuse, speculative decoding, quantized caches, and distributed storage across nodes.[^7] We focus on these two because they directly motivate the attention kernel design in FlashInfer.

## FlashInfer[^5]

Both PagedAttention and RadixAttention store the KV-cache in non-contiguous memory, but their data structures differ. FlashInfer uses a block sparse matrix as a lookup table that can represent both styles (and more).

The block sparse matrix maps query positions (rows) to KV positions (columns). Each row stores which physical pages the kernel needs to load K and V from to compute attention (recall: K = X · W_K, V = X · W_V from Attention Mechanism, and only K and V are cached — see KV Cache). Q is computed on the fly for each position (see Query Optimization). During batched inference, queries from all requests are packed into a single contiguous array without padding (a ragged tensor). Consider a variation of the earlier example with different-length requests:

```
Prompt A: "The cat sat"      (3 tokens)
Prompt B: "The cat ran fast" (4 tokens)

Without ragged tensor (padded to max length):
  A: [Q_"The", Q_"cat", Q_"sat", <pad> ]
  B: [Q_"The", Q_"cat", Q_"ran", Q_"fast"]
  → stored as a 2 × 4 matrix, wasting one slot for A

Ragged tensor (packed, no padding):
  data:   [Q_"The", Q_"cat", Q_"sat", Q_"The", Q_"cat", Q_"ran", Q_"fast"]
           ← request A (3 tokens) →    ← request B (4 tokens) →
  indptr: [0, 3, 7]
```

`indptr` marks where each request starts and ends: request A = `data[0:3]`, request B = `data[3:7]`. Each entry is a vector of shape `d_model`. The rows of the block sparse matrix correspond to positions in this packed array.

The matrix is divided into blocks of size (Br, Bc):

- **Br** (block row size) = query tile size: how many query positions per row block. Br=1 during the decode phase (one new token), larger during the prefill phase.
- **Bc** (block column size) = KV page size: how many KV positions per column block. Determined by the KV-cache management algorithm.

**Example:** Using d_model = 2 for illustration (recall d_model from Sequence Representation). "The" and "cat" are the shared prefix. "sat" is unique to A, "ran" and "fast" are unique to B.

KV cache pages (page size = 1 token, same layout as the PagedAttention example):

```
  P0 ("The", shared):   K=[1,0],  V=[1,1]
  P1 ("cat", shared):   K=[0,1],  V=[2,0]
  P2 ("sat", A-only):   K=[1,1],  V=[0,1]
  P3 ("ran", B-only):   K=[1,-1], V=[1,0]
  P4 ("fast", B-only):  K=[0,-1], V=[0,1]
```

**FlashInfer's block sparse matrix** with Br=1, Bc=1. Each row is one query position, each column is one KV page. Each cell stores the page index to load K, V from. Empty cells mean "no attention":

```
              P0     P1     P2     P3     P4
              (The)  (cat)  (sat)  (ran)  (fast)
             shared shared A-only B-only B-only
            ┌──────┬──────┬──────┬──────┬──────┐
A, pos 0    │  P0  │      │      │      │      │
            ├──────┼──────┼──────┼──────┼──────┤
A, pos 1    │  P0  │  P1  │      │      │      │
            ├──────┼──────┼──────┼──────┼──────┤
A, pos 2    │  P0  │  P1  │  P2  │      │      │
            ├──────┼──────┼──────┼──────┼──────┤
B, pos 0    │  P0  │      │      │      │      │
            ├──────┼──────┼──────┼──────┼──────┤
B, pos 1    │  P0  │  P1  │      │      │      │
            ├──────┼──────┼──────┼──────┼──────┤
B, pos 2    │  P0  │  P1  │      │  P3  │      │
            ├──────┼──────┼──────┼──────┼──────┤
B, pos 3    │  P0  │  P1  │      │  P3  │  P4  │
            └──────┴──────┴──────┴──────┴──────┘
```

This matrix is stored as a lookup table — for each row, it records which columns (physical pages) to access:

```
A, pos 0 → pages [P0]
A, pos 1 → pages [P0, P1]
A, pos 2 → pages [P0, P1, P2]
B, pos 0 → pages [P0]
B, pos 1 → pages [P0, P1]
B, pos 2 → pages [P0, P1, P3]
B, pos 3 → pages [P0, P1, P3, P4]
```

**Kernel workflow for row "A, pos 2":**

Q for position 2 is computed on the fly: Q = [1,1]. The lookup table says: load KV from pages P0, P1, P2.

```
1. Load K from P0, P1, P2:  [[1,0], [0,1], [1,1]]
2. Attention scores = Q · K^T:
     [1,1]·[1,0] = 1,  [1,1]·[0,1] = 1,  [1,1]·[1,1] = 2
     → scores = [1, 1, 2]
3. Causal mask: all valid (positions 0, 1, 2 are all ≤ 2)
4. Softmax([1, 1, 2]) = [0.21, 0.21, 0.58]
5. Load V from P0, P1, P2:  [[1,1], [2,0], [0,1]]
6. Output = 0.21·[1,1] + 0.21·[2,0] + 0.58·[0,1] = [0.63, 0.79]
```

**Causal mask in the Br=1 case:** In step 3, the causal mask didn't mask anything — all loaded KV positions (0, 1, 2) were ≤ the query position (2). This is true for every row: "A, pos 0" only loads P0, "A, pos 1" loads P0 and P1, etc. With Br=1, the lookup table already encodes causality by only listing pages up to each query's position, making the causal mask a no-op.

**Optimization: Br=3.** During prefill, query positions within each request can be grouped into row blocks. Request A (3 tokens) fits in one block of Br=3. Request B (4 tokens) requires ⌈4/3⌉ = 2 blocks: [pos 0-2] and [pos 3]. Each filled block represents multiple query positions looking up KV from the same pages:

```
              P0     P1     P2     P3     P4
              (The)  (cat)  (sat)  (ran)  (fast)
             shared shared A-only B-only B-only
            ┌──────┬──────┬──────┬──────┬──────┐
A, pos 0-2  │  P0  │  P1  │  P2  │      │      │
            ├──────┼──────┼──────┼──────┼──────┤
B, pos 0-2  │  P0  │  P1  │      │  P3  │      │
            ├──────┼──────┼──────┼──────┼──────┤
B, pos 3    │  P0  │  P1  │      │  P3  │  P4  │
            └──────┴──────┴──────┴──────┴──────┘
```

The kernel loads Q for all positions in a block at once and reuses the loaded K, V across all queries in the block. The causal mask is applied inside each block:

```
Scores for A, pos 0-2:
              P0     P1     P2
   pos 0    [ 1     -∞     -∞  ]  ← pos 0 can only attend to pos 0
   pos 1    [ 0      1     -∞  ]  ← pos 1 can attend to pos 0, 1
   pos 2    [ 1      1      2  ]  ← pos 2 can attend to all
```

**Causal mask in the Br=3 case:** Unlike Br=1, the causal mask is now essential. All 3 of A's positions are in one row block and load the same pages (P0, P1, P2). Without the mask, position 0 would incorrectly attend to positions 1 and 2. The -∞ entries ensure these scores become 0 after softmax. The tradeoff: larger Br loads K, V once for multiple queries (more efficient), but some loaded values get masked out (wasted work within the block).

Same results as Br=1, but more efficient — K, V are loaded once and reused.

**Optimization: Bc=2.** With a page size of 2, each page stores KV for 2 token positions. The KV cache is repacked:

```
  P0 ("The"+"cat", shared):       K=[[1,0],[0,1]],   V=[[1,1],[2,0]]
  P1 ("sat", A-only):             K=[[1,1]],          V=[[0,1]]
  P2 ("ran"+"fast", B-only):      K=[[1,-1],[0,-1]],  V=[[1,0],[0,1]]
```

Each filled block is now a (1 × 2) lookup — 1 query position loading KV for 2 token positions at once:

```
              P0           P1       P2
           (The, cat)    (sat)    (ran, fast)
             shared      A-only   B-only
            ┌──────────┬────────┬──────────┐
A, pos 0    │    P0    │        │          │
            ├──────────┼────────┼──────────┤
A, pos 1    │    P0    │        │          │
            ├──────────┼────────┼──────────┤
A, pos 2    │    P0    │   P1   │          │
            ├──────────┼────────┼──────────┤
B, pos 0    │    P0    │        │          │
            ├──────────┼────────┼──────────┤
B, pos 1    │    P0    │        │          │
            ├──────────┼────────┼──────────┤
B, pos 2    │    P0    │        │    P2    │
            ├──────────┼────────┼──────────┤
B, pos 3    │    P0    │        │    P2    │
            └──────────┴────────┴──────────┘
```

Fewer blocks to iterate over, but some loaded KV entries are masked out. For example, when A, pos 0 loads from P0, it gets K, V for both "The" and "cat" — but the causal mask zeroes out the "cat" score since position 1 > position 0.

**Optimization: composable formats.** Consider what happens with the Br=1, Bc=1 matrix when processing the shared prefix. All 7 query positions (A's 3 + B's 4) independently load KV from P0. That's K=[1,0], V=[1,1] loaded from global memory 7 separate times. Same for P1. This is wasteful.

FlashInfer can split the attention computation across multiple block sparse matrices with different block sizes and compose the results:

- **Shared prefix matrix** (Br=7, Bc=1): groups all 7 queries into one row block. K, V for each shared page is loaded into fast shared memory once, and all 7 queries compute their attention scores against it.
- **Unique suffix matrix** (Br=1, Bc=1): each request's unique suffix tokens are handled independently.  
- 
```
Shared prefix matrix (Br=7, Bc=1):

              P0     P1
             (The)  (cat)
            ┌──────┬──────┐
A, pos 0    │      │      │
A, pos 1    │      │      │
A, pos 2    │  P0  │  P1  │  ← one row block, all 7 queries
B, pos 0    │      │      │
B, pos 1    │      │      │
B, pos 2    │      │      │
B, pos 3    │      │      │
            └──────┴──────┘

Kernel for P0 block: load K=[1,0], V=[1,1] once into shared
memory. All 7 queries compute scores against this K and weight
this V — 7× reuse vs. 7 independent global memory loads.
```

```
Unique suffix matrix (Br=1, Bc=1):

              P2     P3     P4
             (sat)  (ran)  (fast)
            ┌──────┬──────┬──────┐
A, pos 2    │  P2  │      │      │
            ├──────┼──────┼──────┤
B, pos 2    │      │  P3  │      │
            ├──────┼──────┼──────┤
B, pos 3    │      │  P3  │  P4  │
            └──────┴──────┴──────┘

Only A pos 2, B pos 2, and B pos 3 have unique suffixes. Other
positions (A pos 0, A pos 1, B pos 0, B pos 1) use only their
shared prefix output.
```

Each matrix produces a partial attention output and a log-sum-exp (LSE) normalizer for each query position. These are merged exactly:

```
For each query position:
  O_shared, LSE_shared = partial attention output from shared prefix matrix
  O_unique, LSE_unique = partial attention output from unique suffix matrix

  LSE_merged = log(exp(LSE_shared) + exp(LSE_unique))
  O_merged   = (exp(LSE_shared) · O_shared + exp(LSE_unique) · O_unique)
                / exp(LSE_merged)
```

This merge is mathematically equivalent to computing attention over all KV positions at once — splitting the softmax denominator across subsets and recombining is exact, not an approximation.

**Fewer memory loads:** Without composable formats, K, V from each shared page is loaded from global memory 7 times (once per query). With the shared prefix matrix (Br=7), K, V is loaded once into shared memory and reused across all 7 queries.

More importantly, the matrix representation enables composable formats (above). Neither PagedAttention nor RadixAttention can do this on their own: PagedAttention has no mechanism to group queries sharing a prefix into a larger kernel tile, and RadixAttention identifies shared prefixes via its radix tree but each query still loads the shared KV from global memory independently. FlashInfer's block sparse matrix makes the sharing pattern explicit in the sparsity structure, enabling the kernel to load shared KV once and reuse it across queries.

## FlashInfer's plan/run API

FlashInfer exposes a two-phase API for attention computation: `plan()` and `run()`.

`plan()` runs on the CPU before each generation step. It takes `kv_indptr` and `kv_indices` — the indirection arrays encoding the block sparse matrix — as inputs, and computes a schedule: how to partition work across sequences, and how to parallelize across the KV dimension.

`run()` launches the GPU kernel, passing `kv_indptr` and `kv_indices` directly to it. `kv_indptr` marks where each sequence's page list begins and ends (the same indptr structure used for the ragged query tensor); `kv_indices` lists the physical page indices for each sequence. The kernel is parameterized on these arrays — a single generic kernel handles any block sparse structure by following the indirection at runtime.

The generation loop therefore looks like this (Listing 1 of the FlashInfer paper[^5]):

```python
attn.plan(batch_structure)      # one-time setup
with torch.cuda.graph(g):
    attn.run(q, kv_cache)       # capture GPU kernel

while not finished:
    attn.plan(batch_structure)  # CPU: recompute schedule
    g.replay()                  # GPU: execute kernel
```

`plan()` is called every step because the batch structure changes: each generated token extends every sequence by one, potentially adding a new KV page.

## SGLang and FlashInfer

SGLang[^4] uses RadixAttention (see [RadixAttention](#radixattention)) to allocate and track KV cache pages, using the radix tree to identify which pages are shared across sequences. SGLang computes `kv_indptr` and `kv_indices` — the block sparse matrix encoding which physical KV pages each sequence attends to — and passes them to FlashInfer's `plan()` and `run()` at each decode step to execute the attention computation.

---

[^1]: In practice, A = (Q · K^T) / √d_k is used where √d_k is a scaling factor. This scaling is for numerical stability: as the dimension d_k grows, the dot products can become very large. The scaling keeps the variance of the dot products constant, preventing this issue.
    
[^2]: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). *Advances in Neural Information Processing Systems (NeurIPS)*.
    
[^3]: Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J., Zhang, H., & Stoica, I. (2023). [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180). *Proceedings of the 29th Symposium on Operating Systems Principles (SOSP)*.
    
[^4]: Zheng, L., Yin, L., Xie, Z., Sun, C., Huang, J., Yu, C. H., Cao, S., Kozyrakis, C., Stoica, I., Gonzalez, J. E., Barrett, C., & Sheng, Y. (2024). [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104). *Advances in Neural Information Processing Systems 37 (NeurIPS)*.
    
[^5]: Ye, Z., Chen, L., Lai, R., Lin, W., Zhang, Y., Wang, S., Chen, T., Kasikci, B., Grover, V., Krishnamurthy, A., & Ceze, L. (2025). [FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving](https://arxiv.org/abs/2501.01005). *Proceedings of Machine Learning and Systems (MLSys)*.
    
[^6]: PagedAttention also introduces indirection, but at block granularity rather than element granularity. The block table is consulted once per block to find its physical location, then memory access within the block is contiguous and coalesced. This amortizes the indirection cost over many elements, unlike fine-grained pointer chasing for every KV cache access.

[^7]: Zhang, B. (2026). [The Five Eras of KVCache](https://www.modular.com/blog/the-five-eras-of-kvcache). *Modular Blog*.
