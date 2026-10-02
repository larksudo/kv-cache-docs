---
next: /basics/paged-attention

---

# KV Cache 本质与显存占用

> 本篇是知识库的地基：KV Cache 到底是什么、为什么它吃显存、优化手段的全景分类。融入 ForceInjection 系列与「一研」系列的增量内容（证据等级 C，关键数字已标注来源）。


> **🎯 面试考察点**（66 家公司真题库 · KV Cache）：
> - 「KV Cache 的原理？为什么 Q 不缓存？大小怎么算？」——17 家公司 34 次（题库排名第 6）；
> - 「为什么训练阶段不用 KV Cache？」；
> - 追问链：公式 → GQA/MLA 怎么改公式 → 三类优化框架 → 每类举两个系统实现。
> 精答见[面试题库·推理系统 Q1/Q3-Q5](/interview/inference-answers)

## 从 Attention 公式说起

自回归生成的每一步，当前 token 的 Query 都要和**所有历史 token** 的 Key/Value 做计算：

```text
Attention(Q, K, V) = softmax(Q × K^T / √d) × V
```

- **Q（Query）**：当前 token 的查询向量，**不需要缓存**——每个 token 现算现用；
- **K（Key）**：所有历史 token 的键向量，**必须缓存**——后续 token 要跟它做点积；
- **V（Value）**：所有历史 token 的值向量，**必须缓存**——点积结果要跟它做加权求和。

为什么 Q 不缓存？因果掩码决定了 `Q_t` 只查询 `[1, t]` 区间，`Q_{t+1}` 永远不需要 `Q_t`——缓存不会被再次使用的东西是纯粹的浪费。以 LLaMA-2 70B（64 Q 头 vs 8 KV 头）、seq=4096 为例：缓存 K+V 约 1.25 GB，而如果缓存 Q 需要额外约 5 GB 永不访问的显存。

K 和 V 必须分开，则是因为「怎么被找到」（K，检索特征）和「提供什么信息」（V，信息载荷）是两件独立的事——模型需要独立调控「谁关注谁」和「传递什么」。

**KV Cache 的本质是用空间换时间**：把 Prefill 的计算结果存下来，避免每个新 token 都重算全部历史前向。

## 显存占用公式

单个 token、单层的 KV 体积：

```text
KV/token/layer = 2 × num_kv_heads × head_dim × sizeof(dtype)
```

（`2` 是 K 和 V 各一份。）

完整展开：

```text
Memory_KV ≈ 2 × bytes_per_element × L(层数) × B(batch) × S(seq) × H_kv × d_head
```

几个基准数字（bf16，全模型合计）：

| 模型 | 注意力 | KV/token/层 | KV/token/全模型 |
|---|---|---|---|
| Mistral-7B (MHA, 32 头) | MHA | 16 KB | 512 KB |
| LLaMA-70B (GQA-8) | GQA | 4 KB | 320 KB |
| Qwen2.5-72B (GQA) | GQA | ~4 KB | ~320 KB |
| DeepSeek-V3 (MLA) | MLA | ~1.1 KB | 68.6 KB |

量级感受：Qwen2.5-7B 在 batch=32、seq=32K 时 KV Cache 约 57 GB——是模型权重（14 GB）的 4 倍；72B 模型 batch=8、seq=32K 时单 KV Cache 就有约 80 GB，一张 H100 装满。**这就是为什么 KV Cache 管理是一个系统问题而不是一个参数问题**。

## 注意力形态决定 KV 体积：MHA/GQA/MQA/MLA/CSA

| 类型 | 单 token KV（80 层基准） | vs MHA 缩减 | 代表 |
|---|---|---|---|
| MHA | 32 KB | 1× | 原始 Transformer |
| GQA | 4 KB | 8× | LLaMA-2/3、Qwen-2 |
| MQA | 0.5 KB | 64× | PaLM、Falcon |
| MLA | ~1.1 KB | ~29× | DeepSeek V2/V3 |
| CSA/HCA | ~0.17 KB | ~190× | DeepSeek V4（新架构） |

- **GQA**：每 8 个 Q 头共享 1 个 KV 头，KV 缩小 8 倍——LLaMA-70B 的 KV（320 KB/token）比 Mistral-7B（512 KB/token）还小；
- **MLA**：K/V 压缩为共享低秩潜向量（DeepSeek-V3：512 维 KV latent ≈ 1 KB + 64 维解耦 RoPE K ≈ 128 B），约为 MHA 的 1/57。RoPE 因旋转不可压缩，单独解耦存储且所有头共享；
- **vLLM 的抽象**：通过 `KVCacheGroupSpec` 为每种注意力类型创建独立 block 池（MLA 是 `MLAAttentionSpec(kv_lora_rank=512, ...)`），这是 vLLM 同时支持多种注意力的关键机制。

## 优化手段全景：三类归一

所有 KV Cache 优化最终都可归入三类（这是评估任何新方案时的检查框架）：

1. **减 per-token 体积**：量化（FP8/INT4）、GQA/MLA、低秩压缩；
2. **减并发驻留**：Offloading（GPU→CPU→SSD 分层）、稀疏注意力（只保留重要 token）；
3. **减重复缓存**：Prefix Caching（跨请求/跨实例复用相同前缀）。

另一个重要的正交视角是**读路径与存路径分离**：

- 稀疏注意力优化「读」（HBM→SM 每步搬多少）；
- KV 压缩优化「存」（HBM 里放多少）；
- 两者互不替代。**「读得少 ≠ 存得少」**：动态选择类稀疏注意力（H2O/NSA-DSA）扫描时仍需全量 KV 可见——省计算不省存储；只有滑动窗口既省读又省存。

## 两个阶段：Prefill 与 Decode

- **Prefill**：一次性处理全部输入 token，Attention 计算量 O(S²)，compute-bound——吃算力；
- **Decode**：每步只处理一个新 token，但要读完整 KV Cache，memory-bound——吃带宽。

这决定了两者的优化方向完全不同：TTFT 受 Prefill 影响（批处理、算子融合），TPOT 受 Decode 影响（KV 访存效率）。PD 分离架构正是利用这个差异把两阶段拆到不同硬件上（详见「P2P 与 PD 分离」篇）。

累计计算量角度：无缓存时生成 L 个 token 的累计 Attention 计算约 O(L³)；有缓存近似 O(P² + P·L + L²)。这个差距就是 KV Cache 的全部价值来源。

## 下一步阅读

- 「PagedAttention 与注意力形态」：KV 在 GPU 显存里如何被组织成 block；
- 「Prefix Cache 原理」：前缀如何被复用；
- 「总体架构」：LMCache/Mooncake/vLLM 如何把这三类优化做成系统。
