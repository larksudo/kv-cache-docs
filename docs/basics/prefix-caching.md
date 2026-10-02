---
prev: /basics/paged-attention
next: /basics/compression

---

# Prefix Cache 原理

> 前缀复用是 KV Cache 管理中收益最大的杠杆（命中率往往超过传输优化的收益）。本篇讲它的机制、索引数据结构选型（Radix Tree vs Hash Chain）、RoPE 带来的位置陷阱，以及从单机 APC 到跨实例共享池的演进路径。


> **🎯 面试考察点**（66 家公司真题库 · Prefix Cache）：
> - 「Prefix Cache 的机制是什么？适用于哪些场景？」；「vLLM APC 与 SGLang RadixAttention 的差别」（索引结构对比是高频追问）；
> - 追问链：hash 链怎么构造 → RoPE 位置陷阱 → CacheBlend 为什么能跨位置 → 跨实例共享怎么做。
> 精答见[面试题库](/interview/inference-answers)，机制深读见[Prefix Cache 命中与数据搬运](/core/prefix-hit)

## 第一性原理

如果两个请求的 token prefix 完全相同，且模型、tokenizer、并行布局、量化格式、adapter 都不变，那么这段 prefix 产生的 KV Cache 也相同。因此缓存系统真正索引的不是文本，而是：

```text
identity = model + tokenizer/revision + quantization/dtype/layout
         + adapter + parallel shard + prefix token hash
```

生产环境 40%-80% 的 token 序列存在可复用共享前缀（system prompt、few-shot 示例、多轮对话历史、agent 工具上下文）。理论加速比 ≈ (P+Q)/2Q：8K 前缀 + 256 查询的请求，命中后约 16.5× 加速。

前缀匹配受因果注意力约束：**一旦某个 chunk 未命中，其后所有 chunk 都必须重算**——不存在「跳着命中」。

## 两种索引：Hash Chain 与 Radix Tree

**vLLM APC（哈希链）**：token 按 block 切分，block hash 链式计算：

```text
h0 = H(model_context + tokens[0:block_size])
h1 = H(h0 + tokens[block_size:2*block_size])
...
```

任一 token 不同，其后所有 hash 全部不同。查找是哈希表精确匹配，实现简单、内存开销低（+2-5%）。

**SGLang RadixAttention（压缩前缀树）**：节点存 token 片段 + KV block 列表 + 子节点映射 + `lock_ref` 引用计数；查找/插入/最长前缀匹配 O(m)；LRU 从叶子淘汰（淘汰中间节点会使所有后代失效），淘汰后单子节点合并。索引粒度 token 级（不需要 block 对齐），内存开销 +5-10%。

| 维度 | RadixAttention (SGLang) | APC (vLLM) |
|---|---|---|
| 数据结构 | Radix Tree | 哈希链 + 哈希表 |
| 索引粒度 | token 级 | block 级 |
| 动态前缀分支 | 任意位置 | 需 block 对齐 |
| 实现复杂度 | 较高（分裂/合并） | 较低 |
| 实测前缀命中率 | 85-95% | 80-90% |

一个有意思的事实：vLLM 的 KV Cache 管理用哈希表，但 **vLLM Router 的 cache-aware 路由组件用 Radix Tree** 追踪各 worker 的前缀分布——两种结构在不同层各得其所。

「幽灵前缀」是 Radix Tree 特有的长期运行问题：根路径长但叶子极少在用时，中间节点因其他活跃子树不能淘汰——需要引用计数 + 定期碎片整理。

## 三个工程约束（vLLM 侧）

1. **Chunked Prefill 下只在第一个 chunk 查找**（`num_computed_tokens == 0` 守卫）：共享前缀跨 chunk 边界时，后续 chunk 的缓存全部浪费。这不是缺陷而是权衡（避免物理页冲突与部分命中重组）。SGLang 的对比做法是 `stash_chunked_request()` 把每个 chunk 的部分 KV 写回 Radix Tree，后续 chunk 重建匹配——跨 chunk 的 HiCache 命中成为可能。vLLM 选了调度简单，SGLang 选了缓存完整。

2. **chunk 大小必须被 block_size 整除**（partial block hashing 可解除此约束）。

3. **投机解码与 prefix caching 的冲突是 block 级的**：投机 token 的 KV 非确定性（同 prompt 两次投机产生不同 KV），hash 无意义；prompt 前缀部分仍可正常缓存；部分 prompt + 部分投机 token 的混块整块无法缓存。

## RoPE 位置陷阱：content-identical ≠ cache-identical

RoPE 把绝对位置不可逆地「烧进」K 向量：同一段 System Prompt 在位置 0-99 和位置 501-600 的 K 完全不同，hash 对不上。Agent 场景（历史截断、多 Agent 协作拼装上下文）会放大这个问题。三种绕过方案：

| 方案 | 命中率 | 额外计算 | 代表 |
|---|---|---|---|
| A: 位置限制匹配 | 低 | 无 | vLLM APC、SGLang 默认 |
| B: nope 哈希 + 重算 RoPE | 高 | 中 | MLA 架构天然支持 |
| C: Rotary Correction | 高 | 低（单次旋转） | **LMCache CacheBlend** |

方案 C 的数学基础是 RoPE 旋转矩阵的正交可加性：从位置 m 校正到 m' 只需乘 (m'-m) 的旋转差矩阵。**LMCache CacheBlend 是目前少数在工业级实现中走通 Rotary Correction 的方案**（EuroSys 2025 最佳论文方向，支持 vLLM/SGLang/Dynamo 多引擎）；MLA 的 KV latent 不含位置信息，天然为方案 B 铺路；SGLang 默认 RadixAttention 属方案 A。

## 从单机到跨实例：APC 的三大局限

vLLM APC 只用 GPU 显存、单实例隔离、重启即丢。这正是 LMCache/Mooncake 要解决的问题，也是本知识库第二部分的主题：

- **LMCache**：chunk key + 多级 StorageManager（L1 CPU ~100μs / L2 P2P ~1ms / L3 磁盘 ~10ms / L4 远程 ~100ms），Waterfall 瀑布式检索；P2P 或 Cache Controller 集中两种跨实例模式；
- **Mooncake Store**：hash 编进 object key 的全局对象池 + Conductor 缓存感知调度；
- **混合注意力模型**（Gemma-3 类）的额外规则：Full Attention 层从左到右匹配、Sliding Window 层从右到左、取较小值为有效匹配长度。

跨引擎/跨精度复用的约束：vLLM Connector API 不内置精度转换，FP16/FP8 下 token-ID-based hash 一致（hash 不依赖 K 值），但跨精度复用需要 producer/consumer 同 dtype——connector 实现者需在传输层自行处理。

## 商业参照：Claude 提示词缓存

Anthropic 把 prefix caching 做成了 API 契约：显式缓存断点（`cache_control`，每请求最多 4 个）、TTL 保证（默认 5 分钟命中刷新，可选 1 小时）、差异化计费（写 1.25×/2.0×，命中读 0.1×）。它与 vLLM APC 的差异是**顶层主动控制 vs 底层被动优化**——后者对用户透明但依赖 LRU，前者倒逼开发者把提示词解耦为「静态冻结区 + 动态变化区」。自建缓存系统时可参考其断点预算、Schema 漂移防护（工具定义用名称+序列化 Schema 做联合缓存键）、缓存断裂诊断（cache_read 跌幅 >5% 且 >2000 tokens 才判定）等工程实践。

## 容量与收益速查

- LMCache 官方实测（Llama-2-7B/A100）：多轮对话 TTFT 降 60-80%、吞吐 2-3×；RAG TTFT 降 70-85%；长文档 TTFT 降 80-95%；
- Mooncake FAST 25 口径：真实 trace 下 prefill 计算节省最高 48%（注意 arXiv 早期版本的 525% 已修正为 59%-498%）;
- vLLM x Mooncake Store 博客：Codex agentic traces 命中率 1.7% → 92.2%，吞吐 3.8×；
- chunk/block 大小推荐 256（哈希开销 <0.1% 总计算）。
