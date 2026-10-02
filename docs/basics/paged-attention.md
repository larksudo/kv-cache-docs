---
prev: /basics/kv-cache-fundamentals
next: /basics/prefix-caching

---

# PagedAttention：KV Cache 的显存管理

> 数学形态的 KV Cache 进入 GPU 显存后会立刻被重新组织。本篇讲 PagedAttention 的分页机制、三种预分配浪费的分类，以及「block 抽象让功能自然生长」——prefix caching、offloading、量化全都长在 block 之上。


> **🎯 面试考察点**（66 家公司真题库 · PagedAttention）：
> - 「PagedAttention 为什么能提升效果？」——×8（京东/字节/小米/拼多多/百度等）；「vLLM 中 PagedAttention 的实现机制与设计动机」；
> - 追问链：三种浪费分类 → block 抽象上长出哪些能力 → K/V 在 Mooncake 里的「能见度递减」。
> 精答见[面试题库 Q2](/interview/inference-answers)；LMcache vs Mooncake 对比见[H2D/D2H 深潜 §5](/core/h2d-d2h-deepdive)

## 为什么需要分页

朴素方案按「最大序列长度」为每个请求预分配连续显存。vLLM 论文（SOSP 2023）给出的实测利用率：

| 方案 | 显存利用率 |
|---|---|
| Orca（按最大长度预留） | 20.4% |
| Orca（按 2 的幂预留） | 32.0% |
| Orca（Oracle，预知真实输出长度） | 38.2% |
| PagedAttention | 接近 100%（浪费 ≤ 1 block/请求） |

注意 Oracle 行：**即使预知每个请求的真实输出长度，预留浪费仍占 61.8%**——这是结构性浪费，不是调度能解决的。同等显存下 PagedAttention 并发提升 2-4×，吞吐提升约 2.5×。

浪费的精确分类（这个三分法也适用于任何资源池设计）：

1. **reservation waste**（预留浪费）：按 max 而不是实际长度分配——PagedAttention 消除的就是它；
2. **internal fragmentation**（内部碎片）：最后一个 block 装不满，浪费 ≤ block_size-1 个 token；
3. **external fragmentation**（外部碎片）：连续分配产生的空洞——分页后不存在。

## 机制：软件 MMU

OS 分页类比的精确对应：

| OS | vLLM |
|---|---|
| 物理页框 | KV block（默认 block_size=16 token） |
| 页表 | Block Table（逻辑 block → 物理 block） |
| MMU 硬件翻译 | attention kernel 里的软件 gather |
| 按需调页 | 按需 block 分配 |

GPU 没有硬件 MMU，vLLM 用 CUDA kernel 实现「软件 MMU」：attention kernel 遍历 block table，按 `physical_block_id` 直接索引显存中的 block。

每层每个方向（K 或 V）的 block 张量形状：

```text
[num_blocks, block_size, num_kv_heads_per_rank, head_dim]
```

一个 block 的字节大小 = `block_size × num_kv_heads_per_rank × head_dim × sizeof(dtype)`。以 LLaMA-70B（tp=8）为例每 block（K 或 V）4 KB；GQA 模型常见 64 KB/block。block table 本身也占内存：seq=32K 时每请求约 16 KB——这就是为什么超大 block_size（如 DeepSeek-V4 推荐 256）在压缩后 per-token 体积很小的模型上反而更优：小 block 会使 block table 条目激增。

## block 抽象之上自然生长的能力

PagedAttention 真正的价值不是消除浪费本身，而是**block 成为统一抽象后，上层能力自然生长**：

1. **Prefix Caching**：block 级共享——相同前缀的请求共享物理 block（详见下一篇）；
2. **Offloading**：block 粒度换入换出——GPU/CPU/SSD 之间搬运的最小单元（LMCache/Mooncake 的整个数据面都建立在这上面）；
3. **量化**：block 内压缩——FP8/INT4 KV Cache 保持 block 结构不变；
4. **Copy-on-Write**：共享 block 被写时复制，逻辑隔离物理共享。

好的抽象不是叠加功能，而是让功能自然生长——这是后续所有系统（vLLM connector、LMCache chunk、Mooncake Slice）反复验证的模式。

## K/V 在传输与存储系统中的「能见度递减」

一个值得单独指出的事实（来自「一研」Mooncake 系列，已对照源码验证）：**Mooncake 不创建 K/V 张量，它只搬运 vLLM 分配好的张量**。从注册到传输到存储，K 和 V 的「身份」逐层模糊：

```text
vLLM 层      K/V 是两个独立 torch.Tensor，形状明确
    ↓ register_kv_caches()
connector 层  K/V 变成 GPU 地址列表（data_ptr + nbytes），不区分名字
    ↓ batch_transfer_sync_write()
传输引擎层    K/V 变成 (src_ptr, dst_ptr, length) 三元组——只知搬多少字节
    ↓ Put()
存储层        K/V 变成 Slice {ptr, size}——连「层」的概念都没有了
```

对 MHA/GQA（CUDA 标准后端），vLLM 每层传入 `(k_cache, v_cache)` 元组，connector 每层注册 2 个地址；对 MLA/Pallas/FlashInfer 后端，vLLM 层面 K/V 就已融合为单张量（每层 1 个地址）——这是 MLA 数学结构决定的，不是缓存系统做的合并。

推论：**缓存系统对 KV 的语义理解越少（越接近字节），通用性越强但复用粒度越粗**。Mooncake 选择「物理寻址优先」，LMCache 选择「语义保留优先」（chunk + shape/dtype/format）——这是两个系统的根本分歧之一，详见「存储引擎中的数据变形」篇。

## 关键数字速查

| 项 | 值 |
|---|---|
| vLLM 默认 block_size | 16 token |
| block GPU 地址 | `张量基地址 + block_id × block_len` |
| LLaMA-70B (tp=8) 每 block | K 或 V 各 4 KB；全模型 160 张量 |
| 4096 token 请求 | 256 个 block；LLaMA-70B 约 160 MB；Mistral-7B 约 2 GB |
| PagedAttention 浪费 | ≤ 1 block/请求 |
