---
prev: /interview/
next: /interview/cuda-systems-answers
---

# 大模型推理系统高频题精答

> 按面试频次排序，答案与本知识库各章结论互相印证（链接直达对应章节深读）。答案按「结论先行 → 展开要点 → 数字锚点」组织，可直接用于口头作答。

## 一、KV Cache 与 PagedAttention（17 家公司，34+15 次）

**Q1：KV Cache 的工作原理？为什么需要？大小怎么算？**

自回归生成第 N 个 token 需要与第 1..N-1 个 token 的 K/V 做注意力。不缓存就要重跑全部历史前向——KV Cache 用空间换时间，把每层的 K/V 存下来增量追加。**只有 K/V 缓存，Q 不缓存**：Q_t 只被自己那一步使用，因果掩码决定它不会被后续 token 再查询——缓存不会被再次使用的东西是纯粹浪费。

大小公式（必背）：

```text
KV 总量 = 2(K+V) × 层数 L × KV头数 H_kv × head_dim × dtype字节 × batch × seq_len
```

锚点数字：LLaMA-70B（GQA-8, bf16）每 token 约 320KB；32K 上下文单请求约 10GB；Qwen2.5-7B batch=32/32K 时 KV 约 57GB，是权重（14GB）的 4 倍。GQA 让 KV 头数小于 Q 头数（8 vs 64），KV 缩小 8 倍。

→ 深读：[KV Cache 本质与显存占用](/basics/kv-cache-fundamentals)

**Q2：PagedAttention 为什么能提升效果？（×8）**

传统方案按最大序列长度预分配连续显存。vLLM 论文实测：Orca 按最大预留的利用率只有 **20.4%**，即使 Oracle 预知真实输出长度也只有 **38.2%**——61.8% 是结构性浪费。PagedAttention 借鉴 OS 分页：物理显存切成 block（默认 16 token），Block Table 记录逻辑→物理映射，attention kernel 软件索引。利用率接近 100%（浪费 ≤1 block/请求），同等显存并发 2-4×、吞吐 2.5×。浪费三分法：预留浪费（分页消除）、内部碎片（≤block_size-1）、外部碎片（分页后不存在）。**block 抽象之上自然生长出 prefix caching（块级共享）、offloading（块粒度换入换出）、量化（块内压缩）**。

→ 深读：[PagedAttention 与注意力形态](/basics/paged-attention)

**Q3：KV Cache 有哪些优化策略？（×5）**

三类归一框架（检查任何新方案）：**减 per-token 体积**（GQA/MLA、FP8/INT4 量化）、**减并发驻留**（offloading 分层 GPU→CPU→SSD、稀疏注意力淘汰）、**减重复缓存**（prefix caching 跨请求复用）。补充两个正交轴：读路径优化（稀疏注意力）与存路径优化（压缩）互不替代——「读得少 ≠ 存得少」；传输侧还有 CacheGen 式压缩（4.3×）。

→ 深读：[KV Cache 压缩与量化](/basics/compression)

**Q4：训练阶段用 KV Cache 吗？为什么？**

训练是 full-sequence teacher forcing，一次前向算完整序列的注意力，中间激活本就驻留用于反向传播——不存在「复用历史前缀」的场景，KV Cache 是推理态概念（增量解码的产物）。预填充（Prefill）可以看成「一次性构建 KV Cache」。

**Q5：MHA/MQA/GQA 在推理阶段的差异？（×4 手写 MHA）**

| | MHA | GQA | MQA | MLA |
|---|---|---|---|---|
| KV 头数 | =Q 头数 | 分组共享（如 64Q/8KV） | 全 Q 共 1 头 | 压缩为低秩潜向量 |
| KV/token（80 层 bf16） | 32KB | 4KB（8×） | 0.5KB（64×） | ~1.1KB（~29×） |
| 质量折损 | 基线 | 小 | 较明显 | 小且带 RoPE 解耦设计 |

线上推理关注 GQA 的原因：显存带宽是 decode 瓶颈，KV 体积直接决定每步访存量，GQA 用极小的质量代价换来 8× 带宽与容量。MLA（DeepSeek-V3）更进一步：512 维 latent + 64 维解耦 RoPE，RoPE 因旋转不可压缩单独存储。

→ 深读：[PagedAttention 与注意力形态](/basics/paged-attention)、[Prefix Cache 原理](/basics/prefix-caching)。被追问「paged KV 具体怎么搬到 CPU、CUDA IPC 共享了什么」时，答法见 [H2D/D2H 接口级深潜](/core/h2d-d2h-deepdive)。

## 二、Prefill/Decode 与调度

**Q6：Prefill 和 Decode 各是什么瓶颈？各有哪些优化？（×高频）**

- **Prefill**：一次算全部输入 token，Attention O(S²)，**compute-bound**——优化方向是算力利用率：FlashAttention（IO 感知分块）、Chunked Prefill（分块调度防长请求阻塞）、算子融合、PD 分离（让 P 用高算力卡）；
- **Decode**：每步只算 1 个新 token 但要读完整 KV，**memory-bound**（MFU 常 <25%）——优化方向是访存：Continuous Batching（攒批摊薄 KV 读取）、KV 量化（减访存字节）、投机解码（一次验证多 token 把算力用起来）、GQA/MLA。

锚点：32K 上下文 70B 模型单步 KV 读取约 10GB，H100 3.35TB/s 带宽下约 3ms——已赶上整个 decode step 预算。prefill 各 token 之间——同 chunk 内 token 有因果依赖但 attention 可并行（矩阵化），chunk 间可流水。

**Q7：Continuous Batching 的设计动机？与 Chunked Prefill 的关系？**

静态 batch 必须等最慢请求完成才换批，GPU 大量空转。Orca 提出迭代级调度：**每步 decode 后都可插入新请求/退出完成请求**。Chunked Prefill 解决另一半：长 prompt 一次性 prefill 会阻塞其他请求的分块申请——把 prefill 也切成 2048-token 块与 decode 混排。两者正交，现代引擎都开。**面试陷阱**：Chunked Prefill 是「同一请求的 prefill 分块」与「不同请求混排」的结合，不是二选一。vLLM 约束：chunk 大小必须被 block_size 整除；prefix cache 只在第一个 chunk 查找（权衡，SGLang 选择跨 chunk 写回 radix tree）。

**Q8：vLLM 的核心设计与调度器流程？（×12 家公司）**

四件套：**PagedAttention**（分页 KV 管理）+ **Continuous Batching**（迭代级调度）+ **Prefix Caching/APC**（block hash 链复用前缀）+ **CUDA Graph**（消除 kernel launch 开销，decode 小批量收益大；piecewise 模式处理 prefill 的动态 shape）。调度器每步：waiting 队列按 FCFS → 尝试分配 → 不够则**抢占**（vLLM V1 只有 Recompute——SWAP 被有意移除，设计假设是 prefix cache 让重算大多命中；被抢占请求插回队首防饥饿）。KV Connector V1 是外部缓存接入点（scheduler 管决策、worker 管搬运）。

→ 深读：[vLLM Connector 对接](/core/vllm-connector)、[vLLM 原生 KV Offload 附录](/core/vllm-connector)

**Q9：vLLM vs SGLang 差别？SGLang 为什么快？**

| | vLLM | SGLang |
|---|---|---|
| 前缀索引 | 哈希链+哈希表（block 粒度） | **RadixAttention 基数树**（token 级，任意位置分支） |
| 前缀命中率 | 80-90% | 85-95%（动态前缀场景优势） |
| 投机/重叠 | 较新 | overlap scheduler 默认开 |
| 结构化生成 | 外挂 | 原生（源自 SGLang 的初衷） |

SGLang 在多轮对话/Tree-of-Thoughts 类共享前缀场景更好：radix tree 前缀去重是天然能力（实测吞吐 2-8×），且每个 chunk 的部分 KV 写回树使跨 chunk 命中成为可能（vLLM 选择了调度简单性）。细节对比见 [Prefix Cache 原理](/basics/prefix-caching)。

**Q10：请求被抢占后怎么处理？**

vLLM V1：`num_computed_tokens` 回退、请求回 waiting 队首、靠 prefix cache 重算快速恢复。LMCache 视角：抢占的 KV 已被外置缓存保存的话可直接恢复（lazy offload）；Mooncake Store 场景 decode offload 增量上下文也可回灌。

## 三、PD 分离（6 家公司，8 次）

**Q11：为什么 PD 分离？KV 传输开销大为什么还值得？（×3 问法）**

两阶段硬件需求错配：Prefill compute-bound（要算力）、Decode memory-bound（要带宽/显存）。混部时互相妥协。分离后 P 用高算力卡快速清空、D 用大显存卡专读 KV。

**传输值得吗——用数字回答**：KV 传输量随序列**线性**增长，而 Prefill 计算量随序列**超线性**（O(S²)）。70B 4K：传输 1.5GB/RDMA 约 30ms vs 本地重算 200ms；405B 32K：64GB/1.28s vs 重算 8s。**模型越大上下文越长，分离收益越大**。KV 传输占 Prefill 计算时间：8B@128K <1%、70B ~5%、DeepSeek-V3 ~8%。

**Q12：PD 分离的调度队列怎么设计？**

参考 SGLang 实现：decode 侧四队列生命周期 **PreallocQueue**（握手 + 有 KV 就预分配，SWA 场景只预分配滑窗尾部）→ **TransferQueue**（poll 传输状态）→ **WaitingQueue**（构造 PrebuiltExtendBatch 跳过 prefill 前向）→ **RunningBatch**。P/D 各自维护心跳与超时（waiting_timeout 默认 300s，超时向 P 发 ABORT 防显存泄漏）。Push vs Pull：LMCache PD 选 Push（P 生成即推）、Mooncake Conductor 选 Pull-with-cache（全局索引指导最优拉取）、vLLM Connector 两套 API 都提供。

**Q13：Mooncake 的 PD 方案设计思路？**

MooncakeConnector 直传：P 把 KV block 地址 + block id 经 ZMQ 元数据发给 D，D 发起 RDMA/NVLink/HCCS 直写自己 paged buffer——零拷贝、支持异构 TP/PP（region 对齐 + 连续块合并）。搭配 MooncakeStore 共享池可组成 MultiConnector：**直传保低延迟 + 池保跨实例复用**。跨 NUMA 代价 +47% 延迟，**P/D 应同 Leaf 交换机部署**；带宽规划公式：1.5GB/30ms = 50GB/s = 400Gb/s。

→ 深读：[P2P 与 PD 分离](/core/p2p-pd)、[Mooncake 内部机制深潜](/core/mooncake-internals)

## 四、量化推理（27 家公司，82 次）

**Q14：per-tensor/per-channel/per-group 量化粒度区别？哪种最细？**

- per-tensor：全张量一个 scale—— outlier 一颗老鼠屎坏一锅粥；
- per-channel：每输出通道一个 scale——隔离通道间分布差异；
- per-group（如 group=128）：通道内再分组——**最细**，outlier 被限制在组内。代价是 scale 元数据（per-token-head 对 70B@32K 仅约 1MB，可忽略）。
精度敏感度排序同理。KV 量化比权重量化更敏感：KV 误差直接进 SoftMax 被指数放大。

**Q15：W4A16 是什么？为什么权重量化到 4bit 激活还要高精度？**

W4A16 = 权重 4bit、激活 16bit。权重是静态的可离线校准；激活含系统性 outlier（LLM.int8 发现的 systematic outliers 集中在少数通道），在线量化误差直接污染每步计算。反量化开销：decode 是 memory-bound，权重反量化发生在加载时，访存量降 4 倍的收益远大于反量化计算——这就是 **AWQ/GPTQ 加了反量化反而更快**的原因（AWQ 按「激活值决定权重重要性」保关键通道，GPTQ 用二阶信息逐列量化误差补偿）。

**Q16：FP8 两种格式区别？SmoothQuant 为什么要做平滑？**

E4M3（4 指数位，精度高动态范围小，前向用）/ E5M2（5 指数位，范围大精度低，梯度/反向用）。SmoothQuant 处理「激活有 outlier、权重平缓」的不对称：数学上等价地把激活的 scale 除以因子 s、权重乘以 s（y=(x/s)·(sW)），把难度从激活迁移到权重，再走 W8A8。超参 α 控制迁移强度；判断模型是否适合看激活的 **input channel** 维度分布是否尖锐。

**Q17：INT8/FP8 量化 KV Cache 提升的是哪方面？**

Decode 是 memory-bound：KV 量化把每步注意力读取的字节数砍半/砍四倍，直接缩短访存时间——**提升的是 memory-bound 阶段的吞吐与容量**（同显存多放 2-4× 上下文），不是计算。FP8 KV Cache 在 vLLM 已 GA；跨精度复用需 producer/consumer 同 dtype（connector 不内置转换）。

→ 深读：[KV Cache 压缩与量化](/basics/compression)

## 五、投机解码与新技术

**Q18：投机解码原理？KV Cache 怎么配合？**

小 draft 模型（或 MTP 头）一次猜 K 个 token → 大模型一次前向并行验证 → 接受最长正确前缀。KV 视角三件事：**placeholder**（为猜测 token 预留槽位）、**回滚**（拒绝时不删 KV，把 `num_computed_tokens` 倒退——乐观写入逻辑回滚）、**草稿 KV**（Eagle 式 self-speculation 几乎零额外 KV）。与 prefix cache 的冲突是 block 级：投机 KV 非确定性无法 hash，但 prompt 前缀部分照常缓存。收益：acceptance 80%、K=5 时每步净增 ~4 token。

**Q19：DeepSeek V3/V3.2/R1 的架构特点？（新增热点）**

V3：MLA + MoE（256+1 专家 K=8，激活 5.5%）+ 无辅助损失负载均衡 + MTP 多 token 预测头。V3.2：**DSA 稀疏注意力**（indexer 选 token，vLLM 已支持 FLASHMLA_SPARSE 后端）。R1：在 V3 基座上 RL 训练推理能力（GRPO）。EPLB 是 MoE 专家并行负载均衡器（热点专家复制）。MTP 属于**训练+推理两侧**：训练时作辅助目标，推理时作投机解码草稿头。

**Q20：Mooncake 的 CMS / Prefix Cache / AF 分离是什么？**

CMS：16KB Count-Min Sketch 做访问频率统计，两岗位——客户端热缓存准入（命中不递增防「富者愈富」）+ Master SSD→DRAM 提升四道闸门。Prefix Cache：vLLM block hash 编进 PoolKey（`model@tp@pcp@dcp@pp@group@chunk_hash`），`batch_is_exist` 最长前缀命中。AF 分离（Attention-FFN）：注意力与 FFN 也拆开池化，Wide EP 场景下 LMSYS 128×H200 跑 Kimi K2 达 2.2k tok/s。

→ 深读：[Prefix Cache 命中与数据搬运](/core/prefix-hit)、[Mooncake 内部机制深潜](/core/mooncake-internals)

**Q21：CUDA Graph 为什么占更多显存？哪个阶段适合用？**

Graph 要求静态内存布局：为容纳动态 batch 需按最大 shape **预录制多尺寸图 + padding dummy block**（最差 ~50% 显存浪费在占位）。Decode batch 小且形状稳定，收益（消除数千次 kernel launch）远大于浪费；Prefill 形状动态大，用 piecewise 图（只录制纯计算段）或不用。**面试加分点**：KV Connector 的异步加载（`wait_for_layer_load` 网络等待）不能进图——`requires_piecewise_for_cudagraph` 就是为此暴露的接口。

**Q22：长上下文推理的挑战与方案？**

显存（320KB/token×128K=40GB/请求）→ GQA/MLA/量化；Prefill O(S²) → FlashAttention/Chunked；Decode 带宽 → 稀疏注意力（H2O/SnapKV/DSA：读得少但注意**休眠 token 风险**——API key 类内容注意力不可见却关键）+ offloading（LMCache/Mooncake 分层）+ Prefix Cache。容量规划锚点：GLM-5 MLA 每 token ~90KB，2000 session×64K ≈ 55TB——必须分层。

→ 深读：[KV Cache 压缩与量化](/basics/compression)、[验证方法与指标](/appendix/benchmark)

## 六、其余高频简答

- **Prefix Cache 机制与场景**：token→chunk→hash 链（任一 token 不同后续全变），适用 system prompt/多轮/agent 工具上下文（生产 40-80% 序列有共享前缀）；陷阱：RoPE 把位置烧进 K，「content-identical ≠ cache-identical」，跨位置复用需 CacheBlend 旋转校正；混合注意力（Gemma-3）Full 层左到右、SWA 层右到左取 min；
- **Tokenizer 优化**：vocab 覆盖与压缩比影响 token 数（间接影响 KV 与 prefill）；快路径（Rust tiktoken 类）与 special token 处理；
- **Beam Search vs Sampling**：Beam 保 K 条最优路径适合确定性任务（翻译），采样（top-p/top-k/temperature）适合开放生成；推理系统实现上 Beam 需要 batch 内维护 K 条序列与交叉 state，工程成本高，线上服务多为采样；
- **LoRA 推理时需要加载 Adapter 吗**：可以合并（merge 进权重，零额外延迟）也可以运行时挂载（多租户热切换场景，`A@B` 低秩增量按需计算）；A 用高斯初始化、B 初始化为零保证初始等价；rank 越大容量越大但过拟合与推理开销增加；
- **LLM 在 NPU 上的瓶颈**：算子生态（CANN 覆盖度）、内存带宽（HBM 代际）、通信（HCCL vs NCCL）、host 内存无 cudaHostRegister 等价物（vllm-ascend 用 pinned/mmap/GVA 三形态）——详见 [SGLang 与 vLLM-Ascend 池化设计](/ecosystem/sglang-ascend-pooling)；
- **RL Rollout 优化**：Rollout 是生成（inference 引擎接 vLLM/SGLang）、量化 rollout（FP8）提吞吐、异步 rollout（训推 overlap）提利用率；注意 MoE routing 在 RL 下可能退化（负载不均→EPLB 失效）。
