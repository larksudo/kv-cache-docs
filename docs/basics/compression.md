---
prev: /basics/prefix-caching
next: /core/architecture

---

# KV Cache 压缩与量化

> 减小 per-token 体积的三条路线：量化、稀疏/淘汰、跨层与跨模型复用。本篇给出统一框架、vLLM 生态的落地现状，以及与缓存系统的交互边界。


> **🎯 面试考察点**（66 家公司真题库 · KV 量化）：- 量化题族——27 家公司 82 次（题库排名第 2）：粒度三档区别、W4A16 为何激活要高精度、FP8 两种格式、KV 量化提升哪方面性能；
> - 追问链：量化误差为何被 SoftMax 放大 → 跨精度缓存复用约束 → TurboQuant/CacheGen。
> 精答见[面试题库 Q14-Q17](/interview/inference-answers)

## 四维冗余模型

KV Cache 的冗余分布在四个维度，总压缩率近似相乘：

```text
C_total ≈ C_token × C_feature × C_structure × C_system
```

| 维度 | 冗余来源 | 代表技术 |
|---|---|---|
| Token 维 | 不是每个 token 都重要 | SnapKV、H2O、KVzap、Lookahead Q-Cache |
| Feature 维 | 数值精度过剩 | KIVI、FP8/INT4 量化、低秩、向量量化 |
| Structure 维 | 跨层相似 | MiniCache、CommonKV、xKV |
| System 维 | 重复存储/搬运 | Offloading、Paging、Prefix Caching |

不可能三角：**压缩率、精度、吞吐难以同时兼顾**；隐藏缺陷是「负样本」问题——全量 KV 能答对的样本，压缩后可能出现幻觉。

## 量化：为什么 KV 比权重更敏感

权重的量化误差在数百层末端才体现；KV 的量化误差**直接作用在 SoftMax 输入上被指数放大**：量化 K → QK^T 偏差 → SoftMax 注意力错配 → V 聚合偏差。因此 KV 量化的粒度要求更高。

vLLM 的 KVQuantMode 五模式：

| 模式 | 精度损失（PPL） | 状态 |
|---|---|---|
| FP8_PER_TENSOR | <0.1 | 当前默认，GA |
| INT8_PER_TOKEN_HEAD | 0.1-0.3 | 可用 |
| FP8_PER_TOKEN_HEAD | 可忽略 | 可用 |
| INT4_PER_TOKEN_HEAD | — | 已合入主干（分组量化） |
| NVFP4（fp4 数据 + fp8 block scale） | 0.3-1.0 | 仅 B200 |

两个粒度「轴」正交：NVFP4 沿 head_dim 每 16 元素一个 block scale（控 outlier 维度）；per-token-head 沿 token×head 维度（隔离 token 分布差异）。scale 元数据开销很小（LLaMA-2 70B@32K 约 1 MB，vs KV 本身 10 GB）。

**与 prefix caching 兼容**：hash 基于 token ID 不依赖 K 值，FP16/FP8 下 hash 一致；但跨精度复用需 producer/consumer 同 dtype，vLLM Connector API 不内置精度转换——PD 混合精度是 connector 实现者的责任（对 LMCache connector 的直接影响）。

## 稀疏与淘汰：读路径的优化

### Attention Sinks：前几个 token 的特殊性

SoftMax「强制消费」（权重和恒为 1，多余注意力必须有去处）+ 因果掩码（前几个 token 对所有后续位置可见）→ 前 4 个 token 成为注意力接收器。这与语义无关、与位置编码无关（RoPE/ALiBi 均如此），是自注意力的数学结构属性。StreamingLLM 的实验：只保留最近 W 个 token 时 PPL 断崖崩塌；保留前 4 个 sink token 后，4K 预训练模型可以跑 4M token。

**滑动窗口失败的根源不是信息丢失，而是破坏了注意力分布的数值稳定性**——去掉 sink 后某个微小正内积的 token 会成为不稳定的「伪 Sink」。

### 淘汰策略谱系

| 方案 | 机制 | 关键数字 |
|---|---|---|
| StreamingLLM | sink + 滑动窗口 | 4M token 稳定推理 |
| H2O | 累积注意力分数贪心（次模保证 (1-1/e) 近似最优） | 保留 20% token 维持精度，吞吐最高 29× |
| SnapKV | 观察窗口（末 32 token）投票 + 按头聚合 + 1D max pooling 平滑 | 16K 输入 8.2× 压缩、3.6× 生成提速 |
| SamKV | Key-Key 语义亲和：block 的 Key mean-pool 与近期 anchor 的 Key 内积 | Qwen2.5-7B 上与 QK^T 真实注意力 top-K 重叠率 100% |

SamKV 绕开了循环悖论（要知道 block 重要需算 QK^T，但算 QK^T 要求 block 已在 GPU）：Key 向量本身是语义空间投影，block 内全 token/head/layer 的 Key mean-pool 得到约 500 B 的探针向量，写入时算一次。

### 「休眠 token」：阿喀琉斯之踵

API key、密码、常量在注意力分布中不可见但关键时刻至关重要——Transactional Attention（2025）实测 SnapKV 在凭据检索任务上准确率接近 0%，**回退 StreamingLLM 同样解决不了**。这是所有注意力驱动淘汰的共同风险。

### 术语澄清：eviction 的两层含义

- 论文语境（H2O/SnapKV）：**永久删除**，不可逆，需要次模最优性保证；
- 工程语境（vLLM/LMCache）：**层级迁移**（GPU→CPU→SSD），可逆，错了纠正即可。

同一个注意力信号，指导「搬到 CPU」的质量门槛远低于指导「永久删除」。一套评分基础设施可同时服务两条决策线。

### vLLM 落地现状（2026 main）

- FP8 KV Cache GA；4-bit 分组量化已合入主干（MLA 有物理形态障碍）；
- **Token Pruning（SnapKV 类）被 vLLM 官方 closed as not planned**——动态稀疏破坏 PagedAttention block 内存连续性假设；
- vLLM V1 的 Preemption 只有 Recompute 模式（SWAP 有意移除，设计假设是 prefix cache 可让恢复大多命中）。

## 跨模型 KV 复用：一个新维度

Prefix caching 的隐含前提是「缓存的产生者和消费者是同一个模型」。换模型后缓存全部作废（无报错，只是永远不命中）——而 L3 共享缓存层 TB 级的字节全是 GPU 算力换来的。agent 负载实测（4300 个 session、35 万步推理）：fresh token 仅占 19%，正常时 81% prefill 由缓存承担，缓存失效全部回到 GPU。

NVIDIA 的跨模型 KV transfer（Qwen3 14B→32B，线性映射）：cross-layer selection + RoPE factoring + per-head ridge 闭式解。结果的光谱很宽：最好的对（14B→32B）retention 97.6%，最差的（Ministral 3 8B→14B）仅 41.6%（GSM8K 1.6%）——**平均分掩盖断层**。且 R² 不能预测 retention（Pearson r = -0.20），拟合前无法预判哪对模型可用；MLA 与混合注意力被根本排除。它做的是「事后补救：用统计方法拟合一个本可在架构设计阶段消除的差异」。

## 与本知识库主线的连接

1. **CacheGen**（LMCache 预研方向）：自适应量化 + 流式算术编码，KV 传输压缩 4.3×（精度损失 <2%，SIGCOMM 2024）——KV 视作流媒体的传输层压缩；
2. **量化边界**（重要澄清）：Mooncake Store 不主动量化 KV Cache，Transfer Engine 只搬运已量化字节；Mooncake EP 的 FP8 是 hidden-state dispatch 载荷，不是 KV 量化（详见「Prefix Cache 命中与数据搬运」篇第 10 章）；
3. **稀疏信号 → offloading 的未来**：注意力分数天然产出「哪些 KV 被反复读取」的信号，offloading 决策可从按时间（LRU/FIFO）升级为按语义重要性——但注意信号是「自我实现的预言」（被卸载的 block 永久失去证明价值的机会，需 occasional recall）。

一句话：**KV Cache 正在从「大模型推理的被动中间状态」演化为「可压缩、可分层、可调度的主动数据管理系统」**——这正是 LMCache 与 Mooncake 存在的理由。
