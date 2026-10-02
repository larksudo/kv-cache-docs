---
prev: /appendix/glossary
next: /interview/inference-answers
---

# 面试题库总览与备考策略

> 来源：AIInfraGuide 面试宝典（caomaolufei.github.io），2026-04 快照，共 **181 场面试、66 家公司、1,775 条题目实例，去重后 1,646 题**。原文快照保存在 `references/raw/interview/`，逐题分类清单在 `references/raw/category_output.md`。本栏目对高频题给出结合本知识库结论的精答；完整题清单见 category_output.md。

## 考什么：频次告诉你的事

四大块占总题量约 **70%**：

1. **CUDA 编程/手写算子**（270 次实例、53 家公司）——绝对第一，几乎每场都有一道手写或优化题；
2. **量化**（82 次、27 家）——INT8/FP8/FP4/W4A16、AWQ/GPTQ/SmoothQuant、量化粒度；
3. **LeetCode/算法手撕**（70 次、27 家）——LRU、链表、二叉树、DP；
4. **大模型推理优化总论 + KV Cache/PagedAttention/vLLM**（合计约 140 次）。

**2025-26 新增热点**：MLA、MoE（含 RL+MoE routing 退化）、DeepSeek 系列架构（V3/V3.2/MTP/DSA）、PD 分离、FlashDecoding、**Agent 自动生成 CUDA kernel**（字节/阿里/综合卷均已出现）。

## TOP 30 高频题族

| # | 题族（代表题干） | 次数 | 公司数 |
|---|---|---|---|
| 1 | CUDA 编程/手写算子与优化 | 270 | 53 |
| 2 | 量化（INT8/FP8/FP4/AWQ/GPTQ/粒度） | 82 | 27 |
| 3 | LeetCode/算法手撕 | 70 | 27 |
| 4 | 大模型推理优化总论 | 69 | 30 |
| 5 | 多线程/锁/协程 | 60 | 30 |
| 6 | KV Cache（原理/大小/优化/量化） | 34 | 17 |
| 7 | 并行策略 DP/TP/PP/SP/3D | 31 | 17 |
| 8 | FlashAttention 原理与版本 | 30 | 19 |
| 9 | Transformer 整体结构/手写 | 30 | 19 |
| 10 | GEMM/矩阵乘优化 | 26 | 20 |
| 11 | 显存估算/显存优化 | 26 | 16 |
| 12 | vLLM 框架原理/调度器 | 25 | 12 |
| 13 | 算子融合（何时有收益/何时变慢） | 25 | 12 |
| 14-15 | 智能指针 / 虚函数与 vtable | 23/23 | 10/11 |
| 16 | MoE（路由/负载均衡/EP） | 19 | 5 |
| 17 | ZeRO/DeepSpeed/FSDP | 18 | 11 |
| 18 | Softmax/Online Softmax 手写 | 17 | 12 |
| 19 | MHA/MQA/GQA/MLA 对比 | 17 | 11 |
| 20 | PagedAttention 原理 | 15 | 9 |
| 21 | LoRA/微调 | 15 | 6 |
| 22 | GPU 架构（H100 vs A100/Tensor Core） | 14 | 11 |
| 23 | AllReduce/集合通信（Ring 推导） | 13 | 8 |
| 24 | NPU/国产芯片（昇腾/CANN/DCU） | 12 | 10 |
| 25-26 | DDP / TP / PP / Norm 系列（各 9-10） | — | — |
| 27 | PD 分离（Mooncake/调度队列/AF 分离） | 8 | 6 |
| 28 | 单例模式（线程安全/DCL） | 8 | 4 |
| 29 | LRU Cache 手写 | 7 | 7 |
| 30 | Continuous Batching / 混合精度 / RDMA-NCCL（各 7） | — | — |

## 公司分布与风格画像

| 公司/机构 | 页数 | 风格特点 |
|---|---|---|
| 百度（17 页） | 262 题 | C++/OS 体系最深（虚函数/内存/进程线程大套题） |
| 阿里巴巴（16） | 133 | 推理系统+量化+Agent 系统设计，MTP/DeepSeek 新架构常客 |
| 字节跳动（13） | 120 | CUDA 手写密度最高，AML/抖音/豆包方向各异 |
| 腾讯（7） | 106 | 系统设计题多（SQL 执行器、分布式推理设计） |
| 快手（9） | 69 | 推理框架+CUDA 均衡覆盖 |
| 美团（4） | 61 | 推理优化+C++ 基础 |
| 蔚来（6） | 56 | C++ 深挖 + LeetCode 3M1H 车载场景 |
| 小米（5） | 50 | CUDA+系统基础 |
| 三星（3） | 40 | 基础广 + 英文自我介绍 |
| MiniMax（4） | 39 | 推理框架细节深（vLLM 源码级） |
| 卓驭（3） | 38 | 车载部署+量化 |
| 海康威视（3） | 33 | 量化+边缘部署 |
| 蚂蚁（4） | 28 | 推理+训练均衡 |
| 太初/壁仞/沐曦/摩尔线程/寒武纪/燧原 等 NPU 厂 | 各 1-3 | 昇腾/DCU/自家芯片适配、CANN、算子移植 |
| 英伟达（2） | 18 | 多线程并发题（打印/时钟夹角）+ 架构 |
| 华为（2） | 23 | CANN/昇腾生态 |
| 智源/智谱/上海AI实验室/阶跃星辰 | 各 1-2 | 前沿架构（MLA/MoE/新注意力） |

## 备考策略（基于频次的优先级）

**第一优先（覆盖 70% 分数）**：
1. 手写熟练：online softmax + FlashAttention 前向、GEMM 分块（含 split-K）、softamax/LayerNorm CUDA 版、转置（shared memory 防 bank conflict）、LRU（哈希+双链表，10 分钟内写完）；
2. 讲清楚：KV Cache 大小公式（`2 × L × H_kv × d × dtype × B × S`）与三类优化（减 per-token/减驻留/减重复）、PagedAttention 三种浪费、Prefill vs Decode 的 compute/memory-bound 分野、GQA/MLA 演进逻辑、量化粒度与 W4A16 vs W8A8 取舍；
3. C++ 基础不丢分：虚函数 vtable、智能指针三件套、四种 cast、内存对齐、移动语义、线程安全单例（DCL+atomic）。

**第二优先**：vLLM 调度器流程（Prefill/Decode 混批 + 抢占 recompute）、Continuous Batching 与 Chunked Prefill 的动机、ZeRO 三阶段显存推导、Ring AllReduce 通信量 2(N-1)/N × 数据量、TP 行列切分与通信位置、PD 分离（为什么值得——传输线性 vs 计算超线性）。

**加分项**：MLA 矩阵吸收、MoE 负载均衡（EPLB/AuxLossless）与 RL 下 routing 退化、DeepSeek 系列演进、TMA/warp specialization（Hopper）、Agent 生成 CUDA kernel 趋势、NPU 平台差异（达芬奇 Cube/CANN）。

**答题模式建议**：架构题按「问题→方案→数字→取舍」四段答（本知识库每章开头的「本篇回答的问题」就是模板）；优化题先定性 compute/memory bound（Roofline）再给方案；手写题先讲思路复杂度再落笔。

## 本栏目结构

- [大模型推理系统高频题精答](/interview/inference-answers)：KV Cache/PagedAttention/PD 分离/vLLM/量化/投机解码等约 40 道核心题的精答（结合本知识库结论）；
- [CUDA、并行与系统题精答](/interview/cuda-systems-answers)：CUDA/算子/并行策略/网络硬件/编程题；
- 扩展方向题库（241 题**逐题精答**，三篇）：[对齐与训练](/interview/extended-alignment)（PPO/DPO/GRPO/LoRA/RL+MoE）、[RAG 与 Agent](/interview/extended-rag-agent)（排障三分法/系统设计）、[ML/CV 与工程基础](/interview/extended-engineering)（OS/网络/云原生/DB/开放题）——二面/三面的差异化考察来源。

另有三篇「机制对比」深度内容沉淀在第二部分（从面试追问中提炼的对比分析）：[H2D/D2H 数据搬运对比](/core/h2d-d2h-deepdive)（gather vs 地址计算）、[异步保存与块保护](/core/async-protection)（引用计数 vs 租约）、[准入控制与防挤占](/core/admission-control)（事务式/反馈式/治理式）。
