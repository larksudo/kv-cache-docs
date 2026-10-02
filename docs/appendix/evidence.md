# 证据索引与版本

> 来源：主报告开头版本表、第 14 章（证据索引）与第 15 章（局限性）。

# LMCache、Mooncake 与 vLLM KV Cache 管理体系深度对比

分析日期：2026-08-28  
分析范围：本地源码快照、官方设计文档、公开论文与官方/社区博客  
本地源码版本：

| 仓库 | HEAD | 最近提交时间 |
|---|---|---|
| LMCache | `a0e4a4c227f33a0ece99aad207b3ab1fae2e2bb4` | 2026-08-27 |
| Mooncake | `0518784d486eefa4343ef38b69b1dccd96b1bad4` | 2026-08-28 |
| vLLM | `5f213ed1592903b7bc38f173d320dac1b2769303` | 2026-08-28 |

> 判读规则：源码与官方仓库文档优先；官方博客用于生产化声明与部署规模；社区/博客资料只作为解释性视角。报告中的“生产可用”指具有部署路径、观测/运维能力或官方/商业案例，不等于任意业务规模下开箱即用。

---


---

## 14. 证据索引

### 14.1 本地源码/官方仓库文档

#### LMCache

| 主题 | 文件 |
|---|---|
| 项目定位 | `LMCache/README.md` |
| vLLM connector | `LMCache/lmcache/integration/vllm/lmcache_connector_v1.py` |
| MP connector | `LMCache/lmcache/integration/vllm/lmcache_mp_connector.py` |
| GPU connector | `LMCache/lmcache/v1/gpu_connector/gpu_connectors.py` |
| KV layout | `LMCache/docs/design/v1/gpu_connector/layout-invariant.md` |
| L2 controller | `LMCache/docs/design/v1/distributed/l2_adapters/overall.md` |
| P2P adapter | `LMCache/docs/design/v1/distributed/l2_adapters/p2p_l2_adapter.md` |
| Transfer Channel | `LMCache/docs/design/v1/distributed/transfer_channel/overall.md` |
| PD reservation | `LMCache/docs/design/v1/pd_async_reservation_design.md` |
| MP transfer | `LMCache/docs/design/v1/multiprocess/engine_driven_transfer_design.md` |
| Worker liveness | `LMCache/docs/design/v1/multiprocess/worker_liveness.md` |
| Coordinator | `LMCache/docs/design/v1/mp_coordinator/README.md` |
| Mooncake L2 | `LMCache/lmcache/v1/distributed/l2_adapters/mooncake_store_l2_adapter.py` |

#### Mooncake

| 主题 | 文件 |
|---|---|
| 项目定位 | `Mooncake/README.md` |
| Transfer Engine | `Mooncake/docs/source/design/transfer-engine/index.md` |
| TENT selector | `Mooncake/docs/source/design/tent/transport-selector.md` |
| Slice spraying | `Mooncake/docs/source/design/tent/slice-spraying.md` |
| Failover | `Mooncake/docs/source/design/tent/failover.md` |
| Store | `Mooncake/docs/source/design/store/mooncake-store.md` |
| SSD offload | `Mooncake/docs/source/design/store/ssd-offload.md` |
| Parallel Tensor IO | `Mooncake/docs/source/design/store/unified-parallel-tensor-io.md` |
| Ascend Direct/HCCS | `Mooncake/docs/source/design/transfer-engine/ascend_direct_transport.md` |
| Ascend transport | `Mooncake/docs/source/design/transfer-engine/ascend_transport.md` |
| vLLM Store worker | `vllm/vllm/.../mooncake/store/worker.py` |
| vLLM direct connector | `vllm/vllm/.../mooncake/mooncake_connector.py` |

#### vLLM

| 主题 | 文件 |
|---|---|
| KV Connector V1 | `vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py` |
| 旧抽象说明 | `vllm/vllm/distributed/kv_transfer/README.md` |
| LMCache wrapper | `vllm/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py` |
| LMCache MP wrapper | `vllm/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py` |
| Mooncake direct | `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py` |
| Mooncake Store | `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py` |
| Mooncake Store usage | `vllm/docs/features/mooncake_store_connector_usage.md` |

### 14.2 已下载外部资料

| 类型 | 文件 |
|---|---|
| FAST25 论文 | `references/raw/mooncake_fast25.pdf` / `.txt` |
| FAST25 slides | `references/raw/mooncake_fast25_slides.pdf` / `.txt` |
| LMCache P2P 官方博客 | `references/raw/lmcache_p2p_blog.html` / `.txt` |
| LMCache MP 官方博客 | `references/raw/lmcache_mp_blog.html` / `.txt` |
| LMCache CoreWeave/Cohere | `references/raw/lmcache_coreweave_blog.html` / `.txt` |
| vLLM x Mooncake Store | `references/raw/vllm_mooncake_store_blog.html` / `.txt` |
| 微信文章 1 | `references/raw/wechat_1.html` / `.txt` |
| 微信文章 2 | `references/raw/wechat_2.html` / `.txt` |
| KV offloading 系列 | `references/raw/09_inference_system__kv_cache__*` |

---

---

## 15. 局限性

1. 本报告基于 2026-08-28 本地快照；这些项目更新极快，具体 PR 可能很快过时。  
2. 没有在本地 RDMA/HCCS/NVLink/GDS 集群运行端到端 benchmark；性能结论来自源码、文档和公开数据，不构成可复现实测。  
3. 微信文章为二手解释，正文已抓取，但部分 HTML 抽取可能保留页面噪音；关键架构判断仍以源码和官方文档为准。  
4. “生产化”判断受公开信息限制；私有部署规模无法验证。  
5. Mooncake 中很多模块处于快速演进状态，TENT、Engram、DFS、PG/EP 的生产边界需要按版本逐项确认。  
6. LMCache 的多引擎/多后端矩阵太大，本报告重点覆盖 vLLM、MP、NIXL、GDS、P2P、Mooncake 等主线，没有穷举所有 backend。  

---

---

## 博客系列引用（2026-09-19 增补）

本知识库第一部分「基础原理」与各篇「增量」附录融入了两个博客系列的重要增量内容（证据等级 C，关键数字多转引自论文/官方口径，作者自注未独立复现；含部分 2026 年新架构如 CSA/HCA、GLM-5，引用前建议回查原始论文）：

**ForceInjection KV Cache 系列（42 篇，已抓取 23+24 篇）**：

- 基础：kv_cache_basics、paged_attention、why_only_kv、attention_kv_cache_formats（MHA/GQA/MQA/MLA/CSA-HCA）、sparse_attention_taxonomy；
- 前缀缓存：radix_attention（Radix Tree vs APC 对比）、prefix_caching、claude_prompt_caching、rope_and_prefix_caching（三方案）；
- 调度：chunked_prefill、spec_decode、cuda_graph（requires_piecewise_for_cudagraph）；
- PD：pd-state-handoff-optimization-map（四轴框架）、disaggregated_prefill_kv_transfer（Push/Pull、Eager/Pipelined）；
- 卸载：kv_offloading（vLLM native vs LMCacheConnector）、layerwise_pipeline、kv_cache_prefetching（三层预取）、sparse_attention_driven_offloading_problems（八问题框架）；
- 压缩：kv_cache_compression（四维冗余模型）、kv_cache_quantization（五模式）；
- 淘汰：attention_sinks_and_eviction、key_key_semantic_affinity（SamKV）；
- 跨模型：cross_model_kv_transfer（NVIDIA 线性映射）；
- 容量：kv_cache_roi、glm5_kv_cache_capacity_planning（三层推演模板）；
- 系统：KVBM_Analysis、hicache_deep_dive、mooncake_architecture、nixl_introduction、tair-kvcache、LMCache 系列 16 篇（**LMCache 系列 16 篇已全部深潜消化**，函数级细节见「LMCache 内部机制深潜」篇：engine/store/retrieve 流水线、Allocate-Evict 循环、四个磁盘后端对比、Controller 三通道协议、四 Connector 零拷贝分级、CacheBlend/CacheGen 实现细节与源码-论文偏差）。

已保存 markdown 快照位于 `references/raw/09_inference_system__*`；线上完整目录见 forceinjection.github.io。

**SGLang 与 vllm-ascend 官方设计文档（2026-09-19 增补）**：SGLang hicache_design/best_practices/runtime-attach-detach/pd-disaggregation/epd/session-radix 七篇 + mem_cache 源码（HiRadixTree 锁与准入、预取三阶段、四种 host 布局、buffer_only 模式、VMM 两段式、DSV4 池族）；vllm-ascend KV_Cache_Pool_Guide 与 layerwise_and_sparse_kv_cache_offloading 两份设计文档 + kv_pool 用户指南（AscendStore 三后端、物理缓冲复用 I+min(B,R)、sparse decode offload、KVPP、CaMemAllocator、UCM）。详见「SGLang 与 vLLM-Ascend 池化设计」篇。

**LMCache 官方博客（blog.lmcache.ai，2026-09-19 补充抓取 21 篇）**：MP 数据传输路径（gather/scatter + 三路径拷贝数）、Device-DAX/CXL 内存层、MI300X 真实 Agent 负载基准（regime crossover/合成基准说谎/三大踩坑）、CacheBlend OpenClaw 实测（98% vs 48% 命中）、静态加密 serde（AES-GCM 29B/chunk）、KV Cache 编辑 SDK（Q ring buffer）、MooncakeStore L2 七 PR 演进（per-op worker pool 35× p99）、SageMaker HyperPod 托管 L2 与 PD 分离（四层栈 LMCache→NIXL→libfabric→EFA）、TurboQuant、Dynamo 1.0 三层集成、「OpenAI API 细腰」与「KV Cache 一等数据对象」观点文。要点已融入「LMCache 内部机制深潜」「生产化边界与公开声明审计」「威胁模型与安全」「验证方法与指标」各篇附录。

**「一研」微信系列**：

- 《KV Cache 在 Mooncake 系统中到底长什么样？——一条数据的六次变身》：`references/raw/wechat_1.html`（UTF-8 提取版 `wechat_1_utf8.txt`；注意原 `wechat_1.txt` 为 GBK 乱码）。其关键数字已逐项源码复核（kMaxSliceSize=16,777,200 ✓、桶 256MB/500key ✓、MooncakeAgentMetadata ✓；split_k_and_v 位置需注意 wheel 包 vs vllm 仓库版本差异）；
- 《LMCache 传输协议详解：从 PCIe 到 RDMA 的协议栈全景》：`references/raw/wechat_2.html`（提取版 `wechat_2_utf8.txt`）。覆盖 PCIe/CUDA Stream/RDMA/NIXL/ZMQ 五层协议栈与 LMCache 实现对照、协议性能对比表（1.5GB KV Cache 各通路传输时间）。

## 一研系列 43 篇 → 章节映射表

> 每篇文章的核心内容落在哪个章节，点击章节名直达。与已有覆盖高度重复的文章（TE 详解/一文读懂/六次变身等）不重复成节，其独有增量仍被吸收进对应位置。

**Mooncake 系列（31 篇）**

| 文章 | 内容落点 |
|---|---|
| 元数据结构与存储（1024 分片/双路径 OpLog/Fork 快照） | [Mooncake 内部机制深潜](/core/mooncake-internals) §1-2（全文骨架） |
| 元数据与数据流（Master 不碰数据，TE 不碰元数据） | 深潜 §8（三不变量 + KV 尺寸速查表 + 部分成功协议） |
| Count-Min Sketch：16KB 管百万 KV | 深潜 §3（两个岗位：客户端准入 + Master 提升） |
| 当故障发生时（水密舱室/心跳探针/弹性恢复） | 深潜 §4 |
| SSD 分层存储（热货架与冷库经济学） | 深潜 §5 |
| Expert Parallel 两篇（EP 详解 + MoE 前置） | 深潜 §6.1 |
| Process Group 后端（缺席跳过的集合通信） | 深潜 §6.2 |
| P2P Store（一火传千灯的权重分发） | 深潜 §6.3 |
| 硬件交互（GPU/DRAM/SSD 数据路径） | 深潜 §7 |
| 四个工程技巧（SIEVE/尾部合并/建连开销） | 深潜 §8 |
| Mooncake 18 问 | 深潜 §9 |
| 部署实践 + 测试实践 | 深潜 §10 |
| 控制平面与数据平面分离架构 | [控制面与数据面](/core/control-data-plane) 附录（模式理论） |
| 性能调优实践 + 部署踩坑实录 | [验证方法与指标](/appendix/benchmark) 附录（调优速查 + 八坑） |
| 安全实践（六道信任边界） | [威胁模型与安全](/ecosystem/security) 附录二（RKey 攻击链） |
| vLLM Mooncake KV Connector 详解 | [vLLM Connector 对接](/core/vllm-connector) 附录（七步流程 + 配置坑） |
| 拓扑感知路由 + NUMA 前置篇 | [物理通路](/core/transfer-paths) 附录二 + 深潜 §4/§7 |
| RDMA 传输实现 + RoCE 三篇（基础/详解/演进史） | 物理通路附录二（RoCE 演进五阶段/DCQCN/IB 四特性） |
| Store 详解（中央仓储） | 深潜 §1 + §8（PyClient API 遗补） |
| PD 分离详解 | [P2P 与 PD 分离](/core/p2p-pd)（开销占比速查） |
| 生态集成（vLLM/SGLang/TRT-LLM） | [横向组件对比](/ecosystem/components) 附录（集成全景 + EPD） |
| Transfer Engine 详解（立交桥） | 已由[物理通路](/core/transfer-paths)与[源码审计](/ecosystem/source-audit)覆盖（重复度最高） |
| 一文读懂 Mooncake | 已由[总结论](/guide/conclusions)与[总体架构](/core/architecture)覆盖 |
| KV Cache 六次变身 | 已由[存储引擎中的数据变形](/core/storage-transforms)全文吸收（关键数字已源码复核） |

**LMCache 系列（12 篇）**

| 文章 | 内容落点 |
|---|---|
| GPU 连接器与内存管理 | [LMCache 内部机制深潜](/core/lmcache-internals) §8（11 种 KV 格式/三层分配器） |
| 设计模式解析（可复用的架构智慧） | 深潜 §7（七大设计模式） |
| D2H 与 H2D 深度解析 | 深潜 §9 尾部（pinned/pageable 精确机制与数字） |
| 存储引擎深潜（分层存储之旅） | 深潜 §9（WeightedSemaphore/TTLLock/12 种 L2 适配器） |
| 性能优化（空间换延迟/异步换吞吐） | 深潜附录（八手段表 + 场景映射） |
| 踩坑与避坑（文档没写的暗礁） | 深潜 §10（10 坑，与 MI300X 三坑互补） |
| 存储与传输全景 + 五层存储体系 | 深潜附录（五层口径对照/L0 显式化/MaruBackend） |
| 一文读懂 LMCache（AI 原生知识） | [总结论](/guide/conclusions) + 深潜官方博客附录（愿景呼应） |
| 透过 vLLM 详解 Offload Memory | [vLLM Connector 对接](/core/vllm-connector) 附录（四态 lookup/ARC/Sleep Mode） |
| 物理网络详解（总线到机架） | [物理通路](/core/transfer-paths) 附录二（同机架原则/PD 带宽公式） |
| 传输协议详解（PCIe 到 RDMA） | 物理通路附录（协议栈全景 + 性能对比表） |

原文快照：`references/raw/yiyan/`（43 篇 HTML + UTF-8 文本）。收益类数字为一研转述口径（证据等级 B/C）。

> **2026-09-19 更新：用户提供完整链接后，一研系列全部 43 篇已批量抓取本地化**（`references/raw/yiyan/`，零失败）——Mooncake 系列 31 篇（元数据结构/18问/工程技巧/控制面数据面/故障体系/CMS/元数据数据流/部署/测试/安全/调优/踩坑/生态/P2P Store/PG/EP 两篇/SSD 分层/Store 详解/PD 分离/拓扑路由两篇/RDMA 与 RoCE 三篇/TE 详解/一文读懂/硬件交互/vLLM Connector）+ LMCache 系列 12 篇（GPU 连接器/设计模式/D2H-H2D/存储引擎深潜/性能优化/踩坑/全景/五层存储/一文读懂/vLLM Offload Memory/物理网络/传输协议）。关键断言已源码验证（1024 分片 ✓、CMS 16KB 双场景 ✓、fork 快照 ✓、OpType 枚举 ✓）。要点已融入「Mooncake 内部机制深潜」「LMCache 内部机制深潜」「vLLM Connector 对接（原生 kv_offload 附录）」「物理通路（网络拓扑附录）」「威胁模型与安全（六道信任边界）」「验证方法与指标（调优速查与踩坑）」。收益数字（TTFT 50%+、吞吐 2-3× 等）为一研转述口径，证据等级 B/C。

另：`references/raw/01_hardware_architecture__*` 含 GPUDirect RDMA/Storage 与 PCIe 综述两篇，为「Host-GPU-RDMA 寻址与零拷贝」篇的底层硬件背景材料。


---

## 面试题库来源（2026-09-19 增补）

AIInfraGuide 面试宝典（caomaolufei.github.io，Astro 静态站）：181 场 AI Infra 方向面试（66 家公司/机构 + 综合面经卷），2026-04 快照。原文快照 `references/raw/interview/`（181 个 txt，页眉含公司/轮次/日期）；去重分类中间数据 `references/raw/category_output.md`（1,646 题 × 八大类全文）与 `merged_categorized.json`。要点融入「面试题库」七页（含扩展方向 241 题逐题精答三篇）与第二部分三篇「机制对比」（H2D/D2H 数据搬运：`multi_layer_kv_transfer` kernel 的 gather/scatter 语义、GPU staging 两跳的三个理由、CudaIPCWrapper/RawCudaIPCWrapper 与 SHM/pickle 三路径对比——全部对照 `gpu_connectors.py`/`mem_kernels.cu`/`ipc_wrapper.py` 源码；异步保存与块保护：LMCache ref_count/pin_count/PinMonitor/TTLLock vs Mooncake store_job_id 台账/lease/zombie 清理，根因「信任域边界」；准入控制：LMCache PD reservation 事务式 vs SGLang no-over-allocation 反馈式 vs Mooncake 四道闸门治理式，根因「各自命门」）（含扩展方向：RAG·Agent·对齐·工程基础）：总览与备考策略（TOP 30 题族/公司画像/优先级）、推理系统高频题精答（22 题族）、CUDA 并行与系统题精答。


---

## MemFabric Hybrid（2026-09-19 增补）

本地克隆 `memfabric_hybrid/`（gitcode.com/Ascend/memfabric_hybrid，华为昇腾内存池化软件）。逐主题源码审计：GVA 统一编址（对称窗口除法算术/devmm ioctl/HalMem 巨页/128T 布局）、xcopy 十引擎矩阵与 batch 设计（8190 desc 单 kernel/16K 段 extend/AIV 直写 RoCE WQE）、三 API 寻址模型（BM rank 窗口/Trans uniqueId/SHM 卡侧对称槽）、acc_offload 四算子（sparse_copy 参数全驻留 device）、控制面（acc_links TCP/config store 三后端/etcd 仅选主/leader 主动退位防脑裂）、QoS 建链属性、ptracer 分位打点、限制清单（A2 HCOMM 动态 MR 快照问题等）。详见「MemFabric Hybrid 深度分析」篇。
