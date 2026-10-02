---
prev: /ecosystem/components
next: /ecosystem/memfabric
---

# SGLang 与 vLLM-Ascend 的 KV Cache 池化设计

> LMCache/Mooncake 之外，两个推理引擎自身也在演化 KV Cache 池化设计。本章从两个仓库的官方设计文档与源码提取：SGLang 的 HiCache 完整设计（分层语义/锁与准入/预取流水线/写回/内存布局/PD 联动/buffer_only 模式）与 vllm-ascend 的 NPU 特有方案（AscendStore 三后端/layerwise 物理缓冲复用/稀疏解码卸载/KVPP/CaMem）。
>
> 来源：`SGLang/docs/docs/advanced_features/*`（hicache_design 等七篇）+ `python/sglang/srt/mem_cache/`（源码级）；`vllm-ascend/docs/source/developer_guide/Design_Documents/`（KV_Cache_Pool_Guide、layerwise_and_sparse_kv_cache_offloading）+ 用户指南。关键机制均标注代码位置。

## 第一部分：SGLang HiCache 设计

### 1.1 分层语义：私有 L1/L2 + 共享 L3

官方设计的核心类比是 CPU 缓存：**L1（GPU）与 L2（host）私有**于每个推理实例，**L3 集群内所有实例共享**——这是「L3 能跨实例去重」的语义基础。

**L3 元数据「按需查询、不同步」原则**（官方设计原文）：HiRadixTree 每个 node 记录一段连续 token 的 KV 所在层（可同时多层存在）；本地层维护精确地址元数据，**L3 层不存储、不同步元数据，访问时实时向 backend 查询**——避免维护分布式元数据的开销，代价是每次访问多一次查询。这与 LMCache Coordinator 的「全局目录 + 事件流投影」和 Mooncake Master 的「集中元数据」形成三种鲜明对照：**树内元数据外置 / 全局事件投影 / 集中分配**。

### 1.2 锁与准入联动（hiradix_cache.py）

- **双轨保护**：`lock_ref`（device 层）与 `host_ref_counter`（host 层）独立计数。`evict()` 只驱逐 `lock_ref==0`；`evict_host()` 只驱逐「device 已释放且 host_ref==0」的节点——L3 预取期间 `protect_host()` 防止 host 页被回收；
- **「命中数确定后才分配」**（源码注释原文："this is the whole point: no over-allocation up front"）：host 内存**延迟到 L3 命中数确定后才精确分配**——这是高并发长序列下防止预取挤占的官方答案（对比：LMCache 的 PD reservation 解决的是同类问题的另一个面）；
- 准入失败降级链：命中数 < prefetch_threshold（256 token）→ revoke；host 分配失败 → 先 evict_host 再试 → 仍失败则截断到页对齐前缀（仍须 ≥ threshold）；
- `load_back` 全有或全无（< 10 token 放弃）；`prefetch_tokens_occupied` 占用率限流作为总闸门。

### 1.3 预取三阶段流水线

队列名即阶段边界：**调度线程 enqueue**（page 对齐 + 限流 + protect_host，host_indices 尚未分配）→ **storage 查询线程**（`batch_exists` → 结果入 `prefetch_hit_queue`；调度线程以 **TP MIN all_reduce** 四队列 qsize 后取最小公共数 drain，保证各 rank 处理同一前缀）→ **I/O 线程**（`batch_get_v2` 每批 128 页 → ack → 取 Full KV 与各 sidecar 池完成页数的 **min**（clamp-to-min）→ 插树 `value=None, host_value=...`）。

三种终止策略的精确语义：best_effort 恒可终止 / wait_complete 恒不 / **timeout 线性公式 `min(max, base + per_ki_token × num_token/1024)`**（通用默认 base=2s、per_ki=0.1s、max=30s；hybrid 路径 1s/0.25s——两套默认值的分歧是版本考古线索）。终止判定由 PP rank0 决定 + `all_reduce(MAX)` 全体一致。

### 1.4 三种写回策略的实现级细节

| 策略 | 触发 | 语义 |
|---|---|---|
| write_through | hit_count ≥ 1 | 命中即 D2H；DMA ack 后 H2L3（仅传 L3 没有的） |
| write_through_selective | hit_count ≥ 2 | 二刷才写——防一次性请求污染 L3 |
| write_back（将弃用） | 驱逐时 | 驱逐才落盘；host 压力不足时 `_drop_subtree_no_host` 整棵丢弃告警 |

四个实现级不变量：① **已备份节点必须构成从根开始的连续前缀，不允许空洞**（父未备份则跳过）；② 节点分裂后 `_concat_split_chain()` 沿父链重组 token/key/hash/host_value；③ **MLA 下仅 TP0 写回**（例外清单：Kimi-K3 Mamba/KDA 状态 TP 分片、mooncake 下 MHA draft 的 rank 专属 key 需每 rank）；④ **每个 rank 必须进入 all_reduce 即使无在途写**——否则 NCCL 序列失配死锁；PP 用链式 isend relay 免全局同步点。

### 1.5 内存布局与 I/O：四种 host 池布局

| 布局 | 形状 | 目的 |
|---|---|---|
| layer_first | (2, L, size, H, D) | GPU 天然布局 |
| page_first | (2, size, L, H, D) | **同页所有层连续**——L2→L3 整页单对象零拷贝 |
| page_first_direct | (2, page_num, L, page_size, H, D) | 页内再按层分组——L2→GPU 每层传输可按页×层聚合 |
| page_head | (2, page_num, H, page_size, L, D) | **头维在外**——配合 `tp_lcm_size` 异构 TP 头分片 |

配套机制：GPU 辅助 I/O kernel 族（`transfer_kv_all_layer_mla(_lf_pf)` 等，比 cudaMemcpyAsync 快至 3×；JIT staged 写回 64 页 staging）；**MLA dense-id 技巧**——等行宽时把层偏移折进 storage_offset 得到层无关 dense id，一张 block 表服务所有层（代价是尾部一页 envelope 越界）；`L2TransferEngine` 逐层发射 + `on_layer_done` 回调（计算 N 层时加载 N+1 层）；page 对齐 4KiB 时 NIXL O_DIRECT 零拷贝直读；host 预算治理——**系统预留 10GiB，预算 = (可用 − 10GiB) / 每机 rank 数**（防同机多 rank 超卖），PP 组 MIN all_reduce 对齐。

### 1.6 存储热插拔（运行时 attach/detach）

HTTP `PUT/DELETE /hicache/storage-backend` → **`is_fully_idle()` 严格检查**（无 running/waiting/grammar/disagg/inflight 任何队列）→ 不满足即 400 拒绝且不改状态（fail fast）。detach 的顺序不变量：**必须先 drain 控制队列再清 ongoing 映射，否则 ack 无法匹配、泄漏 host 页和锁**。DP 跨节点语义：tokenizer fan-out 到所有 DP rank，**all-or-nothing 无自动回滚**——运维守则是失败后先幂等 detach 再重试。

### 1.7 PD 分离联动

- **Decode 请求四阶段生命周期**：PreallocQueue（握手 + 有 KV 就预分配；SWA/DSV4 走 **SWA-tail prealloc** 只为滑窗尾部预分配）→ TransferQueue（poll 传输）→ WaitingQueue（构造 PrebuiltExtendBatch 跳过 prefill 前向）→ RunningBatch；
- **chunked 发送**：每个 chunked prefill chunk 完成即 `send_kv_chunk`（非最后 chunk 截到 page 对齐）；最后 chunk 附状态载荷（Mamba 状态/SWA 窗口索引/DSV4 C128 请求级状态）；
- **双超时语义**：`waiting_timeout`（默认 300s）内未收到传输完成 → Failed + 向 prefill 发 ZMQ ABORT；prefill 侧另有 `bootstrap_timeout`；
- **异构 TP staging buffer**：Triton fused gather 把散落头切片收进连续 GPU 缓冲 → **bulk RDMA（O(layers) 请求替代 O(tokens×layers) 小 RDMA）** → fused scatter——高并发增益 2-5×，与同构基线差 <5%；
- **两种 PD+HiCache 拓扑**（最佳实践）：prefill-only HiCache（SystemPrompt 场景）/ prefill HiCache + decode `--disaggregation-decode-enable-offload-kvcache`（多轮对话：decode 增量 KV 按 `offload_stride` 异步 offload 回 L3 供下轮 prefill 复用，slot 释放延迟到请求结束防跨请求污染）；
- HiCache 门控 receiver：`HiCacheRestoreGatedKVReceiver` 把 KV 传输成功门控在 restore READY 状态机上（PENDING/READY/FAILED）——L3 恢复失败不让 PD 传输白做。

### 1.8 buffer_only 模式：host 不是缓存层而是 staging（新设计）

`--hicache-host-memory-mode buffer_only` 重新定义 host 的角色——**不再作为 L2 缓存，而是瞬时中转**：写路径 = 准入门控 FIFO（**StorageExistenceCache**「相信在存储中」的有界 LRU 信念表，advisory 语义 + parent-cover 门防前缀空洞 + backlog 上限）→ head-of-line D2H staging → **D2H ack 后立即解锁 device**（不等 L3 写完）→ staging 副本写 L3 → L3 ack 释放 staging。读路径 = fetch 停泊为 op-owned bounce，**prefill 准入时才 device alloc** + layer-gated H2D。配套 anchor lock 钉住拼接锚点防驱逐漂移。这个设计把「device 槽位持有时间」与「L3 写入延迟」解耦——是对 write_through 模式的结构性优化。

### 1.9 新式内存池族（mem_cache/ 目录分层契约）

`allocation.py（batch 级策略）→ hybrid_cache/（layer→pool 路由）→ allocator/ → pool/(L1) → pool_host/(L2) → storage/(L3)`，禁止向上 import。值得记录的成员：

- **DeepSeek-V4 池族**：SWA ring + C4（4× 压缩）+ C128（128× 压缩）KV 池 + 压缩状态池 + indexer 池，逐层 `compression_ratios` 决定归属；**KV 行布局 584B/token**（nope FP8 448 + rope BF16 128 + scale 7 + pad 1，页按 576 对齐）；HiCache 栈用「逻辑锚池 + 边车池」组装（锚池无张量纯逻辑页）；
- **HiSparse 双分配器**：`logical_attn_allocator`（全量逻辑槽）+ `hisparse_attn_allocator`（压缩物理槽）+ 映射表——稀疏注意力下 radix 树仍按全量逻辑槽记账，写池时翻译；
- **VMM 两段式定容**（kv_vmm_backing.py）：CUDA cuMemMap 虚拟预留 256GiB VA（free-until-committed）；`ensure_prefix` 先只背书第一页（CUDA graph capture 时 dummy 写全落 slot 0），`finalize` 再背书全量——**先 capture 后定容**，解决「图捕获需要固定地址但容量要等内存探测」的矛盾；
- **DSA 层切分**（CP prefill）：每 rank 只物化自己拥有的层，读非拥有层由 owner 广播进 remote scratch；
- **驱逐策略族**：LRU/LFU/FIFO/MRU/FILO/**Priority**（低优先先逐）/**SLRU**（probationary/protected 两段，threshold=2）。

### 1.10 session radix cache

请求带 `session_id`（只标注引用不拼接上下文）；完成时自动注册可复用叶；`/close_session` 只摘引用回正常驱逐序（软保护非 pin）。防御细节：closed-session tombstone LRU（8192）防迟到 finish 重标 + session generation 防陈旧请求。

## 第二部分：vllm-ascend 的 NPU 池化设计

### 2.1 AscendStore：统一池化连接器与三后端

设计定位（官方 KV_Cache_Pool_Guide）：把 NPU HBM/DRAM/SSD 组成统一池并使前缀跨节点可见。**没有走「LMCache + Mooncake remote backend」的 GPU 路线，而是自研 AscendStoreConnector**（原 MooncakeStoreConnectorV1），理由是「直接支持 MooncakeStore 并利用最适配 NPU 硬件的传输策略」。

三个后端（`backend/` 目录）：

| 后端 | 特色机制 |
|---|---|
| MooncakeBackend | `protocol: ascend`；A3 fabric mem 统一地址直传；嵌入式 Real Client（进程内自动 setup）；SSD offload **按全局 rank 建独立子目录**防副本 bucket 冲突 |
| MemcacheBackend | 华为 MemCache（MemFabric）；layerwise 用 COPY_L2G/G2L/G2H/H2G 方向码；`device_sdma` 懒初始化（首个请求才建 store，避免与权重加载抢内存） |
| YuanrongBackend | openEuler 缘融 Datasystem；Remote H2D 经 HIXL HCCS/RoCE 直达 NPU 显存 |

**B×R 块键设计**（与 GPU 侧的关键差异）：layerwise 模式下块键 `model@block_hash@rank`，**L 层连续排布在同一对象内按字节区间写入——对象数从 B×L×R 降到 B×R**；配套七步区间会话协议（`batch_put_start/copy_put/commit/revoke/get_start/copy_get/get_end`）+ 会话追踪器保证「未提交不可读、chunked prefill 续 chunk 逐层恢复、共享 key 最后一个 owner 才 get_end」。两级命中合并：先查 HBM prefix cache，**只从池加载增量块**。

### 2.2 Layerwise Prefill Offload：物理缓冲复用（本设计最亮的一笔）

核心公式：**物理缓冲数 = I + min(B, R)**（N 个逻辑承重层映射到少量物理 NPU 缓冲；I 独立层、R 可共享层、B 配置的共享缓冲数）——main-KV NPU 占比 ≈ 物理缓冲数 / N。共享层 round-robin 分槽，cache spec 签名不同的层不共桶；MTP 层参与同一规划。

实现方式独特：**改写 vLLM 的 KVCacheTensor 描述符**，把共槽层的逻辑 tensor 合并成一个 `shared_by=[...]` 描述（多层共享同一 NPU 地址）——设备侧池化下沉到了 vLLM 内存描述符层。复用不变式：**按物理槽而非逻辑层名跟踪完成**（多个逻辑层指向同一地址）；Memcache save 与（联合部署时）Decode 远端读都完成才可复用。`AttentionComputeStartGate`：计算流在提交注意力 op 前记录 NPU Event，传输线程等该事件后才提交——传输真正在注意力边界启动而非 Python 调用点。

### 2.3 Sparse Decode Offload：RFC #48203 的昇腾落地

分工哲学：**Prefill 每层计算大→整层搬运与计算重叠；Decode 每步计算小→整层搬运延迟藏不住**。四存储区：Prefill NPU 复用层缓冲 / Prefill host 池 / **Decode host 全量主 KV** / **Decode NPU 只留 indexer 缓存 + 每层 top-k 热缓冲**。

每步流程：新 token K/V 不写全量 NPU 主缓存（`keep_device_kv_cache=false` 时根本不分配）→ TP0 稀疏散拷 D2H 到共享 host 池（**Decode K/V 在 TP 间复制，只有 TP0 写**）→ indexer 选 top-k → **LRU 驻留表**（C++ 算子 + OpenMP 线程池）判命中/分配热槽/驱逐 → 仅 miss 行 H2D → 逻辑 top-k 重映射到物理热槽。**传输量 ∝ top-k miss 数而非序列长度**——这是稀疏注意力信号驱动 offloading 的第一个工业级落地（呼应「基础原理·压缩与量化」篇的推论）。

远程侧 `SfaRemoteD2HConnector`：Decode 用 MemFabric 从 Prefill 的 HBM **拉**主 KV 到自己 host 池、indexer 到 NPU；连续区间合并 + 按块起点轮转 TP 首属主（防小 chunk 都压 TP0）。不等 TP 时贡献组语义：P_TP ≥ D_TP 且整除，主 KV 只由 member 0 传、indexer 传不相交区间、**空块贡献者也要 ack 防死锁**。

### 2.4 NPU 平台机制（与 GPU 的本质差异）

三个「NPU 与 GPU 不同」的论点（写对比时的骨架）：

1. **多逻辑层共享物理缓冲下沉到描述符层**——设备侧池化与连接器层卸载共用同一套复用门（KVPP 同理：MLA/SFA 下 KV 按**层 ownership 分布到 TP 组各 rank**，计算需要时广播进其他 rank 的 2 个 scratch 缓冲——单机容量 6.25×，双机 PP 5.35×）；
2. **host 内存三种形态并存**：Ascend **没有 cudaHostRegister 等价物**——单层用 PyTorch pinned allocator、跨进程共享用 mmap 区、跨 rank 共享用 MemFabric 广播 GVA；三种形态带宽特性需分别压测（代码注释原话）。批量拷贝用自研算子 `swap_blocks_batch`（H2D=0/D2H=1 方向码）；
3. **传输面由启动期配置选择 + 链路级治理**：`memfabric_transfer_protocol` ∈ {sdma, device_rdma(A3), device_urma(950PR/DT)}；**QoS 0-4 级**注入（`MF_DEVICE_UB_QOS`）；**Store/PD 流量分离**（`comm_resource_config` 管 PD、`store.` 前缀管池——如 PD 走 HCCS、Store 走 ROCE 不抢链路）；fabric mem 分配必须 1GB 整倍数。

**CaMemAllocator 可插拔分配器**：基于 CANN aclrtMapMem 的 PyTorch 分配器，`use_memory_pool(tag)` 把张量按 tag（default/weights/kv_cache/sleep_persistent）分池；sleep() 时 offload tag 的分配到 pinned CPU 后全部 unmap 释放显存，wake_up() 重映射回填——**NPU 版 Sleep Mode**，且 wake 可分两段（先 weights 重载权重再 kv_cache）。

### 2.5 其余子系统速览

- **Preempt Offload**：PD 分离 D 节点被抢占时 HBM→CPU 保活重调度后恢复，免回 P 重算长 prompt——CPU 侧**完整复刻一套 vLLM BlockPool/KVCacheCoordinator** 做块管理；
- **Encoder Cache 分数制两级**：NPU+CPU 两级，评分 `score = (freq + clock) × cal_cost`——**cal_cost 是理论重算成本/存储槽**（按视觉模型配置估算），clock 每 64 请求衰减防陈旧热项；晋升按分数分位（0.2）；
- **UCM**（外部统一缓存管理器）：HBM→本地 DRAM→共享存储（NFS/3FS/企业级）三层**持久化** KV，集中式架构（类 3FS 思路），宣称 3-10× 降延迟、TTFT 至多 8×——定位介于 LMCache（插件框架）与 Mooncake（基础设施）之间的第三条路线；
- **LMCache 昇腾**：vllm-ascend 已不再注册 LMCacheAscendConnector——**连接器注册权移交 LMCache-Ascend 插件包**，与自研 AscendStore 形成互补。

## 第三部分：与 LMCache/Mooncake 的对照

| 维度 | SGLang HiCache | vllm-ascend | LMCache | Mooncake |
|---|---|---|---|---|
| 定位 | 引擎内建分层缓存 | NPU 引擎适配层 + 自研连接器 | 引擎外缓存管理框架 | 集群 KV 基础设施 |
| 全局元数据 | **不维护，按需查 backend** | 复用后端（Mooncake/Memcache） | Coordinator 事件投影 | Master 集中分配 |
| host 内存角色 | L2 缓存 或 **buffer_only staging** | 全量主 KV（sparse decode） | L1 全局分配器 | 注册 segment |
| 预取准入 | 命中数确定后才分配 | LRU 驻留表 + top-k miss | EventManager + 清理回调 | CMS 四道闸门 |
| 设备侧池化 | VMM 两段式 + dense-id | **物理缓冲复用 + KVPP 层 ownership** | GPU connector 布局族 | 无（数据面不管设备） |
| 平台广度 | CUDA/ROCm/Ascend(NIXL/mooncake) | 仅 NPU（A2/A3/950） | CUDA/ROCm/XPU/HPU/MUSA | NVIDIA/Ascend/多传输 |

三个值得吸收进主线的洞见：① **SGLang 的「L3 元数据不同步 + 按需查询」**是介于集中式与全同步之间的第三条路，代价转移到每次访问的查询延迟；② **vllm-ascend 的「物理缓冲复用公式 I+min(B,R)」**把设备侧容量问题转化为描述符变换——与 LMCache 的 connector 布局层、Mooncake 的存储层是三个不同高度的解；③ **sparse decode offload 的「传输量 ∝ top-k miss」**首次把稀疏注意力信号用于 KV 池化的生产实现——「基础原理」篇的理论推论已有工业验证。
