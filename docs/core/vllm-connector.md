---
prev: /core/p2p-pd
next: /core/reliability

---

# vLLM Connector 对接

KV Connector V1 的 scheduler/worker 双面接口语义、LMCache 三种接入形态、Mooncake 两类 connector，以及 MultiConnector 组合模式。

> 来源：主报告第 5 章全文。


> **🎯 面试考察点**（66 家公司真题库 · vLLM 框架）：
> - 「vLLM 核心优化技术？」——12 家公司 25 次（题库排名第 12）；「vLLM 和 SGLang 差别？」「请求被抢占后怎么处理？」；
> - 追问链：调度器流程 → Connector V1 双面接口 → 原生 kv_offload 四态模型 → MultiConnector 组合。
> 精答见[面试题库 Q8-Q10](/interview/inference-answers)

## 5. 上层 vLLM 对接

### 5.1 通用 KV Connector V1

vLLM 的接口已经从“搬 tensor”演进为“搬调度决策 + 搬 KV”。`KVConnectorBase_V1` 的关键语义：

1. **lookup 是 side-effect free 且可重入**  
   `get_num_new_matched_tokens()` 可能被多次调用。

2. **分配后才真正绑定 block**  
   `update_state_after_alloc()` 告诉 connector block id，外部系统才知道往哪些 paged GPU buffer 加载。

3. **layerwise 是可选优化**  
   `save_kv_layer()/wait_for_layer_load()` 允许按层流水；但 Mooncake Store 当前不支持 layerwise，LMCache 支持多种 layerwise connector。

4. **异步完成必须显式上报**  
   `get_finished()` 返回完成 save/send 与 load/recv 的 request id。

5. **HMA/hybrid group 成为新复杂度来源**  
   `SupportsHMA.request_finished_all_groups()` 说明 hybrid attention、Mamba、cross-layer 等不再是单一 block list。

### 5.2 LMCache 接入 vLLM

LMCache 有三种接入形态。

#### 5.2.1 vLLM 官方内置 wrapper

`vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py` 中 `LMCacheConnectorV1`：

- `use_native=true` 时使用 vLLM 内置 `lmcache_integration`；
- 否则 lazy import 最新 LMCache 仓库；
- 将 scheduler/worker 方法委托给 `LMCacheConnectorV1Impl`；
- 保存 LMCache KV events 并聚合成 vLLM `BlockStored`。

#### 5.2.2 LMCache 侧最新 adapter

`lmcache/integration/vllm/lmcache_connector_v1.py` 与 `vllm_v1_adapter.py` 实现完整逻辑。LMCache 可以在 lookup 阶段返回外部命中 token 数，在 allocation 后启动 GPU load，在 layerwise 模式中逐层 save/load，并利用 lazy offload 延迟释放 GPU block。

#### 5.2.3 MP connector

`lmcache_mp_connector.py` 是更重的集成：

- scheduler adapter 与 worker adapter 连接独立 MP server；
- worker 注册 KV layout/IPC handle 或 engine-driven context；
- request tracker 维护 `PREFETCHING -> WAITING_FOR_LOAD -> READY`；
- MLA TP 特殊处理；
- heartbeat/recovery/instance id 解决进程故障。

MP connector 不只是“cache 后端”，而是一个独立缓存进程的协议边界。

### 5.3 Mooncake 接入 vLLM

Mooncake 有两类 vLLM connector。

#### 5.3.1 Mooncake Transfer Engine Connector

路径：

```text
Prefill Scheduler -> MooncakeConnectorMetadata
  -> Decode Scheduler/Worker bootstrap
  -> TransferEngine read/write paged KV
```

它处理 direct PD、TP 映射、PP layer 对齐、传输状态、pull 请求超时和请求取消。适合低延迟 PD。

#### 5.3.2 Mooncake Store Connector

路径：

```text
Scheduler lookup hash -> external hit tokens
Worker background thread batch put/get MooncakeDistributedStore
Mooncake Store Master placement + TransferEngine data path
```

适合跨实例共享池。`MooncakeStoreConnector` 明确实现 `SupportsHMA`，但 worker 层不支持 layerwise load/save，而是在 `get_finished()` 中做异步搬运。

### 5.4 MultiConnector 组合

vLLM `MultiConnector` 可以串联多个 connector。vLLM 官方 Mooncake Store 博客给出的推荐结构是：

```text
PD connector + MooncakeStoreConnector
```

含义：

1. Prefill 直接把当前请求 KV 给 Decode；
2. 同时把 KV 写入共享池；
3. 后续请求即使路由到别的节点，也能从池中恢复 prefix；
4. 未来多路径加载可以同时从 producer 和 pool 拉数据。

这是当前最完整的 vLLM 侧模式：**低延迟直传 + 跨实例持久复用**。LMCache 的 MultiConnector/remote backend 思路类似，但 Mooncake 在 Store 数据面上做得更重。

### 5.5 接入对比

| 维度 | LMCache | Mooncake |
|---|---|---|
| connector 数量 | in-process connector + MP connector | Transfer Engine connector + Store connector |
| layerwise | 多个 layerwise GPU connector | Store connector 不支持；direct connector 有 block/layer metadata |
| HMA | 逐步适配，MP connector 限制 hybrid manager | Store connector 实现 SupportsHMA，限制若干 hybrid/mamba 情况 |
| 异构 TP/PP | MP 有 MLA/TP/PP 策略 | direct connector 有显式 TP ratio/region plan |
| KV events | 聚合 BlockStored，进入 vLLM event path | Store connector 聚合 KV events |
| async completion | get_finished 支持异步 save/load | Store worker 后台线程 + get_finished |
| 部署边界 | 可嵌入也可独立 daemon | embedded store / standalone store / master+metadata |

---

## 附：Mooncake direct PD Connector 的七步传输流程与配置细节（一研 vLLM Connector 篇）

> 此前章节覆盖了接口语义；这里补齐一研实测整理的工程细节（多数为本知识库此前未收录）。

**七步流程**（mooncake_connector_v1.py）：① Proxy 注入 `kv_transfer_params{do_remote_decode:True}` + **`max_tokens=1`（Prefill 只算不生成——显存峰值控制的关键技巧）**；② Prefill `request_finished()` 延迟释放块并返回 `{do_remote_prefill, remote_host, remote_port}`；③ Proxy 转发给 Decode；④ Decode `start_load_kv()` 构造 `MooncakeAgentMetadata` 经 ZMQ REQ（msgpack）发 Prefill；⑤ Prefill ROUTER 收到后 `_build_transfer_params()`——**`group_consecutive_contiguous()` NumPy 向量化合并连续 block**（[3,4,5]+[10,11,12] 各合并为一次写入）；⑥ `batch_transfer_sync_write()` RDMA 直写；⑦ TRANS_DONE 确认 + 双方清理。

关键细节：TE 初始化 `initialize(hostname, "P2PHANDSHAKE", ...)`——PD 场景 Proxy 已传地址无需 etcd；**MooncakeBootstrapServer** 运行在全局 rank 0 Prefiller Worker 做集中注册（默认端口 8998）；`send_meta.expire_time` 默认 120s 防 Decode 拉取丢失致 Prefill 显存泄漏；线程模型 kv_producer 10 workers（实际 20 任务超配防异步等待空闲）。

**三个配置坑**：① **PYTHONHASHSEED=0 必须设**（跨进程哈希一致性）；② **MultiConnector 下各 connector 的 block size 等参数必须对齐，否则严重精度问题**；③ Store 模式 enable_offload 需**三端对齐**（master flag + owner client + vLLM JSON）；`cache_prefix` 做命名空间隔离防不同部署互相污染；`enable_cross_layers_blocks` 跨层打包减少 store 次数；SSD offload 通常 standalone-store 模式（外部 mooncake_client 拥有池）。

## 附：vLLM 原生 KV Offload（OffloadingConnector）——与 LMCache 的对比拼图

> 来源：「一研」《透过 vLLM 详解 Offload Memory》。vLLM 原生 `vllm/v1/kv_offload` 是 OffloadingConnector 的内核，理解它才能说清「什么时候还需要 LMCache」。

**核心抽象 `OffloadingManager`**：`lookup()` 返回**四种状态**——HIT / MISS / **HIT_PENDING（正在传输中）/ RETRY（传输已启动稍后再试）**（比 LMCache 的二元命中模型多了两个中间态）；`prepare_store/prepare_load/complete_store` 三段式生命周期。

四个 LMCache 没有的机制：

1. **store_threshold（默认建议 1-2）**：一个块必须被查找至少 N 次才允许 offload——防「刚存进去又要搬回来」的抖动；
2. **ARC 淘汰策略**（可选，默认 LRU）：四象限 T1（最近访问一次）/ T2（频繁）/ B1/B2（幽灵列表）——B1 命中增大 T1 容量偏新面孔、B2 命中偏老主顾，适合访问模式波动大的场景；
3. **级联 Store**（TieringOffloadingManager）：GPU→CPU 主层级→自动级联所有二级存储每级留副本（与 LMCache Write-All 相似）；Load 逐级提升；
4. **SharedOffloadRegion**：多卡 mmap 共享 `/dev/shm/vllm_offload_{id}.mmap`，内存布局**交错排列**（worker0_block0 | worker1_block0 | ...）。

数据搬运：每方向一个 `SingleDirectionOffloadingHandler` + 专属 CUDA Stream，同方向按序串行，Store 方向 wait_stream 计算流。Triton 内核调优：**28KB 以下用 Triton、以上回退 C++ swap_blocks_batch**（H100 Gen5 常量）。`offload_prompt_only` 默认 true（decode 块正在使用，offload 意义小）。

**Sleep Mode 两级别**：Level 1 权重 offload 到 CPU、KV 丢弃；Level 2 全丢但先保存 buffer（RLHF 权重更新场景）。需 `--enable-sleep-mode` + `enable_cumul_allocator=True`。

**选型对照**：vLLM 原生（深度集成、低延迟、省维护）| SGLang HiCache（三级+异步预取）| LMCache（**跨引擎共享、持久化、非前缀复用 CacheBlend、独立进程故障隔离**）| HF Transformers（单卡小批量）。作者观点：vLLM 原生已演进到较成熟，若性能不差可优先；需要跨实例/跨引擎/持久化时上 LMCache。
