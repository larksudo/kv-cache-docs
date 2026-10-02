---
prev: /ecosystem/production-claims
next: /ecosystem/sglang-ascend-pooling
---

# 横向组件对比：Dynamo/KVBM、SGLang HiCache、FlexKV、Tair 与其他

除 LMCache/Mooncake/vLLM 三主线外，KV cache 生态还有一批值得参照的组件。本篇先给出六层比较框架与两个未被源码审计覆盖的系统（Tair KVCache、DeepSpeed），然后是 Dynamo/KVBM、SGLang HiCache、FlexKV 的行号级源码审计。

> 来源：V7 第 1/5/6 章（V7 中 KVBM/HiCache/FlexKV 三章已被 V9 源码审计更正，此处不再保留旧文）+ V9 全文。

## 1. 比较框架

不要把所有叫 “cache” 的系统放在同一层。至少区分六类：

| 层 | 问题 | 代表 |
|---|---|---|
| transport primitive | 内存注册、RDMA/NVLink/HCCS、batch transfer | NIXL、Mooncake TE |
| engine KV manager | 当前请求的 paged KV、block table、prefix | vLLM、SGLang RadixAttention |
| local offloader | GPU -> CPU/NVMe 分层 | Dynamo KVBM、vLLM CPU offload、LMCache LocalCPU |
| distributed cache pool | 跨实例/跨节点复用与生命周期 | Mooncake Store、Tair KVCache、FlexKV |
| fleet control plane | 目录、租户、quota、路由、驱逐 | LMCache Coordinator、Tair KVCM、Mooncake Master |
| training/inference runtime cache | 训练/推理框架内部状态 | DeepSpeed/MII 内部 cache |


---

## 5. Tair KVCache

### 5.1 定位

Tair KVCache 是阿里云面向 LLM 推理的企业级 KVCache 系统，核心组件 Tair KVCache Manager 采用中心化元数据管理，client/connector 负责对接推理引擎，数据面绕过 KVCM 直接读写后端。

### 5.2 架构特点

| 维度 | 特点 | 等级 |
|---|---|---|
| 控制面 | 独立 KVCM，主备选举，元数据持久化 | B/C |
| 数据面 | client 直连后端，metadata plane 与 data plane 分离 | B/C |
| 存储后端 | hf3fs、Mooncake、NFS 等插件化后端 | B/C |
| 元数据 | Valkey/Redis/RocksDB 等外部 KV 存储，分片锁、ReadModifyWrite | B/C |
| 容量 | instance group、quota、水位、TTL/LRU/LFU、异步删除 | B/C |
| 语义 | Radix/prefix、多 block 生命周期、混合注意力优化方向 | B/C |
| 安全 | TLS、访问控制、审计日志、完整性校验、模型版本隔离 | B/C |
| 可观测 | 指标、访问日志、事件日志、健康检查 | B/C |

### 5.3 与 LMCache/Mooncake 对比

| 维度 | Tair KVCache | LMCache | Mooncake |
|---|---|---|---|
| 管理哲学 | 企业级中心化 metadata manager | engine-aware distributed manager | cluster store + TE |
| metadata | 独立 KVCM + 外部持久化 | MP coordinator/event projection | Master snapshot/OpLog |
| 数据面 | client 直连存储后端 | NIXL/GDS/IPC/SHM/Mooncake | 自研 TE |
| 多引擎 | RTP-LLM，扩展 vLLM/SGLang | 多引擎生态最强 | 通过 connector 接入多系统 |
| 多租户 | 企业级 quota/水位/审计能力突出 | cache salt/quota/coordinator | tenant quota/pin |
| 优势场景 | 云厂商多集群、强治理、已有 Tair/阿里云生态 | 多引擎和开源生态 | 超大 RDMA pool、PD、tensor state |

### 5.4 结论

Tair KVCache 在“企业级 KVCache 治理平面”上与 LMCache/Mooncake 正面重叠，尤其元数据持久化、配额、审计、HA 比 LMCache coordinator 更完整。但它不是 LMCache 的简单替代：LMCache 的多引擎 GPU connector 和 MP 故障隔离仍有独立价值。


---

## 6. DeepSpeed cache

### 6.1 定位

DeepSpeed/MII 属于训练/推理 runtime 生态。其 cache 相关能力主要是 runtime 内部的 KV/activation/优化状态管理，不是本报告意义上的跨引擎分布式 KV cache manager。

### 6.2 判断

| 维度 | 判断 | 等级 |
|---|---|---|
| 与 vLLM | 本地 vLLM 主干仅在 benchmark 提示中出现 `deepspeed_mii`，不是 KV Connector 生态的一等集成 | A/C |
| 与 LMCache | DeepSpeed 不是 LMCache 的多引擎 KV cache 管理层 | A/C |
| 与 Mooncake | 没有证据显示 DeepSpeed 依赖 Mooncake Store/TE 作为 KV cache pool | E |
| 主要边界 | engine/runtime optimization，而不是 fleet cache pool | A/C |

### 6.3 结论

DeepSpeed cache 应与训练优化、推理 runtime 内部状态管理一起评估，不应直接与 LMCache/Mooncake/Tair KVCache 排在同一层。


---

# 源码加固审计：Dynamo/KVBM、SGLang HiCache 与 FlexKV（V9）

本文落实前一轮计划，用源码复核 V7 中证据较弱的三个组件。所有仓库已浅克隆到当前工作区，审计日期为 2026-08-29。

## 1. 审计对象与快照

| 目录 | 仓库 | HEAD | 提交时间 |
|---|---|---|---|
| `Dynamo/` | `ai-dynamo/dynamo` | `d7f06b591f3af60270811b7e2d5d7d0a404b934a` | 2026-08-28 |
| `SGLang/` | `sgl-project/sglang` | `51c18d9aa82e7446fe0c661411e26cef063f5a5f` | 2026-08-29 |
| `FlexKV/` | `taco-project/FlexKV` | `a5c8f12867e46ee781b36ec2529e7c813e43cc48` | 2026-08-27 |

工作区状态：

1. Dynamo 使用 sparse checkout，仅保留 KVBM 相关 crate 与参考文档，HEAD 干净；  
2. SGLang 使用 sparse checkout，保留 `mem_cache`、HiCache 文档与测试；  
3. FlexKV 完整 checkout。  

审计方式：

1. `git ls-tree` / `git grep` / `git show` 审计 Dynamo 与 KVBM；  
2. SGLang sparse checkout 后读取 `mem_cache`、HiCache 文档与存储后端；  
3. FlexKV 完整 checkout 后读取 Python、C++、文档与集成目录。  

重要更正：**KVBM 不是独立仓库，而是 Dynamo 主仓中的一组 Rust crate**。FlexKV 也不再停留在“证据不足”，其源码证明它是实现完整度较高的进程内多级 KV cache engine。

---

## 2. Dynamo/KVBM：Dynamo 内的 KV Block Manager

### 2.1 架构定位

KVBM 位于：

```text
Dynamo/lib/bindings/kvbm
Dynamo/lib/kvbm-common
Dynamo/lib/kvbm-config
Dynamo/lib/kvbm-engine
Dynamo/lib/kvbm-consolidator
Dynamo/lib/kvbm-logical
Dynamo/lib/kvbm-physical
Dynamo/lib/kvbm-kernels
```

`lib/kvbm-engine/docs/architecture.md` 明确定义四层模型：

| Tier | 介质 | 角色 |
|---|---|---|
| G1 | GPU HBM | attention 使用的活跃 KV |
| G2 | pinned DRAM | RDMA staging 与中层 cache |
| G3 | NVMe/SSD | warm block |
| G4 | S3/MinIO | cold/archive object |

核心设计是 **logical/physical 分离**：

- `InstanceLeader` 只处理 sequence hash、block identity 和 placement；  
- `PhysicalWorker` 只处理 layout handle、DMA descriptor、local/remote transfer；  
- `CoordinatedWorker` 把 leader 的逻辑请求翻译成本地或远端 worker 调用。  

关键源码：

| 主题 | 证据 |
|---|---|
| Leader 数据结构 | `lib/kvbm-engine/src/leader/instance.rs:67` |
| `find_matches_with_options()` | `lib/kvbm-engine/src/leader/instance.rs:1117` |
| RAII `BlockHolder` | `lib/kvbm-engine/src/leader/session/blocks.rs:51` |
| PhysicalWorker | `lib/kvbm-engine/src/worker/physical.rs:75` |
| CoordinatedWorker | `lib/kvbm-engine/src/worker/coordinated.rs:86` |
| NIXL transfer builder | `lib/kvbm-physical/src/transfer/executor/nixl.rs:33` |
| S3 object client | `lib/kvbm-engine/src/object/s3/client.rs:107` |

### 2.2 生命周期与传输

`lib/kvbm-engine/docs/session.md` 定义两个协议：

1. **Onboard Protocol**：  
   `CreateSession -> G2/G3 Results -> HoldBlocks -> StageBlocks -> RDMA pull -> CloseSession`。  
   用于 initiator 在多个 peer 中查找、hold、staging、RDMA 拉取。

2. **Unified Session Protocol**：  
   `Attach -> StateResponse -> TriggerStaging -> BlocksStaged -> RDMA pull -> BlocksPulled -> Detach`。  
   支持 G2-only 与 G3->G2 staging，并可通过 `YieldControl/AcquireControl` 做控制权迁移。

`BlockHolder` 是 RAII guard：session 持有期间对象不可驱逐；holder drop 后自动释放，即使 session 处理 panic 也不会泄漏。这一点比很多把“查到”直接当作“可安全读”的系统更严谨。

offload 侧采用流水线：

```text
PolicyEvaluator
  -> PreconditionAwaiter
    -> Batcher
      -> TransferExecutor
```

`lib/kvbm-engine/docs/offload.md` 明确以下不变量：

1. container 是取消单元；  
2. cancellation token 随 container 移动；  
3. weak -> strong upgrade 是不可回退 commit point；  
4. upgrade 前最后 sweep cancellation；  
5. forward-pass precondition 未完成前不搬运。

关键源码：

- `lib/kvbm-engine/src/offload/pipeline.rs:265` 定义 `Pipeline`；  
- `lib/kvbm-engine/src/offload/batch.rs:189` 定义 `TransferBatch`；  
- `lib/kvbm-engine/src/offload/policy.rs:277` 起提供 presence/LFU/object lock 等 filter。

### 2.3 并行策略与引擎接入

KVBM 支持：

1. **SPMD TP**：所有 rank 对自己的 KV shard 执行相同 transfer；  
2. **MLA replicated data**：rank 0 承担 G2/G3 搬运，其他 rank 通过 NCCL broadcast 接收；  
3. **vLLM connector**：`lib/bindings/kvbm/python/kvbm/vllm_integration/connector/`；  
4. **TRT-LLM connector**：`lib/bindings/kvbm/python/kvbm/trtllm_integration/connector/`。  

Dynamo 文档给出关键调优项：

| 能力 | 参数 |
|---|---|
| CPU tier 容量 | `DYN_KVBM_CPU_CACHE_GB` |
| disk tier 容量 | `DYN_KVBM_DISK_CACHE_GB` |
| transfer 并发 | `DYN_KVBM_MAX_CONCURRENT_TRANSFERS` |
| NIXL backend | `DYN_KVBM_NIXL_BACKEND_UCX/GDS/...` |
| MLA broadcast | `DYN_KVBM_NCCL_MLA_MODE` |

### 2.4 治理与安全观察

KVBM connector 已经接收 `request.cache_salt`：

- `lib/bindings/kvbm/python/kvbm/vllm_integration/connector_leader.py:276`；  
- `lib/bindings/kvbm/python/kvbm/vllm_integration/kv_cache_manager.py:122`。

S3 object key 也支持 rank/namespace prefix：

- `lib/kvbm-engine/src/object/mod.rs:37`；  
- `lib/kvbm-engine/src/object/s3/client.rs:106`。

但源码中没有看到完整的多租户 authz、per-tenant KMS、租户级 audit event 或跨租户访问控制器。KVBM 更适合把 `cache_salt` 当作 correctness key prefix，而不是把它当作安全隔离边界。

### 2.5 结论

KVBM 是 **Dynamo 生态内的 KV block manager / local-to-remote tiering engine**，不是独立于引擎的通用 KV store。它的强项是：

1. 与 Dynamo router/vLLM/TRT-LLM 的深度集成；  
2. logical/physical 边界清晰；  
3. GPU->CPU->SSD->S3 分层完整；  
4. session hold/staging/RDMA pull 协议明确；  
5. offload cancellation 和 precondition 处理较成熟。

局限是：

1. SGLang 支持仍在演进；  
2. 不是多租户治理系统；  
3. 不提供 Mooncake/Tair 级别的集中对象生命周期与 HA；  
4. 更适合 NVIDIA/Dynamo 栈，而非跨引擎通用缓存层。

---

## 3. SGLang HiCache：RadixAttention 原生分层缓存

### 3.1 架构

核心目录：

```text
SGLang/python/sglang/srt/mem_cache
SGLang/python/sglang/srt/mem_cache/storage
SGLang/python/sglang/srt/mem_cache/unified_cache
SGLang/python/sglang/srt/mem_cache/hybrid_cache
SGLang/docs/docs/advanced_features/hicache_design.mdx
```

HiCache 的模型是：

```text
L1 GPU memory
L2 host memory
L3 distributed storage
```

元数据由 **HiRadixTree** 组织。每个 Radix 节点对应连续 token span，并记录该 span 在哪些层存在。L1/L2 metadata 本地维护；L3 metadata 不做全局连续同步，而是在访问时查询后端。

### 3.2 核心接口

`python/sglang/srt/mem_cache/hicache_storage.py` 定义统一 L3 抽象：

| 行号 | 内容 |
|---|---|
| 22-23 | `STORAGE_BATCH_SIZE = 128` |
| 27 | `HiCacheStorageConfig` |
| 50 | `PrefetchTimeoutConfig` |
| 58 | `PoolName` |
| 83 | `PoolHitPolicy` |
| 95 | `PoolTransfer` |
| 152 | `HiCacheStorage(ABC)` |
| 167 | `batch_exists_v2()` |
| 200 | `batch_get_v2()` |
| 211 | `batch_set_v2()` |
| 331 | `MetadataCache` |
| 363 | `HiCacheFile` |

v2 接口不仅支持 KV pool，还支持 sidecar pool（例如 SWA/Mamba/indexer/compress state），这是它比早期 prefix cache 更接近 hybrid attention cache manager 的证据。

### 3.3 L3 backend 生态

`storage/backend_factory.py` 显示动态注册和内置 backend：

| 行号 | 内容 |
|---|---|
| 16 | `StorageBackendFactory` |
| 44 | `register_backend()` |
| 66 | `create_backend()` |
| 114 | dynamic backend |
| 153 | builtin backend 分发 |
| 165 | mooncake |
| 202-214 | dynamic/sim/mooncake 注册示例 |

源码目录中已有：

1. `storage/mooncake_store/`；  
2. `storage/flexkv/`；  
3. `storage/nixl/`；  
4. `storage/hf3fs/`；  
5. `storage/lmcache/`；  
6. `storage/aibrix_kvcache/`；  
7. `storage/file/`、`storage/shm/`、`storage/mmap/`。

这说明 SGLang HiCache 正在成为 engine 原生的 L3 connector hub，而不是只绑定单一后端。

### 3.4 Mooncake 集成细节

`storage/mooncake_store/mooncake_store.py` 提供了较完整的零拷贝集成：

| 行号 | 主题 |
|---|---|
| 35 | `MooncakeHostTensorAllocator` |
| 86 | tenant id normalization |
| 94 | `MooncakeStoreConfig` |
| 165 | `tenant_id` 配置 |
| 332 | `MooncakeStore(HiCacheStorage, MooncakeBaseStore)` |
| 389 | 初始化 `MooncakeDistributedStore` |
| 409-411 | group semantics 不支持时回退旧 batch put |
| 534-537 | tenant id 不被 Mooncake 支持时显式报错 |
| 700 | `register_mem_host_pool_v2()` |
| 719 | `_tag_keys()` |
| 724-745 | group id / hybrid page component keys |
| 829 | `batch_exists_v2()` |
| 884 | `_batch_io_v2()` |
| 1298 | zero-copy put implementation |

两个重要观察：

1. Mooncake backend 已处理 tenant id、group id、hybrid pool 和 zero-copy batch v2；  
2. 如果 Mooncake 版本不支持 tenant/group 能力，SGLang 会显式拒绝或回退，而不是静默改变语义。

### 3.5 prefetch、write-back 与 rank 同步

`docs/docs/advanced_features/hicache_design.mdx` 明确：

| 机制 | 语义 |
|---|---|
| local match | 遍历 HiRadixTree，page 粒度匹配，节点可分裂 |
| prefetch threshold | 默认 256 token，避免小片段 I/O |
| prefetch strategy | `best_effort`、`wait_complete`、`timeout` |
| timeout | `base + per_ki_token * tokens/1024`，有 max 上限 |
| write policy | `write_through`、`write_through_selective`、`write_back` |
| TP rank sync | prefetch hit 数和成功长度用 `all_reduce(min)` |
| layout | `layer_first`、`page_first`、`page_first_direct` |
| CPU->GPU | layer overlap 与 GPU-assisted I/O kernel |
| MLA | 只由一个 rank 写回，避免重复写 L3 |

与 KVBM 相比，HiCache 的最强点不是通用 RDMA 数据面，而是 **prefix tree 语义与 attention 运行时天然一致**。

### 3.6 namespace / cache salt

HiCache 不是无 namespace 系统：

1. `buffer_mode/pipeline.py:102` 请求上下文带 `cache_salt`；  
2. `pipeline.py:799-808` 明确要求 staged prefetch 的 `extra_key` 与 `cache_salt` 匹配，否则丢弃，避免 wrong-namespace publish；  
3. `unified_cache/unified_tree_core.py:121-123` 区分普通 hash 与 namespace-aware event hash；  
4. `unified_cache/unified_tree_core.py:1949-1953` 返回 anchor node 的 `extra_key/cache_salt`。

但源码中仍未看到完整租户 authz、审计和加密层。HiCache 的 namespace 更像 correctness/prefix isolation，而非安全边界。

### 3.7 结论

HiCache 是 **SGLang 内最自然的 KV cache 管理层**。它的优势：

1. 与 RadixAttention 状态一致；  
2. L1/L2/L3 语义清楚；  
3. hybrid/SWA/Mamba sidecar 能力正在扩展；  
4. backend 插件生态丰富；  
5. Mooncake/FlexKV/NIXL/LMCache 都可以作为 L3。

劣势：

1. 与 SGLang 运行时耦合，跨引擎复用不如 LMCache 通用；  
2. L3 metadata advisory cache 可能引入一致性压力；  
3. 不提供完整租户安全边界；  
4. 复杂 hybrid attention 的实现分散在多个 cache controller 中，测试矩阵会持续变大。

---

## 4. FlexKV：进程内多级 cache engine + 分布式 Radix

### 4.1 架构定位

FlexKV README 定义三大模块：

```text
StorageEngine
GlobalCacheEngine
TransferEngine
```

层级是：

```text
GPU KV cache
  -> CPU memory
    -> local SSD
      -> scalable remote storage / Mooncake Store
```

源码结构非常清晰：

| 目录 | 职责 |
|---|---|
| `flexkv/kvmanager.py` | 对外 KVManager API |
| `flexkv/kvtask.py` | task lifecycle 与 `GlobalCacheEngine` 调用 |
| `flexkv/cache/cache_engine.py` | 全局 cache 决策 |
| `flexkv/cache/redis_meta.py` | Redis metadata |
| `flexkv/cache/radix_remote.py` | distributed radix |
| `flexkv/storage/storage_engine.py` | CPU/SSD/remote storage |
| `flexkv/transfer/transfer_engine.py` | 本地/远端 transfer |
| `flexkv/transfer/layerwise.py` | layerwise transfer |
| `flexkv/transfer/compression/` | nvcomp/ANS compression |
| `csrc/radix_tree.*` | C++ RadixTree |
| `csrc/dist/` | distributed RadixTree、lease、Redis metadata |
| `csrc/gds/` | GPUDirect Storage |
| `csrc/pcfs/` | PCFS |

### 4.2 对外 API 与任务模型

`flexkv/kvmanager.py` 定义 `KVManager`：

| 行号 | 内容 |
|---|---|
| 34 | `class KVManager` |
| 109-120 | 初始化 `RedisMeta` 并连接 metadata service |
| 173 | `get_async()` |
| 199 | `get_match()` |
| 233 | `put_async()` |
| 257 | `put_match()` |
| 277 | `prefetch_async()` |
| 348 | `cancel()` |
| 356 | `wait()` |
| 383 | `reset()` |

`flexkv/kvtask.py` 定义任务生命周期：

| 行号 | 内容 |
|---|---|
| 40-59 | `TaskStatus` / `TaskType`：GET、PUT、BATCH_GET、BATCH_PUT |
| 62 | `KVTask` |
| 126 | `_longest_success_prefix()` |
| 136 | `KVTaskManager` |
| 150-157 | CPU/GDS/SSD/NIXL/kv-sharing 配置约束 |
| 170 | 创建 `GlobalCacheEngine` |
| 305 | `create_get_task()` |
| 344 | `create_put_task()` |
| 378 | `create_prefetch_task()` |
| 701 | abort task plan |
| 735 | cancel task |
| 1053-1041 | async put/get |
| 1447-1488 | reset all tiers，包括 in-flight 风险警告 |

这说明 FlexKV 不是简单同步 get/put，而是有完整的 async task graph、prefix mask、partial success、cancel/abort 和 reset 语义。

### 4.3 namespace 隔离

FlexKV 明确在 hash 输入中加入 namespace：

| 文件:行号 | 内容 |
|---|---|
| `flexkv/common/block.py:11` | `_get_namespace_hash_key()` |
| `flexkv/common/block.py:17` | namespace 元素用 NUL 分隔 |
| `flexkv/common/block.py:23` | `hash_token(token_ids, namespace)` |
| `flexkv/common/block.py:30` | namespace bytes 进入 hasher |
| `flexkv/common/block.py:55-88` | `Block` 保存 namespace 并计算 `namespace_id` |
| `docs/namespace/README_en.md` | 说明 `tenant_A/user_123/LoRA/session salt` |

对应公式：

```text
without namespace: H(token_ids)
with namespace:    H(namespace_id || token_ids)
```

这是 V7 中最需要更正的点：FlexKV 不是“证据不足的通用系统”，它已经有显式 namespace hash 设计。

### 4.4 distributed Radix 与 lease

`csrc/dist/` 显示 FlexKV 采用 local snapshot + Redis GMS + lease：

| 文件:行号 | 机制 |
|---|---|
| `csrc/dist/block_meta.h:9-17` | block hash、parent hash、lease time、lease state |
| `csrc/dist/lease_meta_mempool.h:18-25` | `LeaseMeta`、`lease_time`、`published` |
| `csrc/dist/local_radix_tree.h:43-117` | LocalRadixTree、Redis channel、lease TTL、eviction 队列 |
| `csrc/dist/distributed_radix_tree.h:97-154` | DistributedRadixTree、refresh/rebuild、match |
| `csrc/dist/redis_meta_channel.h:75-100` | Redis block key、lease renew、state update、delete metadata |
| `distributed_radix_tree.cpp:121-196` | 周期 rebuild、三代 index 切换 |
| `distributed_radix_tree.cpp:206-263` | 批量 refresh lease |
| `distributed_radix_tree.cpp:626-690` | match 时验证 hash 与 lease validity |

FlexKV 的设计选择与 Mooncake/Tair 不同：它不把所有 metadata 放在中心 Master，而是让每个节点维护全局 index 的本地 snapshot，通过 Redis 发布 block metadata 和 lease，再周期 rebuild。优点是 lookup 可本地完成；代价是 rebuild interval、lease TTL、网络分区和 stale snapshot 会成为正确性边界。

### 4.5 数据面与 GDS

FlexKV 传输层支持：

1. CPU/SSD/remote 多级；  
2. GPU->CPU 与 CPU->GPU；  
3. GPUDirect Storage；  
4. PCFS；  
5. layerwise transfer；  
6. Mooncake TE 远端 transfer；  
7. nvcomp/ANS 压缩。

关键证据：

| 主题 | 证据 |
|---|---|
| Mooncake wrapper | `flexkv/mooncakeEngineWrapper.py:22-164` |
| TransferManager | `flexkv/transfer_manager.py:38-520` |
| StorageEngine 初始化 | `flexkv/transfer_manager.py:380-501` |
| GDS manager | `csrc/gds/gds_manager.cpp` |
| TP GDS group | `csrc/gds/tp_gds_transfer_thread_group.cpp` |
| PCFS | `csrc/pcfs/pcfs.cpp` |
| compression | `flexkv/transfer/compression/ans/` |
| layerwise | `flexkv/transfer/layerwise.py` |

### 4.6 引擎集成

FlexKV 已经不只是库，它有多个 connector：

| 引擎 | 证据 |
|---|---|
| vLLM | README：vLLM mainline `FlexKVConnectorV1`；仓库 `flexkv/integration/vllm/vllm_v1_adapter.py` |
| SGLang | SGLang 主干 `python/sglang/srt/mem_cache/storage/flexkv/`；FlexKV 仓库也有 connector |
| TensorRT-LLM | `flexkv/integration/tensorrt_llm/` |
| Dynamo | README 与 `docs/dynamo_integration/README_en.md` |

SGLang 侧证据尤其强：

| 文件:行号 | 内容 |
|---|---|
| `storage/flexkv/flexkv_connector.py:64` | `FlexKVConnector` |
| `flexkv_connector.py:247` | `lookup_kv()` |
| `flexkv_connector.py:327` | `retrieve_kv()` |
| `flexkv_connector.py:452` | `store_kv()` |
| `flexkv_connector.py:372` | layerwise load |
| `flexkv_connector.py:604` | prefetch |
| `flexkv_comm.py:50` | PP/CP/TP hierarchical sync |
| `flexkv_comm.py:259` | block-count `all_reduce_min()` |
| `flexkv_radix_cache.py:72` | `FlexKVRadixCache` |
| `flexkv_radix_cache.py:155` | page-aligned `match_prefix()` |

FlexKV 因此应从 V7 的“公开材料不足”升级为“源码可审计的多引擎 KV cache manager”。

### 4.7 治理与风险

FlexKV 已有：

1. namespace hash；  
2. Redis password；  
3. distributed metadata；  
4. lease；  
5. Prometheus metrics；  
6. multi-instance/multi-node 文档。

但仍有明显风险：

| 风险 | 证据 |
|---|---|
| namespace 只是 hash 前缀，不等于 authz | `common/block.py:23-30` |
| Redis 密码不等于完整安全边界 | `csrc/dist/redis_meta_channel.cpp:40-53` |
| `enable_kv_sharing` 与 GDS 当前互斥 | `flexkv/kvtask.py:154-155` |
| reset all tiers 对 in-flight transfer 有警告 | `flexkv/kvtask.py:1447-1465` |
| distributed best practices 仍是 TODO | `docs/dist_reuse/README_en.md` |
| lease snapshot/rebuild 需要压测 stale metadata | `distributed_radix_tree.cpp:121-196` |
| 未发现完整 per-tenant audit/KMS | 关键词扫描无成熟实现 |

### 4.8 结论

FlexKV 是 **本地性能优先、多引擎接入、库形态优先的分布式 KV cache manager**。它的成熟度高于 V7 的保守判断，尤其是：

1. 进程内 API 减少了 client-server hop；  
2. C++ RadixTree + Python task engine 结构清晰；  
3. namespace/lease/distributed Radix 已实现；  
4. GDS、PCFS、Mooncake、compression、layerwise 覆盖面广；  
5. vLLM/SGLang/TRT-LLM/Dynamo 四条集成路径已经打开。

它与 Mooncake 不是简单竞争：FlexKV 可把 Mooncake Store/TE 作为 remote tier。与 LMCache 也不是简单重复：FlexKV 更接近 engine-adjacent cache engine，而 LMCache 更强调独立缓存进程、多引擎 layout 与企业治理。

---

## 5. 三者横向矩阵

| 维度 | Dynamo/KVBM | SGLang HiCache | FlexKV |
|---|---|---|---|
| 主定位 | Dynamo KV block manager | SGLang Radix 分层 cache | 进程内多级 cache engine |
| 核心元数据 | sequence hash / leader / worker | HiRadixTree | C++ RadixTree + distributed snapshot |
| L1 | GPU HBM | GPU | GPU 由引擎管理 |
| L2 | pinned DRAM | host memory | CPU memory |
| L3 | SSD + S3/MinIO | Mooncake/FlexKV/NIXL/HF3FS/LMCache/AIBrix/file | local SSD + PCFS + Mooncake remote |
| remote metadata | G4 object + pubsub/session | backend-specific | Redis + lease + local snapshot rebuild |
| data plane | NIXL/RDMA/NVMe/object API | backend-specific + GPU kernel H2D | native transfer + GDS/PCFS + Mooncake TE |
| prefix tree | sequence hash tree | Radix/HiRadixTree | C++ RadixTree |
| namespace | connector `cache_salt` | `extra_key/cache_salt` | explicit namespace hash |
| async cancel | offload pipeline + token | prefetch policy / timeout | task cancel/abort |
| engine coupling | Dynamo/vLLM/TRT-LLM | SGLang | vLLM/SGLang/TRT-LLM/Dynamo |
| hybrid attention | MLA replication | hybrid/SWA/Mamba sidecar 演进 | Full/SWA、DSv4 sidecar 演进 |
| tenant authz | 弱 | 弱 | 弱 |
| encryption/audit | 未见完整能力 | 未见完整能力 | Redis password/metrics，未见完整 audit/KMS |

## 6. V7 结论更正

V7 中的以下判断需要更新：

| V7 原判断 | V9 更正 |
|---|---|
| KVBM 可能是独立组件，证据 B/C | KVBM 是 Dynamo 主仓 Rust crate，证据 A |
| KVBM 只是 local offloader | KVBM 已有 G1-G4、remote session、RDMA pull、S3 G4 |
| SGLang HiCache 证据 B/C | 设计与源码均可用 A 级证据描述 |
| FlexKV 公开材料不足，E 级 | FlexKV 源码完整，核心能力可用 A 级证据描述 |
| FlexKV 只是分布式 KV store | FlexKV 是库形态 cache engine + async task graph + distributed Radix |
| FlexKV namespace 证据不足 | FlexKV 明确实现 `H(namespace || token_ids)` |

## 7. 新风险清单

| ID | 系统 | 风险 | 优先级 | 建议 |
|---|---|---|---:|---|
| K-01 | KVBM | `cache_salt` 参与正确性，但未见 authz | P1 | 在 Dynamo 前置租户身份，禁止跨租户共享；为 S3 key 增加 KEK/audit |
| K-02 | KVBM | SGLang 支持仍演进 | P2 | 生产矩阵仅承诺 vLLM/TRT-LLM，SGLang 单独验证 |
| K-03 | KVBM | G4 object 语义与 local block holder 的一致性需 fault injection | P1 | 测试 S3 partial upload、GC、object overlap |
| S-01 | SGLang | L3 existence cache 是 advisory，不 authoritative | P1 | 后端 miss 后必须回源查询；测试 stale existence |
| S-02 | SGLang | `cache_salt` 不等于安全边界 | P1 | 引入 tenant authz 和 audit；不要让 request 层自行决定 salt |
| S-03 | SGLang | hybrid pool / group semantics 依赖 Mooncake 版本 | P1 | CI 固定 Mooncake API 版本，并测 fallback |
| F-01 | FlexKV | namespace hash 不做授权 | P1 | API 层校验 tenant 与 namespace 归属 |
| F-02 | FlexKV | distributed Radix snapshot 可能 stale | P1 | 压测 lease TTL/rebuild interval/网络分区下的 false hit |
| F-03 | FlexKV | reset all tiers 与 in-flight transfer 可能冲突 | P0/P1 | reset 前 drain/cancel；保留 generation fence |
| F-04 | FlexKV | Redis password 不覆盖 data plane 完整性 | P1 | Redis TLS/mTLS，value CRC/MAC，密钥轮换 |
| F-05 | FlexKV | `enable_kv_sharing` 与 GDS 互斥 | P2 | 文档明确模式矩阵，避免混部 |

## 8. 更新后的选型判断

### 8.1 NVIDIA/Dynamo 主导平台

优先评估 KVBM + Dynamo KV router。若需要跨实例持久对象池，接 Mooncake Store；若需要库形态 CPU/SSD 加速，接 FlexKV。

### 8.2 SGLang 主导平台

优先使用 HiCache。L3 后端按目标选择：RDMA/Mooncake 用 Mooncake；库形态 CPU/SSD 用 FlexKV；NVIDIA 存储生态用 NIXL；多引擎统一层再考虑 LMCache。

### 8.3 多引擎平台

LMCache 仍是最直接的多引擎管理层。FlexKV 现在应作为强竞争/互补项纳入 POC：如果重点是从 GPU 到 CPU/SSD 的低 hop 库内加速，FlexKV 可能更轻；如果重点是独立缓存进程、企业治理和多后端抽象，LMCache 更合适。

### 8.4 超大集群共享池

Mooncake 仍是数据面与对象生命周期最强的底座。FlexKV/KVBM/HiCache 都可以接 Mooncake，因此不应把三者误解为 Mooncake 的完全替代。

## 9. 下一步源码审计

按风险排序：

1. FlexKV distributed Radix：lease TTL、rebuild、stale snapshot、false positive。  
2. KVBM offload pipeline：upgrade commit boundary、cancel race、G3/G4 一致性。  
3. SGLang HiCache：existence cache stale、partial page、hybrid sidecar 一致性。  
4. FlexKV/KVBM/HiCache 的 namespace 与 tenant authz 差异。  
5. 三者与 Mooncake 的 version compatibility matrix。  
6. V8 安全测试在三个系统上的最小落地脚本。

## 附：KVBM 与 LMCache 在 Dynamo 平台内的关系（增量）

> 来源：ForceInjection《KVBM 深度解析》，与本地 Dynamo 源码交叉印证。

- **独立使用时互斥**：同一 vLLM 实例只能激活一个主 KV 后端——DynamoConnector=KVBM 或 LMCacheConnectorV1=LMCache，二者不可同时作为主缓存管理器。
- **PdConnector 混合模式**：恰好两个连接器槽位，第一个必须是 KVBM 或 LMCache（本地缓存管理），第二个必须是 NIXL（PD 节点间传输），运行时不可动态切换。
- KVBM 的差异化设计：Rust Type State 状态机（MutableBlock→CompleteBlock→ImmutableBlock↔WeakBlock，编译期杜绝未初始化读取/重复释放）、BlockRegistry 基于 PositionalRadixTree 的弱引用自清理、可插拔驱逐后端（LRU/MultiLRU 4 级频率分层/TinyLFU/Lineage 血缘）、以及 `bypass_cpu_mem`（G1 GPU→G3 Disk 走 GDS 直连绕过 CPU）。
- 定位差异：KVBM 追求 Dynamo 生态内的极致单机/集群 tiering 性能与严格生命周期安全；LMCache 追求「prefill-once, reuse-everywhere」的跨实例共享与多引擎接入。

## 附：Mooncake 生态集成全景（一研「生态集成」篇）

Transfer Engine/Store 作为底座被集成的项目全表（超出第 2-4 章已审计的 vLLM/SGLang/LMCache/FlexKV 范围）：**NIXL**（Mooncake 作为其插件后端之一）、**TorchSpec/Speculators**（隐状态投机缓存）、**xLLM**（混合 KV 布局）、**LightX2V**（视频生成长上下文）、**vLLM-Ascend**、**vLLM-Omni**（多模态）、**RBG**（云原生）等；Python 绑定 `PyTensorInfo{data_ptr/tensor_size/metadata/owner}` 支持张量分片与 parallel_read/write；TRT-LLM 走 C++ API。

**EPD（Encoder-Prefill-Decode）**：SGLang 的三段解耦新场景——多模态模型中 ViT encoder 的嵌入向量也经 TE 零拷贝传输到 Prefill 节点，是「KV Cache 传输」范式向「任意张量状态传输」的扩展（与 P2P Store 权重分发、RL checkpoint 传输同一趋势：Mooncake 正从 KV Cache 系统变成张量移动基础设施）。
