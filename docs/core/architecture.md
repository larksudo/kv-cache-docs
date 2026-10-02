---
next: /core/transfer-paths
---

# 总体架构：LMCache、Mooncake 与 vLLM

本篇回答：三个系统各自的层次结构是什么，控制面与数据面如何切分。先看结论摘要（见「总结论」一篇），再进入本篇。

> 来源：主报告第 1 章（总体架构）与第 8 章（控制面与数据面设计），原文保留。

## 1. 总体架构

### 1.1 LMCache

LMCache 的核心层次：

```text
vLLM / SGLang / TRT-LLM
  -> Engine KV Connector / MP Connector
    -> LMCache Engine or MP daemon
      -> GPU Connector (layout + device copy)
      -> Storage Manager / CacheEngine
        -> L1 CPU/DRAM / GDS / local disk
        -> L2 adapters: NIXL Store, FS, Redis/Valkey, Mooncake, InfiniStore, S3, P2P
        -> Transfer Channel: NIXL/UCX RDMA
      -> MP Coordinator / cache controller / observability
```

关键证据：

- `LMCache/README.md` 明确定位为 KV cache management layer，强调独立 daemon、崩溃隔离、分层 offload、多引擎与多存储。
- `lmcache/integration/vllm/lmcache_connector_v1.py` 与 `lmcache_mp_connector.py` 分别承接进程内和多进程模式。
- `lmcache/v1/gpu_connector/gpu_connectors.py` 定义 `GPUConnectorInterface`，再实现 vLLM paged、layerwise、SGLang、TRT-LLM 等路径。
- `lmcache/v1/storage_backend/abstract_backend.py` 定义 `StorageBackendInterface` 与 `AllocatorBackendInterface`。
- `docs/design/v1/distributed/l2_adapters/overall.md` 描述 store/prefetch 控制器、eventfd、L1/L2 锁不变量。
- `docs/design/v1/mp_coordinator/README.md` 描述 Fleet Registry、event ingest、quota、pin/delete/prefetch、key directory 和 blend lookup。

LMCache 的顶层哲学是“把引擎缓存外置成可管理的知识层”。因此它的抽象大量面向工程生态：不同引擎的 KV layout、不同后端、不同租户、可观测和云原生运维。

### 1.2 Mooncake

Mooncake 的核心层次：

```text
vLLM / SGLang / LMDeploy / TensorRT-LLM / RL training
  -> Mooncake Connector / TransferEngine / Store SDK
    -> Mooncake Transfer Engine
      -> Segment / Buffer / BatchTransfer
      -> RDMA / RoCE / TCP / NVMe-oF / NVLink / HIP / Ascend Direct / EFA
    -> Mooncake Store
      -> Master: allocation, metadata, lease, pin, eviction, snapshot/HA
      -> Client/Segment: local DRAM / VRAM / SSD contribution
      -> Replica / slice placement / zero-copy RDMA
    -> Mooncake EP / PG / Reshard / RL extensions
```

关键证据：

- `Mooncake/README.md` 定位为 KV cache-centric disaggregated architecture，包含 Transfer Engine、Store、EP/PG。
- `docs/source/design/transfer-engine/index.md` 给出 Segment、BatchTransfer、RAM/NVMeoF Segment、拓扑矩阵、端点池、故障处理模型。
- `docs/source/design/store/mooncake-store.md` 描述 Master、Client、对象写协议、副本、租约、软/硬 pin、快照与 HA。
- `docs/source/design/tent/*` 描述 Transport Selector、QoS、slice spraying、failover、metrics。
- `docs/source/design/transfer-engine/ascend_direct_transport.md` 显式说明 Ascend H2D/D2H/D2D 可走 HCCS/RDMA，A2/A3 默认 HCCS。

Mooncake 的顶层哲学是“数据位置和传输路径是系统资源”。因此它的抽象大量面向物理拓扑：CPU NUMA、GPU/NPU PCIe 亲和、RDMA NIC、segment、slice 和 batch。

### 1.3 vLLM

vLLM 的 KV transfer 抽象分两层：

1. 旧 README 中的 KV pipe / lookup buffer / connector 抽象；  
2. 当前主干 `KVConnectorBase_V1`，它已经变成 scheduler/worker 双面接口。

`vllm/distributed/kv_transfer/kv_connector/v1/base.py` 将接口分为：

- Scheduler：`get_num_new_matched_tokens()`、`update_state_after_alloc()`、`update_connector_output()`、`request_finished()`、`take_events()`；
- Worker：`register_kv_caches()`、`start_load_kv()`、`wait_for_layer_load()`、`save_kv_layer()`、`wait_for_save()`、`get_finished()`。

这个接口的本质是把“外部命中多少 token”“分配 block 后怎么加载”“何时确认异步保存完成”“请求结束时谁持有 block”暴露给外部系统。LMCache 和 Mooncake 都是在这个接口上实现自己的控制面/数据面策略。

---

---

## 附：架构假设冲突（V2）

### 6.1 “缓存可丢失”与“缓存必须可靠”

推理系统通常说 KV cache miss 可以重算。但在多租户/长上下文/agent 场景，重算成本可能极高，甚至用户语义上要求会话连续。于是系统会从 best-effort cache 演化为带 SLA 的 state store。

LMCache 的定位仍偏“可丢但尽量不丢”；Mooncake Store 已有更多 state store 组件。但两者都不应被宣传为通用数据库。

### 6.2 “对象 immutable”与“逻辑上下文演进”

单块 KV 是 immutable，但一个会话的逻辑上下文会增长。两个系统都需要 group/prefix/chunk 协议：

- LMCache 用 prefix/chunk hash、blend index、非前缀实验；
- Mooncake vLLM Store 用 chunk hash、group id、group TTL。

难点是：尾块未满、partial tail CoW、tokenization 变化、中间插入、非前缀复用。不能简单把多个 object 当一个事务。

### 6.3 “本地优先”与“全局最优”

LMCache MP 假设各实例本地 L1 最快，然后跨 peer/L2 扩展。Mooncake Store 假设 cluster pool 能提升全局命中率。

两者并不矛盾，但控制策略不同：

- LMCache 更适合本地热数据 + 可选远层；
- Mooncake 更适合跨节点冷热数据统一调度。

### 6.4 “引擎 adapter 越多越好”与“基础设施越薄越好”

LMCache 在引擎侧加了很多 adapter，生态移植好，但维护面大。Mooncake 在基础设施侧保持张量/对象语义，引擎 layout 在 connector 内解决，移植面窄但底座重。

这是架构价值观差异，不是简单优劣。
