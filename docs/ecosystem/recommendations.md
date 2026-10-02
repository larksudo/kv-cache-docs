---
prev: /ecosystem/security
next: /ecosystem/selection
---

# 风险登记册与架构改造建议

三系统的 P0-P3 风险登记册、统一 Cache Contract、StoreJob 一等对象、Lease/Fence 分离、cache-aware router 与安全默认值，以及 12 个月路线图。

> 来源：V6 全文。

# 风险登记册与架构改造建议

本文把前五轮发现整理成工程行动项。优先级定义：

| 级别 | 含义 |
|---|---|
| P0 | 可能返回错误数据、死锁、跨租户污染或长期泄漏 |
| P1 | 高压下造成 SLO 违约、重大运维困难 |
| P2 | 增加集成成本、限制规模 |
| P3 | 长期架构债 |

## 1. LMCache 风险登记册

| ID | 风险 | 影响 | 优先级 | 建议 |
|---|---|---|---:|---|
| L-01 | Coordinator registry/directory 重启后从空重建 | 控制面短时不准，quota/eviction/prefetch 决策退化 | P1 | 引入 durable event log 或 checkpoint；暴露 gap metric；恢复前禁止全局驱逐 |
| L-02 | L2 adapter 能力不一致 | 同一上层策略在 Redis/FS/NIXL/Mooncake 上语义不同 | P1 | 在 adapter capability 中声明 capacity/delete/lease/listener/persist；不支持的策略提前拒绝 |
| L-03 | P2P peer 锁释放依赖 fire-and-forget unlock | peer 驱逐或断链时可能长期占用 | P1 | unlock 增加重试/租约；peer 重启后按 incarnation 失效旧锁 |
| L-04 | PD staging rollback 单 key remove 失败只 warning | 可能留下 reservation 或半请求状态 | P0/P1 | rollback 失败升级为 request-level quarantine，并触发审计和一致性检查 |
| L-05 | CUDA IPC handle 生命周期复杂 | worker OOM/restart 后可能泄漏或错绑 | P1 | 继续保留 uuid/reaper；增加 handle fingerprint 与 device/stream 校验 |
| L-06 | MP 协议版本演进 | scheduler/worker/server 混部可能不兼容 | P1 | 协议 handshake 带 schema/version/capability；不匹配 loud fail |
| L-07 | CacheGen/CacheBlend 与生产 prefix cache 边界模糊 | 用户可能把研究特性当商用能力 | P2 | 文档和 CLI 增加 maturity label，如 stable/beta/research |
| L-08 | Mooncake L2 与 LMCache quota/eviction 双层 | 容量和优先级归属不清 | P1 | 定义 source of truth：Mooncake 管池，LMCache 只上报/查询；或 LMCache 只做 policy 前端 |
| L-09 | 多租户 key 仅靠命名约定 | cache salt/tenant 配置错误可能跨租户命中 | P0 | 启动时强制 key schema 版本；按租户验证注入；跨租户读拒绝而非 miss |
| L-10 | observability 分散在 engine/storage/P2P/coordinator | 生产排障成本高 | P2 | 输出统一 request-span 到 store-job/P2P-task 映射 |

## 2. Mooncake 风险登记册

| ID | 风险 | 影响 | 优先级 | 建议 |
|---|---|---|---:|---|
| M-01 | Master 是控制面中心 | 高并发分配、故障切换、运维复杂 | P1 | 持续做 partition/sharding 压测；明确每 Master 对象数/RPC 上限；提供 automatic rebalance |
| M-02 | replication best-effort | 不能当强持久副本使用 | P1 | 文档与 API 返回 requested/achieved replicas；低于 SLA 触发降级或重试 |
| M-03 | TENT submit-stage partial enqueue 不自动 failover | 部分传输已入队时失败处理复杂 | P0/P1 | transport 提供 atomic submit 或 per-request submit status；上层保留 task ledger |
| M-04 | lease expiry 期间读失败 | 上层必须有重试/重算 | P1 | 提供 lease renewal/read fence API；vLLM connector 记录 lease expiry 导致的 miss |
| M-05 | soft pin 不持久化 | Master failover 后热点可能被驱逐 | P2 | 至少记录 pin audit，或允许 durable soft pin 与 TTL 恢复 |
| M-06 | zombie PutStart 空间回收 | client crash 后空间可能延迟释放 | P2 | 暴露 zombie count、oldest age、preemption rate |
| M-07 | group lifecycle 是 best-effort | 一个逻辑 prefix 的部分对象可能被分开驱逐 | P1 | connector 读回前校验 group completeness；失败视为 group miss |
| M-08 | direct PD 状态机复杂 | TP/PP/取消/过期场景容易泄漏 transfer id | P1 | 长压测统计 transfer_id lifetime、sending stuck、expire；增加自动 GC |
| M-09 | HCCS 2MB 对齐 | Ascend block/region 可能需要打包或 padding | P1 | 输出 layout adapter 与 alignment validator；错误对齐 fail-fast |
| M-10 | SSD checkpoint 后 stale seq/CRC | 冷层恢复需要严格校验 | P1 | 压测半写、断电模拟、磁盘满；上报 CRC miss/recovered/discarded |

## 3. vLLM 风险登记册

| ID | 风险 | 影响 | 优先级 | 建议 |
|---|---|---|---:|---|
| V-01 | KV Connector 接口演进快 | 第三方 connector 频繁 broken | P1 | 定义 capability flags 与 compatibility matrix；加入 conformance test |
| V-02 | HMA/hybrid group 支持分散 | 不同 connector 支持边界不清 | P1 | 在 connector factory 输出 supported KV group types；启动前校验模型 |
| V-03 | MultiConnector 冲突语义未完全抽象 | PD + Store 同写/同读失败合并复杂 | P1 | 增加 merge policy：first-hit、parallel-load、producer-priority、cost-based |
| V-04 | KV event 聚合一致性 | 多 worker 不一致可能误报公共 prefix | P1 | event 带 worker quorum/版本；router 只消费 quorum-safe event |
| V-05 | partial tail/CoW 与外部缓存边界 | 尾块保存失败或复用会污染后续前缀 | P0/P1 | 把 partial tail 作为独立 fence 对象，只有 fence 完成才发布 hash |
| V-06 | cache-aware routing 缺口 | 有全局池但 router 仍可能绕开热点 | P2 | Router 消费 KV directory，权衡负载与 transfer cost |
| V-07 | preemption 与 external delivery | producer block 可能先于 save 完成 | P0 | 所有 producer connector 显式声明 reliable delivery 与 recompute policy |

## 4. 架构改造建议

### 4.1 建议引入统一 Cache Contract

一个对象写入共享池前应声明：

```yaml
cache_contract:
  key_schema: v1
  tenant_id: required
  model_fingerprint: required
  tokenizer_fingerprint: required
  parallel_layout: tp/pp/dp/ep
  kv_group_type: mha|gqa|mla|mamba|hybrid
  object_state: partial|complete
  visibility: private|tenant|global
  retention: ttl|session|pinned
  deletion: immediate|best_effort
  consistency: read_committed_immutable
```

好处：

1. LMCache 与 Mooncake 的能力差异变成显式协商；  
2. 不支持该 contract 的 connector 提前失败；  
3. 多租户和安全审计有统一切入点。

### 4.2 把 Store Job 作为一等对象

Mooncake vLLM worker 已经用 `store_job_id`，建议推广成跨系统模型：

```text
StoreJob:
  job_id
  request_generation
  object_keys
  block_ids
  rank_completions
  fence_event
  started_at
  deadline
  state: queued|running|succeeded|failed|cancelled|leaked
```

LMCache 的 pending store、lazy offload、P2P task、Mooncake store job 都可以映射到该模型。事故排查时可直接回答“哪个逻辑对象被哪个 job 固定、何时释放”。

### 4.3 Lease 与 Fence 分离

当前 Mooncake lease 主要保护 store 对象读；vLLM/PD 还需要保护 GPU paged block。两者不应混名：

| 机制 | 保护对象 | 目的 |
|---|---|---|
| Store lease | remote object | 防止 eviction/remove |
| GPU block fence | vLLM paged block | 防止异步 DMA 读写复用内存 |
| P2P read lock | peer L1 object | 防止传输中驱逐 |
| transfer id | PD 请求 | 防止乱序/取消泄漏 |

建议在 metrics 中分开上报持有时间与失败次数。

### 4.4 Cache-aware Router 与 Pool 联动

理想路由不是“先路由再查 cache”，而是联合评分：

```text
score = transfer_cost(hit_node)
      + prefill_remaining_cost(node)
      + queue_delay(node)
      + decode_load(node)
      + pool_fetch_cost(prefix)
      + failure_risk(node)
```

LMCache coordinator 和 Mooncake Master/Conductor 都具备部分信息，但当前谁都不应完全拥有所有状态。短期建议 router 只消费只读 directory/event；长期再考虑集中调度。

### 4.5 安全默认值

1. 多租户部署默认 `cache_salt=tenant_id`；  
2. 公共系统提示单独使用公共 cache namespace；  
3. 用户会话 KV 默认 tenant-private；  
4. 跨租户全局池必须显式 opt-in；  
5. 所有 delete 输出审计事件；  
6. cache 命中日志不保存 prompt 明文，但保存 key hash 版本。

### 4.6 硬件能力声明

Transport 层应暴露：

```yaml
transport_capability:
  transport: rdma|nvlink|hccs|tcp|gds
  gpudirect: true|false
  min_alignment_bytes: 4096|2MB
  max_registered_bytes
  nic_bdf
  gpu_bdf
  numa_distance
  supports_failover
  observed_bandwidth_ewma
```

LMCache 可从 NIXL/Mooncake 获取；Mooncake TE/TENT 可直接生成。这样 connector 不必自己猜 NIC/GPU 亲和。

## 5. 未来 12 个月路线图建议

### 5.1 LMCache

1. Coordinator durable event log 与 gap/replay 指标；  
2. L2 adapter capability contract；  
3. StoreJob/Fence 统一模型；  
4. Mooncake L2 的 quota/eviction 权责测试；  
5. CacheGen/CacheBlend 增加 maturity 与 quality benchmark；  
6. Ascend HCCS 通过后端或 platform plugin 明确支持矩阵。

### 5.2 Mooncake

1. Store group completeness API；  
2. replica SLA 与 degraded read 策略；  
3. TENT atomic submit / per-request submit status；  
4. Master sharding/rebalance 压测；  
5. SSD fault injection 工具；  
6. vLLM decode 读池与 multipath load。

### 5.3 vLLM

1. Connector conformance suite；  
2. MultiConnector conflict/cost policy；  
3. KV event quorum；  
4. HMA support matrix；  
5. cache-aware router reference；  
6. request generation 与 store job 的端到端 trace id。

## 6. 最终工程判断

1. Mooncake 应该继续把资源放在 cluster object/store/transport correctness 和 topology scheduling 上，不要为了多引擎 layout 把对象层改复杂。  
2. LMCache 应该继续把资源放在 engine layout、tenant/operator/observability 和独立缓存进程上，不要试图自研全部 RDMA 拓扑。  
3. vLLM 应该把 connector 能力、对象可见性和 block fence 变成可测试 contract，否则每接入一个新模型都会重现边界 bug。  
4. 三个项目的共同瓶颈正在从“能不能搬 KV”转向“错误状态如何被命名、观测、隔离和恢复”。

