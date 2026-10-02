---
prev: /core/reliability
next: /core/lmcache-internals
---

# 控制面与数据面设计

LMCache 的控制面是「多个专用控制器组合」，Mooncake 是「存储 Master + 传输资源调度器」。本篇对比两者的控制面形态、全局目录、恢复模型与故障面。

> 来源：主报告第 8 章。

## 8. 控制面与数据面设计

### 8.1 LMCache

#### 8.1.1 进程内模式

```text
vLLM Scheduler
  -> lookup/prefix hit, connector metadata
vLLM Worker
  -> GPU connector, storage manager, adapter
```

控制面与数据面在同一引擎进程边界内交互。优点是延迟低、状态简单；缺点是引擎与缓存 fate-sharing。

#### 8.1.2 MP 模式

```text
vLLM Scheduler Adapter -> MP Scheduler MQ
vLLM Worker Adapter    -> MP Worker MQ
MP Cache Server        -> GPU IPC / SHM / engine-driven transfer
                        -> Storage Manager / L2 adapters
MP Coordinator         -> registry/events/quota/pin/prefetch
```

这是 LMCache 最重要的控制面重构：

- 缓存生命周期独立于引擎进程；
- MQ 协议显式区分 scheduler/worker；
- heartbeat/reaper 管理 worker liveness；
- coordinator 只做 fleet policy，不进数据面。

MP coordinator 采用 REST + 事件流：

1. MP server 注册/心跳；
2. cache events 通过 `POST /events` 上报；
3. EventGate 用 incarnation/seq 过滤；
4. KeyDirectory、usage、eviction controller 消费事件；
5. operator 可 pin/delete/prefetch。

这是现代控制面形态，但状态恢复和持久化仍是演进重点。

#### 8.1.3 LMCache 的边界

LMCache 控制面较分散：

- engine connector 管 request；
- storage manager 管 L1/L2；
- P2P controller 管 peer；
- MP coordinator 管 fleet；
- cache controller 管 operator；
- observability 独立收集。

这在插件生态里是合理的，但排障时必须理解多个协议和状态机。

### 8.2 Mooncake

#### 8.2.1 Transfer Engine

```text
Application
  -> submitTransfer / getTransferStatus
    -> TransferEngineImpl
      -> Topology / metadata
      -> MultiTransport / Transport Selector
        -> RdmaTransport / Nvlink / HIP / Ascend Direct / TCP / GDS
```

控制面包括 segment metadata、buffer registration、topology matrix、endpoint/session 状态。数据面是异步 batch，绕开 Master/Coordinator。

TENT 进一步加入：

- priority；
- intent type；
- device mask；
- transport preference；
- QoS queue；
- EWMA bandwidth；
- rail cooldown；
- cross-transport failover。

这让 Mooncake 的数据面不只是“发 RDMA”，而是一个可调度的网络资源池。

#### 8.2.2 Mooncake Store

```text
Client
  -> PutStart / PutEnd / GetReplicaList
Master
  -> allocation, metadata, lease, eviction, snapshot
Client/Segment
  -> TransferEngine data path
SSD worker
  -> offload/restore heartbeat
```

Master 是控制面中心，负责空间、对象状态和租约。数据面 client-to-client 直传，不经过 Master。这个取舍很经典：控制面集中便于全局策略，数据面分布式避免 Master 成为带宽瓶颈。

HA 模式下 Master 使用 etcd election、standby catch-up 和 promotion context。缓存元数据恢复远比简单 in-memory map 重。

### 8.3 对比

| 方面 | LMCache | Mooncake |
|---|---|---|
| 控制面形态 | engine connector + MP MQ + REST coordinator | Master + TE metadata + connector bootstrap |
| 全局目录 | key directory/event projection | object/segment metadata |
| 恢复模型 | coordinator 重启重建，durable replay 演进中 | snapshot + OpLog + leader election |
| 数据面 | NIXL/UCX/GDS/SHM/IPC/Mooncake adapter | 自研 TE + multi-transport |
| 拓扑感知 | 少量在 connector/后端内；主要委托 | CPU/GPU/NIC/NUMA/transport 显式建模 |
| 多租户 | cache salt/quota/operator API | tenant quota/pin/preferred segment |
| 策略执行 | L1/L2/P2P/fleet 多控制器 | Master policy + TE QoS |
| 故障面 | worker/daemon/coordinator/adapter | TE/endpoint/segment/Master |

LMCache 的控制面是“多个专用控制器组合”，Mooncake 是“存储 Master + 传输资源调度器”。前者灵活，后者一致性强。

---

---

## 附：控制面/数据面分离的架构模式理论（一研篇）

Mooncake 的分离不是孤例——它与 SDN（流表项）、服务网格（Pilot-Envoy 代理规则）、云存储（S3-Ceph 对象位置）同构：**四个系统的控制器都不碰数据**。模式化的理论表述：

- **三力权衡**：扩展性 vs 一致性（强一致需二阶段提交）、简单性 vs 解耦度、延迟 vs 可观测性（分离后需心跳/遥测弥补）；
- **描述符设计五原则**：极小、只有地址没有内容、一次性消费、可序列化、自包含——缺字段导致数据面回查控制器，就破坏了分离保证；
- **三种反模式**：代理中转（控制器变收费站）、数据面高频回查、控制器感知数据内容；
- 量化直觉：控制器每秒几千次元数据操作 vs 数据面每秒数 TB——**六个数量级的差**是两者互不干扰的根源；
- 渐进引入三阶段：中转 → 描述符解耦 → 直连（Mooncake 的 PutStart/Transfer/PutEnd 正是第二阶段的教科书实现）。
