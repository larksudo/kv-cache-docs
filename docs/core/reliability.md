---
prev: /core/vllm-connector
next: /core/control-data-plane
---

# 可靠性与一致性：把 KV Cache 管理当数据库看

如果把 KV cache 管理当成数据库，评估维度就不只是带宽，而是事务边界、可见性、并发控制、租约、崩溃恢复、复制、容量配额、故障隔离和可观测。本篇逐项拆解两个系统的取舍。

> 来源：主报告第 7 章全文。

## 7. 把 KV cache 管理当数据库看：可靠性与一致性

如果把 KV cache 管理当成数据库，评估维度就不只是带宽，而是：事务边界、可见性、并发控制、租约、崩溃恢复、复制、容量配额、故障隔离和可观测。

### 7.1 Mooncake 的数据库式设计

#### 7.1.1 写可见性

Mooncake Store 用 `PutStart`/`PutEnd`/`PutRevoke` 分离“分配空间、写副本、发布对象”。文档明确这防止其他 client 读到 partial value。

数据库视角：这是对象级 write commit protocol。虽然不是可写 mutable row，但足够适合 immutable KV cache。

#### 7.1.2 读稳定性

`ExistKey` 或 `GetReplicaList` 成功后授予 lease。lease 活跃时对象不会被 remove/remove-all/evict。若 lease 在 `Get` 完成前过期，读失败而不是返回损坏数据。

数据库视角：这不是两阶段锁，而是面向 immutable object 的 optimistic read lease。它避免了长事务，但把过载时的行为设计成“读失败”，这是一个正确但需要上层容忍的选择。

#### 7.1.3 对象分组

group id 让同一 prefix chunk 的多个对象共享 group TTL，eviction 尽量 all-or-none。文档也明确 grouping 是 lifecycle hint，不是 transactional guarantee。

这是很清醒的设计。一个逻辑 prefix 可能拆成 K/V、TP/PP shard、辅助 metadata；完全原子提交成本很高。Mooncake 选择“共享 TTL + best-effort all-or-none eviction”，承认边界。

#### 7.1.4 pin 与配额

Mooncake 有：

- soft pin：热点对象低优先级驱逐，TTL 后降级；
- hard pin：永不驱逐，创建时设置，只能强制删除；
- tenant quota：写入前扣配额，失败触发 tenant-scoped eviction；
- policy store 持久化，恢复时重建 usage。

数据库视角：这些是资源治理和优先级语义，不是 ACID，但对多租户 cache pool 非常关键。

#### 7.1.5 崩溃与恢复

Mooncake 有多层恢复：

1. Master snapshot：fork COW 拷贝 metadata、segment、allocator state；  
2. HA OpLog：standby apply，恢复操作顺序；  
3. etcd leader election；elected node 还要 standby catch-up、export/validate promotion context、restore/revalidate leadership；  
4. zombie object cleanup：PutStart 后 PutEnd/PutRevoke 缺失，用 preemption 和空间回收兜底；  
5. SSD restart recovery：seq 和 CRC 处理 post-checkpoint 写。

数据库视角：这套设计已经远超普通 cache，接近分布式存储控制面。但它也不是完整强一致数据库：快照之间的 OpLog、best-effort replication、soft pin 不持久化，都说明系统在“性能、可用性和完整性”之间做了工程取舍。

### 7.2 LMCache 的可靠性设计

LMCache 不是没有可靠协议，而是把可靠性拆到不同作用域。

#### 7.2.1 本地 L1/L2

StoreController 在 L2 store 期间持有 L1 read lock；PrefetchController 在 lookup/load 期间持有 L2 lock 和 L1 write lock，写入完成后原子转 read lock。锁矩阵明确，能避免常见 eviction/read race。

数据库视角：这是缓存一致性的基础协议，但作用域主要在单实例 StorageManager 与 adapter。

#### 7.2.2 P2P read

P2P 使用 lookup-and-lock、RDMA read、unlock。只读、超时退化成 miss。它避免了分布式写提交，也没有承诺强一致。

数据库视角：这是一种“peer cache read-through”，不是复制协议。

#### 7.2.3 MP 独立进程

MP daemon 的可靠性设计非常工程化：

- uuid instance id，防止 PID 复用；
- heartbeat lazy start、registration grace、reap timeout；
- leaf lock 保护 context/strategy dict；
- 短分区不重建，恢复 edge 才 re-register；
- clean shutdown 停心跳再 unregister，避免 ghost。

数据库视角：这是进程级 lifecycle protocol，尤其适合推理引擎频繁升级/OOM/重启的场景。它解决的是“缓存宿主与计算宿主故障隔离”，这是 Mooncake Store 不直接针对的问题。

#### 7.2.4 MP coordinator

Fleet coordinator 有 incarnation fencing、sequence dedup、gap detection、event broadcaster、key directory、usage manager、quota 和 eviction controller。容量声明用 `(incarnation, capacity_revision)` 分组，配置事件可以修复丢失。

但文档也明确：

- registry ephemeral，coordinator 重启后重建；
- directory/usage/LRU 从事件流重建；
- durable state 需要外部存储；
- replay integration、registry fencing integration 仍是 follow-up。

数据库视角：这是 CQRS/事件投影风格。设计清晰，但离 durable replicated control plane 还有距离。

#### 7.2.5 PD reservation

`ReservationManager` 保证 receiver staging buffer 能完整容纳一个请求的 total chunks，sender 用 physical staging flow control，abort/failure 回滚所有 chunk。这解决了 throughput 系统里最隐蔽的 admission deadlock。

数据库视角：这是 reservation-based admission control， correctness 价值高于吞吐优化。

### 7.3 数据库视角评分

评分不是综合排名，而是针对“作为 KV cache 数据库”的成熟度。

| 维度 | LMCache | Mooncake | 更强 |
|---|---:|---:|---|
| 单对象可见性 | 依赖本地写入与 adapter 语义 | PutStart/PutEnd 保证 complete 后可见 | Mooncake |
| 读期间防驱逐 | L1/L2/P2P 锁 | lease/group TTL | 平手，作用域不同 |
| 多副本 | 部分后端无明确副本协议 | replica best-effort + slice 不同 segment | Mooncake |
| 强一致性 | 弱 | 对象 immutable complete 后一致；不是 mutable ACID | Mooncake |
| 容量治理 | per-instance + fleet quota/pin | tenant quota + preferred segment + soft/hard pin | LMCache 多租户更细，Mooncake pool 语义更强 |
| 控制面恢复 | coordinator 事件流重建，durable replay 未完成 | snapshot + OpLog + election + standby catch-up | Mooncake |
| 数据面故障 | NIXL/adapter 超时/失败；MP health | multi-NIC、rail cooldown、transport failover | Mooncake |
| 计算进程故障隔离 | MP daemon/reaper/IPC recovery | Store 与引擎可解耦，但 connector 集成仍要处理 worker | LMCache |
| 持久化 | local disk/GDS/persist adapter | DRAM/SSD/metadata snapshot/OpLog | Mooncake |
| 异常语义 | miss/failure 可退化 | 读失败、PutStart 清理、failover | 各有取舍；Mooncake 更显式 |

**总体判断**：Mooncake 更接近“分布式缓存存储引擎”，可靠性协议更完整；LMCache 更接近“引擎外置缓存管理框架”，计算侧故障隔离和插件运维更强。

### 7.4 性能设计差异

#### 7.4.1 LMCache 性能策略

LMCache 的性能主要来自：

1. prefix hit 后减少 prefill；
2. layerwise load/save 与计算重叠；
3. CUDA IPC/SHM 减少进程间拷贝；
4. GDS/NIXL 减少 CPU staging；
5. lazy offload 延迟从 GPU 到 CPU 的复制；
6. 多 L2 adapter 并行 lookup/store；
7. MP 模式把缓存控制线程与引擎解耦。

它的优势是路径短、插件多、在单机多进程和小规模集群里容易获得收益。

#### 7.4.2 Mooncake 性能策略

Mooncake 的性能主要来自：

1. 全局 Store 池提高跨实例命中率；
2. RDMA zero-copy、GPUDirect、multi-NIC slice；
3. NUMA/PCIe/NIC topology-aware path selection；
4. endpoint pool 与 QoS；
5. 对象 slice 分散，配合 multi-rail；
6. SSD io_uring/O_DIRECT/CRC/4KB 对齐；
7. PD direct + Store pool 的 MultiConnector 组合。

它的优势是能吃到物理拓扑和集群级聚合带宽，适合大规模多实例。

#### 7.4.3 带宽数据解读

公开数据不能直接互比，因为负载、模型、硬件和指标不同：

| 来源 | 数据 | 解读 |
|---|---|---|
| Mooncake TE README | 40GB 数据在 4x200G/8x400G RoCE 分别达 87GB/s、190GB/s，约为 TCP 的 2.4x/4.6x | 体现 multi-NIC TE 的传输层价值 |
| Mooncake FAST25 | Kimi real workload 下 SLO 内请求数提升 75%；调度器结合 prefix hit、队列与 TTFT | 体现 KV cache-aware scheduling |
| vLLM x Mooncake Store blog | Codex agentic traces：hit rate 1.7% -> 92.2%，throughput 3.8x，P50 TTFT 46x 改善；12-60 GPU 近线性 | 证明跨实例全局池对 agent/multi-turn 的价值 |
| LMCache P2P blog | 全 workload reuse 下 TTFT 2.028s -> 0.490s，round time 约 5x 改善 | 只说明上限复用场景 |
| LMCache MP blog | MoE 推理性能提升最高 10x | 需要看负载、命中与基线，不能泛化 |

更可靠的结论是：在 agent/multi-turn/load-balanced 场景，全局池的命中率收益往往超过单纯传输优化；在一次性大 prompt/PD 场景，RDMA topology 和 direct transfer 才是决定因素。

---

---

## 附：故障路径矩阵（V2）

### 3.1 LMCache

| 故障 | 系统行为 | 风险 | 审计点 |
|---|---|---|---|
| vLLM worker crash | MP daemon 保留缓存；in-process 模式缓存随进程退出 | IPC handle 可能泄漏 | reaper timeout、registration grace、UNREGISTER 顺序 |
| PID 复用 | 使用 uuid-derived instance id | 旧协议混部可能失败 | instance id 是否全链路传递 |
| 短网络分区 | heartbeat 失败进入 unhealthy；恢复 edge re-register | 短分区不应重建 context | recover callback 与 health event edge |
| L2 store 失败 | best-effort，不重试；L1 锁清理 | 缓存层缺失 | `L2StoreResult.bytes_transferred()` 与清理 |
| L2 lookup 后 eviction | lookup-and-lock/pin | 未 unlock 会占用对象 | phase1/phase2 unlock、shutdown cleanup |
| P2P lookup/load 超时 | 退化为 miss/failure | 无法复用，不损坏数据 | timeout、bitmap、unlock |
| RDMA read 中 peer 驱逐 | peer lookup-and-lock 应保护 | 控制面过期/竞态会损坏读 | lock 生命周期与 task ledger |
| PD staging 满 | receiver reservation 阻塞新请求 | 死锁避免但吞吐下降 | total_chunks、rollback、abort |
| Coordinator 重启 | registry/directory/usage 重建 | 控制面短时不准 | durable replay、gap detection、capacity revision |
| 后端 delete 竞态 | 部分后端 best-effort | read 可能 miss 或 recreate key | listener、pin、delete contract |

### 3.2 Mooncake

| 故障 | 系统行为 | 风险 | 审计点 |
|---|---|---|---|
| Put 中 client crash | zombie PutStart 用 preemption/release timeout 清理 | 空间占用、同 key 写阻塞 | discard/release timeout、preemption 空间 |
| Get 中 lease 过期 | 读失败，不返回损坏数据 | 上层必须重试/重算 | lease TTL、group TTL、读重试策略 |
| replica 部分失败 | best-effort replica，至少一个成功即可返回 | 不能把 replica 当强持久化 | PutStart/PutEnd、replica selection |
| Master crash | snapshot/OpLog/standby catch-up/election | 新 leader 未验证不得服务 | promotion context、empty context 校验 |
| soft pin 恢复 | 不持久化，恢复后降级 | 热点可能被立即驱逐 | operator 是否重建 pin |
| segment client crash | master heartbeat 检测并处理 | 数据不可用 | segment rejoin、metadata 清理 |
| RDMA WC error | rail markFailed + task resubmit | 可恢复故障被放大 | error window、cooldown、budget |
| endpoint/submit 部分失败 | 文档明确 submit-stage 不自动 failover | 任务失败、可能部分已入队 | merge requests、atomic submit 能力 |
| NIC 不能 GPUDirect | TENT 跳过不可用 NIC | 选路错误会降级或失败 | GPUDirect capability、device rotation |
| vLLM block 被复用 | store_job_id + block reference 防护 | job ledger 错误仍会读错数据 | rank completion、partial tail、resume offset |

### 3.3 vLLM

| 故障 | 行为 | 风险 |
|---|---|---|
| preemption before external save | producer 默认 recomputed，除非 connector 声明可靠 delivery | 过度重算或读已释放 block |
| partial tail offload | scheduler 需要与 connector event fence CoW block | 边界 token 不一致 |
| hybrid group | 支持 HMA 的 connector 才能处理多 group finish | 不支持的 connector 必须禁用/限制 |
| connector version mismatch | scheduler/worker metadata 不兼容 | load/save 卡死或状态丢失 |
| external hit 过大 | block allocation 可能不足 | scheduler 与 connector 数量不一致 |
