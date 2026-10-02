---
prev: /core/h2d-d2h-deepdive
next: /core/admission-control
---

# 异步保存与块保护：谁在守护搬运中的数据

> 「机制对比」系列的第二篇（前一篇：[H2D/D2H 数据搬运对比](/core/h2d-d2h-deepdive)）。同一个命题：**KV 的保存/卸载是异步的——DMA/RDMA 还在读这些 block，引擎就想释放、驱逐或复用它们**。不保护就是读到脏数据：静默损坏，比崩溃更可怕。LMCache 与 Mooncake 给出了两套答案，差异的根因是**信任域的边界**。

## 1. 问题为什么必然出现

三条链路都踩这个坑：

- **offload/写池**：save 异步提交后，请求可能立刻结束、block 被 vLLM 释放重用（写时复制都不救你——目标地址已被别的请求写入）；
- **预取/加载**：数据正从远端读入 host 缓冲，此时 LRU 驱逐把目标页收走；
- **PD 直传**：Decode 正从 Prefill 的显存拉 KV，Prefill 的调度器不知道，把物理 block 分配给了新请求。

共同点：**数据的「逻辑所有权」已经转移给了异步任务，但所有参与方（引擎调度器、缓存驱逐器、远端节点）都不知道**。解法就是让它们知道——问题只是用什么机制知道。

## 2. LMCache：客户端引用计数

**核心原语**：`MemoryObj` 携带 `ref_count` / `pin_count`，谁要用数据谁加计数，归零才可驱逐。

- **Allocate-Evict 循环**：分配失败时取逐出候选，`pin_count > 0` 的对象自动跳过；全被 pin 住则无锁 `sleep(0.1)` 忙等待——阻塞式重试保证关键路径请求**绝不分配失败**；
- **写路径锁链**：`write_backup`（D2H）成功后 `inc_lock_ref(node)`，直到 DMA ack（`_finish_write_through_ack(release_lock=True)`）才释放；L3 写入前 `protect_host()`，直到 L3 写 ack；
- **PD 侧**：receiver 的 staging 空间由 [reservation](/core/p2p-pd) 整体预留（那是空间维度的保护，本篇是数据维度）；
- **两个防泄漏补丁**：`__del__` 析构检查未归零引用并告警；**PinMonitor 在 300s 后强制 unpin**——异步回调丢失（请求取消/异常路径）时引用计数会泄漏，强制超时是兜底；
- **MP 模式的崩溃防护**：TTLLock（带 TTL 的读写锁）——**锁超时自动释放**，防进程崩溃后锁永久卡死。

## 3. Mooncake：服务端租约 + 任务台账

Mooncake 的保护分两层，因为**持有数据的双方分属两个系统**：

**引擎侧（GPU block）：store job 台账**。vLLM Mooncake Store connector 用 `store_job_id` 对所有相关 GPU block 加引用，直到**所有 rank 上报完成**才释放。两个精细点：台账按 job 而非 request id 键控——request id 会因抢占/重试被复用，而 store job 是一次性动作；引用范围是「所有已分配 block」——某个 rank 的保存进度可能落后于调度器视角，读到的范围会超出当前 job 的 token 范围。

**缓存侧（对象）：租约**。`GetReplicaList` 成功即授予 lease（默认 TTL 10s），活跃期间对象不可驱逐/删除；`NeedsLeaseRefresh` 在 **TTL 过半**时触发续约（避免「到期→驱逐→重分配」竞态）；lease 过期读失败而不是返回损坏数据。配套：group TTL 让同 prefix 的多对象尽量 all-or-none 驱逐；**zombie 清理**处理 PutStart 后客户端崩溃（两级超时 30s/10min）；HA 侧 `kLeaderWarmup` 等旧 Leader 租约过期防脑裂。

**vllm-ascend 的变体**（物理缓冲复用场景）：完成跟踪按**物理槽而非逻辑层名**——多个逻辑层映射同一 NPU 地址时，只有「该槽的 Memcache save + Decode 远端读都完成」才允许复用；`AscendMultiConnector` 强制完成提供者先于其他 connector 返回，两条路径都完成才开复用门。

## 4. 根因：信任域决定保护机制

| | LMCache | Mooncake |
|---|---|---|
| 保护原语 | 引用计数（客户端、内存计数器） | 租约 + 台账（服务端、显式协议） |
| 适用拓扑 | 单进程/紧耦合 MP（同一信任域内） | 跨进程/跨节点/多租户（信任域边界外） |
| 崩溃语义 | 进程死了计数也没了（无残留，但缓存丢失） | 客户端死了租约超时自动回收（服务端自愈） |
| 代价 | 极低（一个 int 原子操作） | 协议复杂度：TTL 调参（必须 > 传输超时，vllm-ascend 推荐 master lease_ttl 11s > transfer timeout）、续约流量、超时误杀 |
| 已知弱点 | 引用泄漏靠 PinMonitor 300s 兜底（补丁性质） | best-effort replica 不是持久化；lease 过期时读失败需要上层容忍 |

**面试金句**：保护「搬运中的数据」是所有异步系统的共同命题，方案光谱从引用计数到租约到分布式事务——**选择依据是信任域的边界**：计数器只在「崩溃即消亡」的单进程内成立；一旦持有者和保护者分属不同进程，「超时是唯一可信的真相源」（对方可能崩溃、撒谎或网络分区，只有时间不会）。

## 5. 谁更好

按场景答：引擎内/单节点——LMCache 的计数器是对的（简单、零协议开销，Mooncake 若照搬租约反而引入不必要的超时语义）；跨节点池化/多租户——Mooncake 的租约是对的（服务端无法信任客户端计数器，且需要公平性与自愈）。收敛趋势同 H2D 篇：LMCache 的 Mooncake L2 adapter 里，chunk 在 LMCache 域内用计数、跨到 Mooncake 域即转租约——**同一份数据在信任域边界两侧用两套保护机制**，这本身就是对「根因是信任域」的最好印证。

> 🎯 **面试考察点**（66 家公司真题库）：「异步保存为什么会引入 block pinning？」「lease 和引用计数的区别？」「KV 传输期间 block 被释放怎么办？」——追问链：问题必然性 → 两套机制 → 崩溃语义差异 → 信任域根因。相关真题见 [面试题库](/interview/inference-answers)。
