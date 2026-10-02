---
prev: /core/admission-control
next: /core/vllm-connector

---

# P2P 与 PD 分离

LMCache P2P 只读协议、Mooncake direct PD 与 Store 共享池、LMCache 两条 PD 路线（进程内 NIXL 与 MP reservation），以及完整差异对照表。

> 来源：主报告第 4 章全文。行号级证据见「源码审计要点」一篇。


> **🎯 面试考察点**（66 家公司真题库 · PD 分离）：
> - 「PD 分离机制原理？调度队列如何设计？」「Mooncake 的 PD 方案思路」；
> - 追问链：Push vs Pull → 异构 TP 的 region 对齐 → 四条优化轴 → 容错空白。
> 精答见[面试题库 Q11-Q13](/interview/inference-answers)

## 4. P2P 与 PD 分离

### 4.1 LMCache P2P

LMCache P2P 的定位是“LocalCPU 之上的只读共享补充”，不是分布式写缓存。

流程：

```text
peer lookup-and-lock(keys)
  -> peer 返回 TransferChannelAddress
  -> 本地 prepare L1 write buffer
  -> submit_read(local, remote)
  -> poll succeeded_mask
  -> unlock peer objects
```

关键工程决策：

1. **只读，不远程写**  
   文档明确远程写需要远程分配、存在性和状态检查，失败时可能破坏双方。P2P 因此避免成为弱分布式存储。

2. **L1 read lock**  
   peer 锁定 L1 resident keys，防止 RDMA read 期间被驱逐。

3. **MQ 控制面 + Transfer Channel 数据面分离**  
   MQ 做 lookup/unlock，NIXL/UCX/RDMA 做数据搬运。

4. **超时即 miss/failure**  
   lookup/load 超时返回 bitmap 0，由 prefetch controller 修剪。P2P 是性能优化，不伪装成强一致存储。

5. **生产化状态**  
   官方博客标题为 “From Experimental Feature to Production”，作者包含 Tencent 与 Tensormesh；示例测试显示全复用负载 TTFT 约 4 倍提升。该声明可信，但博客也承认 benchmark 是全复用构造场景，不能外推到所有真实工作负载。

优点：

- 消除负载均衡下的 cache silo；
- 直接读 peer L1，延迟低于写共享存储；
- 部署相对轻；
- 不引入远程写一致性。

缺点：

- 容量仍受集群 DRAM 限制；
- peer 是直接依赖，负载/失败会互相影响；
- 没有全局 Master 统一容量和副本；
- 目录、membership、prefetch 的协调仍偏轻量。

### 4.2 Mooncake P2P / PD Direct

Mooncake Transfer Engine 天然支持 P2P read/write。vLLM 的 `MooncakeConnector` 是 direct PD 路径：

- Prefill worker 将 KV block 地址、block id、remote hostname/port/TP 信息封装为 metadata；
- Decode/P 端通过 Mooncake TransferEngine 发起 RDMA/NVLink/HCCS 等搬运；
- scheduler 与 worker 通过 ZMQ/bootstrap 交换请求和传输状态；
- 支持异构 TP：local TP rank 与 remote TP rank 的 KV region 映射、展开、合并与校验；
- 支持 HMA/hybrid memory allocator 的 group/layer metadata。

`mooncake_connector.py` 中的 `TransferRegion`、`_compute_sender_transfer_plan()`、`_align_transfer_regions()`、`_validate_asymmetric_region_lengths()` 显示它已经处理了工程上最麻烦的几类问题：

1. TP 不同导致的 region 长度差异；
2. PP 不同导致 positional matching 错误；
3. producer cache replicated 与 non-replicated 的选择；
4. 连续 block 能否 coalesce；
5. layer name occurrence matching。

优点：

- 一对一低延迟；
- 不需要中间存储对象；
- 可与异构 TP/PP 组合；
- Mooncake TE 提供多 NIC、failover 和 QoS。

缺点：

- 直接连接有配对生命周期和容量压力；
- producer/consumer 版本与 KV region layout 必须匹配；
- 没有 shared pool 的跨实例去重收益。

### 4.3 Mooncake Store 共享池

`MooncakeStoreConnector` 不是只做 PD 一对一，而是把 KV block 写入全局共享池：

- Scheduler 查询 token/block hash，返回 external hit token 数；
- Worker 在后台 Sending/Receiving 线程中执行 batch put/get；
- Mooncake Store key 以 model、tenant/cache prefix、chunk hash 组织；
- vLLM group id 将同一 prefix chunk 的多个 object 关联成 lifecycle group；
- decode 默认不一定写池，`save_decode_cache=true` 可开启；
- `store_job_id` 与 GPU block pinning 保证异步 DMA 期间 block 不被释放。

优点：

- 跨实例、跨 session、跨节点去重；
- 适合 load balancer、agentic、multi-turn；
- 不要求 producer/consumer 在线配对；
- 可以与 SSD/多副本/lease/pin 结合。

缺点：

- 增加控制面与存储层；
- 对网络、master、对象 metadata 有额外依赖；
- 写池和读池有容量/驱逐成本；
- vLLM 官方文档说明 decode 读池/多路径加载仍在演进。

### 4.4 LMCache PD 分离

LMCache PD 有两条主要路径。

#### 4.4.1 进程内 NIXL PD

`examples/disagg_prefill/1p1d` 与 `xpyd` 使用 NIXL 连接 prefill/decode。NIXL 负责 memory registration、metadata exchange、RDMA/UCX 传输。LMCache connector 负责把 vLLM block id/layout 映射到 transfer descriptor。

#### 4.4.2 MP 模式与 reservation-based admission

`docs/design/v1/pd_async_reservation_design.md` 描述了一个非常实际的死锁：

```text
chunked prefill 下，N 个请求各自分配部分 staging chunk，
receiver buffer 满，谁也拿不到剩余空间，谁也无法完成。
```

解决方案：

1. Receiver 在首个 batch 预留 `total_chunks`；
2. 空间不足则等待，不让请求部分进入；
3. 后续 batch 保证可分配；
4. abort/failure 全量回滚；
5. sender 用 physical staging buffer flow control；
6. ProxyNotif 需要 `completed_chunks == total_chunks` 且最后一 batch 完成。

这是典型数据库式 admission control 思维：不能只看当前 batch 成功，还要保证一个逻辑请求整体能完成。

#### 4.4.3 MP CUDA IPC 与 engine-driven

LMCache MP 模式通过独立 daemon 管理 KV：

- 引擎 worker 崩溃不一定带走 cache；
- CUDA IPC 用于 NVIDIA 低延迟路径；
- SHM/pickle 用于非 CUDA/通用 fallback；
- heartbeat、uuid instance id、reaper 防止 PID 复用导致 IPC handle 错绑；
- worker liveness 与 re-register 解决短分区和 OOM kill。

这在工程可靠性上比简单 connector 更进一步，但也增加部署复杂度。

### 4.5 P2P/PD 差异总结

| 维度 | LMCache | Mooncake |
|---|---|---|
| P2P 目标 | 多实例读共享 CPU L1 | Transfer Engine 通用点对点，PD 直传/共享池皆可 |
| 写语义 | peer L1 只读 | RDMA read/write 通用；Store 提供 PutStart/PutEnd |
| 主数据设备 | peer CPU DRAM | GPU/NPU HBM、CPU DRAM、SSD |
| 控制面 | MP MQ + coordinator/controller | connector bootstrap + Mooncake Master/TE metadata |
| PD 重点 | chunk admission、staging flow control、abort rollback | direct transfer plan、heterogeneous TP/PP、store job pinning |
| 全局复用 | P2P/remote backend 可选 | Store 全局 key/prefix pool 是一等能力 |
| 容量假设 | peer DRAM 汇总 | DRAM/VRAM/SSD/NVMe pool |
| 生产声明 | P2P 已宣布转生产 | Kimi 生产、vLLM 官方 x Mooncake Store 案例广泛 |

---

## 附：PD 状态交接的优化地图（四条轴 + 容错空白）

> 来源：ForceInjection 系列《PD 状态交接优化四条轴》，证据等级 B/C。这个框架把上文 LMCache/Mooncake 的所有 PD 机制放进统一坐标系。

状态交接（handoff）覆盖四层：KV 字节过线（传输）、元数据与提交顺序（时序）、D 侧预分配与就绪判定（状态生效）、缓存归属与故障责任（生命周期）。

| 轴 | 问题 | 代表手段 | 证据成熟度 |
|---|---|---|---|
| 压缩 | 字节变少 | 线上量化（CacheGen 4.3×）、稀疏感知传输 | 组件有实测，系统级缺 |
| 重叠 | 传输躲进计算 | chunk-wise 流水线、元数据先行 | 论文+引擎源码 |
| 复用 | 干脆不传 | cache-aware 路由、去重、混合路径 | 多方实测，结论有争议 |
| 隔离 | 传输不伤计算 | 流量规划、路径优化 | 公开材料最少 |
| 容错 | 故障后保留成果 | 状态外置、副本租约 | 仓库与业界均未覆盖 |

三方在四条轴上的位置：

- **重叠**：SGLang PD 默认 `enable_overlap`，每完成一个 prefill chunk 即增量发送（prefill.py:888-892 附近），控制面（bootstrap+ZMQ）与数据面分离 + DecodePreallocQueue 预分配；LMCache 层级流水线见「存储引擎中的数据变形」篇。
- **复用**：Mooncake Conductor 对每个 prefill 实例估计「缓存传输+排队+prefill 执行」三段和，选总 TTFT 最小的实例；LMCache PD backend 用 `already_sent_indexes` 做会话内去重。正方证据：llm-d 实测（8 pod、150 客户共享 6000-token 前缀）cache-aware 路由 8730 tok/s vs 负载均衡 4429 tok/s，TTFT p90 0.542s vs 94.9s。反方证据（Anyscale）：最大化复用会导致 request herding（缓存亲和把同类请求吸到同一实例形成热点）与 session 级不均衡——平衡复用与负载优于最大化复用。
- **容错（三方共同空白）**：组件语义齐了一半（Mooncake Store 多副本 best-effort/租约 gc_ttl_ms/soft-pin；SGLANG_DISAGGREGATION_WAITING_TIMEOUT 默认 300s 防提前回收），但「D 实例故障时把会话迁到另一个 D、从共享 Store 重新装载 KV 而非重新 prefill」这最后一环没有公开端到端实现。对以 KV Store 为中心的形态，「前四条轴节省的成本，在每次故障时都要重新投入」。
- 独到提醒：压缩与重叠两轴收益互相挤压——字节数降下来后重叠窗口也变小，两轴不是独立加法。

容量与传输下限的粗账（GLM-5.3，MLA+DSA，78 层，BF16）：每 token 约 90.5 KB；1000 用户 × 20 session × 32K ≈ 55 TB；单 32K 请求传输约 2.8 GB，200G RDMA 跑满约 110 ms——这是 PD 传输时间的物理下限。

> **PD 传输开销占比速查**（一研 PD 分离篇）：KV 传输占 Prefill 计算时间的比例——LLaMA3-8B 128K 约 2GB 占 <1%；70B 约 40GB 占 ~5%；DeepSeek-V3 约 80GB 占 ~8%。占比随模型规模与上下文长度上升而增大——PD 分离的收益也同步放大（传输线性 vs 计算超线性）。
